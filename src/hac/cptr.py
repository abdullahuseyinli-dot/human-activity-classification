"""Camera-compensated part--trajectory residual model for person activity clips.

The model deliberately treats temporal evidence as a correction to a frozen static
fallback.  Appearance, trajectory, body-region, pose, and semantic-specialist
experts expose separate reliability values and separate posture/motion gates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from hac.vcoco_v3_neural import decode_factorized_logits
from hac.vcoco_v3_temporal import StaticIdentifiabilityStudent, TemporalFactorizedTeacher


class ExpertEvidence(NamedTuple):
    features: torch.Tensor
    reliability: torch.Tensor


@dataclass(frozen=True)
class CPTROutput:
    probabilities: torch.Tensor
    posture_logits: torch.Tensor
    motion_logits: torch.Tensor
    static_probabilities: torch.Tensor
    legacy_probabilities: torch.Tensor
    static_posture_logits: torch.Tensor
    static_motion_logits: torch.Tensor
    expert_names: tuple[str, ...]
    posture_gates: torch.Tensor
    motion_gates: torch.Tensor
    expert_reliability: torch.Tensor
    temporal_residual: torch.Tensor
    learned_temporal_residual: torch.Tensor
    transition_logit: torch.Tensor
    gait_logits: torch.Tensor
    quality_logit: torch.Tensor
    evidence_features: torch.Tensor
    unknown_score: torch.Tensor


class SignedTimeEncoding(nn.Module):
    """Sinusoidal encoding centred on the labelled frame."""

    def __init__(self, feature_dim: int, maximum_length: int = 32) -> None:
        super().__init__()
        if feature_dim < 2 or maximum_length < 2:
            raise ValueError("Time encoding dimensions must be at least two")
        self.feature_dim = int(feature_dim)
        self.maximum_length = int(maximum_length)

    def forward(self, values: torch.Tensor, centre_index: int | torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("Temporal values must have shape [batch, time, features]")
        batch, steps, dimensions = values.shape
        if dimensions != self.feature_dim or steps > self.maximum_length:
            raise ValueError("Temporal values do not match the configured time encoding")
        if isinstance(centre_index, int):
            centres = torch.full((batch,), centre_index, device=values.device, dtype=torch.long)
        else:
            centres = centre_index.to(device=values.device, dtype=torch.long)
            if centres.shape != (batch,):
                raise ValueError("centre_index must be scalar or have shape [batch]")
        positions = torch.arange(steps, device=values.device)[None, :] - centres[:, None]
        positions = positions.to(dtype=torch.float32)
        scales = torch.exp(
            torch.arange(0, dimensions, 2, device=values.device, dtype=torch.float32)
            * (-math.log(10_000.0) / dimensions)
        )
        encoding = torch.zeros(batch, steps, dimensions, device=values.device)
        encoding[:, :, 0::2] = torch.sin(positions[:, :, None] * scales)
        odd = encoding[:, :, 1::2]
        odd.copy_(torch.cos(positions[:, :, None] * scales[: odd.shape[-1]]))
        return values + encoding.to(dtype=values.dtype)


def _ensure_valid_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("A temporal mask must be boolean with shape [batch, time]")
    if torch.all(mask.any(dim=1)):
        return mask
    repaired = mask.clone()
    missing = ~repaired.any(dim=1)
    repaired[missing, repaired.shape[1] // 2] = True
    return repaired


class CenterQueryEncoder(nn.Module):
    """Encode a clip while anchoring the representation to its labelled centre."""

    def __init__(
        self,
        input_dim: int,
        *,
        model_dim: int,
        layers: int,
        heads: int,
        feedforward_dim: int,
        dropout: float,
        maximum_length: int = 32,
    ) -> None:
        super().__init__()
        if input_dim < 1 or model_dim < 1 or layers < 1 or heads < 1:
            raise ValueError("Centre-query dimensions must be positive")
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.time_encoding = SignedTimeEncoding(model_dim, maximum_length)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.query_attention = nn.MultiheadAttention(
            model_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def encode_tokens(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        centre_index: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = _ensure_valid_mask(valid_mask)
        projected = self.input_projection(values)
        encoded = self.encoder(
            self.time_encoding(projected, centre_index),
            src_key_padding_mask=~mask,
        )
        return encoded, mask

    def pool_tokens(
        self,
        encoded: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        centre_index: int | torch.Tensor,
    ) -> torch.Tensor:
        mask = _ensure_valid_mask(valid_mask)
        if isinstance(centre_index, int):
            query = encoded[:, centre_index : centre_index + 1]
        else:
            indices = centre_index.to(encoded.device, dtype=torch.long)
            query = encoded[torch.arange(len(encoded), device=encoded.device), indices][:, None]
        attended, _ = self.query_attention(
            query,
            encoded,
            encoded,
            key_padding_mask=~mask,
            need_weights=False,
        )
        return self.output_norm(query[:, 0] + attended[:, 0])

    def forward(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        centre_index: int | torch.Tensor,
    ) -> ExpertEvidence:
        encoded, mask = self.encode_tokens(
            values,
            valid_mask,
            centre_index=centre_index,
        )
        features = self.pool_tokens(
            encoded,
            mask,
            centre_index=centre_index,
        )
        return ExpertEvidence(features=features, reliability=valid_mask.float().mean(dim=1))


class TrajectoryEncoder(nn.Module):
    """Encode compact per-frame kinematics and a fixed trajectory summary."""

    def __init__(
        self,
        sequence_dim: int,
        summary_dim: int,
        *,
        model_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(sequence_dim, summary_dim, model_dim, heads) < 1:
            raise ValueError("Trajectory dimensions must be positive")
        self.sequence_projection = nn.Sequential(
            nn.LayerNorm(sequence_dim),
            nn.Linear(sequence_dim, model_dim),
            nn.GELU(),
        )
        self.temporal_convolution = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1),
        )
        self.summary_projection = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.MultiheadAttention(
            model_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        summary: torch.Tensor,
        valid_mask: torch.Tensor,
        camera_quality: torch.Tensor,
    ) -> ExpertEvidence:
        mask = _ensure_valid_mask(valid_mask)
        sequence_values = self.sequence_projection(sequence)
        convolved = self.temporal_convolution(sequence_values.transpose(1, 2)).transpose(1, 2)
        sequence_values = sequence_values + convolved
        query = self.summary_projection(summary)[:, None]
        attended, _ = self.attention(
            query,
            sequence_values,
            sequence_values,
            key_padding_mask=~mask,
            need_weights=False,
        )
        features = self.output(torch.cat((query[:, 0], attended[:, 0]), dim=1))
        if camera_quality.shape != valid_mask.shape:
            raise ValueError("camera_quality must align with the trajectory sequence")
        visible_quality = camera_quality * valid_mask.float()
        reliability = visible_quality.sum(dim=1) / valid_mask.float().sum(dim=1).clamp_min(1.0)
        return ExpertEvidence(features=features, reliability=reliability.clamp(0.0, 1.0))


class PartArticulationEncoder(nn.Module):
    """Pool confidence-weighted body-region tokens across space and time."""

    def __init__(
        self,
        input_dim: int,
        part_count: int,
        *,
        model_dim: int,
        heads: int,
        dropout: float,
        maximum_length: int = 32,
    ) -> None:
        super().__init__()
        if min(input_dim, part_count, model_dim, heads) < 1:
            raise ValueError("Part encoder dimensions must be positive")
        self.part_count = int(part_count)
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.part_embedding = nn.Parameter(torch.empty(part_count, model_dim))
        nn.init.normal_(self.part_embedding, std=model_dim**-0.5)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(
            spatial_layer,
            num_layers=1,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.temporal_encoder = CenterQueryEncoder(
            model_dim,
            model_dim=model_dim,
            layers=1,
            heads=heads,
            feedforward_dim=model_dim * 2,
            dropout=dropout,
            maximum_length=maximum_length,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        confidence: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        centre_index: int | torch.Tensor,
    ) -> ExpertEvidence:
        if tokens.ndim != 4:
            raise ValueError("Part tokens must have shape [batch, time, parts, features]")
        if tokens.shape[2] != self.part_count or confidence.shape != tokens.shape[:3]:
            raise ValueError("Part-token confidence does not match the declared body regions")
        batch, steps, parts, _ = tokens.shape
        values = self.projection(tokens) + self.part_embedding[None, None]
        flat_values = values.reshape(batch * steps, parts, -1)
        flat_confidence = confidence.reshape(batch * steps, parts).clamp(0.0, 1.0)
        spatial_mask = flat_confidence <= 0.0
        all_missing = spatial_mask.all(dim=1)
        if torch.any(all_missing):
            spatial_mask = spatial_mask.clone()
            spatial_mask[all_missing, 0] = False
            flat_values = flat_values.clone()
            flat_values[all_missing, 0] = 0.0
        encoded = self.spatial_encoder(flat_values, src_key_padding_mask=spatial_mask)
        weights = flat_confidence.masked_fill(spatial_mask, 0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        frame_values = (encoded * weights[:, :, None]).sum(dim=1).reshape(batch, steps, -1)
        frame_valid = valid_mask & (confidence.max(dim=2).values > 0.0)
        temporal = self.temporal_encoder(
            frame_values,
            frame_valid,
            centre_index=centre_index,
        )
        reliability = (confidence.mean(dim=(1, 2)) * valid_mask.float().mean(dim=1)).clamp(
            0.0, 1.0
        )
        return ExpertEvidence(features=temporal.features, reliability=reliability)


class PoseArticulationEncoder(nn.Module):
    """Optional confidence-aware joint stream for future or externally cached pose."""

    def __init__(
        self,
        joint_count: int,
        *,
        model_dim: int,
        heads: int,
        dropout: float,
        maximum_length: int = 32,
    ) -> None:
        super().__init__()
        if joint_count < 1:
            raise ValueError("joint_count must be positive")
        self.joint_count = int(joint_count)
        self.coordinate_projection = nn.Sequential(
            nn.Linear(3, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.joint_embedding = nn.Parameter(torch.empty(joint_count, model_dim))
        nn.init.normal_(self.joint_embedding, std=model_dim**-0.5)
        self.part_encoder = PartArticulationEncoder(
            model_dim,
            joint_count,
            model_dim=model_dim,
            heads=heads,
            dropout=dropout,
            maximum_length=maximum_length,
        )

    def forward(
        self,
        pose: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        centre_index: int | torch.Tensor,
    ) -> ExpertEvidence:
        if pose.ndim != 4 or pose.shape[2:] != (self.joint_count, 3):
            raise ValueError("pose must have shape [batch, time, joints, 3]")
        confidence = pose[..., 2].clamp(0.0, 1.0)
        coordinates = torch.nan_to_num(pose[..., :2])
        values = torch.cat((coordinates, confidence[..., None]), dim=-1)
        tokens = self.coordinate_projection(values) + self.joint_embedding[None, None]
        return self.part_encoder(
            tokens,
            confidence,
            valid_mask,
            centre_index=centre_index,
        )


class ResidualHead(nn.Module):
    def __init__(self, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.posture = nn.Linear(model_dim, 2)
        self.motion = nn.Linear(model_dim, 2)
        nn.init.zeros_(self.posture.weight)
        nn.init.zeros_(self.posture.bias)
        nn.init.zeros_(self.motion.weight)
        nn.init.zeros_(self.motion.bias)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(values)
        return self.posture(features), self.motion(features)


def probability_diagnostics(probabilities: torch.Tensor) -> torch.Tensor:
    ordered = probabilities.sort(dim=1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)
    return torch.stack(
        (
            probabilities.max(dim=1).values,
            entropy,
            ordered[:, -1] - ordered[:, -2],
            probabilities[:, 1] + probabilities[:, 2],
        ),
        dim=1,
    )


class PartTrajectoryResidualNetwork(nn.Module):
    """Reliability-gated residual mixture anchored to established v3 models."""

    def __init__(
        self,
        static_fallback: StaticIdentifiabilityStudent,
        legacy_temporal: TemporalFactorizedTeacher,
        *,
        frame_input_dim: int,
        quality_dim: int,
        trajectory_sequence_dim: int,
        trajectory_summary_dim: int,
        model_dim: int = 192,
        layers: int = 2,
        heads: int = 4,
        feedforward_dim: int = 384,
        dropout: float = 0.15,
        use_short: bool = False,
        use_long: bool = False,
        use_trajectory: bool = False,
        use_parts: bool = False,
        part_input_dim: int = 768,
        part_count: int = 7,
        use_pose: bool = False,
        pose_joint_count: int = 17,
        use_siglip: bool = False,
        siglip_dim: int = 768,
        expert_roles: Mapping[str, str] | None = None,
        modality_dropout: float = 0.0,
        stochastic_depth: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= modality_dropout < 1.0 or not 0.0 <= stochastic_depth < 1.0:
            raise ValueError("modality dropout and stochastic depth must be in [0, 1)")
        self.static_fallback = static_fallback
        self.legacy_temporal = legacy_temporal
        for baseline in (self.static_fallback, self.legacy_temporal):
            baseline.requires_grad_(False)
            baseline.eval()
        self.model_dim = int(model_dim)
        self.quality_dim = int(quality_dim)
        self.modality_dropout = float(modality_dropout)
        self.stochastic_depth = float(stochastic_depth)
        self.use_short = bool(use_short)
        self.use_long = bool(use_long)
        self.use_trajectory = bool(use_trajectory)
        self.use_parts = bool(use_parts)
        self.use_pose = bool(use_pose)
        self.use_siglip = bool(use_siglip)

        encoder_args = {
            "model_dim": model_dim,
            "layers": layers,
            "heads": heads,
            "feedforward_dim": feedforward_dim,
            "dropout": dropout,
            "maximum_length": 32,
        }
        self.short_encoder = (
            CenterQueryEncoder(frame_input_dim, **encoder_args) if self.use_short else None
        )
        self.long_encoder = (
            CenterQueryEncoder(frame_input_dim, **encoder_args) if self.use_long else None
        )
        self.trajectory_encoder = (
            TrajectoryEncoder(
                trajectory_sequence_dim,
                trajectory_summary_dim,
                model_dim=model_dim,
                heads=heads,
                dropout=dropout,
            )
            if self.use_trajectory
            else None
        )
        self.part_encoder = (
            PartArticulationEncoder(
                part_input_dim,
                part_count,
                model_dim=model_dim,
                heads=heads,
                dropout=dropout,
            )
            if self.use_parts
            else None
        )
        self.pose_encoder = (
            PoseArticulationEncoder(
                pose_joint_count,
                model_dim=model_dim,
                heads=heads,
                dropout=dropout,
            )
            if self.use_pose
            else None
        )
        self.siglip_encoder = (
            nn.Sequential(
                nn.LayerNorm(siglip_dim),
                nn.Linear(siglip_dim, model_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(model_dim),
            )
            if self.use_siglip
            else None
        )

        names = ["legacy_short"]
        if self.use_short:
            names.append("centre_short")
        if self.use_long:
            names.append("centre_long")
        if self.use_trajectory:
            names.append("trajectory")
        if self.use_parts:
            names.append("parts")
        if self.use_pose:
            names.append("pose")
        if self.use_siglip:
            names.append("siglip")
        self.expert_names = tuple(names)
        learned_names = names[1:]
        declared_roles = dict(expert_roles or {})
        unknown_roles = set(declared_roles) - set(learned_names)
        if unknown_roles:
            raise ValueError(f"Roles declared for inactive experts: {sorted(unknown_roles)}")
        invalid_roles = {
            name: role
            for name, role in declared_roles.items()
            if role not in {"both", "posture", "motion"}
        }
        if invalid_roles:
            raise ValueError(f"Unsupported expert roles: {invalid_roles}")
        self.expert_roles = {
            name: declared_roles.get(name, "both") for name in learned_names
        }
        posture_role_mask = [1.0]
        motion_role_mask = [1.0]
        for name in learned_names:
            role = self.expert_roles[name]
            posture_role_mask.append(float(role in {"both", "posture"}))
            motion_role_mask.append(float(role in {"both", "motion"}))
        self.register_buffer(
            "posture_role_mask",
            torch.tensor(posture_role_mask, dtype=torch.float32),
        )
        self.register_buffer(
            "motion_role_mask",
            torch.tensor(motion_role_mask, dtype=torch.float32),
        )
        self.residual_heads = nn.ModuleDict(
            {name: ResidualHead(model_dim, dropout) for name in learned_names}
        )
        static_feature_dim = int(static_fallback.encoder[-1].normalized_shape[0])
        self.static_evidence_projection = nn.Sequential(
            nn.Linear(static_feature_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        gate_input_dim = static_feature_dim + 4 + quality_dim
        self.gate_trunk = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.posture_gate = nn.Linear(model_dim, len(names))
        self.motion_gate = nn.Linear(model_dim, len(names))
        nn.init.zeros_(self.posture_gate.weight)
        nn.init.zeros_(self.motion_gate.weight)
        nn.init.constant_(self.posture_gate.bias, -2.0)
        nn.init.constant_(self.motion_gate.bias, -2.0)
        self.posture_gate.bias.data[0] = 5.0
        self.motion_gate.bias.data[0] = 5.0
        self.auxiliary_norm = nn.LayerNorm(model_dim)
        self.transition_head = nn.Linear(model_dim, 1)
        self.gait_head = nn.Linear(model_dim, 2)
        self.quality_head = nn.Linear(model_dim, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.static_fallback.eval()
        self.legacy_temporal.eval()
        return self

    def _drop_reliability(self, reliability: torch.Tensor) -> torch.Tensor:
        if not self.training or reliability.shape[1] <= 1:
            return reliability
        learned = reliability[:, 1:]
        quality_dropout = (
            self.modality_dropout * (1.0 + (1.0 - learned))
        ).clamp_max(0.95)
        modality_keep = torch.rand_like(learned) >= quality_dropout
        path_keep = torch.rand_like(learned) >= self.stochastic_depth
        kept = learned * modality_keep * path_keep
        return torch.cat((reliability[:, :1], kept), dim=1)

    def forward(
        self,
        *,
        static_features: torch.Tensor,
        short_features: torch.Tensor,
        short_valid_mask: torch.Tensor,
        short_centre_index: int,
        quality_features: torch.Tensor,
        long_features: torch.Tensor | None = None,
        long_valid_mask: torch.Tensor | None = None,
        long_centre_index: int | None = None,
        trajectory_sequence: torch.Tensor | None = None,
        trajectory_summary: torch.Tensor | None = None,
        trajectory_valid_mask: torch.Tensor | None = None,
        camera_quality: torch.Tensor | None = None,
        part_tokens: torch.Tensor | None = None,
        part_confidence: torch.Tensor | None = None,
        part_valid_mask: torch.Tensor | None = None,
        part_centre_index: int | None = None,
        pose: torch.Tensor | None = None,
        pose_valid_mask: torch.Tensor | None = None,
        pose_centre_index: int | None = None,
        siglip_features: torch.Tensor | None = None,
    ) -> CPTROutput:
        if quality_features.ndim != 2 or quality_features.shape[1] != self.quality_dim:
            raise ValueError("quality_features do not match the configured quality dimension")
        with torch.no_grad():
            static = self.static_fallback(static_features)
            legacy_mask = torch.ones_like(short_valid_mask, dtype=torch.bool)
            legacy = self.legacy_temporal(short_features, legacy_mask)

        evidence: dict[str, ExpertEvidence] = {}
        if self.short_encoder is not None:
            evidence["centre_short"] = self.short_encoder(
                short_features,
                short_valid_mask,
                centre_index=short_centre_index,
            )
        if self.long_encoder is not None:
            if long_features is None or long_valid_mask is None or long_centre_index is None:
                raise ValueError("The long-clock expert requires long temporal inputs")
            evidence["centre_long"] = self.long_encoder(
                long_features,
                long_valid_mask,
                centre_index=long_centre_index,
            )
        if self.trajectory_encoder is not None:
            if any(
                value is None
                for value in (
                    trajectory_sequence,
                    trajectory_summary,
                    trajectory_valid_mask,
                    camera_quality,
                )
            ):
                raise ValueError("The trajectory expert requires sequence, summary, mask, and quality")
            evidence["trajectory"] = self.trajectory_encoder(
                trajectory_sequence,
                trajectory_summary,
                trajectory_valid_mask,
                camera_quality,
            )
        if self.part_encoder is not None:
            if any(
                value is None
                for value in (part_tokens, part_confidence, part_valid_mask, part_centre_index)
            ):
                raise ValueError("The part expert requires tokens, confidence, mask, and centre")
            evidence["parts"] = self.part_encoder(
                part_tokens,
                part_confidence,
                part_valid_mask,
                centre_index=part_centre_index,
            )
        if self.pose_encoder is not None:
            if pose is None or pose_valid_mask is None or pose_centre_index is None:
                raise ValueError("The pose expert requires pose values, mask, and centre")
            evidence["pose"] = self.pose_encoder(
                pose,
                pose_valid_mask,
                centre_index=pose_centre_index,
            )
        if self.siglip_encoder is not None:
            if siglip_features is None:
                raise ValueError("The SigLIP specialist requires centre-frame features")
            evidence["siglip"] = ExpertEvidence(
                self.siglip_encoder(siglip_features),
                torch.ones(len(siglip_features), device=siglip_features.device),
            )

        posture_residuals = [legacy.posture_logits - static.posture_logits]
        motion_residuals = [legacy.motion_logits - static.motion_logits]
        reliabilities = [short_valid_mask.float().mean(dim=1)]
        expert_features = [self.static_evidence_projection(static.features)]
        for name in self.expert_names[1:]:
            item = evidence[name]
            posture, motion = self.residual_heads[name](item.features)
            posture_residuals.append(posture)
            motion_residuals.append(motion)
            reliabilities.append(item.reliability)
            expert_features.append(item.features)
        reliability = torch.stack(reliabilities, dim=1).clamp(0.0, 1.0)
        effective_reliability = self._drop_reliability(reliability)
        diagnostics = probability_diagnostics(static.probabilities)
        gate_features = self.gate_trunk(
            torch.cat((static.features, diagnostics, quality_features), dim=1)
        )
        posture_gates = (
            self.posture_gate(gate_features).sigmoid()
            * effective_reliability
            * self.posture_role_mask[None]
        )
        motion_gates = (
            self.motion_gate(gate_features).sigmoid()
            * effective_reliability
            * self.motion_role_mask[None]
        )
        posture_stack = torch.stack(posture_residuals, dim=1)
        motion_stack = torch.stack(motion_residuals, dim=1)
        gated_posture = posture_stack * posture_gates[:, :, None]
        gated_motion = motion_stack * motion_gates[:, :, None]
        posture_delta = gated_posture.sum(dim=1)
        motion_delta = gated_motion.sum(dim=1)
        learned_posture_delta = gated_posture[:, 1:].sum(dim=1)
        learned_motion_delta = gated_motion[:, 1:].sum(dim=1)
        posture_logits = static.posture_logits + posture_delta
        motion_logits = static.motion_logits + motion_delta
        probabilities = decode_factorized_logits(posture_logits, motion_logits)

        feature_stack = torch.stack(expert_features, dim=1)
        shared_gate = 0.5 * (posture_gates + motion_gates)
        learned_gate = shared_gate[:, 1:]
        if feature_stack.shape[1] == 1:
            fused = feature_stack[:, 0]
        else:
            fused = feature_stack[:, 0] + (
                feature_stack[:, 1:] * learned_gate[:, :, None]
            ).sum(dim=1) / learned_gate.sum(dim=1, keepdim=True).clamp_min(1.0)
        fused = self.auxiliary_norm(fused)
        unknown_score = (
            0.65 * (1.0 - probabilities.max(dim=1).values)
            + 0.35 * (1.0 - reliability.mean(dim=1))
        ).clamp(0.0, 1.0)
        return CPTROutput(
            probabilities=probabilities,
            posture_logits=posture_logits,
            motion_logits=motion_logits,
            static_probabilities=static.probabilities,
            legacy_probabilities=legacy.probabilities,
            static_posture_logits=static.posture_logits,
            static_motion_logits=static.motion_logits,
            expert_names=self.expert_names,
            posture_gates=posture_gates,
            motion_gates=motion_gates,
            expert_reliability=reliability,
            temporal_residual=torch.cat((posture_delta, motion_delta), dim=1),
            learned_temporal_residual=torch.cat(
                (learned_posture_delta, learned_motion_delta), dim=1
            ),
            transition_logit=self.transition_head(fused).squeeze(1),
            gait_logits=self.gait_head(fused),
            quality_logit=self.quality_head(fused).squeeze(1),
            evidence_features=fused,
            unknown_score=unknown_score,
        )


def _factorized_loss(
    output: CPTROutput,
    labels: torch.Tensor,
    *,
    label_smoothing: float,
    posture_weight: torch.Tensor | None,
    motion_weight: torch.Tensor | None,
) -> torch.Tensor:
    posture_targets = (labels != 0).long()
    posture = F.cross_entropy(
        output.posture_logits,
        posture_targets,
        weight=posture_weight,
        label_smoothing=label_smoothing,
    )
    upright = labels != 0
    if torch.any(upright):
        motion = F.cross_entropy(
            output.motion_logits[upright],
            (labels[upright] == 2).long(),
            weight=motion_weight,
            label_smoothing=label_smoothing,
        )
    else:
        motion = output.motion_logits.sum() * 0.0
    return posture + motion


def cptr_primary_loss_per_sample(
    output: CPTROutput,
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.02,
    posture_weight: torch.Tensor | None = None,
    motion_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return hierarchical supervised loss per example for grouped robust training."""

    posture = F.cross_entropy(
        output.posture_logits,
        (labels != 0).long(),
        weight=posture_weight,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    motion = torch.zeros_like(posture)
    upright = labels != 0
    if torch.any(upright):
        motion[upright] = F.cross_entropy(
            output.motion_logits[upright],
            (labels[upright] == 2).long(),
            weight=motion_weight,
            label_smoothing=label_smoothing,
            reduction="none",
        )
    return posture + motion


def _symmetric_kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.clamp_min(1e-8)
    right = right.clamp_min(1e-8)
    return 0.5 * (
        F.kl_div(left.log(), right.detach(), reduction="batchmean")
        + F.kl_div(right.log(), left.detach(), reduction="batchmean")
    )


def cptr_loss(
    output: CPTROutput,
    labels: torch.Tensor,
    *,
    transition_targets: torch.Tensor,
    gait_targets: torch.Tensor,
    occlusion_targets: torch.Tensor,
    null_output: CPTROutput | None = None,
    reversed_output: CPTROutput | None = None,
    jittered_output: CPTROutput | None = None,
    label_smoothing: float = 0.02,
    posture_weight: torch.Tensor | None = None,
    motion_weight: torch.Tensor | None = None,
    transition_weight: float = 0.10,
    gait_weight: float = 0.05,
    quality_weight: float = 0.025,
    motion_null_weight: float = 0.15,
    reversal_weight: float = 0.05,
    camera_invariance_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
    primary = _factorized_loss(
        output,
        labels,
        label_smoothing=label_smoothing,
        posture_weight=posture_weight,
        motion_weight=motion_weight,
    )
    transition = F.binary_cross_entropy_with_logits(
        output.transition_logit,
        transition_targets.float(),
    )
    gait_rows = gait_targets >= 0
    if torch.any(gait_rows):
        gait = F.cross_entropy(output.gait_logits[gait_rows], gait_targets[gait_rows].long())
    else:
        gait = output.gait_logits.sum() * 0.0
    quality = F.binary_cross_entropy_with_logits(
        output.quality_logit,
        occlusion_targets.float(),
    )
    null_loss = output.probabilities.sum() * 0.0
    if null_output is not None:
        null_loss = _symmetric_kl(
            null_output.probabilities,
            null_output.legacy_probabilities,
        )
        null_loss = null_loss + null_output.learned_temporal_residual.square().mean()
    reversal = output.probabilities.sum() * 0.0
    if reversed_output is not None:
        stable = ~transition_targets.bool()
        if torch.any(stable):
            reversal = _symmetric_kl(
                output.probabilities[stable],
                reversed_output.probabilities[stable],
            )
    camera = output.probabilities.sum() * 0.0
    if jittered_output is not None:
        camera = _symmetric_kl(output.probabilities, jittered_output.probabilities)
    total = (
        primary
        + transition_weight * transition
        + gait_weight * gait
        + quality_weight * quality
        + motion_null_weight * null_loss
        + reversal_weight * reversal
        + camera_invariance_weight * camera
    )
    return {
        "loss": total,
        "primary_loss": primary,
        "transition_loss": transition,
        "gait_loss": gait,
        "quality_loss": quality,
        "motion_null_loss": null_loss,
        "reversal_loss": reversal,
        "camera_invariance_loss": camera,
    }


def trainable_parameter_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "trainable_fraction": float(trainable / total) if total else 0.0,
    }
