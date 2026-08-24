"""Temporal teacher, static student, and evidence-routing primitives for v3."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from hac.vcoco_v3_neural import FactorizedClassifier, decode_factorized_logits

TEMPORAL_CLASS_NAMES = ("sitting", "standing", "walking_running")


@dataclass(frozen=True)
class TemporalOutput:
    probabilities: torch.Tensor
    posture_logits: torch.Tensor
    motion_logits: torch.Tensor
    pooled_features: torch.Tensor
    attention_weights: torch.Tensor


@dataclass(frozen=True)
class StaticStudentOutput:
    probabilities: torch.Tensor
    posture_logits: torch.Tensor
    motion_logits: torch.Tensor
    identifiability_logit: torch.Tensor
    features: torch.Tensor


class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, feature_dim: int, maximum_length: int = 128) -> None:
        super().__init__()
        if feature_dim < 2 or maximum_length < 2:
            raise ValueError("Temporal encoding dimensions must be at least two")
        positions = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        scales = torch.exp(
            torch.arange(0, feature_dim, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / feature_dim)
        )
        encoding = torch.zeros(maximum_length, feature_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * scales)
        if feature_dim > 1:
            encoding[:, 1::2] = torch.cos(positions * scales[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("Temporal values must have shape [batch, time, features]")
        if values.shape[1] > self.encoding.shape[0]:
            raise ValueError("Clip is longer than the configured temporal encoding")
        return values + self.encoding[: values.shape[1]].to(values.dtype).unsqueeze(0)


class MaskedAttentionPooling(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(feature_dim))
        nn.init.normal_(self.query, std=feature_dim**-0.5)

    def forward(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if valid_mask.shape != values.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean with shape [batch, time]")
        if not torch.all(valid_mask.any(dim=1)):
            raise ValueError("Every clip must contain at least one valid frame")
        scores = torch.einsum("btd,d->bt", values, self.query) / math.sqrt(values.shape[-1])
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=1)
        pooled = torch.einsum("bt,btd->bd", weights, values)
        return pooled, weights


class TemporalFactorizedTeacher(nn.Module):
    """Classify a short sequence of frozen or jointly adapted frame embeddings."""

    def __init__(
        self,
        input_dim: int,
        *,
        model_dim: int = 256,
        layers: int = 2,
        attention_heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        maximum_length: int = 64,
    ) -> None:
        super().__init__()
        if min(input_dim, model_dim, layers, attention_heads, feedforward_dim) < 1:
            raise ValueError("Temporal model dimensions must be positive")
        if model_dim % attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.time_encoding = SinusoidalTimeEncoding(model_dim, maximum_length)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.pooling = MaskedAttentionPooling(model_dim)
        self.classifier = FactorizedClassifier(model_dim, dropout)

    def forward(
        self,
        frame_features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> TemporalOutput:
        if frame_features.ndim != 3:
            raise ValueError("frame_features must have shape [batch, time, feature_dim]")
        if valid_mask is None:
            valid_mask = torch.ones(
                frame_features.shape[:2], dtype=torch.bool, device=frame_features.device
            )
        values = self.time_encoding(self.input_projection(frame_features))
        values = self.temporal_encoder(values, src_key_padding_mask=~valid_mask)
        pooled, attention = self.pooling(values, valid_mask)
        posture_logits, motion_logits = self.classifier(pooled)
        return TemporalOutput(
            probabilities=decode_factorized_logits(posture_logits, motion_logits),
            posture_logits=posture_logits,
            motion_logits=motion_logits,
            pooled_features=pooled,
            attention_weights=attention,
        )


class StaticIdentifiabilityStudent(nn.Module):
    """Predict activity and whether a short clip is likely to add useful evidence."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 256,
        geometry_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or geometry_dim < 0:
            raise ValueError("Static student dimensions are invalid")
        self.geometry_dim = int(geometry_dim)
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim + geometry_dim),
            nn.Linear(input_dim + geometry_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = FactorizedClassifier(hidden_dim, dropout)
        self.identifiability = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        static_features: torch.Tensor,
        geometry: torch.Tensor | None = None,
    ) -> StaticStudentOutput:
        if static_features.ndim != 2:
            raise ValueError("static_features must have shape [batch, feature_dim]")
        if self.geometry_dim:
            if geometry is None or geometry.shape != (static_features.shape[0], self.geometry_dim):
                raise ValueError("The declared geometry features are required")
            inputs = torch.cat((static_features, geometry), dim=1)
        else:
            if geometry is not None:
                raise ValueError("This student was configured without geometry")
            inputs = static_features
        features = self.encoder(inputs)
        posture_logits, motion_logits = self.classifier(features)
        probabilities = decode_factorized_logits(posture_logits, motion_logits)
        ordered = probabilities.sort(dim=1).values
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
        diagnostics = torch.stack(
            (
                probabilities.max(dim=1).values,
                entropy,
                ordered[:, -1] - ordered[:, -2],
                probabilities[:, 1] + probabilities[:, 2],
            ),
            dim=1,
        )
        identifiability_logit = self.identifiability(torch.cat((features, diagnostics), dim=1))
        return StaticStudentOutput(
            probabilities=probabilities,
            posture_logits=posture_logits,
            motion_logits=motion_logits,
            identifiability_logit=identifiability_logit.squeeze(1),
            features=features,
        )


def _hierarchical_supervised_loss(
    posture_logits: torch.Tensor,
    motion_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float,
    posture_weight: torch.Tensor | None = None,
    motion_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    posture_targets = (labels != 0).long()
    posture = F.cross_entropy(
        posture_logits,
        posture_targets,
        weight=posture_weight,
        label_smoothing=float(label_smoothing),
    )
    upright = labels != 0
    if torch.any(upright):
        motion = F.cross_entropy(
            motion_logits[upright],
            (labels[upright] == 2).long(),
            weight=motion_weight,
            label_smoothing=float(label_smoothing),
        )
    else:
        motion = motion_logits.sum() * 0.0
    return posture + motion


def temporal_teacher_loss(
    output: TemporalOutput,
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    posture_weight: torch.Tensor | None = None,
    motion_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return _hierarchical_supervised_loss(
        output.posture_logits,
        output.motion_logits,
        labels,
        label_smoothing=label_smoothing,
        posture_weight=posture_weight,
        motion_weight=motion_weight,
    )


def static_student_supervised_loss(
    output: StaticStudentOutput,
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    posture_weight: torch.Tensor | None = None,
    motion_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return _hierarchical_supervised_loss(
        output.posture_logits,
        output.motion_logits,
        labels,
        label_smoothing=label_smoothing,
        posture_weight=posture_weight,
        motion_weight=motion_weight,
    )


def static_student_distillation_loss(
    output: StaticStudentOutput,
    labels: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    *,
    identifiability_targets: torch.Tensor | None,
    supervised_weight: float = 0.6,
    distillation_weight: float = 0.3,
    identifiability_weight: float = 0.1,
    temperature: float = 2.0,
    label_smoothing: float = 0.0,
    identifiability_positive_weight: torch.Tensor | None = None,
    posture_weight: torch.Tensor | None = None,
    motion_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Train a static student from labels and cross-fitted temporal predictions."""

    weights = (supervised_weight, distillation_weight, identifiability_weight)
    if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
        raise ValueError("Distillation loss weights must be nonnegative with a positive sum")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    supervised = _hierarchical_supervised_loss(
        output.posture_logits,
        output.motion_logits,
        labels,
        label_smoothing=label_smoothing,
        posture_weight=posture_weight,
        motion_weight=motion_weight,
    )
    teacher = teacher_probabilities.detach().clamp_min(1e-8)
    softened_teacher = (teacher.log() / temperature).softmax(dim=1)
    softened_student_log = (output.probabilities.clamp_min(1e-8).log() / temperature).log_softmax(
        dim=1
    )
    distillation = (
        F.kl_div(
            softened_student_log,
            softened_teacher,
            reduction="batchmean",
        )
        * temperature**2
    )
    if identifiability_targets is None:
        if identifiability_weight > 0.0:
            raise ValueError(
                "Identifiability targets are required when their loss weight is positive"
            )
        identifiability = output.identifiability_logit.sum() * 0.0
    else:
        identifiability = F.binary_cross_entropy_with_logits(
            output.identifiability_logit,
            identifiability_targets.float(),
            pos_weight=identifiability_positive_weight,
        )
    normalizer = sum(weights)
    total = (
        supervised_weight * supervised
        + distillation_weight * distillation
        + identifiability_weight * identifiability
    ) / normalizer
    return {
        "loss": total,
        "supervised_loss": supervised,
        "distillation_loss": distillation,
        "identifiability_loss": identifiability,
    }


def teacher_advantage_targets(
    labels: np.ndarray,
    static_probabilities: np.ndarray,
    teacher_probabilities: np.ndarray,
    *,
    minimum_log_likelihood_gain: float = 0.20,
) -> np.ndarray:
    """Derive motion-evidence targets only from out-of-fold predictions.

    The caller is responsible for supplying predictions generated without fitting on
    the scored recording or track. The target is positive when the temporal model is
    correct and improves true-class log likelihood by the declared margin.
    """

    labels = np.asarray(labels, dtype=int)
    static = np.asarray(static_probabilities, dtype=float)
    teacher = np.asarray(teacher_probabilities, dtype=float)
    if static.shape != teacher.shape or static.shape != (len(labels), 3):
        raise ValueError("Static and teacher probabilities must have shape [n, 3]")
    if minimum_log_likelihood_gain < 0.0:
        raise ValueError("minimum_log_likelihood_gain cannot be negative")
    rows = np.arange(len(labels))
    gain = np.log(np.clip(teacher[rows, labels], 1e-12, 1.0)) - np.log(
        np.clip(static[rows, labels], 1e-12, 1.0)
    )
    return ((teacher.argmax(axis=1) == labels) & (gain >= minimum_log_likelihood_gain)).astype(
        np.int64
    )


def aps_nonconformity_scores(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Conservative adaptive-prediction-set scores for calibration examples."""

    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(labels), len(TEMPORAL_CLASS_NAMES)):
        raise ValueError("Probability matrix and labels do not align")
    order = np.argsort(-probabilities, axis=1, kind="stable")
    sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)
    cumulative = np.cumsum(sorted_probabilities, axis=1)
    true_positions = (order == labels[:, None]).argmax(axis=1)
    return cumulative[np.arange(len(labels)), true_positions]


def fit_aps_threshold(scores: np.ndarray, *, miscoverage: float) -> float:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError("Calibration scores must be a finite nonempty vector")
    if not 0.0 < miscoverage < 1.0:
        raise ValueError("miscoverage must be in (0, 1)")
    quantile_level = min(1.0, math.ceil((len(scores) + 1) * (1.0 - miscoverage)) / len(scores))
    return float(np.quantile(scores, quantile_level, method="higher"))


def aps_prediction_sets(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Return a boolean class-membership matrix using a locked APS threshold."""

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(TEMPORAL_CLASS_NAMES):
        raise ValueError("probabilities must have shape [n, 3]")
    if not 0.0 <= threshold <= 1.0 + 1e-12:
        raise ValueError("APS threshold must be in [0, 1]")
    order = np.argsort(-probabilities, axis=1, kind="stable")
    sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)
    cumulative_before = np.cumsum(sorted_probabilities, axis=1) - sorted_probabilities
    sorted_membership = cumulative_before < threshold
    membership = np.zeros_like(sorted_membership, dtype=bool)
    np.put_along_axis(membership, order, sorted_membership, axis=1)
    return membership


def route_by_budget(scores: np.ndarray, *, clip_fraction: float) -> np.ndarray:
    """Route the highest predicted temporal-benefit scores under a fixed budget."""

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("Routing scores must be a finite vector")
    if not 0.0 <= clip_fraction <= 1.0:
        raise ValueError("clip_fraction must be in [0, 1]")
    count = int(math.ceil(len(scores) * clip_fraction))
    routed = np.zeros(len(scores), dtype=bool)
    if count:
        ranked = np.argsort(-scores, kind="stable")
        routed[ranked[:count]] = True
    return routed


def routed_probabilities(
    static_probabilities: np.ndarray,
    temporal_probabilities: np.ndarray,
    routed: np.ndarray,
) -> np.ndarray:
    static = np.asarray(static_probabilities, dtype=float)
    temporal = np.asarray(temporal_probabilities, dtype=float)
    routed = np.asarray(routed, dtype=bool)
    if static.shape != temporal.shape or static.shape[0] != len(routed):
        raise ValueError("Static, temporal, and routing rows do not align")
    output = static.copy()
    output[routed] = temporal[routed]
    return output


def evaluate_routing_curve(
    labels: np.ndarray,
    static_probabilities: np.ndarray,
    temporal_probabilities: np.ndarray,
    routing_scores: np.ndarray,
    *,
    clip_fractions: Iterable[float],
    advantage_targets: np.ndarray | None = None,
) -> list[dict[str, float]]:
    labels = np.asarray(labels, dtype=int)
    targets = None if advantage_targets is None else np.asarray(advantage_targets, dtype=int)
    output = []
    for fraction in clip_fractions:
        routed = route_by_budget(routing_scores, clip_fraction=float(fraction))
        probabilities = routed_probabilities(static_probabilities, temporal_probabilities, routed)
        predictions = probabilities.argmax(axis=1)
        row = {
            "requested_clip_fraction": float(fraction),
            "observed_clip_fraction": float(routed.mean()) if len(routed) else 0.0,
            "accuracy": float((predictions == labels).mean()),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        }
        if targets is not None:
            row["routed_advantage_precision"] = (
                float(targets[routed].mean()) if np.any(routed) else math.nan
            )
            row["advantage_recall"] = (
                float(targets[routed].sum() / targets.sum()) if targets.sum() else math.nan
            )
        output.append(row)
    return output


def uniform_clip_indices(
    frame_count: int,
    *,
    center_index: int,
    samples: int,
    span_frames: int,
) -> np.ndarray:
    """Uniformly sample a centred temporal window, repeating only at clip boundaries."""

    if frame_count < 1 or samples < 1 or span_frames < 1:
        raise ValueError("frame_count, samples, and span_frames must be positive")
    if not 0 <= center_index < frame_count:
        raise ValueError("center_index is outside the clip")
    half = (span_frames - 1) / 2.0
    positions = np.linspace(center_index - half, center_index + half, samples)
    return np.rint(np.clip(positions, 0, frame_count - 1)).astype(np.int64)


def validate_temporal_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the provider-derived manifest before any temporal model fit."""

    required = {
        "sample_id",
        "recording_id",
        "track_id",
        "label",
        "split",
        "frame_count",
        "center_frame_index",
        "frames_per_second",
        "feature_path",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Temporal manifest is missing columns: {missing}")
    output = frame.copy()
    if output["sample_id"].astype(str).duplicated().any():
        raise ValueError("Temporal sample identifiers must be unique")
    allowed_labels = set(TEMPORAL_CLASS_NAMES)
    unknown = sorted(set(output["label"].astype(str)) - allowed_labels)
    if unknown:
        raise ValueError(f"Temporal manifest contains unknown labels: {unknown}")
    allowed_splits = {"train", "validation", "calibration", "confirmation"}
    unknown_splits = sorted(set(output["split"].astype(str)) - allowed_splits)
    if unknown_splits:
        raise ValueError(f"Temporal manifest contains unknown splits: {unknown_splits}")
    frame_count = pd.to_numeric(output["frame_count"], errors="coerce")
    center = pd.to_numeric(output["center_frame_index"], errors="coerce")
    frames_per_second = pd.to_numeric(output["frames_per_second"], errors="coerce")
    if frame_count.isna().any() or center.isna().any() or frames_per_second.isna().any():
        raise ValueError("Temporal frame indices must be numeric")
    if (frame_count < 1).any() or (center < 0).any() or (center >= frame_count).any():
        raise ValueError("Temporal frame bounds are invalid")
    if (frames_per_second <= 0).any():
        raise ValueError("Temporal frame rates must be positive")
    recording_splits = output.groupby("recording_id")["split"].nunique()
    if (recording_splits > 1).any():
        raise ValueError("A recording crosses temporal split boundaries")
    track_keys = output["recording_id"].astype(str) + "::" + output["track_id"].astype(str)
    track_splits = output.assign(_track_key=track_keys).groupby("_track_key")["split"].nunique()
    if (track_splits > 1).any():
        raise ValueError("A person track crosses temporal split boundaries")
    return output.reset_index(drop=True)


def grouped_recording_splits(
    labels: Sequence[int],
    recording_ids: Sequence[str],
    *,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(recording_ids, dtype=str)
    if len(labels) != len(groups):
        raise ValueError("Temporal labels and recording groups do not align")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    dummy = np.zeros((len(labels), 1), dtype=np.uint8)
    splits = list(splitter.split(dummy, labels, groups))
    for fit_index, held_index in splits:
        if set(groups[fit_index]).intersection(groups[held_index]):
            raise RuntimeError("A recording crossed a grouped temporal fold")
    return splits


class TemporalFeatureDataset(Dataset):
    """Load one declared short window from each provider-derived feature archive."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        uniform_samples: int,
        window_seconds: float,
        manifest_directory: str | Path,
    ) -> None:
        if uniform_samples < 1 or window_seconds <= 0.0:
            raise ValueError("Temporal sampling parameters must be positive")
        self.frame = validate_temporal_manifest(frame).reset_index(drop=True)
        self.uniform_samples = int(uniform_samples)
        self.window_seconds = float(window_seconds)
        self.manifest_directory = Path(manifest_directory).resolve()
        self.class_to_index = {name: index for index, name in enumerate(TEMPORAL_CLASS_NAMES)}
        self._packed_stores: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def _path(self, value: str) -> Path:
        path = Path(str(value))
        return path.resolve() if path.is_absolute() else (self.manifest_directory / path).resolve()

    def _packed_store(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._packed_stores.get(path)
        if cached is not None:
            return cached
        declaration = json.loads(path.read_text(encoding="utf-8"))
        if declaration.get("status") != "VCOCO_V3_PACKED_TEMPORAL_FEATURE_STORE_COMPLETE":
            raise RuntimeError(f"Packed temporal store is incomplete: {path}")
        arrays = declaration.get("arrays", {})
        loaded = tuple(
            np.load((path.parent / arrays[name]["path"]).resolve(), mmap_mode="r")
            for name in ("tight", "context", "geometry")
        )
        if len({value.shape[0] for value in loaded}) != 1:
            raise RuntimeError(f"Packed temporal store arrays do not align: {path}")
        self._packed_stores[path] = loaded
        return loaded

    def _features(self, row: pd.Series, path: Path) -> tuple[np.ndarray, ...]:
        if path.suffix.lower() == ".json":
            if "feature_index" not in row or pd.isna(row["feature_index"]):
                raise RuntimeError("Packed temporal rows require feature_index")
            index = int(row["feature_index"])
            store = self._packed_store(path)
            if not 0 <= index < store[0].shape[0]:
                raise RuntimeError(f"Packed temporal feature index is outside the store: {index}")
            return tuple(np.asarray(values[index], dtype=np.float32) for values in store)
        with np.load(path, allow_pickle=False) as payload:
            return (
                np.asarray(payload["tight"], dtype=np.float32),
                np.asarray(payload["context"], dtype=np.float32),
                np.asarray(payload["geometry"], dtype=np.float32),
            )

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        path = self._path(str(row["feature_path"]))
        tight, context, geometry = self._features(row, path)
        expected_frames = int(row["frame_count"])
        if tight.shape[0] != expected_frames or context.shape[0] != expected_frames:
            raise RuntimeError(f"Temporal feature count drift: {path}")
        if geometry.shape != (expected_frames, 6):
            raise RuntimeError(f"Temporal geometry shape drift: {path}")
        span_frames = max(1, int(round(float(row["frames_per_second"]) * self.window_seconds)))
        indices = uniform_clip_indices(
            expected_frames,
            center_index=int(row["center_frame_index"]),
            samples=self.uniform_samples,
            span_frames=span_frames,
        )
        combined = np.concatenate((tight, context, geometry), axis=1)
        center = int(row["center_frame_index"])
        return {
            "clip_features": torch.from_numpy(combined[indices].copy()),
            "static_features": torch.from_numpy(combined[center].copy()),
            "valid_mask": torch.ones(self.uniform_samples, dtype=torch.bool),
            "label": self.class_to_index[str(row["label"])],
            "sample_id": str(row["sample_id"]),
            "recording_id": str(row["recording_id"]),
            "track_id": str(row["track_id"]),
        }


def pose_velocity_summary(
    pose: np.ndarray,
    *,
    frames_per_second: float,
    frame_indices: np.ndarray | None = None,
    confidence_threshold: float = 0.2,
) -> np.ndarray:
    """Summarize normalized pose translation and articulation as a linear control."""

    values = np.asarray(pose, dtype=float)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[2] not in {2, 3}:
        raise ValueError("pose must have shape [time>=2, joints, 2 or 3]")
    if frames_per_second <= 0.0 or not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("Pose timing or confidence threshold is invalid")
    coordinates = values[..., :2]
    if values.shape[2] == 3:
        valid = (values[..., 2] >= confidence_threshold) & np.all(np.isfinite(coordinates), axis=2)
    else:
        valid = np.all(np.isfinite(coordinates), axis=2)
    coordinates = np.where(valid[..., None], coordinates, np.nan)
    if frame_indices is None:
        timing = np.arange(values.shape[0], dtype=float) / float(frames_per_second)
    else:
        indices = np.asarray(frame_indices, dtype=float)
        if indices.shape != (values.shape[0],) or not np.all(np.isfinite(indices)):
            raise ValueError("frame_indices must be a finite vector aligned with pose")
        if np.any(np.diff(indices) < 0.0):
            raise ValueError("frame_indices must be nondecreasing")
        timing = indices / float(frames_per_second)
    counts = valid.sum(axis=1)
    center = np.full((values.shape[0], 2), np.nan, dtype=float)
    observed = counts > 0
    center[observed] = np.nansum(coordinates[observed], axis=1) / counts[observed, None]
    finite_center = np.all(np.isfinite(center), axis=1)
    if finite_center.sum() < 2:
        return np.asarray([0.0] * 11 + [float(1.0 - valid.mean())], dtype=np.float32)
    center = center[finite_center]
    center_timing = timing[finite_center]
    distinct = np.r_[True, np.diff(center_timing) > 0.0]
    center = center[distinct]
    center_timing = center_timing[distinct]
    if len(center) < 2:
        return np.asarray([0.0] * 11 + [float(1.0 - valid.mean())], dtype=np.float32)
    center_dt = np.diff(center_timing)
    center_velocity = np.diff(center, axis=0) / center_dt[:, None]
    center_speed = np.linalg.norm(center_velocity, axis=1)
    net_displacement = np.linalg.norm(center[-1] - center[0])
    path_length = np.linalg.norm(np.diff(center, axis=0), axis=1).sum()

    interval_dt = np.diff(timing)
    usable_intervals = interval_dt > 0.0
    joint_velocity = (
        np.diff(coordinates, axis=0)[usable_intervals] / interval_dt[usable_intervals, None, None]
    )
    joint_speed = np.linalg.norm(joint_velocity, axis=2)
    finite_joint_speed = joint_speed[np.isfinite(joint_speed)]
    if not len(finite_joint_speed):
        finite_joint_speed = np.asarray([0.0])
    velocity_timing = (center_timing[1:] + center_timing[:-1]) / 2.0
    acceleration_dt = np.diff(velocity_timing)
    acceleration = (
        np.diff(center_velocity, axis=0) / acceleration_dt[:, None]
        if len(acceleration_dt)
        else np.empty((0, 2), dtype=float)
    )
    acceleration_norm = (
        np.linalg.norm(acceleration, axis=1) if len(acceleration) else np.asarray([0.0])
    )
    output = np.asarray(
        [
            center_velocity[:, 0].mean(),
            center_velocity[:, 1].mean(),
            center_speed.mean(),
            center_speed.std(),
            center_speed.max(),
            net_displacement,
            path_length,
            net_displacement / max(path_length, 1e-8),
            np.median(finite_joint_speed),
            np.quantile(finite_joint_speed, 0.9),
            acceleration_norm.mean(),
            1.0 - valid.mean(),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
