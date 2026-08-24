"""Shared construction, loading, and evaluation helpers for CPTR experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, log_loss
from torch.utils.data import DataLoader

from hac.cptr import PartTrajectoryResidualNetwork, trainable_parameter_summary
from hac.polar import sha256_file
from hac.vcoco_v3_temporal import StaticIdentifiabilityStudent, TemporalFactorizedTeacher


def classification_summary(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True).clip(min=1e-12)
    predictions = probabilities.argmax(axis=1)
    class_values = f1_score(
        labels,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": float(class_values.mean()),
        "accuracy": float((predictions == labels).mean()),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1, 2])),
        "sitting_f1": float(class_values[0]),
        "standing_f1": float(class_values[1]),
        "walking_running_f1": float(class_values[2]),
        "worst_class_f1": float(class_values.min()),
    }


def better_metrics(candidate: dict[str, float], incumbent: dict[str, float] | None) -> bool:
    if incumbent is None:
        return True
    candidate_key = (
        candidate["macro_f1"],
        candidate["standing_f1"],
        candidate["walking_running_f1"],
        -candidate["log_loss"],
    )
    incumbent_key = (
        incumbent["macro_f1"],
        incumbent["standing_f1"],
        incumbent["walking_running_f1"],
        -incumbent["log_loss"],
    )
    return candidate_key > incumbent_key


def load_frozen_v3_baselines(
    seed: int,
    *,
    input_dim: int,
    v3_grid: dict,
    v3_root: Path,
    device: torch.device,
) -> tuple[StaticIdentifiabilityStudent, TemporalFactorizedTeacher, dict[str, str]]:
    architecture = v3_grid["architecture"]
    static = StaticIdentifiabilityStudent(
        input_dim,
        hidden_dim=int(architecture["static_hidden_dim"]),
        dropout=float(architecture["dropout"]),
    )
    teacher = TemporalFactorizedTeacher(
        input_dim,
        model_dim=int(architecture["temporal_model_dim"]),
        layers=int(architecture["temporal_layers"]),
        attention_heads=int(architecture["attention_heads"]),
        feedforward_dim=int(architecture["feedforward_dim"]),
        dropout=float(architecture["dropout"]),
        maximum_length=16,
    )
    static_path = v3_root / "static" / f"seed-{seed}" / "best_checkpoint.pt"
    teacher_path = (
        v3_root
        / "teacher"
        / "temporal_8f_050s"
        / f"seed-{seed}"
        / "best_checkpoint.pt"
    )
    for model, path in ((static, static_path), (teacher, teacher_path)):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval().requires_grad_(False)
    return static, teacher, {
        "static": sha256_file(static_path),
        "teacher": sha256_file(teacher_path),
    }


def build_cptr_model(
    candidate: dict,
    protocol: dict,
    *,
    input_dim: int,
    part_input_dim: int,
    pose_joint_count: int,
    siglip_dim: int,
    static: StaticIdentifiabilityStudent,
    teacher: TemporalFactorizedTeacher,
) -> PartTrajectoryResidualNetwork:
    architecture = protocol["architecture"]
    return PartTrajectoryResidualNetwork(
        static,
        teacher,
        frame_input_dim=input_dim,
        quality_dim=8,
        trajectory_sequence_dim=21,
        trajectory_summary_dim=58,
        model_dim=int(architecture["model_dim"]),
        layers=int(architecture["temporal_layers"]),
        heads=int(architecture["attention_heads"]),
        feedforward_dim=int(architecture["feedforward_dim"]),
        dropout=float(architecture["dropout"]),
        use_short=bool(candidate["use_short"]),
        use_long=bool(candidate["use_long"]),
        use_trajectory=bool(candidate["use_trajectory"]),
        use_parts=bool(candidate["use_parts"]),
        part_input_dim=int(part_input_dim),
        part_count=7,
        use_pose=bool(candidate["use_pose"]),
        pose_joint_count=int(pose_joint_count),
        use_siglip=bool(candidate["use_siglip"]),
        siglip_dim=int(siglip_dim),
        expert_roles=candidate.get("expert_roles"),
        modality_dropout=float(protocol["augmentation"]["modality_dropout"]),
        stochastic_depth=float(architecture["stochastic_depth"]),
    )


@torch.inference_mode()
def evaluate_cptr(
    model: PartTrajectoryResidualNetwork,
    loader: DataLoader,
    device: torch.device,
    *,
    kwargs_builder,
) -> dict:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    static_probabilities: list[np.ndarray] = []
    posture_gates: list[np.ndarray] = []
    motion_gates: list[np.ndarray] = []
    reliabilities: list[np.ndarray] = []
    unknown_scores: list[np.ndarray] = []
    transitions: list[np.ndarray] = []
    occlusions: list[np.ndarray] = []
    sample_ids: list[str] = []
    recording_ids: list[str] = []
    track_ids: list[str] = []
    for batch in loader:
        kwargs = kwargs_builder(batch, device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(**kwargs)
        labels.append(batch["label"].numpy())
        probabilities.append(output.probabilities.float().cpu().numpy())
        static_probabilities.append(output.static_probabilities.float().cpu().numpy())
        posture_gates.append(output.posture_gates.float().cpu().numpy())
        motion_gates.append(output.motion_gates.float().cpu().numpy())
        reliabilities.append(output.expert_reliability.float().cpu().numpy())
        unknown_scores.append(output.unknown_score.float().cpu().numpy())
        transitions.append(batch["transition_target"].numpy())
        occlusions.append(batch["occlusion_target"].numpy())
        sample_ids.extend(map(str, batch["sample_id"]))
        recording_ids.extend(map(str, batch["recording_id"]))
        track_ids.extend(map(str, batch["track_id"]))
    target = np.concatenate(labels)
    predicted = np.concatenate(probabilities)
    predicted = predicted / predicted.sum(axis=1, keepdims=True).clip(min=1e-12)
    static_predicted = np.concatenate(static_probabilities)
    static_predicted = static_predicted / static_predicted.sum(axis=1, keepdims=True).clip(
        min=1e-12
    )
    transition_values = np.concatenate(transitions).astype(bool)
    occlusion_values = np.concatenate(occlusions).astype(bool)
    subgroup_metrics: dict[str, dict[str, float] | None] = {}
    for name, mask in (
        ("transition", transition_values),
        ("non_transition", ~transition_values),
        ("window_occluded", occlusion_values),
        ("window_clear", ~occlusion_values),
    ):
        subgroup_metrics[name] = (
            classification_summary(target[mask], predicted[mask]) if np.sum(mask) >= 3 else None
        )
    return {
        "labels": target,
        "probabilities": predicted,
        "static_probabilities": static_predicted,
        "posture_gates": np.concatenate(posture_gates),
        "motion_gates": np.concatenate(motion_gates),
        "expert_reliability": np.concatenate(reliabilities),
        "unknown_scores": np.concatenate(unknown_scores),
        "transition_targets": transition_values,
        "occlusion_targets": occlusion_values,
        "sample_ids": np.asarray(sample_ids),
        "recording_ids": np.asarray(recording_ids),
        "track_ids": np.asarray(track_ids),
        "metrics": classification_summary(target, predicted),
        "subgroups": subgroup_metrics,
    }


def model_evidence(model: PartTrajectoryResidualNetwork) -> dict:
    return {
        "expert_names": list(model.expert_names),
        "expert_roles": {"legacy_short": "both", **model.expert_roles},
        "parameters": trainable_parameter_summary(model),
    }
