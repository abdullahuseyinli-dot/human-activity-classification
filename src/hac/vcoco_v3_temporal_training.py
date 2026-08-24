"""Shared feature-level training helpers for the v3 temporal experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from hac.metrics import classification_metrics
from hac.vcoco_v3_models import locomotion_f1
from hac.vcoco_v3_temporal import (
    StaticIdentifiabilityStudent,
    TemporalFactorizedTeacher,
    TemporalFeatureDataset,
    static_student_supervised_loss,
    temporal_teacher_loss,
)


def make_temporal_loader(
    frame: pd.DataFrame,
    *,
    candidate: dict,
    manifest_directory: Path,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        TemporalFeatureDataset(
            frame,
            uniform_samples=int(candidate["uniform_samples"]),
            window_seconds=float(candidate["window_seconds"]),
            manifest_directory=manifest_directory,
        ),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        generator=generator,
    )


def hierarchical_class_weights(
    labels: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = np.asarray(labels, dtype=int)
    posture = (labels != 0).astype(int)
    motion = (labels[labels != 0] == 2).astype(int)

    def calculate(values: np.ndarray) -> torch.Tensor:
        counts = np.bincount(values, minlength=2).astype(float)
        if np.any(counts == 0):
            raise RuntimeError("A temporal fit partition is missing a hierarchical class")
        weights = counts.sum() / (2.0 * counts)
        weights /= weights.mean()
        return torch.as_tensor(weights, dtype=torch.float32, device=device)

    return calculate(posture), calculate(motion)


def build_temporal_development_model(
    role: str,
    *,
    input_dimensions: int,
    architecture: dict,
    maximum_length: int,
) -> nn.Module:
    if role == "teacher":
        return TemporalFactorizedTeacher(
            input_dimensions,
            model_dim=int(architecture["temporal_model_dim"]),
            layers=int(architecture["temporal_layers"]),
            attention_heads=int(architecture["attention_heads"]),
            feedforward_dim=int(architecture["feedforward_dim"]),
            dropout=float(architecture["dropout"]),
            maximum_length=int(maximum_length),
        )
    if role == "static":
        return StaticIdentifiabilityStudent(
            input_dimensions,
            hidden_dim=int(architecture["static_hidden_dim"]),
            dropout=float(architecture["dropout"]),
        )
    raise ValueError(f"Unknown temporal development role: {role}")


def forward_temporal_development(
    model: nn.Module,
    role: str,
    batch: dict,
    device: torch.device,
):
    if role == "teacher":
        return model(
            batch["clip_features"].to(device, non_blocking=True),
            batch["valid_mask"].to(device, non_blocking=True),
        )
    if role == "static":
        return model(batch["static_features"].to(device, non_blocking=True))
    raise ValueError(f"Unknown temporal development role: {role}")


def temporal_development_loss(
    output,
    role: str,
    labels: torch.Tensor,
    *,
    label_smoothing: float,
    posture_weight: torch.Tensor,
    motion_weight: torch.Tensor,
) -> torch.Tensor:
    function = temporal_teacher_loss if role == "teacher" else static_student_supervised_loss
    return function(
        output,
        labels,
        label_smoothing=float(label_smoothing),
        posture_weight=posture_weight,
        motion_weight=motion_weight,
    )


@torch.inference_mode()
def evaluate_temporal_development(
    model: nn.Module,
    role: str,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    labels = []
    probabilities = []
    sample_ids = []
    recording_ids = []
    track_ids = []
    identifiability_scores = []
    for batch in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = forward_temporal_development(model, role, batch, device)
        labels.append(batch["label"].numpy())
        probabilities.append(output.probabilities.float().cpu().numpy())
        if hasattr(output, "identifiability_logit"):
            identifiability_scores.append(
                output.identifiability_logit.sigmoid().float().cpu().numpy()
            )
        sample_ids.extend(map(str, batch["sample_id"]))
        recording_ids.extend(map(str, batch["recording_id"]))
        track_ids.extend(map(str, batch["track_id"]))
    target = np.concatenate(labels)
    predicted = np.concatenate(probabilities)
    metrics = classification_metrics(target, predicted)
    metrics["locomotion_f1"] = locomotion_f1(target, predicted)
    return {
        "labels": target,
        "probabilities": predicted,
        "sample_ids": np.asarray(sample_ids),
        "recording_ids": np.asarray(recording_ids),
        "track_ids": np.asarray(track_ids),
        "identifiability_scores": (
            np.concatenate(identifiability_scores) if identifiability_scores else None
        ),
        "metrics": metrics,
    }
