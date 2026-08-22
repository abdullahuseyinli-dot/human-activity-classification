"""Reproducible training utilities for the leakage-audited POLAR study."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

from .metrics import classification_metrics

TASK_LABELS = {
    "label_4": ("sitting", "standing", "walking", "running"),
    "label_3": ("sitting", "standing", "walking_running"),
}


def stable_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Compute probabilities in FP32 even when inference used mixed precision."""

    return torch.softmax(logits.float(), dim=1)


def normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("Probabilities must be a finite non-negative matrix")
    row_sums = values.sum(axis=1, keepdims=True)
    if (row_sums <= 0.0).any():
        raise ValueError("Probability rows must have positive mass")
    return values / row_sums


@torch.inference_mode()
def evaluate_classifier(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    *,
    return_features: bool = False,
) -> dict:
    model.eval()
    losses, logits, probabilities, labels, features = [], [], [], [], []
    image_ids, image_paths = [], []
    for batch in loader:
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            if return_features:
                batch_logits, batch_features = model(inputs, return_features=True)
            else:
                batch_logits = model(inputs)
                batch_features = None
            loss = criterion(batch_logits, targets)
        batch_logits = batch_logits.float()
        batch_probabilities = stable_probabilities(batch_logits)
        losses.append(float(loss.item()))
        logits.append(batch_logits.cpu().numpy())
        probabilities.append(batch_probabilities.cpu().numpy())
        labels.append(targets.cpu().numpy())
        if batch_features is not None:
            features.append(batch_features.float().cpu().numpy())
        image_ids.extend(str(value) for value in batch["image_id"])
        image_paths.extend(str(value) for value in batch["image_path"])

    logits_array = np.concatenate(logits)
    probabilities_array = normalize_probability_rows(np.concatenate(probabilities))
    labels_array = np.concatenate(labels)
    return {
        "loss": float(np.mean(losses)),
        "labels": labels_array,
        "logits": logits_array,
        "probabilities": probabilities_array,
        "predictions": probabilities_array.argmax(axis=1),
        "features": np.concatenate(features) if features else None,
        "image_ids": image_ids,
        "image_paths": image_paths,
        "metrics": classification_metrics(labels_array, probabilities_array),
    }


def validate_development_manifest(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    """Validate and normalize a train/validation-only manifest."""

    if task not in TASK_LABELS:
        raise ValueError(f"Unknown task: {task}")
    required = {"image_id", "image_path", "split", task}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Development manifest is missing columns: {sorted(missing)}")
    output = frame.copy()
    output["image_id"] = output["image_id"].astype(str)
    output["split"] = output["split"].astype(str)
    splits = set(output["split"])
    if splits != {"train", "val"}:
        raise ValueError(f"Development manifest must contain only train and val, found {splits}")
    if output["image_id"].duplicated().any():
        raise ValueError("Development image identifiers must be unique")
    allowed = set(TASK_LABELS[task])
    observed = set(output[task].astype(str))
    if observed != allowed:
        raise ValueError(f"Unexpected {task} labels: {sorted(observed)}")
    output["label"] = output[task].astype(str)
    return output.sort_values("image_id", ignore_index=True)


def _proportional_quotas(counts: pd.Series, total: int) -> dict[str, int]:
    if total < len(counts) or total > int(counts.sum()):
        raise ValueError("Subset size must include every class and cannot exceed the training set")
    exact = counts.astype(float) * (float(total) / float(counts.sum()))
    quotas = np.floor(exact).astype(int)
    quotas[quotas == 0] = 1
    while int(quotas.sum()) > total:
        removable = [label for label in counts.index if quotas[label] > 1]
        if not removable:
            raise RuntimeError("Unable to reduce stratified quotas")
        label = min(removable, key=lambda value: (exact[value] - quotas[value], value))
        quotas[label] -= 1
    while int(quotas.sum()) < total:
        eligible = [label for label in counts.index if quotas[label] < counts[label]]
        if not eligible:
            raise RuntimeError("Unable to fill stratified quotas")
        label = max(eligible, key=lambda value: (exact[value] - quotas[value], value))
        quotas[label] += 1
    return {str(label): int(quotas[label]) for label in counts.index}


def nested_stratified_subset(
    frame: pd.DataFrame,
    size: int | None,
    *,
    label_column: str = "label",
    seed: int = 20260822,
) -> pd.DataFrame:
    """Return deterministic class-wise prefixes for the declared scale study."""

    if size is None or int(size) >= len(frame):
        return frame.sort_values("image_id", ignore_index=True).copy()
    counts = frame[label_column].astype(str).value_counts().sort_index()
    quotas = _proportional_quotas(counts, int(size))
    selected = []
    for class_index, label in enumerate(counts.index):
        group = frame[frame[label_column].astype(str).eq(label)].copy()
        rng = np.random.default_rng(int(seed) + class_index)
        order = rng.permutation(len(group))
        selected.append(group.iloc[order[: quotas[str(label)]]])
    return pd.concat(selected).sort_values("image_id", ignore_index=True)


def inverse_frequency_weights(labels: Iterable[int], num_classes: int) -> torch.Tensor:
    values = np.asarray(list(labels), dtype=int)
    counts = np.bincount(values, minlength=int(num_classes)).astype(np.float64)
    if (counts == 0).any():
        raise ValueError("Every class needs at least one training example")
    weights = counts.sum() / (len(counts) * counts)
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


def optimizer_parameter_groups(
    model: nn.Module,
    *,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
    model_kind: str | None = None,
    layer_decay: float | None = None,
) -> list[dict]:
    """Build head/backbone groups while exempting bias and norm vectors from decay."""

    if layer_decay is not None and not 0.0 < float(layer_decay) <= 1.0:
        raise ValueError("layer_decay must be in (0, 1]")
    if layer_decay is not None and model_kind is None:
        raise ValueError("model_kind is required with layer_decay")

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    maximum_layer = 0
    if layer_decay is not None and model_kind in {"dinov2_small", "dinov2_base"}:
        block_indices = [
            int(match.group(1))
            for name in trainable_names
            if (match := re.search(r"encoder\.layer\.(\d+)", name))
        ]
        maximum_layer = max(block_indices, default=-1) + 1
    elif layer_decay is not None and model_kind == "convnext_small":
        feature_indices = [
            int(match.group(1))
            for name in trainable_names
            if (match := re.search(r"backbone\.features\.(\d+)", name))
        ]
        maximum_layer = max(feature_indices, default=0)

    groups: dict[tuple[str, bool, int], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        scope = "head" if "classifier" in name or name.startswith("head.") else "backbone"
        use_decay = parameter.ndim > 1 and not name.endswith(".bias")
        layer_id = maximum_layer
        if scope == "backbone" and layer_decay is not None:
            if model_kind in {"dinov2_small", "dinov2_base"}:
                match = re.search(r"encoder\.layer\.(\d+)", name)
                if match:
                    layer_id = int(match.group(1)) + 1
                elif "embeddings" in name:
                    layer_id = 0
            elif model_kind == "convnext_small":
                match = re.search(r"backbone\.features\.(\d+)", name)
                if match:
                    layer_id = int(match.group(1))
        groups.setdefault((scope, use_decay, layer_id), []).append(parameter)

    output = []
    for (scope, use_decay, layer_id), parameters in groups.items():
        if not parameters:
            continue
        learning_rate = float(head_lr)
        scale = 1.0
        if scope == "backbone":
            scale = (
                float(layer_decay) ** (maximum_layer - layer_id)
                if layer_decay is not None
                else 1.0
            )
            learning_rate = float(backbone_lr) * scale
        layer_name = f"_layer_{layer_id}" if scope == "backbone" and layer_decay else ""
        output.append(
            {
                "params": parameters,
                "lr": learning_rate,
                "weight_decay": float(weight_decay if use_decay else 0.0),
                "group_name": f"{scope}{layer_name}_{'decay' if use_decay else 'no_decay'}",
                "lr_scale": scale,
            }
        )
    if not output:
        raise ValueError("Model has no trainable parameters")
    return output


def warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float = 0.10,
    minimum_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    warmup_steps = int(round(total_steps * warmup_fraction))

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(minimum_ratio + (1.0 - minimum_ratio) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def is_better_validation(
    metrics: dict[str, float],
    incumbent: dict[str, float] | None,
    *,
    tolerance: float = 1e-12,
) -> bool:
    if incumbent is None:
        return True
    macro_delta = float(metrics["macro_f1"]) - float(incumbent["macro_f1"])
    if macro_delta > tolerance:
        return True
    if abs(macro_delta) <= tolerance:
        return float(metrics["log_loss"]) < float(incumbent["log_loss"]) - tolerance
    return False
