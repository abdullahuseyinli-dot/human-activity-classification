"""Reproducible training utilities for the leakage-audited POLAR study."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

TASK_LABELS = {
    "label_4": ("sitting", "standing", "walking", "running"),
    "label_3": ("sitting", "standing", "walking_running"),
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
) -> list[dict]:
    """Build head/backbone groups while exempting bias and norm vectors from decay."""

    groups: dict[tuple[str, bool], list[nn.Parameter]] = {
        ("head", True): [],
        ("head", False): [],
        ("backbone", True): [],
        ("backbone", False): [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        scope = "head" if "classifier" in name or name.startswith("head.") else "backbone"
        use_decay = parameter.ndim > 1 and not name.endswith(".bias")
        groups[(scope, use_decay)].append(parameter)

    output = []
    for (scope, use_decay), parameters in groups.items():
        if not parameters:
            continue
        output.append(
            {
                "params": parameters,
                "lr": float(head_lr if scope == "head" else backbone_lr),
                "weight_decay": float(weight_decay if use_decay else 0.0),
                "group_name": f"{scope}_{'decay' if use_decay else 'no_decay'}",
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
