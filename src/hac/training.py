"""Small, testable training primitives shared by scripts and notebooks."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_batch(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0 or len(inputs) < 2:
        return inputs, labels, labels, 1.0
    weight = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(inputs.size(0), device=inputs.device)
    mixed = weight * inputs + (1.0 - weight) * inputs[permutation]
    return mixed, labels, labels[permutation], weight


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    mixup_alpha: float = 0.0,
    grad_accum_steps: int = 1,
    scaler=None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    use_amp = scaler is not None and device.type == "cuda"
    for step, batch in enumerate(loader):
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        mixed, labels_a, labels_b, weight = mixup_batch(inputs, labels, mixup_alpha)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(mixed)
            raw_loss = weight * criterion(logits, labels_a) + (1.0 - weight) * criterion(
                logits, labels_b
            )
            window_start = (step // int(grad_accum_steps)) * int(grad_accum_steps)
            window_size = min(int(grad_accum_steps), len(loader) - window_start)
            loss = raw_loss / window_size
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        should_step = (step + 1) % int(grad_accum_steps) == 0 or step + 1 == len(loader)
        if should_step:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(raw_loss.detach().item()))
    return {"loss": float(np.mean(losses))}
