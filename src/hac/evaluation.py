"""Inference, embedding extraction, and OOF temperature scaling."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from .metrics import classification_metrics


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-6)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    *,
    iterations: int = 1000,
    learning_rate: float = 0.01,
) -> tuple[float, float]:
    scaler = TemperatureScaler().to(device)
    logits_tensor = torch.as_tensor(logits, dtype=torch.float32, device=device)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam([scaler.log_temperature], lr=float(learning_rate))
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_loss = float("inf")
    for _ in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(scaler(logits_tensor), labels_tensor)
        loss.backward()
        optimizer.step()
        if float(loss.item()) < best_loss:
            best_loss = float(loss.item())
            best_state = deepcopy(scaler.state_dict())
    if best_state is None:
        raise RuntimeError("Temperature optimization produced no valid state")
    scaler.load_state_dict(best_state)
    return float(scaler.temperature.detach().cpu().item()), best_loss


@torch.inference_mode()
def evaluate(
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
    use_amp = device.type == "cuda"
    for batch in loader:
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            if return_features:
                batch_logits, batch_features = model(inputs, return_features=True)
            else:
                batch_logits = model(inputs)
                batch_features = None
            loss = criterion(batch_logits, targets)
        batch_probabilities = torch.softmax(batch_logits, dim=1)
        losses.append(float(loss.item()))
        logits.append(batch_logits.float().cpu().numpy())
        probabilities.append(batch_probabilities.float().cpu().numpy())
        labels.append(targets.cpu().numpy())
        if batch_features is not None:
            features.append(batch_features.float().cpu().numpy())
        image_ids.extend(str(value) for value in batch["image_id"])
        image_paths.extend(str(value) for value in batch["image_path"])

    logits_array = np.concatenate(logits)
    probabilities_array = np.concatenate(probabilities)
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
