"""Classification and probability-quality metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=float)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, *, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=False)):
        mask = (confidence >= left) & (
            (confidence <= right) if index == bins - 1 else (confidence < right)
        )
        if mask.any():
            value += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return value


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or len(y_true) != len(probabilities):
        raise ValueError("Expected labels [n] and probabilities [n, classes]")
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError("Probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probability rows must sum to one")
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "log_loss": float(
            log_loss(y_true, probabilities, labels=np.arange(probabilities.shape[1]))
        ),
        "brier_score": multiclass_brier_score(y_true, probabilities),
        "ece": expected_calibration_error(y_true, probabilities),
    }
