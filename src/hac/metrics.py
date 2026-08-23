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


def adaptive_expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, *, bins: int = 15
) -> float:
    """ECE over approximately equal-count confidence bins."""

    if bins < 1:
        raise ValueError("bins must be positive")
    confidence = np.asarray(probabilities, dtype=float).max(axis=1)
    correct = np.asarray(probabilities).argmax(axis=1) == np.asarray(y_true, dtype=int)
    value = 0.0
    for indices in np.array_split(np.argsort(confidence, kind="stable"), bins):
        if len(indices):
            value += float(len(indices) / len(confidence)) * abs(
                float(correct[indices].mean()) - float(confidence[indices].mean())
            )
    return value


def classwise_expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, *, bins: int = 15
) -> float:
    """Mean one-vs-rest ECE across all declared classes."""

    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    class_values = []
    for class_index in range(values.shape[1]):
        confidence = values[:, class_index]
        positive = labels == class_index
        calibration_error = 0.0
        for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=False)):
            mask = (confidence >= left) & (
                (confidence <= right) if index == bins - 1 else (confidence < right)
            )
            if mask.any():
                calibration_error += float(mask.mean()) * abs(
                    float(positive[mask].mean()) - float(confidence[mask].mean())
                )
        class_values.append(calibration_error)
    return float(np.mean(class_values))


def selective_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    coverages: tuple[float, ...] = (1.0, 0.95, 0.90, 0.80, 0.70),
) -> tuple[list[dict[str, float | int]], float]:
    """Return fixed-confidence risk/coverage points and area under the risk curve."""

    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(probabilities, dtype=float)
    if len(labels) != len(values) or not len(labels):
        raise ValueError("Selective metrics require aligned non-empty inputs")
    if any(not 0.0 < coverage <= 1.0 for coverage in coverages):
        raise ValueError("coverages must be in (0, 1]")
    confidence = values.max(axis=1)
    predictions = values.argmax(axis=1)
    order = np.argsort(-confidence, kind="stable")
    cumulative_errors = np.cumsum(predictions[order] != labels[order])
    retained = np.arange(1, len(labels) + 1)
    risk = cumulative_errors / retained
    aurc = float(risk.mean())
    rows = []
    for coverage in coverages:
        count = max(1, int(np.ceil(float(coverage) * len(labels))))
        indices = order[:count]
        metrics = classification_metrics(labels[indices], values[indices])
        rows.append(
            {
                "requested_coverage": float(coverage),
                "realized_coverage": float(count / len(labels)),
                "retained": int(count),
                "confidence_threshold": float(confidence[indices].min()),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "risk": float(1.0 - metrics["accuracy"]),
            }
        )
    return rows, aurc


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
        "adaptive_ece": adaptive_expected_calibration_error(y_true, probabilities),
        "classwise_ece": classwise_expected_calibration_error(y_true, probabilities),
    }
