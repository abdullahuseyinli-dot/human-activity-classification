import numpy as np
import pytest

from hac.metrics import classification_metrics, selective_classification_metrics


def test_perfect_probabilities_have_perfect_hard_metrics():
    labels = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[labels] * 0.9 + 0.1 / 3.0
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    metrics = classification_metrics(labels, probabilities)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["adaptive_ece"] >= 0.0
    assert metrics["classwise_ece"] >= 0.0


def test_invalid_probability_rows_are_rejected():
    with pytest.raises(ValueError, match="sum to one"):
        classification_metrics(np.array([0]), np.array([[0.2, 0.2, 0.2]]))


def test_selective_metrics_retain_highest_confidence_first():
    labels = np.asarray([0, 1, 1, 0])
    probabilities = np.asarray(
        [[0.99, 0.01], [0.05, 0.95], [0.55, 0.45], [0.40, 0.60]], dtype=float
    )

    rows, aurc = selective_classification_metrics(labels, probabilities, coverages=(1.0, 0.5))

    assert rows[1]["retained"] == 2
    assert rows[1]["accuracy"] == 1.0
    assert 0.0 <= aurc <= 1.0
