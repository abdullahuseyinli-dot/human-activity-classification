import numpy as np
import pytest

from hac.metrics import classification_metrics


def test_perfect_probabilities_have_perfect_hard_metrics():
    labels = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[labels] * 0.9 + 0.1 / 3.0
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    metrics = classification_metrics(labels, probabilities)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_invalid_probability_rows_are_rejected():
    with pytest.raises(ValueError, match="sum to one"):
        classification_metrics(np.array([0]), np.array([[0.2, 0.2, 0.2]]))
