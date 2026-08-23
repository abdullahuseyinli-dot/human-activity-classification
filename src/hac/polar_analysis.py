"""Validation and uncertainty analysis for POLAR prediction artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .metrics import classification_metrics
from .polar_training import normalize_probability_rows


@dataclass(frozen=True)
class PredictionArtifact:
    """A validated, path-independent set of model probabilities."""

    image_ids: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray
    class_names: tuple[str, ...]


def load_prediction_artifact(path: Path) -> PredictionArtifact:
    """Load and validate the common prediction ``npz`` contract."""

    with np.load(path, allow_pickle=True) as payload:
        required = {"image_ids", "labels", "probabilities", "class_names"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"Prediction artifact is missing keys: {sorted(missing)}")
        image_ids = np.asarray([str(value) for value in payload["image_ids"]])
        labels = np.asarray(payload["labels"], dtype=int)
        probabilities = normalize_probability_rows(payload["probabilities"])
        class_names = tuple(str(value) for value in payload["class_names"])

    if image_ids.ndim != 1 or labels.ndim != 1:
        raise ValueError("image_ids and labels must be one-dimensional")
    if len(image_ids) != len(labels) or len(labels) != len(probabilities):
        raise ValueError("Prediction artifact rows do not align")
    if len(np.unique(image_ids)) != len(image_ids):
        raise ValueError("Prediction artifact image identifiers must be unique")
    if probabilities.shape[1] != len(class_names):
        raise ValueError("Probability columns and class names do not align")
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= len(class_names):
        raise ValueError("Prediction artifact contains an out-of-range label")
    return PredictionArtifact(image_ids, labels, probabilities, class_names)


def align_prediction_artifacts(
    artifacts: Mapping[str, PredictionArtifact],
) -> dict[str, PredictionArtifact]:
    """Align artifacts to sorted image identifiers and prove label agreement."""

    if not artifacts:
        raise ValueError("At least one prediction artifact is required")
    common_ids: set[str] | None = None
    for artifact in artifacts.values():
        values = set(artifact.image_ids.tolist())
        common_ids = values if common_ids is None else common_ids & values
    if common_ids is None or not common_ids:
        raise ValueError("Prediction artifacts have no common image identifiers")
    expected_rows = {name: len(value.image_ids) for name, value in artifacts.items()}
    if any(count != len(common_ids) for count in expected_rows.values()):
        raise ValueError(
            "Prediction artifacts must contain the same image identifiers; "
            f"found row counts {expected_rows} and {len(common_ids)} common rows"
        )

    ordered_ids = np.asarray(sorted(common_ids))
    aligned: dict[str, PredictionArtifact] = {}
    reference_labels: np.ndarray | None = None
    reference_classes: tuple[str, ...] | None = None
    for name, artifact in artifacts.items():
        index = {value: row for row, value in enumerate(artifact.image_ids.tolist())}
        order = np.asarray([index[value] for value in ordered_ids], dtype=int)
        candidate = PredictionArtifact(
            image_ids=ordered_ids.copy(),
            labels=artifact.labels[order],
            probabilities=artifact.probabilities[order],
            class_names=artifact.class_names,
        )
        if reference_labels is None:
            reference_labels = candidate.labels
            reference_classes = candidate.class_names
        elif not np.array_equal(candidate.labels, reference_labels):
            raise ValueError(f"Ground-truth labels differ for artifact {name}")
        elif candidate.class_names != reference_classes:
            raise ValueError(f"Class order differs for artifact {name}")
        aligned[name] = candidate
    return aligned


def simplex_weights(names: Iterable[str], step: float = 0.05) -> list[dict[str, float]]:
    """Enumerate an exact probability simplex on a declared grid."""

    names = tuple(names)
    if len(names) < 2:
        raise ValueError("A blend requires at least two models")
    units = int(round(1.0 / float(step)))
    if units < 1 or not np.isclose(units * float(step), 1.0, atol=1e-12):
        raise ValueError("Blend step must divide one exactly")

    output: list[dict[str, float]] = []

    def visit(position: int, remaining: int, values: list[int]) -> None:
        if position == len(names) - 1:
            allocation = [*values, remaining]
            output.append(
                {name: float(value / units) for name, value in zip(names, allocation, strict=True)}
            )
            return
        for value in range(remaining + 1):
            visit(position + 1, remaining - value, [*values, value])

    visit(0, units, [])
    return output


def blend_probabilities(
    artifacts: Mapping[str, PredictionArtifact], weights: Mapping[str, float]
) -> np.ndarray:
    if set(artifacts) != set(weights):
        raise ValueError("Blend weights must name every and only supplied artifact")
    total = float(sum(weights.values()))
    if not np.isclose(total, 1.0, atol=1e-12) or any(value < 0.0 for value in weights.values()):
        raise ValueError("Blend weights must be non-negative and sum to one")
    probability_sum = sum(
        float(weights[name]) * artifact.probabilities for name, artifact in artifacts.items()
    )
    return normalize_probability_rows(probability_sum)


def select_probability_blend(
    artifacts: Mapping[str, PredictionArtifact], *, step: float = 0.05
) -> tuple[dict[str, float], dict[str, float], np.ndarray]:
    """Select macro-F1 first, then log loss and ECE, on the declared grid."""

    aligned = align_prediction_artifacts(artifacts)
    labels = next(iter(aligned.values())).labels
    best: tuple[tuple[float, float, float, tuple[float, ...]], dict, dict, np.ndarray] | None = None
    for weights in simplex_weights(aligned, step):
        probabilities = blend_probabilities(aligned, weights)
        metrics = classification_metrics(labels, probabilities)
        ordered_weights = tuple(weights[name] for name in aligned)
        rank = (-metrics["macro_f1"], metrics["log_loss"], metrics["ece"], ordered_weights)
        if best is None or rank < best[0]:
            best = (rank, weights, metrics, probabilities)
    if best is None:
        raise RuntimeError("Blend search generated no candidates")
    return best[1], best[2], best[3]


def per_class_metrics(
    labels: np.ndarray, probabilities: np.ndarray, class_names: Iterable[str]
) -> list[dict[str, float | int | str]]:
    names = tuple(class_names)
    predictions = np.asarray(probabilities).argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(len(names)),
        zero_division=0,
    )
    return [
        {
            "class": name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(names)
    ]


def confusion_metrics(
    labels: np.ndarray, probabilities: np.ndarray, class_names: Iterable[str]
) -> dict[str, list]:
    names = tuple(class_names)
    matrix = confusion_matrix(
        labels,
        np.asarray(probabilities).argmax(axis=1),
        labels=np.arange(len(names)),
    )
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    return {
        "class_names": list(names),
        "counts": matrix.tolist(),
        "row_normalized": normalized.tolist(),
    }


def complementarity_metrics(
    labels: np.ndarray, left: np.ndarray, right: np.ndarray
) -> dict[str, float | int]:
    """Summarize pairwise correctness and probability complementarity."""

    left = normalize_probability_rows(left)
    right = normalize_probability_rows(right)
    if left.shape != right.shape or len(labels) != len(left):
        raise ValueError("Complementarity inputs do not align")
    left_prediction = left.argmax(axis=1)
    right_prediction = right.argmax(axis=1)
    left_correct = left_prediction == labels
    right_correct = right_prediction == labels
    disagreement = left_prediction != right_prediction
    both_wrong = ~left_correct & ~right_correct
    left_only = left_correct & ~right_correct
    right_only = right_correct & ~left_correct
    either_correct = left_correct | right_correct
    probability_correlation = float(np.corrcoef(left.ravel(), right.ravel())[0, 1])
    return {
        "rows": int(len(labels)),
        "prediction_disagreement_rate": float(disagreement.mean()),
        "double_fault_rate": float(both_wrong.mean()),
        "oracle_accuracy": float(either_correct.mean()),
        "left_unique_correct_count": int(left_only.sum()),
        "right_unique_correct_count": int(right_only.sum()),
        "left_unique_correct_rate": float(left_only.mean()),
        "right_unique_correct_rate": float(right_only.mean()),
        "flattened_probability_pearson_correlation": probability_correlation,
    }


def _macro_f1(labels: np.ndarray, predictions: np.ndarray, classes: int) -> float:
    encoded = labels * classes + predictions
    matrix = np.bincount(encoded, minlength=classes * classes).reshape(classes, classes)
    true_positive = np.diag(matrix).astype(float)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    scores = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros(classes, dtype=float),
        where=denominator != 0,
    )
    return float(scores.mean())


def stratified_paired_bootstrap(
    labels: np.ndarray,
    left_probabilities: np.ndarray,
    right_probabilities: np.ndarray | None = None,
    *,
    resamples: int = 10_000,
    seed: int = 20_260_822,
) -> dict[str, float | int]:
    """Bootstrap macro-F1 or a paired macro-F1 delta within each class."""

    labels = np.asarray(labels, dtype=int)
    left_predictions = np.asarray(left_probabilities).argmax(axis=1)
    right_predictions = (
        None if right_probabilities is None else np.asarray(right_probabilities).argmax(axis=1)
    )
    if len(labels) != len(left_predictions):
        raise ValueError("Bootstrap labels and left predictions do not align")
    if right_predictions is not None and len(labels) != len(right_predictions):
        raise ValueError("Bootstrap labels and right predictions do not align")
    if resamples < 1:
        raise ValueError("resamples must be positive")

    classes = int(labels.max(initial=0)) + 1
    groups = [np.flatnonzero(labels == class_index) for class_index in range(classes)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Stratified bootstrap requires every class")
    generator = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=float)
    for iteration in range(resamples):
        indices = np.concatenate(
            [generator.choice(group, size=len(group), replace=True) for group in groups]
        )
        left_value = _macro_f1(labels[indices], left_predictions[indices], classes)
        if right_predictions is None:
            values[iteration] = left_value
        else:
            right_value = _macro_f1(labels[indices], right_predictions[indices], classes)
            values[iteration] = left_value - right_value
    point_left = _macro_f1(labels, left_predictions, classes)
    point = (
        point_left
        if right_predictions is None
        else point_left - _macro_f1(labels, right_predictions, classes)
    )
    return {
        "point_estimate": float(point),
        "ci_95_low": float(np.quantile(values, 0.025)),
        "ci_95_high": float(np.quantile(values, 0.975)),
        "resamples": int(resamples),
        "seed": int(seed),
    }
