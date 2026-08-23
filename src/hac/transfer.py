"""Probability adaptation and clustered uncertainty for transfer experiments."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .polar_analysis import _macro_f1
from .polar_training import normalize_probability_rows


def probability_logits(probabilities: np.ndarray) -> np.ndarray:
    """Return centered log probabilities suitable for affine calibration."""

    values = normalize_probability_rows(probabilities)
    logits = np.log(np.clip(values, 1e-12, 1.0))
    return logits - logits.mean(axis=1, keepdims=True)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=np.float64)
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return normalize_probability_rows(exponent)


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Fit one positive temperature by target-domain negative log likelihood."""

    logits = probability_logits(probabilities)
    labels = np.asarray(labels, dtype=int)

    def objective(log_temperature: float) -> float:
        calibrated = softmax(logits / np.exp(float(log_temperature)))
        return float(-np.log(np.clip(calibrated[np.arange(len(labels)), labels], 1e-12, 1)).mean())

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return softmax(probability_logits(probabilities) / float(temperature))


def apply_prior_ratio(
    probabilities: np.ndarray,
    source_prior: np.ndarray,
    target_prior: np.ndarray,
) -> np.ndarray:
    """Apply the standard label-prior likelihood ratio to calibrated outputs."""

    source = np.asarray(source_prior, dtype=np.float64)
    target = np.asarray(target_prior, dtype=np.float64)
    if source.ndim != 1 or target.shape != source.shape:
        raise ValueError("source and target priors must be aligned vectors")
    if (source <= 0.0).any() or (target < 0.0).any() or target.sum() <= 0.0:
        raise ValueError("priors must have valid positive mass")
    source = source / source.sum()
    target = target / target.sum()
    return normalize_probability_rows(
        normalize_probability_rows(probabilities) * (target / source)[None, :]
    )


def estimate_label_shift_em(
    probabilities: np.ndarray,
    source_prior: np.ndarray,
    *,
    tolerance: float = 1e-10,
    maximum_iterations: int = 10_000,
) -> tuple[np.ndarray, dict]:
    """Estimate an unlabeled target prior with the Saerens EM update."""

    values = normalize_probability_rows(probabilities)
    source = np.asarray(source_prior, dtype=np.float64)
    if source.ndim != 1 or len(source) != values.shape[1] or (source <= 0.0).any():
        raise ValueError("source_prior must be positive and aligned with probabilities")
    source = source / source.sum()
    target = source.copy()
    converged = False
    iterations_run = 0
    maximum_update = float("inf")
    for _iteration in range(1, int(maximum_iterations) + 1):
        iterations_run = _iteration
        adjusted = apply_prior_ratio(values, source, target)
        updated = adjusted.mean(axis=0)
        maximum_update = float(np.abs(updated - target).max())
        if maximum_update <= tolerance:
            target = updated
            converged = True
            break
        target = updated
    return target / target.sum(), {
        "converged": converged,
        "iterations": iterations_run,
        "maximum_absolute_update": maximum_update,
    }


def image_cluster_paired_bootstrap(
    labels: np.ndarray,
    challenger_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    image_ids: np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 20_260_823,
) -> dict[str, float | int]:
    """Paired macro-F1 bootstrap that resamples images and retains all people."""

    labels = np.asarray(labels, dtype=int)
    challenger = np.asarray(challenger_probabilities).argmax(axis=1)
    baseline = np.asarray(baseline_probabilities).argmax(axis=1)
    groups = np.asarray([str(value) for value in image_ids])
    if not (len(labels) == len(challenger) == len(baseline) == len(groups)):
        raise ValueError("Cluster bootstrap inputs do not align")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Cluster bootstrap requires at least two images")
    group_index = {group: index for index, group in enumerate(unique_groups)}
    encoded_groups = np.asarray([group_index[group] for group in groups], dtype=int)
    classes = (
        int(max(labels.max(initial=0), challenger.max(initial=0), baseline.max(initial=0))) + 1
    )
    challenger_by_group = np.zeros((len(unique_groups), classes, classes), dtype=np.int64)
    baseline_by_group = np.zeros_like(challenger_by_group)
    np.add.at(challenger_by_group, (encoded_groups, labels, challenger), 1)
    np.add.at(baseline_by_group, (encoded_groups, labels, baseline), 1)

    def batched_macro_f1(matrices: np.ndarray) -> np.ndarray:
        true_positive = np.diagonal(matrices, axis1=1, axis2=2).astype(np.float64)
        denominator = matrices.sum(axis=1) + matrices.sum(axis=2)
        scores = np.divide(
            2.0 * true_positive,
            denominator,
            out=np.zeros_like(true_positive),
            where=denominator != 0,
        )
        return scores.mean(axis=1)

    generator = np.random.default_rng(seed)
    deltas = np.empty(int(resamples), dtype=np.float64)
    batch_size = 256
    for start in range(0, int(resamples), batch_size):
        stop = min(start + batch_size, int(resamples))
        sampled = generator.integers(0, len(unique_groups), size=(stop - start, len(unique_groups)))
        challenger_matrices = challenger_by_group[sampled].sum(axis=1)
        baseline_matrices = baseline_by_group[sampled].sum(axis=1)
        deltas[start:stop] = batched_macro_f1(challenger_matrices) - batched_macro_f1(
            baseline_matrices
        )
    point = _macro_f1(labels, challenger, classes) - _macro_f1(labels, baseline, classes)
    return {
        "point_estimate": float(point),
        "ci_95_low": float(np.quantile(deltas, 0.025)),
        "ci_95_high": float(np.quantile(deltas, 0.975)),
        "resamples": int(resamples),
        "clusters": int(len(unique_groups)),
        "seed": int(seed),
    }
