"""Leakage-safe nested grouped evaluation for V-COCO v3 cached features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hac.metrics import classification_metrics

CLASS_NAMES = ("sitting", "standing", "walking_running")
_CUDA_SVM_FIT_AUDIT: list[dict[str, float | int | bool | str]] = []


def reset_cuda_svm_fit_audit() -> None:
    """Clear process-local CUDA SVM optimization records before an evaluation."""

    _CUDA_SVM_FIT_AUDIT.clear()


def restore_cuda_svm_fit_audit(
    records: list[dict[str, float | int | bool | str]],
) -> None:
    """Restore optimization records from a validated evaluator checkpoint."""

    _CUDA_SVM_FIT_AUDIT.extend(dict(record) for record in records)


def cuda_svm_fit_audit() -> list[dict[str, float | int | bool | str]]:
    """Return a copy of the process-local CUDA SVM optimization records."""

    return [dict(record) for record in _CUDA_SVM_FIT_AUDIT]


@dataclass(frozen=True)
class FactorizedHead:
    posture: object
    motion: object


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    component_c: float | None
    meta_c: float | None
    svm_c: float | None
    class_weight: str


class CudaPrimalLinearSVM(ClassifierMixin, BaseEstimator):
    """Deterministic CUDA optimizer for the linear squared-hinge control."""

    def __init__(
        self,
        *,
        c_value: float,
        class_weight: str,
        maximum_iterations: int,
        tolerance: float,
        seed: int,
    ) -> None:
        self.c_value = float(c_value)
        self.class_weight = str(class_weight)
        self.maximum_iterations = int(maximum_iterations)
        self.tolerance = float(tolerance)
        self.seed = int(seed)
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.optimization_: dict[str, float | int | bool] = {}

    def fit(self, features: np.ndarray, labels: np.ndarray) -> CudaPrimalLinearSVM:
        if not torch.cuda.is_available():
            raise RuntimeError("The declared linear-SVM control requires CUDA")
        values = np.asarray(features, dtype=np.float32)
        targets = np.asarray(labels, dtype=np.int64)
        if values.ndim != 2 or len(values) != len(targets):
            raise ValueError("Linear-SVM features and labels do not align")
        if set(np.unique(targets)) != set(range(len(CLASS_NAMES))):
            raise ValueError("The linear-SVM fit requires all activity classes")
        self.classes_ = np.arange(len(CLASS_NAMES), dtype=np.int64)

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        device = torch.device("cuda")
        x = torch.as_tensor(values, device=device)
        y = torch.as_tensor(targets, device=device)
        weights = torch.ones(len(targets), dtype=torch.float32, device=device)
        if self.class_weight == "balanced":
            counts = torch.bincount(y, minlength=len(CLASS_NAMES)).to(torch.float32)
            class_weights = len(targets) / (len(CLASS_NAMES) * counts)
            weights = class_weights[y]
        elif self.class_weight != "none":
            raise ValueError(f"Unknown class weight: {self.class_weight}")

        coefficients = torch.zeros(
            (values.shape[1], len(CLASS_NAMES)),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        intercept = torch.zeros(
            len(CLASS_NAMES), dtype=torch.float32, device=device, requires_grad=True
        )
        signed_targets = torch.full(
            (len(targets), len(CLASS_NAMES)),
            -1.0,
            dtype=torch.float32,
            device=device,
        )
        signed_targets.scatter_(1, y[:, None], 1.0)
        optimizer = torch.optim.LBFGS(
            [coefficients, intercept],
            lr=1.0,
            max_iter=self.maximum_iterations,
            tolerance_grad=self.tolerance,
            tolerance_change=1e-9,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            scores = x @ coefficients + intercept
            margins = torch.clamp(1.0 - signed_targets * scores, min=0.0)
            objective = 0.5 * coefficients.square().sum()
            objective = objective + self.c_value * (
                weights[:, None] * margins.square()
            ).sum()
            objective.backward()
            return objective

        initial_objective = float(closure().detach().cpu())
        optimizer.zero_grad(set_to_none=True)
        optimizer.step(closure)
        final_objective = float(closure().detach().cpu())
        state = optimizer.state[coefficients]
        gradient_max = max(
            float(coefficients.grad.detach().abs().max().cpu()),
            float(intercept.grad.detach().abs().max().cpu()),
        )
        iterations = int(state.get("n_iter", self.maximum_iterations))
        evaluations = int(state.get("func_evals", 0))
        if not np.isfinite(final_objective) or not np.isfinite(gradient_max):
            raise RuntimeError("The CUDA linear-SVM optimizer produced non-finite values")
        self.coef_ = coefficients.detach().cpu().numpy()
        self.intercept_ = intercept.detach().cpu().numpy()
        self.optimization_ = {
            "iterations": iterations,
            "function_evaluations": evaluations,
            "initial_objective": initial_objective,
            "final_objective": final_objective,
            "final_gradient_max": gradient_max,
            "iteration_limit_reached": iterations >= self.maximum_iterations,
        }
        _CUDA_SVM_FIT_AUDIT.append(
            {
                "rows": len(targets),
                "dimensions": values.shape[1],
                "c_value": self.c_value,
                "class_weight": self.class_weight,
                "seed": self.seed,
                **self.optimization_,
            }
        )
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("The CUDA linear SVM has not been fitted")
        device = torch.device("cuda")
        values = torch.as_tensor(np.asarray(features, dtype=np.float32), device=device)
        coefficients = torch.as_tensor(self.coef_, device=device)
        intercept = torch.as_tensor(self.intercept_, device=device)
        with torch.inference_mode():
            return (values @ coefficients + intercept).cpu().numpy()


def normalize_probability_rows(values: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(values, dtype=float), 1e-12, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def decode_factorized(posture_seated: np.ndarray, upright_locomoting: np.ndarray) -> np.ndarray:
    seated = np.clip(np.asarray(posture_seated, dtype=float), 0.0, 1.0)
    locomoting = np.clip(np.asarray(upright_locomoting, dtype=float), 0.0, 1.0)
    upright = 1.0 - seated
    return normalize_probability_rows(
        np.column_stack([seated, upright * (1.0 - locomoting), upright * locomoting])
    )


def grouped_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    dummy = np.zeros((len(labels), 1), dtype=np.uint8)
    output = list(splitter.split(dummy, labels, groups))
    for train_index, held_index in output:
        if set(groups[train_index]).intersection(groups[held_index]):
            raise RuntimeError("A source image crossed a grouped fold boundary")
        if len(np.unique(labels[train_index])) != len(CLASS_NAMES):
            raise RuntimeError("A training fold does not contain all activity classes")
    return output


def _logistic(c_value: float, class_weight: str, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            class_weight="balanced" if class_weight == "balanced" else None,
            max_iter=3_000,
            random_state=seed,
            solver="lbfgs",
        ),
    )


def fit_probability_head(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    factorized: bool,
    c_value: float,
    class_weight: str,
    seed: int,
) -> object:
    if not factorized:
        model = _logistic(c_value, class_weight, seed)
        model.fit(features, labels)
        return model
    posture = _logistic(c_value, class_weight, seed + 101)
    posture.fit(features, (labels != 0).astype(int))
    upright = labels != 0
    motion = _logistic(c_value, class_weight, seed + 202)
    motion.fit(features[upright], (labels[upright] == 2).astype(int))
    return FactorizedHead(posture=posture, motion=motion)


def predict_probability_head(model: object, features: np.ndarray) -> np.ndarray:
    if isinstance(model, FactorizedHead):
        seated = model.posture.predict_proba(features)[:, 0]
        locomoting = model.motion.predict_proba(features)[:, 1]
        return decode_factorized(seated, locomoting)
    return normalize_probability_rows(model.predict_proba(features))


def geometry_features(rows) -> np.ndarray:
    area = np.clip(rows["bbox_area_fraction"].to_numpy(dtype=float), 1e-8, 1.0)
    aspect = np.clip(rows["bbox_aspect_ratio"].to_numpy(dtype=float), 1e-6, None)
    height = np.clip(rows["person_pixel_height"].to_numpy(dtype=float), 1.0, None)
    center_x = rows["bbox_center_x_fraction"].to_numpy(dtype=float)
    center_y = rows["bbox_center_y_fraction"].to_numpy(dtype=float)
    edge_distance = np.minimum.reduce([center_x, 1.0 - center_x, center_y, 1.0 - center_y])
    return np.column_stack(
        [np.log(area), np.log(aspect), center_x, center_y, np.log(height), edge_distance]
    ).astype(np.float32)


def probability_reliability_features(probabilities: list[np.ndarray]) -> np.ndarray:
    """Construct confidence, entropy, margin, and cross-view divergence features."""

    outputs = []
    clipped = [normalize_probability_rows(value) for value in probabilities]
    for value in clipped:
        ordered = np.sort(value, axis=1)
        outputs.extend(
            [
                value.max(axis=1, keepdims=True),
                (-np.sum(value * np.log(np.clip(value, 1e-12, 1.0)), axis=1, keepdims=True)),
                (ordered[:, -1] - ordered[:, -2])[:, None],
            ]
        )
    for left_index, left in enumerate(clipped):
        for right in clipped[left_index + 1 :]:
            midpoint = 0.5 * (left + right)
            divergence = 0.5 * np.sum(
                left * np.log(np.clip(left / midpoint, 1e-12, None)), axis=1
            ) + 0.5 * np.sum(right * np.log(np.clip(right / midpoint, 1e-12, None)), axis=1)
            outputs.append(divergence[:, None])
    return np.concatenate(outputs, axis=1).astype(np.float32)


def stack_features(
    probabilities: list[np.ndarray],
    geometry: np.ndarray,
    *,
    reliability: bool,
) -> np.ndarray:
    blocks = [np.log(np.clip(value, 1e-8, 1.0)) for value in probabilities]
    blocks.append(np.asarray(geometry, dtype=np.float32))
    if reliability:
        blocks.append(probability_reliability_features(probabilities))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def fit_probability_stack(
    train_features: dict[str, np.ndarray],
    target_features: dict[str, np.ndarray],
    train_geometry: np.ndarray,
    target_geometry: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    component_names: tuple[str, ...],
    component_c: float,
    meta_c: float,
    class_weight: str,
    factorized: bool,
    reliability: bool,
    stack_folds: int,
    seed: int,
) -> np.ndarray:
    splits = grouped_splits(labels, groups, folds=stack_folds, seed=seed)
    train_probabilities = []
    target_probabilities = []
    for component_index, name in enumerate(component_names):
        features = train_features[name]
        target = target_features[name]
        oof = np.zeros((len(labels), len(CLASS_NAMES)), dtype=np.float64)
        for fold, (fit_index, held_index) in enumerate(splits):
            head = fit_probability_head(
                features[fit_index],
                labels[fit_index],
                factorized=factorized,
                c_value=component_c,
                class_weight=class_weight,
                seed=seed + 1_000 * component_index + fold,
            )
            oof[held_index] = predict_probability_head(head, features[held_index])
        final_head = fit_probability_head(
            features,
            labels,
            factorized=factorized,
            c_value=component_c,
            class_weight=class_weight,
            seed=seed + 1_000 * component_index + 99,
        )
        train_probabilities.append(normalize_probability_rows(oof))
        target_probabilities.append(predict_probability_head(final_head, target))

    meta_train = stack_features(train_probabilities, train_geometry, reliability=reliability)
    meta_target = stack_features(target_probabilities, target_geometry, reliability=reliability)
    meta = fit_probability_head(
        meta_train,
        labels,
        factorized=factorized,
        c_value=meta_c,
        class_weight=class_weight,
        seed=seed + 50_000,
    )
    return predict_probability_head(meta, meta_target)


def fit_calibrated_linear_svm(
    train_features: dict[str, np.ndarray],
    target_features: dict[str, np.ndarray],
    train_geometry: np.ndarray,
    target_geometry: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    component_names: tuple[str, ...],
    c_value: float,
    class_weight: str,
    maximum_iterations: int,
    tolerance: float,
    stack_folds: int,
    seed: int,
) -> np.ndarray:
    features = np.concatenate(
        [*(train_features[name] for name in component_names), train_geometry], axis=1
    )
    target = np.concatenate(
        [*(target_features[name] for name in component_names), target_geometry], axis=1
    )

    def estimator(fold_seed: int):
        return make_pipeline(
            StandardScaler(),
            CudaPrimalLinearSVM(
                c_value=c_value,
                class_weight=class_weight,
                maximum_iterations=maximum_iterations,
                tolerance=tolerance,
                seed=fold_seed,
            ),
        )

    oof_scores = np.zeros((len(labels), len(CLASS_NAMES)), dtype=np.float64)
    for fold, (fit_index, held_index) in enumerate(
        grouped_splits(labels, groups, folds=stack_folds, seed=seed)
    ):
        model = estimator(seed + fold)
        model.fit(features[fit_index], labels[fit_index])
        oof_scores[held_index] = model.decision_function(features[held_index])
    calibrator = LogisticRegression(C=1.0, max_iter=3_000, random_state=seed, solver="lbfgs")
    calibrator.fit(oof_scores, labels)
    final = estimator(seed + 99)
    final.fit(features, labels)
    return normalize_probability_rows(calibrator.predict_proba(final.decision_function(target)))


def fit_box_augmented_factorized(
    train_tight: np.ndarray,
    train_contexts: dict[str, np.ndarray],
    target_tight: np.ndarray,
    target_contexts: dict[str, np.ndarray],
    train_geometry: np.ndarray,
    target_geometry: np.ndarray,
    labels: np.ndarray,
    *,
    c_value: float,
    class_weight: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit one factorized head on every declared train-box intervention."""

    if not train_contexts or set(train_contexts) != set(target_contexts):
        raise ValueError("Train and target box-intervention views must match")
    augmented_features = []
    augmented_labels = []
    for name in train_contexts:
        augmented_features.append(
            np.concatenate([train_tight, train_contexts[name], train_geometry], axis=1)
        )
        augmented_labels.append(labels)
    model = fit_probability_head(
        np.concatenate(augmented_features, axis=0),
        np.concatenate(augmented_labels),
        factorized=True,
        c_value=c_value,
        class_weight=class_weight,
        seed=seed,
    )
    by_intervention = {
        name: predict_probability_head(
            model,
            np.concatenate([target_tight, values, target_geometry], axis=1),
        )
        for name, values in target_contexts.items()
    }
    aggregate = normalize_probability_rows(np.stack(list(by_intervention.values())).mean(axis=0))
    return aggregate, by_intervention


def _format_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def enumerate_candidates(grid: dict, family: str) -> list[Candidate]:
    if family not in grid["families"]:
        raise ValueError(f"Unknown candidate family: {family}")
    candidates = []
    if family == "dino_siglip_linear_svm_control":
        hyperparameters = grid["hyperparameters"]["linear_svm_control"]
        for c_value in hyperparameters["C"]:
            for class_weight in hyperparameters["class_weight"]:
                candidates.append(
                    Candidate(
                        candidate_id=(
                            f"{family}__c-{_format_float(float(c_value))}__cw-{class_weight}"
                        ),
                        family=family,
                        component_c=None,
                        meta_c=None,
                        svm_c=float(c_value),
                        class_weight=str(class_weight),
                    )
                )
        return candidates
    hyperparameters = grid["hyperparameters"]["probability_stacks"]
    for component_c in hyperparameters["component_C"]:
        for meta_c in hyperparameters["meta_C"]:
            for class_weight in hyperparameters["class_weight"]:
                candidates.append(
                    Candidate(
                        candidate_id=(
                            f"{family}__cc-{_format_float(float(component_c))}"
                            f"__mc-{_format_float(float(meta_c))}__cw-{class_weight}"
                        ),
                        family=family,
                        component_c=float(component_c),
                        meta_c=float(meta_c),
                        svm_c=None,
                        class_weight=str(class_weight),
                    )
                )
    return candidates


def fit_candidate(
    candidate: Candidate,
    declaration: dict,
    train_features: dict[str, np.ndarray],
    target_features: dict[str, np.ndarray],
    train_geometry: np.ndarray,
    target_geometry: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    stack_folds: int,
    seed: int,
) -> np.ndarray:
    components = tuple(map(str, declaration["components"]))
    if candidate.family == "dino_siglip_linear_svm_control":
        return fit_calibrated_linear_svm(
            train_features,
            target_features,
            train_geometry,
            target_geometry,
            labels,
            groups,
            component_names=components,
            c_value=float(candidate.svm_c),
            class_weight=candidate.class_weight,
            maximum_iterations=int(declaration["maximum_iterations"]),
            tolerance=float(declaration["tolerance"]),
            stack_folds=stack_folds,
            seed=seed,
        )
    return fit_probability_stack(
        train_features,
        target_features,
        train_geometry,
        target_geometry,
        labels,
        groups,
        component_names=components,
        component_c=float(candidate.component_c),
        meta_c=float(candidate.meta_c),
        class_weight=candidate.class_weight,
        factorized=bool(declaration["factorized"]),
        reliability=bool(declaration["reliability_features"]),
        stack_folds=stack_folds,
        seed=seed,
    )


def locomotion_f1(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        f1_score(
            np.asarray(labels) == 2,
            np.asarray(probabilities).argmax(axis=1) == 2,
            zero_division=0,
        )
    )


def evaluate_candidate_inner(
    candidate: Candidate,
    declaration: dict,
    features: dict[str, np.ndarray],
    geometry: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    inner_folds: int,
    stack_folds: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    probabilities = np.zeros((len(labels), len(CLASS_NAMES)), dtype=np.float64)
    for fold, (fit_index, held_index) in enumerate(
        grouped_splits(labels, groups, folds=inner_folds, seed=seed)
    ):
        probabilities[held_index] = fit_candidate(
            candidate,
            declaration,
            {name: values[fit_index] for name, values in features.items()},
            {name: values[held_index] for name, values in features.items()},
            geometry[fit_index],
            geometry[held_index],
            labels[fit_index],
            groups[fit_index],
            stack_folds=stack_folds,
            seed=seed + 10_000 * (fold + 1),
        )
    metrics = classification_metrics(labels, probabilities)
    metrics["locomotion_f1"] = locomotion_f1(labels, probabilities)
    return probabilities, metrics


def candidate_rank_key(candidate: Candidate, metrics: dict[str, float]) -> tuple:
    return (
        -metrics["macro_f1"],
        -metrics["locomotion_f1"],
        metrics["log_loss"],
        candidate.candidate_id,
    )


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for index, name in enumerate(ordered):
        value = min(1.0, (total - index) * float(p_values[name]))
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def paired_cluster_bootstrap(
    labels: np.ndarray,
    challenger_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    groups: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict:
    """Paired image-cluster intervals for macro and class F1 differences."""

    labels = np.asarray(labels, dtype=int)
    challenger = np.asarray(challenger_probabilities).argmax(axis=1)
    baseline = np.asarray(baseline_probabilities).argmax(axis=1)
    groups = np.asarray(groups).astype(str)
    unique_groups = np.unique(groups)
    classes = len(CLASS_NAMES)
    group_index = {value: index for index, value in enumerate(unique_groups)}
    encoded = np.asarray([group_index[value] for value in groups], dtype=int)
    challenger_confusion = np.zeros((len(unique_groups), classes, classes), dtype=np.int64)
    baseline_confusion = np.zeros_like(challenger_confusion)
    np.add.at(challenger_confusion, (encoded, labels, challenger), 1)
    np.add.at(baseline_confusion, (encoded, labels, baseline), 1)

    def class_f1(matrices: np.ndarray) -> np.ndarray:
        true_positive = np.diagonal(matrices, axis1=1, axis2=2).astype(float)
        denominator = matrices.sum(axis=1) + matrices.sum(axis=2)
        return np.divide(
            2.0 * true_positive,
            denominator,
            out=np.zeros_like(true_positive),
            where=denominator != 0,
        )

    observed_challenger = class_f1(challenger_confusion.sum(axis=0, keepdims=True))[0]
    observed_baseline = class_f1(baseline_confusion.sum(axis=0, keepdims=True))[0]
    rng = np.random.default_rng(seed)
    deltas = np.empty((resamples, classes), dtype=float)
    batch_size = 256
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        sampled = rng.integers(0, len(unique_groups), size=(stop - start, len(unique_groups)))
        challenger_f1 = class_f1(challenger_confusion[sampled].sum(axis=1))
        baseline_f1 = class_f1(baseline_confusion[sampled].sum(axis=1))
        deltas[start:stop] = challenger_f1 - baseline_f1
    macro_deltas = deltas.mean(axis=1)

    def summarize(values: np.ndarray, point: float) -> dict[str, float]:
        probability_nonpositive = (np.count_nonzero(values <= 0.0) + 1) / (len(values) + 1)
        probability_nonnegative = (np.count_nonzero(values >= 0.0) + 1) / (len(values) + 1)
        return {
            "point_estimate": float(point),
            "ci_95_low": float(np.quantile(values, 0.025)),
            "ci_95_high": float(np.quantile(values, 0.975)),
            "two_sided_p": float(
                min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative))
            ),
        }

    return {
        "macro_f1": summarize(
            macro_deltas, float((observed_challenger - observed_baseline).mean())
        ),
        "per_class_f1": {
            name: summarize(deltas[:, index], observed_challenger[index] - observed_baseline[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "clusters": len(unique_groups),
        "resamples": resamples,
        "seed": seed,
    }
