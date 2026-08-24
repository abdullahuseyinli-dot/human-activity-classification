"""CUDA-native linear probability heads for the V-COCO v3 studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hac.metrics import classification_metrics
from hac.vcoco_v3_models import (
    CLASS_NAMES,
    Candidate,
    decode_factorized,
    grouped_splits,
    locomotion_f1,
    normalize_probability_rows,
    stack_features,
)

_CUDA_LOGISTIC_FIT_AUDIT: list[dict[str, float | int | bool | str]] = []


def reset_cuda_logistic_fit_audit() -> None:
    """Clear process-local optimization records."""

    _CUDA_LOGISTIC_FIT_AUDIT.clear()


def restore_cuda_logistic_fit_audit(
    records: list[dict[str, float | int | bool | str]],
) -> None:
    """Restore records from a source-validated evaluator checkpoint."""

    _CUDA_LOGISTIC_FIT_AUDIT.extend(dict(record) for record in records)


def cuda_logistic_fit_audit() -> list[dict[str, float | int | bool | str]]:
    """Return a defensive copy of the optimization records."""

    return [dict(record) for record in _CUDA_LOGISTIC_FIT_AUDIT]


@dataclass(frozen=True)
class CudaFactorizedHead:
    posture: object
    motion: object


class CudaStandardizedLogisticRegression:
    """L2-regularized binary or multinomial logistic regression fitted on CUDA.

    Standardization is learned inside each training fold on the GPU. The objective
    follows the scaling used by scikit-learn's LBFGS logistic estimator: weighted
    mean log loss plus an L2 penalty of ``1 / (2 * C * weight_sum)``.
    """

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
        self.classes_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.optimization_: dict[str, float | int | bool] = {}

    def fit(
        self, features: np.ndarray, labels: np.ndarray
    ) -> CudaStandardizedLogisticRegression:
        if not torch.cuda.is_available():
            raise RuntimeError("The declared logistic-regression head requires CUDA")
        if self.c_value <= 0.0:
            raise ValueError("Logistic-regression C must be positive")
        values = np.asarray(features, dtype=np.float32)
        targets = np.asarray(labels)
        if values.ndim != 2 or targets.ndim != 1 or len(values) != len(targets):
            raise ValueError("Logistic-regression features and labels do not align")
        classes, encoded = np.unique(targets, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("Logistic regression requires at least two classes")
        self.classes_ = classes

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        device = torch.device("cuda")
        x = torch.as_tensor(values, dtype=torch.float32, device=device)
        y = torch.as_tensor(encoded, dtype=torch.long, device=device)
        mean = x.mean(dim=0)
        scale = x.var(dim=0, correction=0).sqrt()
        scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
        x = (x - mean) / scale

        sample_weight = torch.ones(len(y), dtype=torch.float32, device=device)
        if self.class_weight == "balanced":
            counts = torch.bincount(y, minlength=len(classes)).to(torch.float32)
            weights = len(y) / (len(classes) * counts)
            sample_weight = weights[y]
        elif self.class_weight != "none":
            raise ValueError(f"Unknown class weight: {self.class_weight}")
        weight_sum = sample_weight.sum()
        regularization = 1.0 / (self.c_value * float(weight_sum.detach().cpu()))

        if len(classes) == 2:
            coefficients = torch.zeros(
                values.shape[1], dtype=torch.float32, device=device, requires_grad=True
            )
            intercept = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
        else:
            coefficients = torch.zeros(
                (values.shape[1], len(classes)),
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            intercept = torch.zeros(
                len(classes), dtype=torch.float32, device=device, requires_grad=True
            )

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
            if len(classes) == 2:
                logits = x @ coefficients + intercept[0]
                losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y.to(torch.float32), reduction="none"
                )
            else:
                logits = x @ coefficients + intercept
                losses = torch.nn.functional.cross_entropy(logits, y, reduction="none")
            objective = (sample_weight * losses).sum() / weight_sum
            objective = objective + 0.5 * regularization * coefficients.square().sum()
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
            raise RuntimeError("The CUDA logistic optimizer produced non-finite values")

        self.mean_ = mean.detach().cpu().numpy()
        self.scale_ = scale.detach().cpu().numpy()
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
        _CUDA_LOGISTIC_FIT_AUDIT.append(
            {
                "rows": len(targets),
                "dimensions": values.shape[1],
                "classes": len(classes),
                "c_value": self.c_value,
                "class_weight": self.class_weight,
                "seed": self.seed,
                **self.optimization_,
            }
        )
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if any(
            value is None
            for value in (self.classes_, self.mean_, self.scale_, self.coef_, self.intercept_)
        ):
            raise RuntimeError("The CUDA logistic head has not been fitted")
        device = torch.device("cuda")
        x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=device)
        mean = torch.as_tensor(self.mean_, device=device)
        scale = torch.as_tensor(self.scale_, device=device)
        coefficients = torch.as_tensor(self.coef_, device=device)
        intercept = torch.as_tensor(self.intercept_, device=device)
        with torch.inference_mode():
            x = (x - mean) / scale
            if len(self.classes_) == 2:
                positive = torch.sigmoid(x @ coefficients + intercept[0])
                probabilities = torch.column_stack((1.0 - positive, positive))
            else:
                probabilities = torch.softmax(x @ coefficients + intercept, dim=1)
        return probabilities.cpu().numpy()

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("The CUDA logistic head has not been fitted")
        return self.classes_[self.predict_proba(features).argmax(axis=1)]


def fit_probability_head_cuda(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    factorized: bool,
    c_value: float,
    class_weight: str,
    seed: int,
    maximum_iterations: int,
    tolerance: float,
) -> object:
    def estimator(fit_seed: int) -> CudaStandardizedLogisticRegression:
        return CudaStandardizedLogisticRegression(
            c_value=c_value,
            class_weight=class_weight,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
            seed=fit_seed,
        )

    if not factorized:
        model = estimator(seed)
        model.fit(features, labels)
        return model
    posture = estimator(seed + 101)
    posture.fit(features, (labels != 0).astype(int))
    upright = labels != 0
    motion = estimator(seed + 202)
    motion.fit(features[upright], (labels[upright] == 2).astype(int))
    return CudaFactorizedHead(posture=posture, motion=motion)


def predict_probability_head_cuda(model: object, features: np.ndarray) -> np.ndarray:
    if isinstance(model, CudaFactorizedHead):
        seated = model.posture.predict_proba(features)[:, 0]
        locomoting = model.motion.predict_proba(features)[:, 1]
        return decode_factorized(seated, locomoting)
    return normalize_probability_rows(model.predict_proba(features))


def fit_probability_stack_cuda_many(
    train_features: dict[str, np.ndarray],
    target_feature_sets: dict[str, dict[str, np.ndarray]],
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
    maximum_iterations: int,
    tolerance: float,
) -> dict[str, np.ndarray]:
    if not target_feature_sets:
        raise ValueError("At least one CUDA probability-stack target is required")
    splits = grouped_splits(labels, groups, folds=stack_folds, seed=seed)
    train_probabilities = []
    target_probabilities = {name: [] for name in target_feature_sets}
    for component_index, name in enumerate(component_names):
        features = train_features[name]
        oof = np.zeros((len(labels), len(CLASS_NAMES)), dtype=np.float64)
        for fold, (fit_index, held_index) in enumerate(splits):
            head = fit_probability_head_cuda(
                features[fit_index],
                labels[fit_index],
                factorized=factorized,
                c_value=component_c,
                class_weight=class_weight,
                seed=seed + 1_000 * component_index + fold,
                maximum_iterations=maximum_iterations,
                tolerance=tolerance,
            )
            oof[held_index] = predict_probability_head_cuda(head, features[held_index])
        final_head = fit_probability_head_cuda(
            features,
            labels,
            factorized=factorized,
            c_value=component_c,
            class_weight=class_weight,
            seed=seed + 1_000 * component_index + 99,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
        )
        train_probabilities.append(normalize_probability_rows(oof))
        for condition, target_features in target_feature_sets.items():
            target_probabilities[condition].append(
                predict_probability_head_cuda(final_head, target_features[name])
            )

    meta_train = stack_features(train_probabilities, train_geometry, reliability=reliability)
    meta = fit_probability_head_cuda(
        meta_train,
        labels,
        factorized=factorized,
        c_value=meta_c,
        class_weight=class_weight,
        seed=seed + 50_000,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
    )
    return {
        condition: predict_probability_head_cuda(
            meta,
            stack_features(probabilities, target_geometry, reliability=reliability),
        )
        for condition, probabilities in target_probabilities.items()
    }


def fit_probability_stack_cuda(
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
    maximum_iterations: int,
    tolerance: float,
) -> np.ndarray:
    return fit_probability_stack_cuda_many(
        train_features,
        {"target": target_features},
        train_geometry,
        target_geometry,
        labels,
        groups,
        component_names=component_names,
        component_c=component_c,
        meta_c=meta_c,
        class_weight=class_weight,
        factorized=factorized,
        reliability=reliability,
        stack_folds=stack_folds,
        seed=seed,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
    )["target"]


def fit_box_augmented_factorized_cuda(
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
    maximum_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not train_contexts or set(train_contexts) != set(target_contexts):
        raise ValueError("Train and target box-intervention views must match")
    augmented_features = []
    augmented_labels = []
    for name in train_contexts:
        augmented_features.append(
            np.concatenate([train_tight, train_contexts[name], train_geometry], axis=1)
        )
        augmented_labels.append(labels)
    model = fit_probability_head_cuda(
        np.concatenate(augmented_features, axis=0),
        np.concatenate(augmented_labels),
        factorized=True,
        c_value=c_value,
        class_weight=class_weight,
        seed=seed,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
    )
    by_intervention = {
        name: predict_probability_head_cuda(
            model,
            np.concatenate([target_tight, values, target_geometry], axis=1),
        )
        for name, values in target_contexts.items()
    }
    aggregate = normalize_probability_rows(np.stack(list(by_intervention.values())).mean(axis=0))
    return aggregate, by_intervention


def fit_candidate_cuda(
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
    maximum_iterations: int,
    tolerance: float,
) -> np.ndarray:
    if candidate.svm_c is not None:
        raise ValueError("The CUDA probability-head path does not accept SVM candidates")
    return fit_probability_stack_cuda(
        train_features,
        target_features,
        train_geometry,
        target_geometry,
        labels,
        groups,
        component_names=tuple(map(str, declaration["components"])),
        component_c=float(candidate.component_c),
        meta_c=float(candidate.meta_c),
        class_weight=candidate.class_weight,
        factorized=bool(declaration["factorized"]),
        reliability=bool(declaration["reliability_features"]),
        stack_folds=stack_folds,
        seed=seed,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
    )


def fit_candidate_cuda_many(
    candidate: Candidate,
    declaration: dict,
    train_features: dict[str, np.ndarray],
    target_feature_sets: dict[str, dict[str, np.ndarray]],
    train_geometry: np.ndarray,
    target_geometry: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    stack_folds: int,
    seed: int,
    maximum_iterations: int,
    tolerance: float,
) -> dict[str, np.ndarray]:
    """Fit one CUDA stack and score several matched target interventions."""

    if candidate.svm_c is not None:
        raise ValueError("The CUDA probability-head path does not accept SVM candidates")
    return fit_probability_stack_cuda_many(
        train_features,
        target_feature_sets,
        train_geometry,
        target_geometry,
        labels,
        groups,
        component_names=tuple(map(str, declaration["components"])),
        component_c=float(candidate.component_c),
        meta_c=float(candidate.meta_c),
        class_weight=candidate.class_weight,
        factorized=bool(declaration["factorized"]),
        reliability=bool(declaration["reliability_features"]),
        stack_folds=stack_folds,
        seed=seed,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
    )


def evaluate_candidate_inner_cuda(
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
    maximum_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, float]]:
    probabilities = np.zeros((len(labels), len(CLASS_NAMES)), dtype=np.float64)
    for fold, (fit_index, held_index) in enumerate(
        grouped_splits(labels, groups, folds=inner_folds, seed=seed)
    ):
        probabilities[held_index] = fit_candidate_cuda(
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
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
        )
    metrics = classification_metrics(labels, probabilities)
    metrics["locomotion_f1"] = locomotion_f1(labels, probabilities)
    return probabilities, metrics
