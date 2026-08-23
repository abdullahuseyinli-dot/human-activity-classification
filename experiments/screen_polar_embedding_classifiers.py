"""Compare suitable classifiers on the strongest frozen POLAR representation."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_features import load_aligned_feature_view

CLASS_NAMES = ["sitting", "standing", "walking", "running"]
MODEL_KIND = "dinov2_base"
VIEWS = ["full_frame", "person_context_10"]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Candidate:
    family: str
    parameters: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("polar_study_protocol.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--calibration-folds", type=int, default=5)
    return parser.parse_args()


def git_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def declared_candidates() -> list[Candidate]:
    candidates = [
        Candidate("multinomial_logistic", {"C": c_value, "class_weight": weight})
        for c_value in (0.001, 0.01, 0.1, 1.0)
        for weight in (None, "balanced")
    ]
    candidates.extend(
        Candidate("linear_svm", {"C": c_value, "class_weight": weight})
        for c_value in (0.0001, 0.001, 0.01, 0.1, 1.0)
        for weight in (None, "balanced")
    )
    candidates.extend(
        Candidate(
            "rbf_svm",
            {
                "C": c_value,
                "gamma_multiplier": gamma_multiplier,
                "class_weight": weight,
            },
        )
        for c_value in (1.0, 10.0, 100.0)
        for gamma_multiplier in (0.25, 1.0, 4.0)
        for weight in (None, "balanced")
    )
    candidates.extend(
        Candidate("shrinkage_lda", {"shrinkage": shrinkage})
        for shrinkage in ("auto", 0.01, 0.1, 0.5)
    )
    return candidates


def build_estimator(candidate: Candidate, feature_count: int, seed: int) -> BaseEstimator:
    params = candidate.parameters
    if candidate.family == "multinomial_logistic":
        classifier: BaseEstimator = LogisticRegression(
            C=float(params["C"]),
            class_weight=params["class_weight"],
            max_iter=3000,
            random_state=seed,
            solver="lbfgs",
            tol=1e-5,
        )
    elif candidate.family == "linear_svm":
        classifier = LinearSVC(
            C=float(params["C"]),
            class_weight=params["class_weight"],
            dual="auto",
            max_iter=20000,
            random_state=seed,
            tol=1e-4,
        )
    elif candidate.family == "rbf_svm":
        gamma = float(params["gamma_multiplier"]) / float(feature_count)
        classifier = SVC(
            C=float(params["C"]),
            cache_size=4096,
            class_weight=params["class_weight"],
            gamma=gamma,
            kernel="rbf",
            probability=False,
            random_state=seed,
        )
    elif candidate.family == "shrinkage_lda":
        classifier = LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage=params["shrinkage"],
        )
    else:
        raise ValueError(f"Unknown classifier family: {candidate.family}")
    return Pipeline([("scale", StandardScaler()), ("classifier", classifier)])


def inner_cv_metrics(
    candidate: Candidate,
    features: np.ndarray,
    labels: np.ndarray,
    folds: StratifiedKFold,
    seed: int,
) -> dict[str, float]:
    predictions = np.full(len(labels), -1, dtype=int)
    started = time.perf_counter()
    for train_indices, holdout_indices in folds.split(features, labels):
        estimator = build_estimator(candidate, features.shape[1], seed)
        estimator.fit(features[train_indices], labels[train_indices])
        predictions[holdout_indices] = estimator.predict(features[holdout_indices])
    if (predictions < 0).any():
        raise RuntimeError("Inner cross-validation did not predict every training row")
    return {
        "inner_accuracy": float(accuracy_score(labels, predictions)),
        "inner_macro_f1": float(f1_score(labels, predictions, average="macro")),
        "inner_balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "inner_fit_seconds": float(time.perf_counter() - started),
    }


def complexity_rank(candidate: Candidate) -> tuple[float, float, float]:
    params = candidate.parameters
    class_weight_rank = 0.0 if params.get("class_weight") is None else 1.0
    if candidate.family == "shrinkage_lda":
        shrinkage = params["shrinkage"]
        return (0.0, 0.0 if shrinkage == "auto" else float(shrinkage), 0.0)
    return (
        float(params.get("C", 0.0)),
        float(params.get("gamma_multiplier", 0.0)),
        class_weight_rank,
    )


def calibrated_final_estimator(
    candidate: Candidate,
    feature_count: int,
    seed: int,
    calibration_folds: int,
) -> BaseEstimator:
    estimator = build_estimator(candidate, feature_count, seed)
    if candidate.family not in {"linear_svm", "rbf_svm"}:
        return estimator
    calibration = StratifiedKFold(
        n_splits=calibration_folds,
        shuffle=True,
        random_state=seed,
    )
    return CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=calibration,
        n_jobs=1,
        ensemble=True,
    )


def serializable_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        key: "none" if value is None else value
        for key, value in sorted(parameters.items())
    }


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "1.3.0":
        raise RuntimeError("Classifier screen requires locked POLAR protocol 1.3.0")
    screen_protocol = protocol.get("frozen_classifier_screen", {})
    if screen_protocol.get("test_rows_read") != 0 or screen_protocol.get(
        "test_used_for_selection"
    ):
        raise RuntimeError("Classifier-screen protocol violates the held-out test gate")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest.resolve()
    manifest_hash = sha256_file(manifest_path)
    manifest = pd.read_csv(manifest_path, dtype={"image_id": str})
    manifest = manifest.sort_values("image_id").reset_index(drop=True)
    if set(manifest["split"].astype(str)) != {"train", "val"}:
        raise ValueError("Classifier screen requires development train and validation rows only")
    if "label_4" not in manifest:
        raise ValueError("Manifest is missing label_4")

    feature_blocks = []
    cache_provenance = {}
    for view in VIEWS:
        block, provenance = load_aligned_feature_view(
            args.feature_root.resolve(), MODEL_KIND, view, manifest, manifest_hash
        )
        feature_blocks.append(block)
        cache_provenance[view] = provenance
    features = np.concatenate(feature_blocks, axis=1)
    label_to_index = {label: index for index, label in enumerate(CLASS_NAMES)}
    labels = manifest["label_4"].map(label_to_index).to_numpy(dtype=int)
    train_mask = manifest["split"].astype(str).eq("train").to_numpy()
    val_mask = manifest["split"].astype(str).eq("val").to_numpy()
    train_features, train_labels = features[train_mask], labels[train_mask]
    val_features, val_labels = features[val_mask], labels[val_mask]
    inner_folds = StratifiedKFold(
        n_splits=args.inner_folds,
        shuffle=True,
        random_state=args.seed,
    )

    inner_rows = []
    candidates_by_family: dict[str, list[tuple[Candidate, dict[str, float]]]] = {}
    for candidate in declared_candidates():
        metrics = inner_cv_metrics(
            candidate,
            train_features,
            train_labels,
            inner_folds,
            args.seed,
        )
        row = {
            "family": candidate.family,
            **serializable_parameters(candidate.parameters),
            **metrics,
        }
        inner_rows.append(row)
        candidates_by_family.setdefault(candidate.family, []).append((candidate, metrics))
        print(
            f"[inner] {candidate.family} {serializable_parameters(candidate.parameters)} "
            f"macro_f1={metrics['inner_macro_f1']:.6f}",
            flush=True,
        )

    validation_rows = []
    selected_candidates = {}
    for family, family_candidates in candidates_by_family.items():
        candidate, inner_metrics = min(
            family_candidates,
            key=lambda item: (
                -item[1]["inner_macro_f1"],
                -item[1]["inner_balanced_accuracy"],
                complexity_rank(item[0]),
            ),
        )
        final_estimator = calibrated_final_estimator(
            candidate,
            features.shape[1],
            args.seed,
            args.calibration_folds,
        )
        started = time.perf_counter()
        final_estimator.fit(train_features, train_labels)
        probabilities = np.asarray(final_estimator.predict_proba(val_features), dtype=float)
        fit_seconds = time.perf_counter() - started
        classes = np.asarray(final_estimator.classes_, dtype=int)
        if not np.array_equal(classes, np.arange(len(CLASS_NAMES))):
            raise RuntimeError(f"Unexpected class ordering for {family}: {classes.tolist()}")
        metrics = classification_metrics(val_labels, probabilities)
        parameters = serializable_parameters(candidate.parameters)
        row = {
            "family": family,
            **parameters,
            "inner_macro_f1": inner_metrics["inner_macro_f1"],
            "inner_balanced_accuracy": inner_metrics["inner_balanced_accuracy"],
            "validation_rows": int(val_mask.sum()),
            "train_rows": int(train_mask.sum()),
            "feature_dimensions": int(features.shape[1]),
            "final_fit_seconds": float(fit_seconds),
            **metrics,
        }
        validation_rows.append(row)
        selected_candidates[family] = parameters
        np.savez_compressed(
            output_dir / f"validation_{family}.npz",
            probabilities=probabilities,
            labels=val_labels,
            image_ids=manifest.loc[val_mask, "image_id"].astype(str).to_numpy(),
            class_names=np.asarray(CLASS_NAMES),
        )
        joblib.dump(final_estimator, output_dir / f"development_{family}.joblib", compress=3)
        print(
            f"[validation] {family} macro_f1={metrics['macro_f1']:.6f} "
            f"log_loss={metrics['log_loss']:.6f} ece={metrics['ece']:.6f}",
            flush=True,
        )

    inner_frame = pd.DataFrame(inner_rows).sort_values(
        ["family", "inner_macro_f1", "inner_balanced_accuracy"],
        ascending=[True, False, False],
        ignore_index=True,
    )
    validation_frame = pd.DataFrame(validation_rows).sort_values(
        ["macro_f1", "log_loss", "family"],
        ascending=[False, True, True],
        ignore_index=True,
    )
    inner_frame.to_csv(output_dir / "classifier_inner_cv.csv", index=False)
    validation_frame.to_csv(output_dir / "classifier_validation.csv", index=False)
    write_json(
        output_dir / "classifier_screen_provenance.json",
        {
            "status": "DEVELOPMENT_ONLY_CLASSIFIER_SCREEN",
            "protocol_version": "1.3.0",
            "protocol_path": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "model_kind": MODEL_KIND,
            "views": VIEWS,
            "fusion": "feature_concatenation",
            "cache_provenance": cache_provenance,
            "feature_dimensions": int(features.shape[1]),
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(val_mask.sum()),
            "inner_folds": int(args.inner_folds),
            "calibration_folds": int(args.calibration_folds),
            "seed": int(args.seed),
            "selected_candidates": selected_candidates,
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "git_revision_at_start": git_revision(repository_root),
            "implementation_sha256": {
                "experiments/screen_polar_embedding_classifiers.py": sha256_file(
                    Path(__file__).resolve()
                ),
                "src/hac/metrics.py": sha256_file(repository_root / "src/hac/metrics.py"),
                "src/hac/polar_features.py": sha256_file(
                    repository_root / "src/hac/polar_features.py"
                ),
            },
            "test_rows_read": 0,
            "test_used_for_selection": False,
        },
    )
    print(f"[done] {output_dir / 'classifier_validation.csv'}", flush=True)


if __name__ == "__main__":
    main()
