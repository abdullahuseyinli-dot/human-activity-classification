"""Fit one locked frozen-feature POLAR classifier on all clean development rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from screen_polar_embedding_classifiers import Candidate, calibrated_final_estimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hac.polar import sha256_file
from hac.polar_features import load_aligned_feature_view
from hac.polar_training import TASK_LABELS, validate_development_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--probe-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-kind", required=True)
    parser.add_argument(
        "--representation",
        choices=["final_cls", "last4_cls_mean_patch"],
        required=True,
    )
    parser.add_argument("--view", action="append", required=True, dest="views")
    parser.add_argument("--task", choices=sorted(TASK_LABELS), required=True)
    parser.add_argument(
        "--classifier",
        choices=["standardized_multinomial_logistic", "calibrated_rbf_svm"],
        required=True,
    )
    parser.add_argument("--c-value", type=float, required=True)
    parser.add_argument("--class-weight", choices=["none", "balanced"], required=True)
    parser.add_argument("--gamma-multiplier", type=float, default=1.0)
    parser.add_argument("--calibration-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def configuration(args: argparse.Namespace) -> dict:
    shared = {
        "model_kind": args.model_kind,
        "representation": args.representation,
        "views": args.views,
        "task": args.task,
        "classifier": args.classifier,
        "C": args.c_value,
        "class_weight": args.class_weight,
        "seed": args.seed,
    }
    if args.classifier == "standardized_multinomial_logistic":
        return {
            **shared,
            "solver": "lbfgs",
            "max_iter": 2_000,
            "tol": 1e-5,
        }
    return {
        **shared,
        "kernel": "rbf",
        "gamma_multiplier": args.gamma_multiplier,
        "calibration": "sigmoid",
        "calibration_folds": args.calibration_folds,
        "calibration_ensemble": True,
        "cache_size_mb": 4_096,
    }


def build_estimator(args: argparse.Namespace, feature_count: int):
    class_weight = None if args.class_weight == "none" else args.class_weight
    if args.classifier == "standardized_multinomial_logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=args.c_value,
                        class_weight=class_weight,
                        max_iter=2_000,
                        random_state=args.seed,
                        solver="lbfgs",
                        tol=1e-5,
                    ),
                ),
            ]
        )
    candidate = Candidate(
        "rbf_svm",
        {
            "C": args.c_value,
            "gamma_multiplier": args.gamma_multiplier,
            "class_weight": class_weight,
        },
    )
    return calibrated_final_estimator(
        candidate,
        feature_count,
        args.seed,
        args.calibration_folds,
    )


def fitted_estimator_details(
    estimator, classifier: str, feature_count: int, gamma_multiplier: float
) -> dict:
    if classifier == "standardized_multinomial_logistic":
        iterations = int(np.max(estimator.named_steps["classifier"].n_iter_))
        return {"iterations": iterations, "converged": iterations < 2_000}
    support_vectors = [
        int(np.asarray(calibrated.estimator.named_steps["classifier"].n_support_).sum())
        for calibrated in estimator.calibrated_classifiers_
    ]
    return {
        "resolved_gamma": float(gamma_multiplier / feature_count),
        "support_vector_counts_by_calibration_fold": support_vectors,
        "mean_support_vectors": float(np.mean(support_vectors)),
    }


def validate_lock(args: argparse.Namespace, config: dict) -> str:
    lock_path = args.selection_lock.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "FINAL_SELECTION_LOCKED_PRE_TEST":
        raise RuntimeError("Final fitting requires a FINAL_SELECTION_LOCKED_PRE_TEST lock")
    if lock.get("test_rows_read") != 0 or lock.get("test_used_for_selection"):
        raise RuntimeError("Selection lock violates the test gate")
    try:
        expected = lock["final_probe_fits"][args.probe_id]["configuration"]
    except KeyError as error:
        raise RuntimeError(f"Probe id is absent from selection lock: {args.probe_id}") from error
    if config != expected:
        raise RuntimeError(
            "Final-probe arguments differ from the selection lock: "
            f"expected={expected!r}, observed={config!r}"
        )
    return sha256_file(lock_path)


def main() -> None:
    args = parse_args()
    if args.c_value <= 0.0:
        raise ValueError("c-value must be positive")
    if args.gamma_multiplier <= 0.0:
        raise ValueError("gamma-multiplier must be positive")
    if args.calibration_folds < 2:
        raise ValueError("calibration-folds must be at least two")
    if len(set(args.views)) != len(args.views):
        raise ValueError("Views must be unique")
    config = configuration(args)
    selection_lock_hash = validate_lock(args, config)
    manifest_path = args.manifest.resolve()
    manifest_hash = sha256_file(manifest_path)
    manifest = validate_development_manifest(
        pd.read_csv(manifest_path, dtype={"image_id": str}), args.task
    )
    class_names = list(TASK_LABELS[args.task])
    class_to_index = {name: index for index, name in enumerate(class_names)}
    labels = manifest["label"].map(class_to_index).to_numpy(dtype=int)

    feature_views = []
    cache_provenance = {}
    for view in args.views:
        features, provenance = load_aligned_feature_view(
            args.feature_root.resolve(), args.model_kind, view, manifest, manifest_hash
        )
        feature_views.append(features)
        cache_provenance[view] = provenance
        if provenance.get("representation", "final_cls") != args.representation:
            raise RuntimeError(f"Feature representation differs for {view}: {provenance}")
    features = np.concatenate(feature_views, axis=1)

    request_core = {
        "status": "LOCKED_FINAL_DEVELOPMENT_PROBE_REQUEST",
        "probe_id": args.probe_id,
        "configuration": config,
        "selection_lock_sha256": selection_lock_hash,
        "manifest_sha256": manifest_hash,
        "feature_cache_sha256": {
            view: {
                "features": sha256_file(
                    args.feature_root.resolve() / args.model_kind / view / "features.npy"
                ),
                "rows": sha256_file(
                    args.feature_root.resolve() / args.model_kind / view / "rows.csv"
                ),
                "provenance": sha256_file(
                    args.feature_root.resolve() / args.model_kind / view / "provenance.json"
                ),
            }
            for view in args.views
        },
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "test_rows_read": 0,
    }
    request_hash = hashlib.sha256(
        json.dumps(request_core, sort_keys=True).encode("utf-8")
    ).hexdigest()
    request = {**request_core, "request_sha256": request_hash}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "request.json"
    if request_path.is_file():
        existing = json.loads(request_path.read_text(encoding="utf-8"))
        if existing.get("request_sha256") != request_hash:
            raise RuntimeError(f"Existing final-probe request differs: {request_path}")
    else:
        write_json(request_path, request)

    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETE" and existing.get("request_sha256") == request_hash:
            print(json.dumps(existing, indent=2, sort_keys=True), flush=True)
            return

    pipeline = build_estimator(args, features.shape[1])
    started = time.perf_counter()
    pipeline.fit(features, labels)
    fit_seconds = time.perf_counter() - started
    pipeline_path = output_dir / "pipeline.joblib"
    joblib.dump(pipeline, pipeline_path, compress=3)
    classes = np.asarray(pipeline.classes_, dtype=int)
    if not np.array_equal(classes, np.arange(len(class_names))):
        raise RuntimeError(f"Unexpected fitted class ordering: {classes.tolist()}")
    summary = {
        "status": "COMPLETE",
        "stage": "LOCKED_FINAL_TRAIN_PLUS_VALIDATION_PROBE_FIT",
        "probe_id": args.probe_id,
        "configuration": config,
        "selection_lock_sha256": selection_lock_hash,
        "manifest_sha256": manifest_hash,
        "request_sha256": request_hash,
        "development_rows": len(manifest),
        "class_names": class_names,
        "class_counts": manifest["label"].value_counts().sort_index().to_dict(),
        "feature_dimensions": features.shape[1],
        "feature_cache_provenance": cache_provenance,
        **fitted_estimator_details(
            pipeline, args.classifier, features.shape[1], args.gamma_multiplier
        ),
        "fit_seconds": fit_seconds,
        "pipeline_sha256": sha256_file(pipeline_path),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "test_rows_read": 0,
        "test_used_for_selection": False,
    }
    write_json(summary_path, summary)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
