"""Screen leakage-safe classifiers on controlled V-COCO v2 feature caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_analysis import confusion_metrics, per_class_metrics
from hac.polar_training import normalize_probability_rows
from hac.transfer import image_cluster_paired_bootstrap

CLASS_NAMES = ("sitting", "standing", "walking_running")
LOGISTIC_C = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
SVM_C = (0.001, 0.01, 0.1, 1.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cache", action="append", required=True, help="name=cache_directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args()


def parse_caches(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Each cache must use name=directory syntax")
        name, raw_path = value.split("=", 1)
        if not name or name in output:
            raise ValueError(f"Invalid or duplicate cache name: {name!r}")
        output[name] = Path(raw_path).resolve()
    return output


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
        return float(value)
    return value


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_cache(path: Path, protocol_hash: str) -> tuple[pd.DataFrame, np.ndarray, dict]:
    provenance_path = path / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "VCOCO_V2_DEVELOPMENT_FEATURE_CACHE_COMPLETE":
        raise RuntimeError(f"Incomplete feature cache: {path}")
    if provenance.get("protocol_lock_sha256") != protocol_hash:
        raise RuntimeError(f"Feature cache protocol drift: {path}")
    if provenance.get("test_rows_read") != 0 or provenance.get("test_predictions_run"):
        raise RuntimeError(f"Feature cache violated the target-test gate: {path}")
    rows_path = path / "rows.csv"
    features_path = path / "features.npy"
    if sha256_file(rows_path) != provenance["artifact_sha256"]["rows.csv"]:
        raise RuntimeError(f"Feature-row hash drift: {path}")
    if sha256_file(features_path) != provenance["artifact_sha256"]["features.npy"]:
        raise RuntimeError(f"Feature-array hash drift: {path}")
    rows = pd.read_csv(rows_path, dtype={"person_id": str, "image_id": str})
    features = np.load(features_path, mmap_mode="r")
    if len(rows) != len(features) or rows["person_id"].duplicated().any():
        raise RuntimeError(f"Feature cache rows do not align: {path}")
    return rows, np.asarray(features, dtype=np.float32), provenance


def label_indices(rows: pd.DataFrame) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = rows["label_3"].map(mapping)
    if labels.isna().any():
        raise RuntimeError("Feature cache contains an unknown class")
    return labels.to_numpy(dtype=int)


def geometry_features(rows: pd.DataFrame) -> np.ndarray:
    area = np.clip(rows["bbox_area_fraction"].to_numpy(dtype=float), 1e-8, 1.0)
    aspect = np.clip(rows["bbox_aspect_ratio"].to_numpy(dtype=float), 1e-6, None)
    height = np.clip(rows["person_pixel_height"].to_numpy(dtype=float), 1.0, None)
    return np.column_stack(
        [
            np.log(area),
            np.log(aspect),
            rows["bbox_center_x_fraction"].to_numpy(dtype=float),
            rows["bbox_center_y_fraction"].to_numpy(dtype=float),
            np.log(height),
        ]
    ).astype(np.float32)


def grouped_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    balanced: bool,
) -> tuple[object, np.ndarray, dict]:
    splitter = GroupKFold(n_splits=5)
    candidates = []
    for c_value in LOGISTIC_C:
        oof = np.zeros((len(labels), len(CLASS_NAMES)), dtype=np.float64)

        def fit_fold(
            train_index: np.ndarray,
            held_index: np.ndarray,
            fold_c: float = float(c_value),
        ):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=fold_c,
                    class_weight="balanced" if balanced else None,
                    max_iter=2_000,
                    random_state=42,
                    solver="lbfgs",
                ),
            )
            model.fit(features[train_index], labels[train_index])
            return held_index, model.predict_proba(features[held_index])

        folds = list(splitter.split(features, labels, groups))
        fitted = Parallel(n_jobs=len(folds))(
            delayed(fit_fold)(train_index, held_index) for train_index, held_index in folds
        )
        for held_index, probabilities in fitted:
            oof[held_index] = probabilities
        metrics = classification_metrics(labels, oof)
        candidates.append({"C": c_value, "metrics": metrics, "probabilities": oof})
    selected = sorted(
        candidates,
        key=lambda row: (-row["metrics"]["macro_f1"], row["metrics"]["log_loss"], row["C"]),
    )[0]
    final = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(selected["C"]),
            class_weight="balanced" if balanced else None,
            max_iter=2_000,
            random_state=42,
            solver="lbfgs",
        ),
    )
    final.fit(features, labels)
    return (
        final,
        selected["probabilities"],
        {
            "selected_C": selected["C"],
            "inner_cv": [{"C": row["C"], **row["metrics"]} for row in candidates],
        },
    )


def grouped_linear_svm(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    balanced: bool,
) -> tuple[tuple[object, LogisticRegression], np.ndarray, dict]:
    splitter = GroupKFold(n_splits=5)
    candidates = []
    for c_value in SVM_C:
        oof_scores = np.zeros((len(labels), len(CLASS_NAMES)), dtype=np.float64)

        def fit_fold(
            train_index: np.ndarray,
            held_index: np.ndarray,
            fold_c: float = float(c_value),
        ):
            model = make_pipeline(
                StandardScaler(),
                LinearSVC(
                    C=fold_c,
                    class_weight="balanced" if balanced else None,
                    dual="auto",
                    max_iter=20_000,
                    random_state=42,
                ),
            )
            model.fit(features[train_index], labels[train_index])
            return held_index, model.decision_function(features[held_index])

        folds = list(splitter.split(features, labels, groups))
        fitted = Parallel(n_jobs=len(folds))(
            delayed(fit_fold)(train_index, held_index) for train_index, held_index in folds
        )
        for held_index, scores in fitted:
            oof_scores[held_index] = scores
        macro = float(f1_score(labels, oof_scores.argmax(axis=1), average="macro"))
        candidates.append({"C": c_value, "macro_f1": macro, "scores": oof_scores})
    selected = sorted(candidates, key=lambda row: (-row["macro_f1"], row["C"]))[0]
    calibrator = LogisticRegression(C=1.0, max_iter=2_000, random_state=42, solver="lbfgs")
    calibrator.fit(selected["scores"], labels)
    oof_probabilities = calibrator.predict_proba(selected["scores"])
    final = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=float(selected["C"]),
            class_weight="balanced" if balanced else None,
            dual="auto",
            max_iter=20_000,
            random_state=42,
        ),
    )
    final.fit(features, labels)
    return (
        (final, calibrator),
        oof_probabilities,
        {
            "selected_C": selected["C"],
            "inner_cv": [{"C": row["C"], "macro_f1": row["macro_f1"]} for row in candidates],
            "probability_calibration": "multinomial_logistic_on_grouped_oof_scores",
        },
    )


def predict_model(model, classifier: str, features: np.ndarray) -> np.ndarray:
    if classifier == "logistic":
        return normalize_probability_rows(model.predict_proba(features))
    if classifier == "linear_svm":
        estimator, calibrator = model
        return normalize_probability_rows(
            calibrator.predict_proba(estimator.decision_function(features))
        )
    raise ValueError(f"Unknown classifier: {classifier}")


def factor_metrics(rows: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    posture_labels = rows["posture_label"].map({"seated": 0, "upright": 1}).to_numpy(dtype=int)
    posture_probabilities = normalize_probability_rows(
        np.column_stack([probabilities[:, 0], probabilities[:, 1] + probabilities[:, 2]])
    )
    motion_labels = rows["motion_label"].map({"stationary": 0, "locomoting": 1}).to_numpy(dtype=int)
    motion_probabilities = normalize_probability_rows(
        np.column_stack([probabilities[:, 0] + probabilities[:, 1], probabilities[:, 2]])
    )
    return {
        "posture_macro_f1": float(
            f1_score(posture_labels, posture_probabilities.argmax(axis=1), average="macro")
        ),
        "motion_macro_f1": float(
            f1_score(motion_labels, motion_probabilities.argmax(axis=1), average="macro")
        ),
    }


def baseline_probabilities(path: Path, val_rows: pd.DataFrame) -> np.ndarray:
    with np.load(path, allow_pickle=True) as payload:
        person_ids = [str(value) for value in payload["person_ids"]]
        index = {value: row for row, value in enumerate(person_ids)}
        order = np.asarray([index[value] for value in val_rows["person_id"]], dtype=int)
        return normalize_probability_rows(payload["person_dinov2_base_top4"][order])


def main() -> None:
    args = parse_args()
    lock_path = args.protocol_lock.resolve()
    protocol_hash = sha256_file(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("Feature screening requires the locked v2 protocol")
    caches = parse_caches(args.cache)

    summary_rows = []
    class_rows = []
    confusions = {}
    uncertainty = {}
    fit_details = {}
    prediction_artifacts = {}
    reference_rows = None
    reference_baseline = None

    for cache_name, cache_path in caches.items():
        rows, features, provenance = load_cache(cache_path, protocol_hash)
        if reference_rows is None:
            reference_rows = rows
        elif not rows["person_id"].equals(reference_rows["person_id"]):
            raise RuntimeError(f"Cache row order differs: {cache_name}")
        train_mask = rows["split"].eq("train").to_numpy()
        val_mask = rows["split"].eq("val").to_numpy()
        train_rows = rows[train_mask].reset_index(drop=True)
        val_rows = rows[val_mask].reset_index(drop=True)
        train_labels = label_indices(train_rows)
        val_labels = label_indices(val_rows)
        train_groups = train_rows["image_id"].astype(str).to_numpy()
        if reference_baseline is None:
            reference_baseline = baseline_probabilities(args.predictions.resolve(), val_rows)
            baseline_metrics = classification_metrics(val_labels, reference_baseline)
            summary_rows.append(
                {
                    "cache": "v1_dinov2_base_top4",
                    "feature_variant": "polar_adapted_person_context_25",
                    "classifier": "locked_v1",
                    "class_balance": "source_only",
                    **{f"val_{key}": value for key, value in baseline_metrics.items()},
                    **{
                        f"val_{key}": value
                        for key, value in factor_metrics(val_rows, reference_baseline).items()
                    },
                }
            )
            prediction_artifacts["v1_dinov2_base_top4"] = reference_baseline

        geometry = geometry_features(rows)
        variants = {
            "visual": features,
            "visual_plus_geometry": np.concatenate([features, geometry], axis=1),
        }
        if cache_name == next(iter(caches)):
            variants["geometry_only"] = geometry

        confusions[cache_name] = {}
        uncertainty[cache_name] = {}
        fit_details[cache_name] = {}
        for variant_name, variant_features in variants.items():
            train_features = variant_features[train_mask]
            val_features = variant_features[val_mask]
            for classifier in ("logistic", "linear_svm"):
                for balanced in (False, True):
                    if classifier == "logistic":
                        model, oof_probabilities, details = grouped_logistic(
                            train_features,
                            train_labels,
                            train_groups,
                            balanced=balanced,
                        )
                    else:
                        model, oof_probabilities, details = grouped_linear_svm(
                            train_features,
                            train_labels,
                            train_groups,
                            balanced=balanced,
                        )
                    val_probabilities = predict_model(model, classifier, val_features)
                    method = f"{variant_name}__{classifier}__{'balanced' if balanced else 'plain'}"
                    train_metrics = classification_metrics(train_labels, oof_probabilities)
                    val_metrics = classification_metrics(val_labels, val_probabilities)
                    factors = factor_metrics(val_rows, val_probabilities)
                    summary_rows.append(
                        {
                            "cache": cache_name,
                            "model_kind": provenance["model_kind"],
                            "view": provenance["view"],
                            "preprocess": provenance["preprocess"],
                            "image_size": provenance["image_size"],
                            "feature_variant": variant_name,
                            "classifier": classifier,
                            "class_balance": "balanced" if balanced else "unweighted",
                            **{f"train_oof_{key}": value for key, value in train_metrics.items()},
                            **{f"val_{key}": value for key, value in val_metrics.items()},
                            **{f"val_{key}": value for key, value in factors.items()},
                        }
                    )
                    class_rows.extend(
                        {"cache": cache_name, "method": method, **row}
                        for row in per_class_metrics(val_labels, val_probabilities, CLASS_NAMES)
                    )
                    confusions[cache_name][method] = confusion_metrics(
                        val_labels, val_probabilities, CLASS_NAMES
                    )
                    uncertainty[cache_name][method] = image_cluster_paired_bootstrap(
                        val_labels,
                        val_probabilities,
                        reference_baseline,
                        val_rows["image_id"].astype(str).to_numpy(),
                        resamples=args.bootstrap_resamples,
                    )
                    fit_details[cache_name][method] = details
                    prediction_artifacts[f"{cache_name}__{method}"] = val_probabilities

    summary = pd.DataFrame(summary_rows).sort_values(
        ["val_macro_f1", "val_log_loss", "cache"],
        ascending=[False, True, True],
        ignore_index=True,
    )
    per_class = pd.DataFrame(class_rows)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "feature_screen_summary.csv"
    per_class_path = output_dir / "feature_screen_per_class.csv"
    summary.to_csv(summary_path, index=False)
    per_class.to_csv(per_class_path, index=False)
    write_json(output_dir / "feature_screen_confusions.json", confusions)
    write_json(output_dir / "feature_screen_uncertainty.json", uncertainty)
    write_json(output_dir / "feature_screen_fit_details.json", fit_details)
    np.savez_compressed(
        output_dir / "feature_screen_val_probabilities.npz",
        person_ids=reference_rows.loc[reference_rows["split"].eq("val"), "person_id"].to_numpy(),
        image_ids=reference_rows.loc[reference_rows["split"].eq("val"), "image_id"].to_numpy(),
        labels=label_indices(reference_rows[reference_rows["split"].eq("val")]),
        class_names=np.asarray(CLASS_NAMES),
        **prediction_artifacts,
    )
    result = {
        "status": "VCOCO_V2_CONTROLLED_FEATURE_SCREEN_COMPLETE",
        "best_validation_result": summary.iloc[0].to_dict(),
        "cache_count": len(caches),
        "configuration_count": len(summary) - 1,
        "test_rows_read": 0,
        "test_predictions_run": False,
        "protocol_lock_sha256": protocol_hash,
        "artifact_sha256": {
            summary_path.name: sha256_file(summary_path),
            per_class_path.name: sha256_file(per_class_path),
        },
    }
    write_json(output_dir / "summary.json", result)
    print(json.dumps(json_safe(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
