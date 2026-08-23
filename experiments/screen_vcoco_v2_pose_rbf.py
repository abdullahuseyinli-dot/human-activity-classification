"""Screen RBF-SVM pose-oracle baselines on locked V-COCO development data."""

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
from sklearn.svm import SVC

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_analysis import per_class_metrics
from hac.polar_training import normalize_probability_rows
from hac.transfer import image_cluster_paired_bootstrap

CLASS_NAMES = ("sitting", "standing", "walking_running")
C_VALUES = (0.1, 1.0, 10.0, 100.0)
GAMMA_VALUES = ("scale", 0.01)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    return parser.parse_args()


def labels(rows: pd.DataFrame) -> np.ndarray:
    output = rows["label_3"].map({name: index for index, name in enumerate(CLASS_NAMES)})
    if output.isna().any():
        raise RuntimeError("Unknown class label")
    return output.to_numpy(dtype=int)


def geometry(rows: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.log(np.clip(rows["bbox_area_fraction"].to_numpy(float), 1e-8, 1.0)),
            np.log(np.clip(rows["bbox_aspect_ratio"].to_numpy(float), 1e-6, None)),
            rows["bbox_center_x_fraction"].to_numpy(float),
            rows["bbox_center_y_fraction"].to_numpy(float),
            np.log(np.clip(rows["person_pixel_height"].to_numpy(float), 1.0, None)),
        ]
    ).astype(np.float32)


def svc(c_value: float, gamma: str | float, balanced: bool, seed: int):
    return make_pipeline(
        StandardScaler(),
        SVC(
            C=float(c_value),
            gamma=gamma,
            kernel="rbf",
            class_weight="balanced" if balanced else None,
            probability=False,
            random_state=seed,
        ),
    )


def fit_rbf(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    val_features: np.ndarray,
    *,
    balanced: bool,
) -> tuple[np.ndarray, dict]:
    folds = list(GroupKFold(n_splits=5).split(train_features, train_labels, train_groups))
    candidates = []
    for c_value in C_VALUES:
        for gamma in GAMMA_VALUES:

            def fit_fold(
                fit_index: np.ndarray,
                held_index: np.ndarray,
                fold_c: float = float(c_value),
                fold_gamma: str | float = gamma,
            ):
                model = svc(fold_c, fold_gamma, balanced, 42)
                model.fit(train_features[fit_index], train_labels[fit_index])
                return held_index, model.decision_function(train_features[held_index])

            fitted = Parallel(n_jobs=len(folds))(
                delayed(fit_fold)(fit_index, held_index) for fit_index, held_index in folds
            )
            oof_scores = np.zeros((len(train_labels), len(CLASS_NAMES)), dtype=np.float64)
            for held_index, scores in fitted:
                oof_scores[held_index] = scores
            candidates.append(
                {
                    "C": c_value,
                    "gamma": gamma,
                    "macro_f1": float(
                        f1_score(train_labels, oof_scores.argmax(axis=1), average="macro")
                    ),
                    "scores": oof_scores,
                }
            )
    selected = sorted(
        candidates,
        key=lambda row: (-row["macro_f1"], float(row["C"]), str(row["gamma"])),
    )[0]
    calibrator = LogisticRegression(C=1.0, max_iter=3_000, random_state=42, solver="lbfgs")
    calibrator.fit(selected["scores"], train_labels)
    final = svc(float(selected["C"]), selected["gamma"], balanced, 42)
    final.fit(train_features, train_labels)
    probabilities = normalize_probability_rows(
        calibrator.predict_proba(final.decision_function(val_features))
    )
    details = {
        "selected_C": selected["C"],
        "selected_gamma": selected["gamma"],
        "probability_calibration": "multinomial_logistic_on_grouped_oof_scores",
        "inner_cv": [
            {key: value for key, value in row.items() if key != "scores"} for row in candidates
        ],
    }
    return probabilities, details


def baseline(path: Path, val_rows: pd.DataFrame) -> np.ndarray:
    with np.load(path, allow_pickle=True) as payload:
        index = {str(value): row for row, value in enumerate(payload["person_ids"])}
        order = np.asarray([index[value] for value in val_rows["person_id"]], dtype=int)
        return normalize_probability_rows(payload["person_dinov2_base_top4"][order])


def main() -> None:
    args = parse_args()
    lock_path = args.protocol_lock.resolve()
    protocol_hash = sha256_file(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("RBF screening requires the locked V-COCO v2 protocol")
    cache = args.cache.resolve()
    provenance = json.loads((cache / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("protocol_lock_sha256") != protocol_hash:
        raise RuntimeError("Pose cache protocol drift")
    if provenance.get("model_kind") != "coco_gt_pose_oracle":
        raise RuntimeError("This screen is restricted to the declared COCO pose oracle")
    if provenance.get("test_rows_read") != 0 or provenance.get("test_predictions_run"):
        raise RuntimeError("Pose cache violated the target-test gate")
    rows_path, features_path = cache / "rows.csv", cache / "features.npy"
    if sha256_file(rows_path) != provenance["artifact_sha256"]["rows.csv"]:
        raise RuntimeError("Pose row drift")
    if sha256_file(features_path) != provenance["artifact_sha256"]["features.npy"]:
        raise RuntimeError("Pose feature drift")
    rows = pd.read_csv(rows_path, dtype={"person_id": str, "image_id": str})
    pose = np.asarray(np.load(features_path, mmap_mode="r"), dtype=np.float32)
    train_mask = rows["split"].eq("train").to_numpy()
    val_mask = rows["split"].eq("val").to_numpy()
    train_rows = rows[train_mask].reset_index(drop=True)
    val_rows = rows[val_mask].reset_index(drop=True)
    train_labels = labels(train_rows)
    val_labels = labels(val_rows)
    groups = train_rows["image_id"].astype(str).to_numpy()
    box = geometry(rows)
    variants = {
        "pose": pose,
        "pose_geometry": np.concatenate([pose, box], axis=1),
        "geometry": box,
    }
    reference = baseline(args.predictions.resolve(), val_rows)
    summary_rows = []
    class_rows = []
    uncertainty = {}
    fit_details = {}
    probabilities = {
        "locked_v1_dinov2": reference,
    }
    for name, features in variants.items():
        for balanced in (False, True):
            method = f"{name}__rbf_svm__{'balanced' if balanced else 'plain'}"
            prediction, details = fit_rbf(
                features[train_mask],
                train_labels,
                groups,
                features[val_mask],
                balanced=balanced,
            )
            probabilities[method] = prediction
            metrics = classification_metrics(val_labels, prediction)
            summary_rows.append({"method": method, **metrics})
            class_rows.extend(
                {"method": method, **row}
                for row in per_class_metrics(val_labels, prediction, CLASS_NAMES)
            )
            uncertainty[method] = image_cluster_paired_bootstrap(
                val_labels,
                prediction,
                reference,
                val_rows["image_id"].astype(str).to_numpy(),
                resamples=args.bootstrap_resamples,
            )
            fit_details[method] = details
    summary = pd.DataFrame(summary_rows).sort_values(
        ["macro_f1", "log_loss", "method"], ascending=[False, True, True], ignore_index=True
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "pose_rbf_summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output_dir / "pose_rbf_per_class.csv", index=False)
    for name, payload in (("uncertainty", uncertainty), ("fit_details", fit_details)):
        (output_dir / f"pose_rbf_{name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    np.savez_compressed(
        output_dir / "pose_rbf_val_probabilities.npz",
        person_ids=val_rows["person_id"].to_numpy(),
        image_ids=val_rows["image_id"].to_numpy(),
        labels=val_labels,
        class_names=np.asarray(CLASS_NAMES),
        **probabilities,
    )
    result = {
        "status": "VCOCO_V2_POSE_RBF_DEVELOPMENT_COMPLETE",
        "diagnostic_scope": provenance["diagnostic_scope"],
        "best_validation_result": summary.iloc[0].to_dict(),
        "protocol_lock_sha256": protocol_hash,
        "test_rows_read": 0,
        "test_predictions_run": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
