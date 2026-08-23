"""Evaluate controlled multiview and factorized heads on locked V-COCO development data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_analysis import per_class_metrics
from hac.polar_training import normalize_probability_rows
from hac.transfer import image_cluster_paired_bootstrap

CLASS_NAMES = ("sitting", "standing", "walking_running")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tight-cache", type=Path, required=True)
    parser.add_argument("--context-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--C", dest="c_value", type=float, required=True)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    return parser.parse_args()


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
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
        raise RuntimeError(f"Feature row drift: {path}")
    if sha256_file(features_path) != provenance["artifact_sha256"]["features.npy"]:
        raise RuntimeError(f"Feature array drift: {path}")
    rows = pd.read_csv(rows_path, dtype={"person_id": str, "image_id": str})
    features = np.asarray(np.load(features_path, mmap_mode="r"), dtype=np.float32)
    if len(rows) != len(features) or rows["person_id"].duplicated().any():
        raise RuntimeError(f"Feature rows do not align: {path}")
    return rows, features, provenance


def label_indices(rows: pd.DataFrame) -> np.ndarray:
    values = rows["label_3"].map({name: index for index, name in enumerate(CLASS_NAMES)})
    if values.isna().any():
        raise RuntimeError("Unknown three-class label")
    return values.to_numpy(dtype=int)


def geometry_features(rows: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.log(np.clip(rows["bbox_area_fraction"].to_numpy(float), 1e-8, 1.0)),
            np.log(np.clip(rows["bbox_aspect_ratio"].to_numpy(float), 1e-6, None)),
            rows["bbox_center_x_fraction"].to_numpy(float),
            rows["bbox_center_y_fraction"].to_numpy(float),
            np.log(np.clip(rows["person_pixel_height"].to_numpy(float), 1.0, None)),
        ]
    ).astype(np.float32)


def estimator(c_value: float, class_weight: str, *, random_state: int = 42):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            class_weight="balanced" if class_weight == "balanced" else None,
            max_iter=3_000,
            random_state=random_state,
            solver="lbfgs",
        ),
    )


def crossfit_flat(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    val_features: np.ndarray,
    *,
    c_value: float,
    class_weight: str,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupKFold(n_splits=5)
    oof = np.zeros((len(train_labels), len(CLASS_NAMES)), dtype=np.float64)
    for fold, (fit_index, held_index) in enumerate(
        splitter.split(train_features, train_labels, train_groups)
    ):
        model = estimator(c_value, class_weight, random_state=42 + fold)
        model.fit(train_features[fit_index], train_labels[fit_index])
        oof[held_index] = model.predict_proba(train_features[held_index])
    model = estimator(c_value, class_weight)
    model.fit(train_features, train_labels)
    return normalize_probability_rows(oof), normalize_probability_rows(
        model.predict_proba(val_features)
    )


def decode_factorized(posture_seated: np.ndarray, upright_locomoting: np.ndarray) -> np.ndarray:
    seated = np.clip(np.asarray(posture_seated, dtype=float), 0.0, 1.0)
    locomoting = np.clip(np.asarray(upright_locomoting, dtype=float), 0.0, 1.0)
    upright = 1.0 - seated
    return normalize_probability_rows(
        np.column_stack([seated, upright * (1.0 - locomoting), upright * locomoting])
    )


def crossfit_factorized(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    val_features: np.ndarray,
    *,
    c_value: float,
    class_weight: str,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupKFold(n_splits=5)
    oof = np.zeros((len(train_labels), len(CLASS_NAMES)), dtype=np.float64)
    posture_labels = (train_labels != 0).astype(int)
    for fold, (fit_index, held_index) in enumerate(
        splitter.split(train_features, train_labels, train_groups)
    ):
        posture = estimator(c_value, class_weight, random_state=142 + fold)
        posture.fit(train_features[fit_index], posture_labels[fit_index])
        upright_fit = fit_index[train_labels[fit_index] != 0]
        motion = estimator(c_value, class_weight, random_state=242 + fold)
        motion.fit(train_features[upright_fit], (train_labels[upright_fit] == 2).astype(int))
        oof[held_index] = decode_factorized(
            1.0 - posture.predict_proba(train_features[held_index])[:, 1],
            motion.predict_proba(train_features[held_index])[:, 1],
        )
    posture = estimator(c_value, class_weight, random_state=142)
    posture.fit(train_features, posture_labels)
    upright_train = train_labels != 0
    motion = estimator(c_value, class_weight, random_state=242)
    motion.fit(train_features[upright_train], (train_labels[upright_train] == 2).astype(int))
    validation = decode_factorized(
        1.0 - posture.predict_proba(val_features)[:, 1],
        motion.predict_proba(val_features)[:, 1],
    )
    return normalize_probability_rows(oof), validation


def baseline_probabilities(path: Path, val_rows: pd.DataFrame) -> np.ndarray:
    with np.load(path, allow_pickle=True) as payload:
        person_ids = [str(value) for value in payload["person_ids"]]
        index = {value: row for row, value in enumerate(person_ids)}
        order = np.asarray([index[value] for value in val_rows["person_id"]], dtype=int)
        return normalize_probability_rows(payload["person_dinov2_base_top4"][order])


def factor_metrics(rows: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    posture = rows["posture_label"].map({"seated": 0, "upright": 1}).to_numpy(dtype=int)
    motion = rows["motion_label"].map({"stationary": 0, "locomoting": 1}).to_numpy(dtype=int)
    predictions = probabilities.argmax(axis=1)
    return {
        "posture_macro_f1": float(f1_score(posture, (predictions != 0).astype(int), average="macro")),
        "motion_macro_f1": float(f1_score(motion, (predictions == 2).astype(int), average="macro")),
    }


def main() -> None:
    args = parse_args()
    if args.c_value <= 0.0 or args.bootstrap_resamples < 1:
        raise ValueError("C and bootstrap resamples must be positive")
    lock_path = args.protocol_lock.resolve()
    protocol_hash = sha256_file(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("The v2 protocol must be locked before model fitting")

    tight_rows, tight, tight_provenance = load_cache(args.tight_cache.resolve(), protocol_hash)
    context_rows, context, context_provenance = load_cache(
        args.context_cache.resolve(), protocol_hash
    )
    if not tight_rows["person_id"].equals(context_rows["person_id"]):
        raise RuntimeError("Tight and context caches use different row order")
    if tight_provenance["model_kind"] != context_provenance["model_kind"]:
        raise RuntimeError("Multiview comparison requires the same backbone")

    train_mask = tight_rows["split"].eq("train").to_numpy()
    val_mask = tight_rows["split"].eq("val").to_numpy()
    train_rows = tight_rows[train_mask].reset_index(drop=True)
    val_rows = tight_rows[val_mask].reset_index(drop=True)
    train_labels = label_indices(train_rows)
    val_labels = label_indices(val_rows)
    groups = train_rows["image_id"].astype(str).to_numpy()
    geometry = geometry_features(tight_rows)
    train_geometry, val_geometry = geometry[train_mask], geometry[val_mask]
    feature_sets = {
        "tight": (tight[train_mask], tight[val_mask]),
        "context": (context[train_mask], context[val_mask]),
        "tight_context": (
            np.concatenate([tight[train_mask], context[train_mask]], axis=1),
            np.concatenate([tight[val_mask], context[val_mask]], axis=1),
        ),
        "tight_context_geometry": (
            np.concatenate([tight[train_mask], context[train_mask], train_geometry], axis=1),
            np.concatenate([tight[val_mask], context[val_mask], val_geometry], axis=1),
        ),
    }

    train_oof = {}
    validation = {}
    for name, (train_features, val_features) in feature_sets.items():
        train_oof[f"flat_{name}"], validation[f"flat_{name}"] = crossfit_flat(
            train_features,
            train_labels,
            groups,
            val_features,
            c_value=args.c_value,
            class_weight=args.class_weight,
        )
    validation["late_mean_tight_context"] = normalize_probability_rows(
        0.5 * validation["flat_tight"] + 0.5 * validation["flat_context"]
    )

    stack_train = np.concatenate(
        [
            np.log(np.clip(train_oof["flat_tight"], 1e-8, 1.0)),
            np.log(np.clip(train_oof["flat_context"], 1e-8, 1.0)),
            train_geometry,
        ],
        axis=1,
    )
    stack_val = np.concatenate(
        [
            np.log(np.clip(validation["flat_tight"], 1e-8, 1.0)),
            np.log(np.clip(validation["flat_context"], 1e-8, 1.0)),
            val_geometry,
        ],
        axis=1,
    )
    stacker = estimator(1.0, args.class_weight, random_state=342)
    stacker.fit(stack_train, train_labels)
    validation["scale_conditioned_stacking"] = normalize_probability_rows(
        stacker.predict_proba(stack_val)
    )

    for name in ("tight", "context", "tight_context_geometry"):
        train_features, val_features = feature_sets[name]
        _, validation[f"factorized_{name}"] = crossfit_factorized(
            train_features,
            train_labels,
            groups,
            val_features,
            c_value=args.c_value,
            class_weight=args.class_weight,
        )

    baseline = baseline_probabilities(args.predictions.resolve(), val_rows)
    validation["locked_v1_dinov2"] = baseline
    summary_rows = []
    per_class_rows = []
    uncertainty = {}
    for name, probabilities in validation.items():
        summary_rows.append(
            {
                "method": name,
                **classification_metrics(val_labels, probabilities),
                **factor_metrics(val_rows, probabilities),
            }
        )
        per_class_rows.extend(
            {"method": name, **row}
            for row in per_class_metrics(val_labels, probabilities, CLASS_NAMES)
        )
        if name != "locked_v1_dinov2":
            uncertainty[name] = image_cluster_paired_bootstrap(
                val_labels,
                probabilities,
                baseline,
                val_rows["image_id"].astype(str).to_numpy(),
                resamples=args.bootstrap_resamples,
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["macro_f1", "log_loss", "method"], ascending=[False, True, True], ignore_index=True
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "factorized_fusion_summary.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(
        output_dir / "factorized_fusion_per_class.csv", index=False
    )
    write_json(output_dir / "factorized_fusion_uncertainty.json", uncertainty)
    np.savez_compressed(
        output_dir / "factorized_fusion_val_probabilities.npz",
        person_ids=val_rows["person_id"].to_numpy(),
        image_ids=val_rows["image_id"].to_numpy(),
        labels=val_labels,
        class_names=np.asarray(CLASS_NAMES),
        **validation,
    )
    result = {
        "status": "VCOCO_V2_FACTORIZED_FUSION_DEVELOPMENT_COMPLETE",
        "model_kind": tight_provenance["model_kind"],
        "tight_view": tight_provenance["view"],
        "context_view": context_provenance["view"],
        "C": args.c_value,
        "class_weight": args.class_weight,
        "best_validation_result": summary.iloc[0].to_dict(),
        "protocol_lock_sha256": protocol_hash,
        "test_rows_read": 0,
        "test_predictions_run": False,
    }
    write_json(output_dir / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
