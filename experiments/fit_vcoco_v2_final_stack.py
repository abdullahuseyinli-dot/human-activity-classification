"""Verify selection, then refit the locked V-COCO v2 stack on all development data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from evaluate_vcoco_v2_factorized_fusion import (
    crossfit_flat,
    estimator,
    geometry_features,
    label_indices,
    load_cache,
)

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_training import normalize_probability_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--tight-cache", type=Path, required=True)
    parser.add_argument("--context-cache", type=Path, required=True)
    parser.add_argument("--development-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--C", dest="c_value", type=float, default=0.01)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock_path = args.protocol_lock.resolve()
    protocol_hash = sha256_file(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("Final fit requires the locked V-COCO v2 protocol")
    tight_rows, tight, tight_provenance = load_cache(args.tight_cache.resolve(), protocol_hash)
    context_rows, context, context_provenance = load_cache(
        args.context_cache.resolve(), protocol_hash
    )
    if not tight_rows["person_id"].equals(context_rows["person_id"]):
        raise RuntimeError("Selected feature caches do not align")
    if tight_provenance["model_kind"] != "dinov2_base" or context_provenance[
        "model_kind"
    ] != "dinov2_base":
        raise RuntimeError("Final selection requires the declared DINOv2-B caches")
    train_mask = tight_rows["split"].eq("train").to_numpy()
    val_mask = tight_rows["split"].eq("val").to_numpy()
    train_rows = tight_rows[train_mask].reset_index(drop=True)
    val_rows = tight_rows[val_mask].reset_index(drop=True)
    train_labels = label_indices(train_rows)
    val_labels = label_indices(val_rows)
    train_groups = train_rows["image_id"].astype(str).to_numpy()
    geometry = geometry_features(tight_rows)
    train_geometry = geometry[train_mask]
    val_geometry = geometry[val_mask]

    tight_oof, _ = crossfit_flat(
        tight[train_mask],
        train_labels,
        train_groups,
        tight[val_mask],
        c_value=args.c_value,
        class_weight=args.class_weight,
    )
    context_oof, _ = crossfit_flat(
        context[train_mask],
        train_labels,
        train_groups,
        context[val_mask],
        c_value=args.c_value,
        class_weight=args.class_weight,
    )
    train_stack = np.concatenate(
        [
            np.log(np.clip(tight_oof, 1e-8, 1.0)),
            np.log(np.clip(context_oof, 1e-8, 1.0)),
            train_geometry,
        ],
        axis=1,
    )
    selection_stacker = estimator(1.0, args.class_weight, random_state=342)
    selection_stacker.fit(train_stack, train_labels)
    selection_tight_model = estimator(args.c_value, args.class_weight, random_state=42)
    selection_context_model = estimator(args.c_value, args.class_weight, random_state=42)
    selection_tight_model.fit(tight[train_mask], train_labels)
    selection_context_model.fit(context[train_mask], train_labels)
    tight_val = normalize_probability_rows(
        selection_tight_model.predict_proba(tight[val_mask])
    )
    context_val = normalize_probability_rows(
        selection_context_model.predict_proba(context[val_mask])
    )
    val_stack = np.concatenate(
        [
            np.log(np.clip(tight_val, 1e-8, 1.0)),
            np.log(np.clip(context_val, 1e-8, 1.0)),
            val_geometry,
        ],
        axis=1,
    )
    val_probabilities = normalize_probability_rows(selection_stacker.predict_proba(val_stack))
    development_path = args.development_predictions.resolve()
    with np.load(development_path, allow_pickle=True) as payload:
        expected_ids = [str(value) for value in payload["person_ids"]]
        if expected_ids != val_rows["person_id"].astype(str).tolist():
            raise RuntimeError("Selected development prediction order drift")
        expected = np.asarray(payload["scale_conditioned_stacking"], dtype=np.float64)
    maximum_difference = float(np.abs(val_probabilities - expected).max())
    if maximum_difference > 1e-10:
        raise RuntimeError(f"Final fit does not reproduce selection predictions: {maximum_difference}")

    development_rows = tight_rows.reset_index(drop=True)
    development_labels = label_indices(development_rows)
    development_groups = development_rows["image_id"].astype(str).to_numpy()
    development_geometry = geometry_features(development_rows)
    tight_oof, _ = crossfit_flat(
        tight,
        development_labels,
        development_groups,
        tight,
        c_value=args.c_value,
        class_weight=args.class_weight,
    )
    context_oof, _ = crossfit_flat(
        context,
        development_labels,
        development_groups,
        context,
        c_value=args.c_value,
        class_weight=args.class_weight,
    )
    development_stack = np.concatenate(
        [
            np.log(np.clip(tight_oof, 1e-8, 1.0)),
            np.log(np.clip(context_oof, 1e-8, 1.0)),
            development_geometry,
        ],
        axis=1,
    )
    tight_model = estimator(args.c_value, args.class_weight, random_state=42)
    context_model = estimator(args.c_value, args.class_weight, random_state=42)
    stacker = estimator(1.0, args.class_weight, random_state=342)
    tight_model.fit(tight, development_labels)
    context_model.fit(context, development_labels)
    stacker.fit(development_stack, development_labels)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "vcoco_v2_multiview_stack.joblib"
    artifact = {
        "status": "VCOCO_V2_FINAL_DEVELOPMENT_STACK_FIT",
        "fit_scope": "locked_train_plus_validation_after_selection",
        "development_people": int(len(development_rows)),
        "development_images": int(development_rows["image_id"].nunique()),
        "class_names": ["sitting", "standing", "walking_running"],
        "model_kind": "dinov2_base",
        "views": ["person_tight", "person_context_25"],
        "preprocess": "aspect_preserving_pad_224",
        "base_C": args.c_value,
        "base_class_weight": args.class_weight,
        "stack_C": 1.0,
        "geometry_schema": [
            "log_bbox_area_fraction",
            "log_bbox_aspect_ratio",
            "bbox_center_x_fraction",
            "bbox_center_y_fraction",
            "log_person_pixel_height",
        ],
        "tight_model": tight_model,
        "context_model": context_model,
        "stacker": stacker,
    }
    joblib.dump(artifact, artifact_path, compress=3)
    result = {
        "status": "VCOCO_V2_FINAL_DEVELOPMENT_STACK_FIT_COMPLETE",
        "selection_validation_metrics": classification_metrics(val_labels, val_probabilities),
        "validation_prediction_replay_maximum_absolute_difference": maximum_difference,
        "final_fit_scope": "locked_train_plus_validation_after_selection",
        "final_training_people": int(len(development_rows)),
        "final_training_images": int(development_rows["image_id"].nunique()),
        "final_training_split_counts": {
            str(name): int(count)
            for name, count in development_rows["split"].value_counts().sort_index().items()
        },
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "development_predictions_sha256": sha256_file(development_path),
        "tight_cache_provenance_sha256": sha256_file(args.tight_cache.resolve() / "provenance.json"),
        "context_cache_provenance_sha256": sha256_file(
            args.context_cache.resolve() / "provenance.json"
        ),
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
