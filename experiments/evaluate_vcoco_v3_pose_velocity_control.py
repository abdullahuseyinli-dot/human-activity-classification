"""Fit the optional pose/velocity SVM control without opening confirmation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.vcoco_v3_cuda_heads import (
    cuda_logistic_fit_audit,
    reset_cuda_logistic_fit_audit,
)
from hac.vcoco_v3_models import cuda_svm_fit_audit, locomotion_f1, reset_cuda_svm_fit_audit
from hac.vcoco_v3_pose_control import (
    PoseControlUnavailableError,
    build_pose_svm,
    extract_pose_control_features,
    fit_pose_score_calibrator,
    pose_decision_scores,
    predict_pose_probabilities,
)
from hac.vcoco_v3_temporal import grouped_recording_splits, validate_temporal_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", type=Path, default=Path("experiments/vcoco_v3_temporal_grid.json")
    )
    parser.add_argument(
        "--temporal-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/temporal_grid_lock.json"),
    )
    parser.add_argument(
        "--manifest-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/temporal_manifest_lock.json"),
    )
    parser.add_argument(
        "--development-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/development_final/summary.json"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".runs/vcoco_v3/temporal/pose_control")
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def softmax_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    values -= values.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def unavailable_summary(
    *,
    reason: str,
    sources: dict[str, str],
    development_feature_count: int,
    pose_feature_count: int,
) -> dict:
    return {
        "status": "VCOCO_V3_POSE_CONTROL_UNAVAILABLE",
        "reason": reason,
        "development_feature_count": int(development_feature_count),
        "pose_feature_count": int(pose_feature_count),
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": sources,
        "artifact_sha256": {},
    }


def main() -> None:
    args = parse_args()
    grid_path = args.grid.resolve()
    lock_path = args.temporal_lock.resolve()
    manifest_lock_path = args.manifest_lock.resolve()
    development_path = args.development_summary.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest_lock = json.loads(manifest_lock_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_TEMPORAL_GRID_LOCKED_BEFORE_FIT":
        raise RuntimeError("The temporal grid is not locked")
    if manifest_lock.get("status") != "VCOCO_V3_TEMPORAL_MANIFEST_LOCKED":
        raise RuntimeError("The temporal manifest is not locked")
    if development.get("status") != "VCOCO_V3_TEMPORAL_DEVELOPMENT_COMPLETE":
        raise RuntimeError("Temporal development is incomplete")
    if manifest_lock.get("confirmation_feature_arrays_opened") != 0:
        raise RuntimeError("Confirmation features were opened before pose-control development")
    if manifest_lock["source_sha256"].get("manifest") != sha256_file(manifest_path):
        raise RuntimeError("The temporal manifest changed after it was locked")
    sources = {
        "temporal_grid": sha256_file(grid_path),
        "temporal_grid_lock": sha256_file(lock_path),
        "temporal_manifest_lock": sha256_file(manifest_lock_path),
        "development_summary": sha256_file(development_path),
        "manifest": sha256_file(manifest_path),
    }
    feature_evidence = manifest_lock["development_feature_sha256"]
    pose_count = sum(bool(item.get("pose_available")) for item in feature_evidence.values())
    if pose_count != len(feature_evidence):
        summary = unavailable_summary(
            reason="normalized_pose_not_available_for_every_development_sample",
            sources=sources,
            development_feature_count=len(feature_evidence),
            pose_feature_count=pose_count,
        )
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return

    frame = validate_temporal_manifest(
        pd.read_csv(
            manifest_path,
            dtype={"sample_id": str, "recording_id": str, "track_id": str},
        )
    )
    partitions = {
        split: frame[frame["split"].eq(split)].reset_index(drop=True)
        for split in ("train", "validation", "calibration")
    }
    if any(not len(rows) for rows in partitions.values()):
        raise RuntimeError(
            "Pose-control train, validation, and calibration splits must be nonempty"
        )
    candidate = development["selected_teacher"]
    declared_candidates = {item["candidate_id"]: item for item in lock["teacher_candidates"]}
    if candidate != declared_candidates.get(candidate["candidate_id"]):
        raise RuntimeError("The selected temporal window differs from the locked grid")
    pose_grid = grid["pose_velocity_svm"]
    if pose_grid.get("execution_backend") != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Pose-control fitting requires the declared CUDA backend")
    maximum_iterations = int(pose_grid["maximum_iterations"])
    tolerance = float(pose_grid["gradient_tolerance"])
    calibrator_iterations = int(pose_grid["calibrator_maximum_iterations"])
    calibrator_tolerance = float(pose_grid["calibrator_gradient_tolerance"])
    confidence_threshold = float(pose_grid["confidence_threshold"])
    try:
        features = {
            split: extract_pose_control_features(
                rows,
                candidate=candidate,
                manifest_directory=manifest_path.parent,
                confidence_threshold=confidence_threshold,
            )
            for split, rows in partitions.items()
        }
    except PoseControlUnavailableError as error:
        summary = unavailable_summary(
            reason=str(error),
            sources=sources,
            development_feature_count=len(feature_evidence),
            pose_feature_count=pose_count,
        )
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return

    reset_cuda_svm_fit_audit()
    reset_cuda_logistic_fit_audit()
    selection_rows = []
    for c_value in pose_grid["C"]:
        for class_weight in pose_grid["class_weight"]:
            model = build_pose_svm(
                c_value=float(c_value),
                class_weight=str(class_weight),
                seed=int(grid["split_policy"]["validation_seed"]),
                maximum_iterations=maximum_iterations,
                tolerance=tolerance,
            )
            model.fit(features["train"].values, features["train"].labels)
            probabilities = softmax_scores(
                pose_decision_scores(model, features["validation"].values)
            )
            metrics = classification_metrics(features["validation"].labels, probabilities)
            metrics["locomotion_f1"] = locomotion_f1(features["validation"].labels, probabilities)
            selection_rows.append(
                {"C": float(c_value), "class_weight": str(class_weight), **metrics}
            )
    selection = pd.DataFrame(selection_rows).sort_values(
        ["macro_f1", "locomotion_f1", "accuracy", "C", "class_weight"],
        ascending=[False, False, False, True, False],
        ignore_index=True,
    )
    selected = selection.iloc[0]
    selected_c = float(selected["C"])
    selected_class_weight = str(selected["class_weight"])

    development_values = np.concatenate(
        (features["train"].values, features["validation"].values), axis=0
    )
    development_labels = np.concatenate((features["train"].labels, features["validation"].labels))
    development_groups = np.concatenate(
        (features["train"].recording_ids, features["validation"].recording_ids)
    )
    folds = grouped_recording_splits(
        development_labels,
        development_groups,
        folds=int(pose_grid["calibration_folds"]),
        seed=int(grid["split_policy"]["calibration_seed"]),
    )
    out_of_fold_scores = np.full((len(development_labels), 3), np.nan, dtype=float)
    for fold_index, (fit_index, held_index) in enumerate(folds):
        model = build_pose_svm(
            c_value=selected_c,
            class_weight=selected_class_weight,
            seed=int(grid["split_policy"]["calibration_seed"]) + fold_index,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
        )
        model.fit(development_values[fit_index], development_labels[fit_index])
        out_of_fold_scores[held_index] = pose_decision_scores(model, development_values[held_index])
    if not np.all(np.isfinite(out_of_fold_scores)):
        raise RuntimeError("Grouped pose-control calibration did not score every development row")
    score_calibrator = fit_pose_score_calibrator(
        out_of_fold_scores,
        development_labels,
        seed=int(grid["split_policy"]["calibration_seed"]),
        maximum_iterations=calibrator_iterations,
        tolerance=calibrator_tolerance,
    )
    final_model = build_pose_svm(
        c_value=selected_c,
        class_weight=selected_class_weight,
        seed=int(grid["split_policy"]["calibration_seed"]),
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
    )
    final_model.fit(development_values, development_labels)
    bundle = {
        "version": "vcoco-v3-pose-control-2-cuda",
        "teacher_candidate": candidate,
        "confidence_threshold": confidence_threshold,
        "selected_C": selected_c,
        "selected_class_weight": selected_class_weight,
        "svm": final_model,
        "score_calibrator": score_calibrator,
    }
    calibration_probabilities = predict_pose_probabilities(bundle, features["calibration"].values)

    selection_path = output_dir / "validation_grid.csv"
    predictions_path = output_dir / "calibration_predictions.npz"
    bundle_path = output_dir / "model.joblib"
    optimization_path = output_dir / "cuda_optimization.json"
    selection.to_csv(selection_path, index=False)
    np.savez_compressed(
        predictions_path,
        sample_ids=features["calibration"].sample_ids,
        recording_ids=features["calibration"].recording_ids,
        track_ids=features["calibration"].track_ids,
        labels=features["calibration"].labels,
        probabilities=calibration_probabilities,
    )
    joblib.dump(bundle, bundle_path, compress=3)
    write_json(
        optimization_path,
        {
            "device": torch.cuda.get_device_name(0),
            "svm_records": cuda_svm_fit_audit(),
            "logistic_records": cuda_logistic_fit_audit(),
        },
    )
    summary = {
        "status": "VCOCO_V3_POSE_CONTROL_COMPLETE",
        "selected_C": selected_c,
        "selected_class_weight": selected_class_weight,
        "selected_teacher": candidate,
        "development_samples": len(development_labels),
        "development_recordings": int(pd.Series(development_groups).nunique()),
        "calibration_samples": len(features["calibration"].labels),
        "calibration_recordings": int(pd.Series(features["calibration"].recording_ids).nunique()),
        "score_calibration": "recording_grouped_out_of_fold_multinomial_logistic",
        "training_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "calibration_samples_used_for_model_selection": 0,
        "confirmation_samples_read": 0,
        "source_sha256": sources,
        "artifact_sha256": {
            selection_path.name: sha256_file(selection_path),
            predictions_path.name: sha256_file(predictions_path),
            bundle_path.name: sha256_file(bundle_path),
            optimization_path.name: sha256_file(optimization_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
