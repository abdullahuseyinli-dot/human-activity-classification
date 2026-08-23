"""Lock the V-COCO v2 development champion before any official-test prediction."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_analysis import per_class_metrics
from hac.transfer import image_cluster_paired_bootstrap

CLASS_NAMES = ("sitting", "standing", "walking_running")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--factor-predictions", type=Path, required=True)
    parser.add_argument("--feature-predictions", type=Path, required=True)
    parser.add_argument("--augmix-predictions", type=Path, required=True)
    parser.add_argument("--final-fit", type=Path, required=True)
    parser.add_argument("--final-fit-summary", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=False, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def align_neural(path: Path, person_ids: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=True) as payload:
        index = {str(value): row for row, value in enumerate(payload["person_ids"])}
        order = np.asarray([index[value] for value in person_ids], dtype=int)
        return np.asarray(payload["probabilities"][order], dtype=np.float64)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol_path = args.protocol_lock.resolve()
    protocol_hash = sha256_file(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("Final selection requires the locked V-COCO v2 protocol")
    if protocol["test_access"]["model_predictions_run"]:
        raise RuntimeError("Protocol reports prior official-test predictions")
    val_path = args.val_manifest.resolve()
    test_path = args.test_manifest.resolve()
    if sha256_file(val_path) != protocol["artifact_sha256"]["vcoco_val_clean.csv"]:
        raise RuntimeError("Locked validation manifest drift")
    if sha256_file(test_path) != protocol["artifact_sha256"]["vcoco_test_clean.csv"]:
        raise RuntimeError("Locked test manifest drift")
    val_rows = pd.read_csv(val_path, dtype={"person_id": str, "image_id": str})
    labels = val_rows["label_3"].map({name: index for index, name in enumerate(CLASS_NAMES)})
    if labels.isna().any():
        raise RuntimeError("Unknown validation class")
    labels_array = labels.to_numpy(dtype=int)
    person_ids = val_rows["person_id"].astype(str).tolist()
    image_ids = val_rows["image_id"].astype(str).to_numpy()

    factor_path = args.factor_predictions.resolve()
    with np.load(factor_path, allow_pickle=True) as payload:
        if [str(value) for value in payload["person_ids"]] != person_ids:
            raise RuntimeError("Factor prediction order drift")
        champion = np.asarray(payload["scale_conditioned_stacking"], dtype=np.float64)
        factorized = np.asarray(
            payload["factorized_tight_context_geometry"], dtype=np.float64
        )
        flat_factor_control = np.asarray(payload["flat_tight_context_geometry"], dtype=np.float64)
        historical = np.asarray(payload["locked_v1_dinov2"], dtype=np.float64)
    feature_path = args.feature_predictions.resolve()
    with np.load(feature_path, allow_pickle=True) as payload:
        if [str(value) for value in payload["person_ids"]] != person_ids:
            raise RuntimeError("Feature-screen prediction order drift")
        single_view = np.asarray(
            payload["tight_pad__visual_plus_geometry__linear_svm__balanced"], dtype=np.float64
        )
    augmix = align_neural(args.augmix_predictions.resolve(), person_ids)

    comparisons = {
        "champion_minus_historical_v1": image_cluster_paired_bootstrap(
            labels_array, champion, historical, image_ids, resamples=10_000
        ),
        "champion_minus_single_view_dino": image_cluster_paired_bootstrap(
            labels_array, champion, single_view, image_ids, resamples=10_000
        ),
        "champion_minus_augmix_lpft": image_cluster_paired_bootstrap(
            labels_array, champion, augmix, image_ids, resamples=10_000
        ),
        "factorized_minus_flat_same_features": image_cluster_paired_bootstrap(
            labels_array, factorized, flat_factor_control, image_ids, resamples=10_000
        ),
    }
    promotion = comparisons["champion_minus_single_view_dino"]
    if promotion["point_estimate"] < 0.01 or promotion["ci_95_low"] <= 0.0:
        raise RuntimeError("Selected champion does not satisfy the locked promotion rule")
    champion_metrics = classification_metrics(labels_array, champion)
    champion_classes = per_class_metrics(labels_array, champion, CLASS_NAMES)
    if min(float(row["f1"]) for row in champion_classes) < 0.70:
        raise RuntimeError("Selected champion has an unacceptable class collapse")

    final_fit_path = args.final_fit.resolve()
    final_fit_summary_path = args.final_fit_summary.resolve()
    final_fit_summary = json.loads(final_fit_summary_path.read_text(encoding="utf-8"))
    if (
        final_fit_summary.get("status") != "VCOCO_V2_FINAL_DEVELOPMENT_STACK_FIT_COMPLETE"
        or final_fit_summary.get("protocol_lock_sha256") != protocol_hash
        or final_fit_summary.get("artifact_sha256") != sha256_file(final_fit_path)
        or final_fit_summary.get("final_fit_scope")
        != "locked_train_plus_validation_after_selection"
        or final_fit_summary.get("final_training_people")
        != int(protocol["split_summary"]["train"]["people"])
        + int(protocol["split_summary"]["val"]["people"])
        or final_fit_summary.get("test_predictions_run")
    ):
        raise RuntimeError("Final development fit evidence is invalid")
    checkpoints = [path.resolve() for path in args.baseline_checkpoint]
    if len(checkpoints) != 3:
        raise RuntimeError("The historical DINO baseline requires exactly three seed checkpoints")
    evidence_paths = [path.resolve() for path in args.evidence]
    for path in evidence_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    implementation_paths = [
        Path(__file__).resolve(),
        root / "experiments" / "fit_vcoco_v2_final_stack.py",
        root / "experiments" / "cache_vcoco_v2_features.py",
        root / "experiments" / "evaluate_vcoco_v2_factorized_fusion.py",
        root / "src" / "hac" / "augmentations.py",
        root / "src" / "hac" / "metrics.py",
        root / "src" / "hac" / "polar.py",
        root / "src" / "hac" / "transfer.py",
    ]
    result = {
        "status": "VCOCO_V2_FINAL_SELECTION_LOCKED_PRE_TEST",
        "protocol_lock_sha256": protocol_hash,
        "repository_revision_at_lock": git_revision(root),
        "selection": {
            "method": "scale_conditioned_stacking",
            "model_kind": "dinov2_base",
            "model_revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
            "views": ["person_tight", "person_context_25"],
            "preprocess": "aspect_preserving_pad_224",
            "base_classifier": "multinomial_logistic",
            "base_C": 0.01,
            "class_weight": "none",
            "stacker_C": 1.0,
            "stacker_inputs": "cross_fitted_tight_and_context_log_probabilities_plus_geometry",
            "validation_metrics": champion_metrics,
            "validation_per_class": champion_classes,
        },
        "development_comparisons": comparisons,
        "decision_notes": {
            "augmix_lpft_status": "TIED_WITH_CHAMPION_INTERVAL_INCLUDES_ZERO",
            "factorized_head_status": "SUPPORTED_OVER_FLAT_SAME_FEATURES",
            "manual_harmonized_annotation_status": protocol["ontology"][
                "manual_annotation_status"
            ],
        },
        "final_fit": {
            "path": str(final_fit_path),
            "sha256": sha256_file(final_fit_path),
            "summary_sha256": sha256_file(final_fit_summary_path),
            "scope": final_fit_summary["final_fit_scope"],
            "training_people": final_fit_summary["final_training_people"],
            "training_images": final_fit_summary["final_training_images"],
        },
        "historical_baseline": {
            "method": "polar_dinov2_base_top4_three_seed_mean_original_preprocess",
            "checkpoints": [
                {"path": str(path), "sha256": sha256_file(path)} for path in checkpoints
            ],
        },
        "final_test": {
            "manifest_sha256": sha256_file(test_path),
            "expected_people": int(protocol["split_summary"]["test"]["people"]),
            "expected_images": int(protocol["split_summary"]["test"]["images"]),
            "authorized_candidates": ["scale_conditioned_stacking", "historical_v1_dino"],
            "bootstrap_unit": "source_image",
            "bootstrap_resamples": 10_000,
            "access_policy": "one_pipeline_run_after_this_lock",
        },
        "source_sha256": {
            "factor_predictions": sha256_file(factor_path),
            "feature_predictions": sha256_file(feature_path),
            "augmix_predictions": sha256_file(args.augmix_predictions.resolve()),
            **{str(path.relative_to(root)): sha256_file(path) for path in implementation_paths},
        },
        "evidence_sha256": {str(path): sha256_file(path) for path in evidence_paths},
        "test_access": {
            "model_predictions_before_lock": False,
            "official_test_labels_used_for_selection": False,
            "official_test_open_count": 0,
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vcoco_v2_final_selection_lock.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**result, "selection_lock_sha256": sha256_file(output_path)}, indent=2))


if __name__ == "__main__":
    main()
