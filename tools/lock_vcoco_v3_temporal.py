"""Lock temporal candidates after upstream development and external manifests pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", type=Path, default=Path("experiments/vcoco_v3_temporal_grid.json")
    )
    parser.add_argument(
        "--neural-grid-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/neural/neural_grid_lock.json"),
    )
    parser.add_argument(
        "--neural-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/neural/final/summary.json"),
    )
    parser.add_argument(
        "--representation-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/representations/evaluation/summary.json"),
    )
    parser.add_argument(
        "--representation-metrics",
        type=Path,
        default=Path(".runs/vcoco_v3/representations/evaluation/nested_source_tag_metrics.csv"),
    )
    parser.add_argument(
        "--manifest-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/temporal_manifest_lock.json"),
    )
    parser.add_argument(
        "--protocol-amendment",
        type=Path,
        default=Path(".runs/vcoco_v3/protocol/external_cuda_amendment_lock.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/temporal_grid_lock.json"),
    )
    return parser.parse_args()


def validate_grid(grid: dict) -> None:
    if grid.get("status") != "DECLARED_BEFORE_TEMPORAL_FITTING":
        raise RuntimeError("The temporal grid is not in its pre-fit state")
    candidates = grid.get("teacher_candidates", ())
    identifiers = [candidate.get("candidate_id") for candidate in candidates]
    if len(candidates) < 2 or len(identifiers) != len(set(identifiers)):
        raise ValueError("Temporal teacher candidate identifiers must be unique")
    if len(set(grid.get("training", {}).get("seeds", ()))) < 5:
        raise ValueError("Temporal comparisons require five declared seeds")
    training = grid.get("training", {})
    if training.get("execution_backend") != "cuda":
        raise ValueError("Temporal model fitting must use the declared CUDA backend")
    if training.get("cpu_fallback_permitted") is not False:
        raise ValueError("Temporal CPU fallback must remain disabled")
    routing = grid.get("routing", {})
    clip_fractions = [float(value) for value in routing.get("clip_budget_fractions", ())]
    if (
        not clip_fractions
        or clip_fractions != sorted(set(clip_fractions))
        or clip_fractions[0] != 0.0
        or clip_fractions[-1] != 1.0
        or any(not 0.0 <= value <= 1.0 for value in clip_fractions)
    ):
        raise ValueError("Temporal routing budgets must be unique, ordered, and span [0, 1]")
    if routing.get("calibrator_solver") != "pytorch_cuda_lbfgs_logistic":
        raise ValueError("Temporal routing calibration must use the declared CUDA solver")
    if float(routing.get("calibrator_C", 0.0)) <= 0.0:
        raise ValueError("Temporal routing calibrator C must be positive")
    if int(routing.get("calibrator_maximum_iterations", 0)) < 1:
        raise ValueError("Temporal routing calibrator iterations must be positive")
    if float(routing.get("calibrator_gradient_tolerance", 0.0)) <= 0.0:
        raise ValueError("Temporal routing calibrator tolerance must be positive")
    students = grid.get("student_candidates", ())
    student_ids = [candidate.get("candidate_id") for candidate in students]
    if len(students) != 2 or len(student_ids) != len(set(student_ids)):
        raise ValueError("The distilled and identifiability student grid changed")
    for candidate in students:
        weights = [
            float(candidate["supervised_weight"]),
            float(candidate["teacher_distribution_weight"]),
            float(candidate["identifiability_weight"]),
        ]
        if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
            raise ValueError("Temporal student loss weights are invalid")
    if int(grid.get("training", {}).get("teacher_crossfit_folds", 0)) < 3:
        raise ValueError("Temporal distillation requires grouped cross-fitting")
    if grid.get("distillation", {}).get("teacher_targets_for_training") != (
        "recording-grouped_out_of_fold_only"
    ):
        raise ValueError("Teacher targets must remain recording-grouped and out-of-fold")
    if not grid.get("selection", {}).get("confirmation_models_locked_before_open"):
        raise ValueError("Confirmation models must be locked before the one-open evaluation")
    pose = grid.get("pose_velocity_svm", {})
    if pose.get("optional_when_pose_unavailable") is not True:
        raise ValueError("The pose control must remain optional when normalized pose is absent")
    c_values = [float(value) for value in pose.get("C", ())]
    if (
        not c_values
        or len(c_values) != len(set(c_values))
        or any(value <= 0.0 for value in c_values)
    ):
        raise ValueError("Pose-control SVM C values must be unique and positive")
    if set(map(str, pose.get("class_weight", ()))) != {"none", "balanced"}:
        raise ValueError("Pose-control class weighting must compare none and balanced")
    if int(pose.get("calibration_folds", 0)) < 3:
        raise ValueError("Pose-control calibration requires grouped cross-fitting")
    if pose.get("calibration_method") != ("multinomial_logistic_on_grouped_out_of_fold_svm_scores"):
        raise ValueError("Pose-control score calibration must remain out of fold")
    if pose.get("execution_backend") != "cuda":
        raise ValueError("Pose-control fitting must use CUDA")
    if pose.get("solver") != "pytorch_cuda_lbfgs_ovr_squared_hinge":
        raise ValueError("Pose-control fitting must use the declared CUDA SVM solver")
    if int(pose.get("maximum_iterations", 0)) < 1 or float(
        pose.get("gradient_tolerance", 0.0)
    ) <= 0.0:
        raise ValueError("Pose-control CUDA SVM optimization settings are invalid")
    if int(pose.get("calibrator_maximum_iterations", 0)) < 1 or float(
        pose.get("calibrator_gradient_tolerance", 0.0)
    ) <= 0.0:
        raise ValueError("Pose-control CUDA calibrator settings are invalid")
    confidence = float(pose.get("confidence_threshold", -1.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Pose-control confidence threshold must be in [0, 1]")


def main() -> None:
    args = parse_args()
    grid_path = args.grid.resolve()
    neural_lock_path = args.neural_grid_lock.resolve()
    representation_path = args.representation_summary.resolve()
    representation_metrics_path = args.representation_metrics.resolve()
    manifest_path = args.manifest_lock.resolve()
    amendment_path = args.protocol_amendment.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    validate_grid(grid)
    neural_lock = json.loads(neural_lock_path.read_text(encoding="utf-8"))
    representation = json.loads(representation_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if (
        amendment.get("status")
        != "VCOCO_V3_EXTERNAL_CUDA_AMENDMENT_LOCKED_BEFORE_TARGET_FITTING"
    ):
        raise RuntimeError("The external CUDA protocol amendment is not locked")
    if amendment["source_sha256"].get("okutama_temporal_grid") != sha256_file(grid_path):
        raise RuntimeError("The amended Okutama temporal grid changed")
    if amendment["source_sha256"].get("temporal_locker_source") != sha256_file(
        Path(__file__).resolve()
    ):
        raise RuntimeError("The amended temporal locker implementation changed")
    if representation.get("status") != "VCOCO_V3_MATCHED_REPRESENTATION_DEVELOPMENT_COMPLETE":
        raise RuntimeError("The matched representation stage is incomplete")
    if manifest.get("status") != "VCOCO_V3_TEMPORAL_MANIFEST_LOCKED":
        raise RuntimeError("The temporal manifest is not locked")
    if manifest.get("confirmation_feature_arrays_opened") != 0:
        raise RuntimeError("Confirmation features were opened before the temporal pipeline lock")
    neural_status = neural_lock.get("status")
    if neural_status == "VCOCO_V3_NEURAL_GRID_LOCKED_BEFORE_FIT":
        neural_summary_path = args.neural_summary.resolve()
        neural = json.loads(neural_summary_path.read_text(encoding="utf-8"))
        if neural.get("status") != "VCOCO_V3_NEURAL_DEVELOPMENT_COMPLETE":
            raise RuntimeError("Eligible neural development is not complete")
        backbone = str(neural_lock["selected_backbone"]["model_kind"])
        neural_evidence = sha256_file(neural_summary_path)
    elif neural_status == "VCOCO_V3_NEURAL_STAGE_NOT_ELIGIBLE":
        if (
            sha256_file(representation_metrics_path)
            != representation["artifact_sha256"][representation_metrics_path.name]
        ):
            raise RuntimeError("The frozen representation metrics changed")
        metrics = pd.read_csv(representation_metrics_path)
        dino = metrics[metrics["family"].isin({"dinov2_base", "dinov3_base"})].sort_values(
            ["macro_f1", "locomotion_f1", "log_loss", "family"],
            ascending=[False, False, True, True],
            ignore_index=True,
        )
        if len(dino) != 2:
            raise RuntimeError("The matched screen does not contain both temporal DINO backbones")
        backbone = str(dino.iloc[0]["family"])
        neural_evidence = None
    else:
        raise RuntimeError("Neural eligibility has not been resolved")

    seeds = list(map(int, grid["training"]["seeds"]))
    teacher_candidates = [
        {**candidate, "frame_backbone": backbone} for candidate in grid["teacher_candidates"]
    ]
    student_candidates = list(grid["student_candidates"])
    result = {
        "status": "VCOCO_V3_TEMPORAL_GRID_LOCKED_BEFORE_FIT",
        "grid_version": grid["grid_version"],
        "frame_backbone": backbone,
        "input_dimensions": 2 * int(manifest["embedding_dimensions"]) + 6,
        "teacher_candidates": teacher_candidates,
        "student_candidates": student_candidates,
        "seeds": seeds,
        "static_run_count": len(seeds),
        "teacher_screen_run_count": len(teacher_candidates) * len(seeds),
        "crossfit_run_count_after_teacher_selection": (
            2 * int(grid["training"]["teacher_crossfit_folds"]) * len(seeds)
        ),
        "student_run_count": len(student_candidates) * len(seeds),
        "confirmation_feature_arrays_opened": 0,
        "official_v2_test_rows_read": 0,
        "source_sha256": {
            "temporal_grid": sha256_file(grid_path),
            "neural_grid_lock": sha256_file(neural_lock_path),
            "neural_summary": neural_evidence,
            "representation_summary": sha256_file(representation_path),
            "representation_metrics": sha256_file(representation_metrics_path),
            "temporal_manifest_lock": sha256_file(manifest_path),
            "external_cuda_amendment": sha256_file(amendment_path),
        },
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
