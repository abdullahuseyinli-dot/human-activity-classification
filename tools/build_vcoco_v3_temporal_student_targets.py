"""Combine cross-fit train targets with untouched validation teacher distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hac.polar import sha256_file
from hac.vcoco_v3_temporal import teacher_advantage_targets


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
        "--teacher-selection",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/teacher_selection_lock.json"),
    )
    parser.add_argument(
        "--crossfit-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/crossfit_locked/summary.json"),
    )
    parser.add_argument(
        "--crossfit-targets",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/crossfit_locked/crossfit_targets.npz"),
    )
    parser.add_argument(
        "--development-run-root",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/development"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".runs/vcoco_v3/temporal/student_targets")
    )
    return parser.parse_args()


def load_validation(path: Path, expected_role: str, expected_candidate: str, seed: int) -> dict:
    summary_path = path / "summary.json"
    predictions_path = path / "validation_predictions.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "status": "VCOCO_V3_TEMPORAL_DEVELOPMENT_RUN_COMPLETE",
        "model_role": expected_role,
        "candidate_id": expected_candidate,
        "seed": seed,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise RuntimeError(f"Temporal validation target source drift at {path}: {field}")
    if sha256_file(predictions_path) != summary["artifact_sha256"][predictions_path.name]:
        raise RuntimeError(f"Temporal validation target hash drift at {path}")
    payload = np.load(predictions_path, allow_pickle=False)
    return {
        "summary_path": summary_path,
        "predictions_path": predictions_path,
        "sample_ids": payload["sample_ids"].astype(str),
        "recording_ids": payload["recording_ids"].astype(str),
        "labels": payload["labels"].astype(int),
        "probabilities": payload["probabilities"].astype(float),
    }


def main() -> None:
    args = parse_args()
    grid_path = args.grid.resolve()
    lock_path = args.temporal_lock.resolve()
    selection_path = args.teacher_selection.resolve()
    crossfit_summary_path = args.crossfit_summary.resolve()
    crossfit_targets_path = args.crossfit_targets.resolve()
    development_root = args.development_run_root.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    crossfit_summary = json.loads(crossfit_summary_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_TEMPORAL_GRID_LOCKED_BEFORE_FIT":
        raise RuntimeError("The temporal grid is not locked")
    if selection.get("status") != "VCOCO_V3_TEMPORAL_TEACHER_SELECTED":
        raise RuntimeError("The temporal teacher is not selected")
    if crossfit_summary.get("status") != "VCOCO_V3_TEMPORAL_CROSSFIT_TARGETS_LOCKED":
        raise RuntimeError("The temporal cross-fit targets are not locked")
    if (
        sha256_file(crossfit_targets_path)
        != crossfit_summary["artifact_sha256"][crossfit_targets_path.name]
    ):
        raise RuntimeError("The cross-fit student targets changed")
    train = np.load(crossfit_targets_path, allow_pickle=False)
    seeds = list(map(int, lock["seeds"]))
    teacher_id = selection["selected_teacher"]["candidate_id"]
    static_runs = []
    teacher_runs = []
    evidence = {}
    for seed in seeds:
        static_path = development_root / "static" / f"seed-{seed}"
        teacher_path = development_root / "teacher" / teacher_id / f"seed-{seed}"
        static_run = load_validation(static_path, "static", "static_center_frame", seed)
        teacher_run = load_validation(teacher_path, "teacher", teacher_id, seed)
        static_runs.append(static_run)
        teacher_runs.append(teacher_run)
        for path, run in ((static_path, static_run), (teacher_path, teacher_run)):
            evidence[path.relative_to(development_root).as_posix()] = {
                "summary": sha256_file(run["summary_path"]),
                "predictions": sha256_file(run["predictions_path"]),
            }
    reference = static_runs[0]
    for run in static_runs[1:] + teacher_runs:
        for field in ("sample_ids", "recording_ids", "labels"):
            if not np.array_equal(run[field], reference[field]):
                raise RuntimeError("Temporal student validation rows differ across sources")
    static_validation = np.stack([run["probabilities"] for run in static_runs]).mean(axis=0)
    teacher_validation = np.stack([run["probabilities"] for run in teacher_runs]).mean(axis=0)
    margin = float(grid["distillation"]["teacher_advantage_minimum_log_likelihood_gain"])
    validation_targets = teacher_advantage_targets(
        reference["labels"],
        static_validation,
        teacher_validation,
        minimum_log_likelihood_gain=margin,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "student_targets.npz"
    np.savez_compressed(
        output_path,
        train_sample_ids=train["sample_ids"].astype(str),
        train_recording_ids=train["recording_ids"].astype(str),
        train_labels=train["labels"].astype(int),
        train_static_probabilities=train["static_probabilities"].astype(float),
        train_teacher_probabilities=train["teacher_probabilities"].astype(float),
        train_advantage_targets=train["teacher_advantage_targets"].astype(int),
        validation_sample_ids=reference["sample_ids"],
        validation_recording_ids=reference["recording_ids"],
        validation_labels=reference["labels"],
        validation_static_probabilities=static_validation,
        validation_teacher_probabilities=teacher_validation,
        validation_advantage_targets=validation_targets,
    )
    summary = {
        "status": "VCOCO_V3_TEMPORAL_STUDENT_TARGETS_LOCKED",
        "teacher_candidate_id": teacher_id,
        "seeds": seeds,
        "train_samples": int(len(train["sample_ids"])),
        "validation_samples": len(reference["sample_ids"]),
        "train_advantage_fraction": float(train["teacher_advantage_targets"].mean()),
        "validation_advantage_fraction": float(validation_targets.mean()),
        "minimum_log_likelihood_gain": margin,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "temporal_grid": sha256_file(grid_path),
            "temporal_grid_lock": sha256_file(lock_path),
            "teacher_selection": sha256_file(selection_path),
            "crossfit_summary": sha256_file(crossfit_summary_path),
            "crossfit_targets": sha256_file(crossfit_targets_path),
        },
        "validation_run_artifact_sha256": evidence,
        "artifact_sha256": {output_path.name: sha256_file(output_path)},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
