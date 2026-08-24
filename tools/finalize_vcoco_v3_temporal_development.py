"""Lock temporal development models and the validation-tested routing mechanism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.vcoco_v3_models import locomotion_f1


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
        "--student-target-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/student_targets/summary.json"),
    )
    parser.add_argument(
        "--student-targets",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/student_targets/student_targets.npz"),
    )
    parser.add_argument(
        "--student-run-root",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/students"),
    )
    parser.add_argument(
        "--development-run-root",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/development"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".runs/vcoco_v3/temporal/development_final")
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_student_run(path: Path, candidate: str, seed: int) -> dict:
    summary_path = path / "summary.json"
    predictions_path = path / "validation_predictions.npz"
    checkpoint_path = path / "best_checkpoint.pt"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "status": "VCOCO_V3_TEMPORAL_STUDENT_RUN_COMPLETE",
        "candidate_id": candidate,
        "seed": seed,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise RuntimeError(f"Temporal student run drift at {path}: {field}")
    for artifact in (predictions_path, checkpoint_path):
        if sha256_file(artifact) != summary["artifact_sha256"][artifact.name]:
            raise RuntimeError(f"Temporal student artifact drift at {path}: {artifact.name}")
    payload = np.load(predictions_path, allow_pickle=False)
    return {
        "summary": summary,
        "summary_path": summary_path,
        "predictions_path": predictions_path,
        "checkpoint_path": checkpoint_path,
        "sample_ids": payload["sample_ids"].astype(str),
        "recording_ids": payload["recording_ids"].astype(str),
        "labels": payload["labels"].astype(int),
        "probabilities": payload["probabilities"].astype(float),
        "identifiability_scores": payload["identifiability_scores"].astype(float),
        "advantage_targets": payload["advantage_targets"].astype(int),
    }


def cluster_bootstrap_ap_gain(
    targets: np.ndarray,
    scores: np.ndarray,
    recordings: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    targets = np.asarray(targets, dtype=int)
    scores = np.asarray(scores, dtype=float)
    recordings = np.asarray(recordings, dtype=str)
    unique = np.unique(recordings)
    by_recording = [np.flatnonzero(recordings == value) for value in unique]
    observed = float(average_precision_score(targets, scores) - targets.mean())
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled_groups = rng.integers(0, len(unique), size=len(unique))
        sampled = np.concatenate([by_recording[value] for value in sampled_groups])
        sampled_targets = targets[sampled]
        if len(np.unique(sampled_targets)) < 2:
            values[index] = 0.0
        else:
            values[index] = average_precision_score(sampled_targets, scores[sampled]) - float(
                sampled_targets.mean()
            )
    return {
        "point_estimate": observed,
        "ci_95_low": float(np.quantile(values, 0.025)),
        "ci_95_high": float(np.quantile(values, 0.975)),
        "resamples": int(resamples),
        "recordings": len(unique),
        "seed": int(seed),
    }


def main() -> None:
    args = parse_args()
    grid_path = args.grid.resolve()
    lock_path = args.temporal_lock.resolve()
    teacher_selection_path = args.teacher_selection.resolve()
    target_summary_path = args.student_target_summary.resolve()
    targets_path = args.student_targets.resolve()
    student_root = args.student_run_root.resolve()
    development_root = args.development_run_root.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    teacher_selection = json.loads(teacher_selection_path.read_text(encoding="utf-8"))
    target_summary = json.loads(target_summary_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_TEMPORAL_GRID_LOCKED_BEFORE_FIT":
        raise RuntimeError("The temporal grid is not locked")
    if teacher_selection.get("status") != "VCOCO_V3_TEMPORAL_TEACHER_SELECTED":
        raise RuntimeError("The temporal teacher is not selected")
    if target_summary.get("status") != "VCOCO_V3_TEMPORAL_STUDENT_TARGETS_LOCKED":
        raise RuntimeError("The temporal student targets are not locked")
    if sha256_file(targets_path) != target_summary["artifact_sha256"][targets_path.name]:
        raise RuntimeError("The temporal student targets changed")
    targets = np.load(targets_path, allow_pickle=False)
    seeds = list(map(int, lock["seeds"]))
    metrics_rows = []
    student_aggregates = {}
    fixed_epochs = {}
    evidence = {}
    for candidate in lock["student_candidates"]:
        candidate_id = candidate["candidate_id"]
        runs = []
        fixed_epochs[candidate_id] = {}
        for seed in seeds:
            path = student_root / candidate_id / f"seed-{seed}"
            run = load_student_run(path, candidate_id, seed)
            runs.append(run)
            fixed_epochs[candidate_id][str(seed)] = int(run["summary"]["best_epoch"]) + 1
            evidence[path.relative_to(student_root).as_posix()] = {
                "summary": sha256_file(run["summary_path"]),
                "checkpoint": sha256_file(run["checkpoint_path"]),
                "predictions": sha256_file(run["predictions_path"]),
            }
        reference = runs[0]
        for run in runs[1:]:
            for field in ("sample_ids", "recording_ids", "labels", "advantage_targets"):
                if not np.array_equal(run[field], reference[field]):
                    raise RuntimeError(f"Temporal student seed rows differ: {candidate_id}")
        probabilities = np.stack([run["probabilities"] for run in runs]).mean(axis=0)
        scores = np.stack([run["identifiability_scores"] for run in runs]).mean(axis=0)
        metrics = classification_metrics(reference["labels"], probabilities)
        metrics["locomotion_f1"] = locomotion_f1(reference["labels"], probabilities)
        metrics_rows.append({"family": candidate_id, **metrics})
        student_aggregates[candidate_id] = {
            "sample_ids": reference["sample_ids"],
            "recording_ids": reference["recording_ids"],
            "labels": reference["labels"],
            "advantage_targets": reference["advantage_targets"],
            "probabilities": probabilities,
            "identifiability_scores": scores,
        }
    metric_frame = pd.DataFrame(metrics_rows).sort_values(
        ["macro_f1", "locomotion_f1", "log_loss", "family"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    classification_student = str(metric_frame.iloc[0]["family"])
    routing_student = "identifiability_conditioned_static"
    routing = student_aggregates[routing_student]
    if not np.array_equal(routing["sample_ids"], targets["validation_sample_ids"].astype(str)):
        raise RuntimeError("Student validation rows do not align with locked teacher targets")
    ap = float(
        average_precision_score(routing["advantage_targets"], routing["identifiability_scores"])
    )
    prevalence = float(routing["advantage_targets"].mean())
    ap_uncertainty = cluster_bootstrap_ap_gain(
        routing["advantage_targets"],
        routing["identifiability_scores"],
        routing["recording_ids"],
        resamples=int(grid["selection"]["bootstrap_resamples"]),
        seed=int(grid["split_policy"]["validation_seed"]) + 70_000,
    )
    routing_eligible = bool(
        ap - prevalence >= float(grid["routing"]["minimum_average_precision_gain_over_prevalence"])
        and ap_uncertainty["ci_95_low"] > 0.0
    )

    teacher_id = teacher_selection["selected_teacher"]["candidate_id"]
    teacher_epochs = {}
    static_epochs = {}
    for seed in seeds:
        teacher_summary_path = (
            development_root / "teacher" / teacher_id / f"seed-{seed}" / "summary.json"
        )
        static_summary_path = development_root / "static" / f"seed-{seed}" / "summary.json"
        teacher_summary = json.loads(teacher_summary_path.read_text(encoding="utf-8"))
        static_summary = json.loads(static_summary_path.read_text(encoding="utf-8"))
        teacher_epochs[str(seed)] = int(teacher_summary["best_epoch"]) + 1
        static_epochs[str(seed)] = int(static_summary["best_epoch"]) + 1
        for path in (teacher_summary_path, static_summary_path):
            key = path.parent.relative_to(development_root).as_posix()
            evidence.setdefault(key, {})["summary"] = sha256_file(path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "student_validation_metrics.csv"
    probabilities_path = output_dir / "student_validation_predictions.npz"
    uncertainty_path = output_dir / "identifiability_ap_uncertainty.json"
    metric_frame.to_csv(metrics_path, index=False)
    np.savez_compressed(
        probabilities_path,
        sample_ids=routing["sample_ids"],
        recording_ids=routing["recording_ids"],
        labels=routing["labels"],
        advantage_targets=routing["advantage_targets"],
        **{
            f"{candidate_id}_probabilities": values["probabilities"]
            for candidate_id, values in student_aggregates.items()
        },
        identifiability_scores=routing["identifiability_scores"],
        static_probabilities=targets["validation_static_probabilities"],
        teacher_probabilities=targets["validation_teacher_probabilities"],
    )
    write_json(uncertainty_path, ap_uncertainty)
    summary = {
        "status": "VCOCO_V3_TEMPORAL_DEVELOPMENT_COMPLETE",
        "selected_teacher": teacher_selection["selected_teacher"],
        "classification_student": classification_student,
        "routing_student": routing_student,
        "routing_validation_average_precision": ap,
        "routing_validation_prevalence": prevalence,
        "routing_validation_ap_gain": ap - prevalence,
        "routing_eligible_for_calibration": routing_eligible,
        "fixed_epochs": {
            "static": static_epochs,
            "teacher": teacher_epochs,
            "students": fixed_epochs,
        },
        "seeds": seeds,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "temporal_grid": sha256_file(grid_path),
            "temporal_grid_lock": sha256_file(lock_path),
            "teacher_selection": sha256_file(teacher_selection_path),
            "student_target_summary": sha256_file(target_summary_path),
            "student_targets": sha256_file(targets_path),
        },
        "run_artifact_sha256": evidence,
        "artifact_sha256": {
            metrics_path.name: sha256_file(metrics_path),
            probabilities_path.name: sha256_file(probabilities_path),
            uncertainty_path.name: sha256_file(uncertainty_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
