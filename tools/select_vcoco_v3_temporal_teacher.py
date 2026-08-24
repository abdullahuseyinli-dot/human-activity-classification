"""Aggregate five-seed validation predictions and lock one temporal teacher window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.vcoco_v3_models import locomotion_f1, paired_cluster_bootstrap


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
        "--run-root", type=Path, default=Path(".runs/vcoco_v3/temporal/development")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/teacher_selection_lock.json"),
    )
    return parser.parse_args()


def run_path(root: Path, role: str, candidate: str, seed: int) -> Path:
    if role == "static":
        return root / "static" / f"seed-{seed}"
    return root / "teacher" / candidate / f"seed-{seed}"


def load_run(path: Path, role: str, candidate: str, seed: int) -> dict:
    summary_path = path / "summary.json"
    prediction_path = path / "validation_predictions.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "status": "VCOCO_V3_TEMPORAL_DEVELOPMENT_RUN_COMPLETE",
        "model_role": role,
        "candidate_id": candidate,
        "seed": seed,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise RuntimeError(f"Temporal development run drift at {path}: {field}")
    if sha256_file(prediction_path) != summary["artifact_sha256"][prediction_path.name]:
        raise RuntimeError(f"Temporal validation probability drift at {path}")
    payload = np.load(prediction_path, allow_pickle=False)
    return {
        "summary_path": summary_path,
        "checkpoint_path": path / "best_checkpoint.pt",
        "prediction_path": prediction_path,
        "sample_ids": payload["sample_ids"].astype(str),
        "recording_ids": payload["recording_ids"].astype(str),
        "labels": payload["labels"].astype(int),
        "probabilities": payload["probabilities"].astype(float),
    }


def aggregate_seed_runs(runs: list[dict]) -> dict:
    reference = runs[0]
    for run in runs[1:]:
        for field in ("sample_ids", "recording_ids", "labels"):
            if not np.array_equal(run[field], reference[field]):
                raise RuntimeError("Temporal validation rows differ across seeds")
    return {
        "sample_ids": reference["sample_ids"],
        "recording_ids": reference["recording_ids"],
        "labels": reference["labels"],
        "probabilities": np.stack([run["probabilities"] for run in runs]).mean(axis=0),
    }


def main() -> None:
    args = parse_args()
    grid_path = args.grid.resolve()
    lock_path = args.temporal_lock.resolve()
    run_root = args.run_root.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_TEMPORAL_GRID_LOCKED_BEFORE_FIT":
        raise RuntimeError("The temporal grid is not locked")
    if lock["source_sha256"]["temporal_grid"] != sha256_file(grid_path):
        raise RuntimeError("The temporal grid changed after locking")
    seeds = list(map(int, lock["seeds"]))
    evidence = {}
    static_runs = []
    for seed in seeds:
        path = run_path(run_root, "static", "static_center_frame", seed)
        run = load_run(path, "static", "static_center_frame", seed)
        static_runs.append(run)
        evidence[path.relative_to(run_root).as_posix()] = {
            "summary": sha256_file(run["summary_path"]),
            "checkpoint": sha256_file(run["checkpoint_path"]),
            "predictions": sha256_file(run["prediction_path"]),
        }
    static = aggregate_seed_runs(static_runs)
    static_metrics = classification_metrics(static["labels"], static["probabilities"])
    static_metrics["locomotion_f1"] = locomotion_f1(static["labels"], static["probabilities"])

    rows = []
    aggregates = {}
    for candidate in lock["teacher_candidates"]:
        candidate_id = candidate["candidate_id"]
        runs = []
        for seed in seeds:
            path = run_path(run_root, "teacher", candidate_id, seed)
            run = load_run(path, "teacher", candidate_id, seed)
            runs.append(run)
            evidence[path.relative_to(run_root).as_posix()] = {
                "summary": sha256_file(run["summary_path"]),
                "checkpoint": sha256_file(run["checkpoint_path"]),
                "predictions": sha256_file(run["prediction_path"]),
            }
        aggregate = aggregate_seed_runs(runs)
        for field in ("sample_ids", "recording_ids", "labels"):
            if not np.array_equal(aggregate[field], static[field]):
                raise RuntimeError(f"Static and temporal validation rows differ: {candidate_id}")
        metrics = classification_metrics(aggregate["labels"], aggregate["probabilities"])
        metrics["locomotion_f1"] = locomotion_f1(aggregate["labels"], aggregate["probabilities"])
        rows.append({"candidate_id": candidate_id, **metrics})
        aggregates[candidate_id] = aggregate
    metrics_frame = pd.DataFrame(rows).sort_values(
        ["macro_f1", "locomotion_f1", "log_loss", "candidate_id"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    selected_id = str(metrics_frame.iloc[0]["candidate_id"])
    selected = next(
        candidate
        for candidate in lock["teacher_candidates"]
        if candidate["candidate_id"] == selected_id
    )
    uncertainty = paired_cluster_bootstrap(
        static["labels"],
        aggregates[selected_id]["probabilities"],
        static["probabilities"],
        static["recording_ids"],
        resamples=int(grid["selection"]["bootstrap_resamples"]),
        seed=int(grid["split_policy"]["validation_seed"]) + 50_000,
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path.with_name("teacher_validation_metrics.csv")
    uncertainty_path = output_path.with_name("teacher_vs_static_uncertainty.json")
    metrics_frame.to_csv(metrics_path, index=False)
    uncertainty_path.write_text(
        json.dumps(uncertainty, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result = {
        "status": "VCOCO_V3_TEMPORAL_TEACHER_SELECTED",
        "selected_teacher": selected,
        "static_validation_metrics": static_metrics,
        "selected_teacher_validation_metrics": {
            key: value
            for key, value in metrics_frame.iloc[0].to_dict().items()
            if key != "candidate_id"
        },
        "seeds": seeds,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "temporal_grid": sha256_file(grid_path),
            "temporal_grid_lock": sha256_file(lock_path),
            "teacher_validation_metrics": sha256_file(metrics_path),
            "teacher_vs_static_uncertainty": sha256_file(uncertainty_path),
        },
        "run_artifact_sha256": evidence,
    }
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
