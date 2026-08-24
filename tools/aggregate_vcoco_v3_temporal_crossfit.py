"""Lock aligned out-of-fold static and temporal targets for student training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.vcoco_v3_models import locomotion_f1
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path(".runs/vcoco_v3/temporal/crossfit"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".runs/vcoco_v3/temporal/crossfit_locked")
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_path(root: Path, role: str, fold: int, seed: int) -> Path:
    return root / role / f"fold-{fold}" / f"seed-{seed}"


def load_run(path: Path, role: str, fold: int, seed: int) -> dict:
    summary_path = path / "summary.json"
    predictions_path = path / "held_predictions.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "status": "VCOCO_V3_TEMPORAL_CROSSFIT_RUN_COMPLETE",
        "model_role": role,
        "fold": fold,
        "seed": seed,
        "validation_samples_read": 0,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise RuntimeError(f"Temporal cross-fit run drift at {path}: {field}")
    if sha256_file(predictions_path) != summary["artifact_sha256"][predictions_path.name]:
        raise RuntimeError(f"Temporal cross-fit probability drift at {path}")
    payload = np.load(predictions_path, allow_pickle=False)
    return {
        "summary_path": summary_path,
        "predictions_path": predictions_path,
        "sample_ids": payload["sample_ids"].astype(str),
        "recording_ids": payload["recording_ids"].astype(str),
        "track_ids": payload["track_ids"].astype(str),
        "labels": payload["labels"].astype(int),
        "probabilities": payload["probabilities"].astype(float),
    }


def aggregate_role(
    root: Path,
    role: str,
    *,
    folds: int,
    seeds: list[int],
    evidence: dict,
) -> dict:
    blocks = []
    for fold in range(folds):
        runs = []
        for seed in seeds:
            path = run_path(root, role, fold, seed)
            run = load_run(path, role, fold, seed)
            runs.append(run)
            evidence[path.relative_to(root).as_posix()] = {
                "summary": sha256_file(run["summary_path"]),
                "held_predictions": sha256_file(run["predictions_path"]),
            }
        reference = runs[0]
        for run in runs[1:]:
            for field in ("sample_ids", "recording_ids", "track_ids", "labels"):
                if not np.array_equal(run[field], reference[field]):
                    raise RuntimeError(f"Cross-fit {role} seed rows differ in fold {fold}")
        blocks.append(
            {
                **{
                    field: reference[field]
                    for field in ("sample_ids", "recording_ids", "track_ids", "labels")
                },
                "probabilities": np.stack([run["probabilities"] for run in runs]).mean(axis=0),
            }
        )
    result = {
        field: np.concatenate([block[field] for block in blocks])
        for field in ("sample_ids", "recording_ids", "track_ids", "labels", "probabilities")
    }
    if len(set(result["sample_ids"])) != len(result["sample_ids"]):
        raise RuntimeError(f"Cross-fit {role} folds contain duplicate samples")
    return result


def main() -> None:
    args = parse_args()
    grid_path = args.grid.resolve()
    lock_path = args.temporal_lock.resolve()
    selection_path = args.teacher_selection.resolve()
    manifest_path = args.manifest.resolve()
    run_root = args.run_root.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_TEMPORAL_GRID_LOCKED_BEFORE_FIT":
        raise RuntimeError("The temporal grid is not locked")
    if selection.get("status") != "VCOCO_V3_TEMPORAL_TEACHER_SELECTED":
        raise RuntimeError("The temporal teacher is not selected")
    folds = int(grid["training"]["teacher_crossfit_folds"])
    seeds = list(map(int, lock["seeds"]))
    evidence = {}
    static = aggregate_role(run_root, "static", folds=folds, seeds=seeds, evidence=evidence)
    teacher = aggregate_role(run_root, "teacher", folds=folds, seeds=seeds, evidence=evidence)
    static_positions = {value: index for index, value in enumerate(static["sample_ids"])}
    if set(static_positions) != set(teacher["sample_ids"]):
        raise RuntimeError("Static and teacher cross-fit sample sets differ")
    order = np.asarray([static_positions[value] for value in teacher["sample_ids"]])
    for field in ("recording_ids", "track_ids", "labels"):
        if not np.array_equal(static[field][order], teacher[field]):
            raise RuntimeError(f"Static and teacher cross-fit {field} differ")
    static_probabilities = static["probabilities"][order]
    train_frame = pd.read_csv(manifest_path, dtype={"sample_id": str})
    expected_train = set(train_frame.loc[train_frame["split"].eq("train"), "sample_id"])
    if set(teacher["sample_ids"]) != expected_train:
        raise RuntimeError("Cross-fit predictions do not cover the locked temporal training split")
    margin = float(grid["distillation"]["teacher_advantage_minimum_log_likelihood_gain"])
    targets = teacher_advantage_targets(
        teacher["labels"],
        static_probabilities,
        teacher["probabilities"],
        minimum_log_likelihood_gain=margin,
    )
    static_metrics = classification_metrics(teacher["labels"], static_probabilities)
    static_metrics["locomotion_f1"] = locomotion_f1(teacher["labels"], static_probabilities)
    teacher_metrics = classification_metrics(teacher["labels"], teacher["probabilities"])
    teacher_metrics["locomotion_f1"] = locomotion_f1(teacher["labels"], teacher["probabilities"])

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets_path = output_dir / "crossfit_targets.npz"
    np.savez_compressed(
        targets_path,
        sample_ids=teacher["sample_ids"],
        recording_ids=teacher["recording_ids"],
        track_ids=teacher["track_ids"],
        labels=teacher["labels"],
        static_probabilities=static_probabilities,
        teacher_probabilities=teacher["probabilities"],
        teacher_advantage_targets=targets,
    )
    summary = {
        "status": "VCOCO_V3_TEMPORAL_CROSSFIT_TARGETS_LOCKED",
        "samples": len(teacher["sample_ids"]),
        "recordings": int(pd.Series(teacher["recording_ids"]).nunique()),
        "folds": folds,
        "seeds": seeds,
        "static_oof_metrics": static_metrics,
        "teacher_oof_metrics": teacher_metrics,
        "teacher_advantage_positive_count": int(targets.sum()),
        "teacher_advantage_fraction": float(targets.mean()),
        "teacher_advantage_minimum_log_likelihood_gain": margin,
        "validation_samples_read": 0,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "temporal_grid": sha256_file(grid_path),
            "temporal_grid_lock": sha256_file(lock_path),
            "teacher_selection": sha256_file(selection_path),
            "manifest": sha256_file(manifest_path),
        },
        "run_artifact_sha256": evidence,
        "artifact_sha256": {targets_path.name: sha256_file(targets_path)},
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
