"""Aggregate every declared inner neural run and lock one candidate per outer fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.vcoco_v3_models import CLASS_NAMES, grouped_splits, locomotion_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=Path("experiments/vcoco_v3_neural_grid.json"))
    parser.add_argument(
        "--neural-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/neural/neural_grid_lock.json"),
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path(".runs/polar_v2/locked_protocol/vcoco_train_clean.csv"),
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path(".runs/polar_v2/locked_protocol/vcoco_val_clean.csv"),
    )
    parser.add_argument("--run-root", type=Path, default=Path(".runs/vcoco_v3/neural/inner"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runs/vcoco_v3/neural/inner_selection_lock.json"),
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_directory(root: Path, outer: int, inner: int, candidate: str, seed: int) -> Path:
    return root / f"outer-{outer}" / f"inner-{inner}" / candidate / f"seed-{seed}"


def load_development(train_path: Path, val_path: Path) -> pd.DataFrame:
    frames = []
    for split, path in (("train", train_path), ("val", val_path)):
        frame = pd.read_csv(path.resolve(), dtype={"person_id": str, "image_id": str})
        frame["split"] = split
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    if output["person_id"].duplicated().any():
        raise RuntimeError("Development person identifiers are not unique")
    return output


def load_run(path: Path, *, candidate: str, outer: int, inner: int, seed: int) -> dict:
    summary_path = path / "summary.json"
    probabilities_path = path / "held_predictions.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "status": "VCOCO_V3_NEURAL_RUN_COMPLETE",
        "role": "inner_screen",
        "candidate_id": candidate,
        "outer_fold": outer,
        "inner_fold": inner,
        "seed": seed,
        "official_v2_test_rows_read": 0,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise RuntimeError(f"Inner neural run field drift at {path}: {field}")
    if sha256_file(probabilities_path) != summary["artifact_sha256"][probabilities_path.name]:
        raise RuntimeError(f"Inner prediction hash drift at {path}")
    payload = np.load(probabilities_path, allow_pickle=False)
    return {
        "summary": summary,
        "summary_path": summary_path,
        "probabilities_path": probabilities_path,
        "person_ids": payload["person_ids"].astype(str),
        "image_ids": payload["image_ids"].astype(str),
        "labels": payload["labels"].astype(int),
        "probabilities": payload["probabilities"].astype(float),
    }


def main() -> None:
    args = parse_args()
    grid_path = args.grid.resolve()
    lock_path = args.neural_lock.resolve()
    run_root = args.run_root.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_NEURAL_GRID_LOCKED_BEFORE_FIT":
        raise RuntimeError("The neural run grid is not locked")
    if sha256_file(grid_path) != lock["source_sha256"]["neural_grid"]:
        raise RuntimeError("The neural grid changed after locking")
    development = load_development(args.train_manifest, args.val_manifest)
    label_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = development["label_3"].map(label_mapping).to_numpy(dtype=int)
    groups = development["image_id"].astype(str).to_numpy(dtype=str)
    people = development["person_id"].astype(str).to_numpy(dtype=str)
    cross_validation = grid["cross_validation"]
    outer_splits = grouped_splits(
        labels,
        groups,
        folds=int(cross_validation["outer_folds"]),
        seed=int(cross_validation["outer_seed"]),
    )
    candidate_ids = [candidate["candidate_id"] for candidate in lock["candidates"]]
    seeds = list(map(int, cross_validation["screening_seeds"]))
    inner_folds = int(cross_validation["inner_folds"])
    selection_rows = []
    selected_by_outer = {}
    evidence = {}
    for outer_fold, (outer_train, _) in enumerate(outer_splits):
        expected_people = set(people[outer_train])
        evaluated = []
        for candidate in candidate_ids:
            candidate_people = []
            candidate_images = []
            candidate_labels = []
            candidate_probabilities = []
            phase_counts: dict[str, int] = {}
            for inner_fold in range(inner_folds):
                seed_runs = []
                for seed in seeds:
                    path = run_directory(run_root, outer_fold, inner_fold, candidate, seed)
                    run = load_run(
                        path,
                        candidate=candidate,
                        outer=outer_fold,
                        inner=inner_fold,
                        seed=seed,
                    )
                    seed_runs.append(run)
                    phase = str(run["summary"]["best_phase"])
                    phase_counts[phase] = phase_counts.get(phase, 0) + 1
                    relative = path.relative_to(run_root).as_posix()
                    evidence[relative] = {
                        "summary": sha256_file(run["summary_path"]),
                        "held_predictions": sha256_file(run["probabilities_path"]),
                    }
                reference = seed_runs[0]
                for run in seed_runs[1:]:
                    for field in ("person_ids", "image_ids", "labels"):
                        if not np.array_equal(run[field], reference[field]):
                            raise RuntimeError(
                                f"Seed prediction rows differ for outer {outer_fold}, "
                                f"inner {inner_fold}, {candidate}"
                            )
                candidate_people.append(reference["person_ids"])
                candidate_images.append(reference["image_ids"])
                candidate_labels.append(reference["labels"])
                candidate_probabilities.append(
                    np.stack([run["probabilities"] for run in seed_runs]).mean(axis=0)
                )
            person_ids = np.concatenate(candidate_people)
            image_ids = np.concatenate(candidate_images)
            target_labels = np.concatenate(candidate_labels)
            probabilities = np.concatenate(candidate_probabilities)
            if len(set(person_ids)) != len(person_ids) or set(person_ids) != expected_people:
                raise RuntimeError(
                    f"Inner folds do not partition outer training people for {candidate}"
                )
            metrics = classification_metrics(target_labels, probabilities)
            metrics["locomotion_f1"] = locomotion_f1(target_labels, probabilities)
            row = {
                "outer_fold": outer_fold,
                "candidate_id": candidate,
                **metrics,
                "people": len(person_ids),
                "source_images": int(pd.Series(image_ids).nunique()),
                "probe_checkpoint_count": phase_counts.get("probe", 0),
                "adapt_checkpoint_count": phase_counts.get("adapt", 0),
                "selected": False,
            }
            selection_rows.append(row)
            evaluated.append(row)
        selected = min(
            evaluated,
            key=lambda row: (
                -row["macro_f1"],
                -row["locomotion_f1"],
                row["log_loss"],
                row["candidate_id"],
            ),
        )
        selected["selected"] = True
        selected_by_outer[str(outer_fold)] = {
            "candidate_id": selected["candidate_id"],
            "macro_f1": float(selected["macro_f1"]),
            "locomotion_f1": float(selected["locomotion_f1"]),
            "log_loss": float(selected["log_loss"]),
            "probe_checkpoint_count": int(selected["probe_checkpoint_count"]),
            "adapt_checkpoint_count": int(selected["adapt_checkpoint_count"]),
        }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path.with_name("inner_candidate_metrics.csv")
    pd.DataFrame(selection_rows).to_csv(metrics_path, index=False)
    result = {
        "status": "VCOCO_V3_NEURAL_INNER_SELECTION_LOCKED",
        "selected_by_outer_fold": selected_by_outer,
        "screening_seeds": seeds,
        "candidate_count": len(candidate_ids),
        "expected_run_count": int(lock["inner_screen_run_count"]),
        "observed_run_count": len(evidence),
        "official_v2_test_rows_read": 0,
        "official_v2_test_predictions_run": False,
        "human_pilot_labels_used_for_selection": False,
        "source_sha256": {
            "neural_grid": sha256_file(grid_path),
            "neural_grid_lock": sha256_file(lock_path),
            "train_manifest": sha256_file(args.train_manifest.resolve()),
            "val_manifest": sha256_file(args.val_manifest.resolve()),
            "inner_candidate_metrics": sha256_file(metrics_path),
        },
        "run_artifact_sha256": evidence,
    }
    if result["observed_run_count"] != result["expected_run_count"]:
        raise RuntimeError("The observed neural inner-run count does not match the locked grid")
    write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
