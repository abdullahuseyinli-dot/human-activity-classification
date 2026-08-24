"""Aggregate selected outer neural runs and compare them with the matched frozen model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_analysis import per_class_metrics
from hac.vcoco_v3_models import CLASS_NAMES, locomotion_f1, paired_cluster_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/vcoco_v3_protocol.json"))
    parser.add_argument("--grid", type=Path, default=Path("experiments/vcoco_v3_neural_grid.json"))
    parser.add_argument(
        "--neural-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/neural/neural_grid_lock.json"),
    )
    parser.add_argument(
        "--selection-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/neural/inner_selection_lock.json"),
    )
    parser.add_argument(
        "--representation-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/representations/evaluation/summary.json"),
    )
    parser.add_argument(
        "--representation-probabilities",
        type=Path,
        default=Path(".runs/vcoco_v3/representations/evaluation/nested_oof_probabilities.npz"),
    )
    parser.add_argument("--run-root", type=Path, default=Path(".runs/vcoco_v3/neural/outer"))
    parser.add_argument("--output-dir", type=Path, default=Path(".runs/vcoco_v3/neural/final"))
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_directory(root: Path, outer: int, candidate: str, seed: int) -> Path:
    return root / f"outer-{outer}" / candidate / f"seed-{seed}"


def load_outer_run(path: Path, *, outer: int, candidate: str, seed: int) -> dict:
    summary_path = path / "summary.json"
    probabilities_path = path / "held_predictions.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "status": "VCOCO_V3_NEURAL_RUN_COMPLETE",
        "role": "outer_fit",
        "candidate_id": candidate,
        "outer_fold": outer,
        "inner_fold": None,
        "seed": seed,
        "official_v2_test_rows_read": 0,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise RuntimeError(f"Outer neural run field drift at {path}: {field}")
    if sha256_file(probabilities_path) != summary["artifact_sha256"][probabilities_path.name]:
        raise RuntimeError(f"Outer prediction hash drift at {path}")
    payload = np.load(probabilities_path, allow_pickle=False)
    return {
        "summary": summary,
        "summary_path": summary_path,
        "probabilities_path": probabilities_path,
        "person_ids": payload["person_ids"].astype(str),
        "image_ids": payload["image_ids"].astype(str),
        "labels": payload["labels"].astype(int),
        "probabilities": payload["probabilities"].astype(float),
        "gate_weights": payload["gate_weights"].astype(float),
    }


def aligned_baseline(
    path: Path,
    *,
    family: str,
    person_ids: np.ndarray,
) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if family not in payload.files:
        raise RuntimeError(f"Frozen probability artifact does not contain {family}")
    baseline_ids = payload["person_ids"].astype(str)
    if len(set(baseline_ids)) != len(baseline_ids):
        raise RuntimeError("Frozen representation person identifiers are not unique")
    positions = {person_id: index for index, person_id in enumerate(baseline_ids)}
    if set(person_ids) != set(baseline_ids):
        raise RuntimeError("Neural and frozen representation people do not align")
    return payload[family][np.asarray([positions[value] for value in person_ids])].astype(float)


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    grid_path = args.grid.resolve()
    lock_path = args.neural_lock.resolve()
    selection_path = args.selection_lock.resolve()
    representation_summary_path = args.representation_summary.resolve()
    representation_probabilities_path = args.representation_probabilities.resolve()
    run_root = args.run_root.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    representation_summary = json.loads(representation_summary_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_NEURAL_GRID_LOCKED_BEFORE_FIT":
        raise RuntimeError("The neural grid is not locked")
    if selection.get("status") != "VCOCO_V3_NEURAL_INNER_SELECTION_LOCKED":
        raise RuntimeError("The inner neural selection is not locked")
    if representation_summary.get("status") != (
        "VCOCO_V3_MATCHED_REPRESENTATION_DEVELOPMENT_COMPLETE"
    ):
        raise RuntimeError("The matched frozen representation stage is incomplete")
    if (
        sha256_file(representation_probabilities_path)
        != representation_summary["artifact_sha256"][representation_probabilities_path.name]
    ):
        raise RuntimeError("Frozen representation probability evidence changed")

    seeds = list(map(int, grid["cross_validation"]["outer_fit_seeds"]))
    outer_count = int(grid["cross_validation"]["outer_folds"])
    fold_records = []
    evidence = {}
    phase_counts: dict[str, int] = {}
    for outer in range(outer_count):
        selected = selection["selected_by_outer_fold"][str(outer)]["candidate_id"]
        seed_runs = []
        for seed in seeds:
            path = run_directory(run_root, outer, selected, seed)
            run = load_outer_run(path, outer=outer, candidate=selected, seed=seed)
            seed_runs.append(run)
            phase = str(run["summary"]["best_phase"])
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            evidence[path.relative_to(run_root).as_posix()] = {
                "summary": sha256_file(run["summary_path"]),
                "held_predictions": sha256_file(run["probabilities_path"]),
            }
        reference = seed_runs[0]
        for run in seed_runs[1:]:
            for field in ("person_ids", "image_ids", "labels"):
                if not np.array_equal(run[field], reference[field]):
                    raise RuntimeError(f"Outer seed rows differ in fold {outer}")
        fold_records.append(
            {
                "outer_fold": outer,
                "candidate_id": selected,
                "person_ids": reference["person_ids"],
                "image_ids": reference["image_ids"],
                "labels": reference["labels"],
                "probabilities_by_seed": {
                    seed: run["probabilities"] for seed, run in zip(seeds, seed_runs, strict=True)
                },
                "gate_weights": np.stack([run["gate_weights"] for run in seed_runs]).mean(axis=0),
            }
        )

    person_ids = np.concatenate([record["person_ids"] for record in fold_records])
    image_ids = np.concatenate([record["image_ids"] for record in fold_records])
    labels = np.concatenate([record["labels"] for record in fold_records])
    gate_weights = np.concatenate([record["gate_weights"] for record in fold_records])
    if len(set(person_ids)) != len(person_ids):
        raise RuntimeError("Outer neural folds contain duplicate people")
    probabilities_by_seed = {
        seed: np.concatenate([record["probabilities_by_seed"][seed] for record in fold_records])
        for seed in seeds
    }
    neural_probabilities = np.stack(list(probabilities_by_seed.values())).mean(axis=0)
    backbone = str(lock["selected_backbone"]["model_kind"])
    baseline_probabilities = aligned_baseline(
        representation_probabilities_path,
        family=backbone,
        person_ids=person_ids,
    )

    neural_metrics = classification_metrics(labels, neural_probabilities)
    neural_metrics["locomotion_f1"] = locomotion_f1(labels, neural_probabilities)
    baseline_metrics = classification_metrics(labels, baseline_probabilities)
    baseline_metrics["locomotion_f1"] = locomotion_f1(labels, baseline_probabilities)
    seed_rows = []
    for seed, probabilities in probabilities_by_seed.items():
        metrics = classification_metrics(labels, probabilities)
        metrics["locomotion_f1"] = locomotion_f1(labels, probabilities)
        seed_rows.append({"seed": seed, **metrics})
    seed_metrics = pd.DataFrame(seed_rows)

    rules = protocol["promotion_rules"]
    comparison = paired_cluster_bootstrap(
        labels,
        neural_probabilities,
        baseline_probabilities,
        image_ids,
        resamples=int(rules["bootstrap_resamples"]),
        seed=int(grid["cross_validation"]["outer_seed"]) + 90_000,
    )
    macro = comparison["macro_f1"]
    motion = comparison["per_class_f1"]["walking_running"]
    class_deltas = [comparison["per_class_f1"][name]["point_estimate"] for name in CLASS_NAMES]
    general = rules["general_candidate"]
    specialist = rules["locomotion_specialist"]
    promotion = {
        "reference_family": backbone,
        "general_candidate": bool(
            macro["point_estimate"] >= float(general["minimum_macro_f1_gain"])
            and macro["ci_95_low"] > 0.0
            and min(class_deltas) >= -float(general["maximum_allowed_class_f1_regression"])
        ),
        "locomotion_specialist": bool(
            motion["point_estimate"] >= float(specialist["minimum_locomotion_f1_gain"])
            and motion["ci_95_low"] > 0.0
            and macro["point_estimate"] >= -float(specialist["macro_f1_noninferiority_margin"])
        ),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "neural_vs_frozen_metrics.csv"
    per_class_path = output_dir / "neural_vs_frozen_per_class.csv"
    seed_path = output_dir / "neural_seed_metrics.csv"
    probabilities_path = output_dir / "nested_oof_probabilities.npz"
    comparison_path = output_dir / "paired_uncertainty.json"
    promotion_path = output_dir / "promotion_decision.json"
    pd.DataFrame(
        [
            {"family": f"frozen_{backbone}", **baseline_metrics},
            {"family": "selected_multiview_neural", **neural_metrics},
        ]
    ).to_csv(metrics_path, index=False)
    pd.DataFrame(
        [
            {"family": family, **row}
            for family, probabilities in (
                (f"frozen_{backbone}", baseline_probabilities),
                ("selected_multiview_neural", neural_probabilities),
            )
            for row in per_class_metrics(labels, probabilities, CLASS_NAMES)
        ]
    ).to_csv(per_class_path, index=False)
    seed_metrics.to_csv(seed_path, index=False)
    np.savez_compressed(
        probabilities_path,
        person_ids=person_ids,
        image_ids=image_ids,
        labels=labels,
        class_names=np.asarray(CLASS_NAMES),
        frozen_probabilities=baseline_probabilities,
        neural_probabilities=neural_probabilities,
        gate_weights=gate_weights,
        **{f"seed_{seed}": values for seed, values in probabilities_by_seed.items()},
    )
    write_json(comparison_path, comparison)
    write_json(promotion_path, promotion)
    summary = {
        "status": "VCOCO_V3_NEURAL_DEVELOPMENT_COMPLETE",
        "endpoint": "source_tag_macro_f1",
        "reference_family": backbone,
        "selected_candidate_by_outer_fold": {
            fold: value["candidate_id"]
            for fold, value in selection["selected_by_outer_fold"].items()
        },
        "outer_folds": outer_count,
        "seeds": seeds,
        "people": len(person_ids),
        "source_images": int(pd.Series(image_ids).nunique()),
        "neural_macro_f1": float(neural_metrics["macro_f1"]),
        "neural_locomotion_f1": float(neural_metrics["locomotion_f1"]),
        "seed_macro_f1_mean": float(seed_metrics["macro_f1"].mean()),
        "seed_macro_f1_sample_std": float(seed_metrics["macro_f1"].std(ddof=1)),
        "best_phase_counts": phase_counts,
        "promotion": promotion,
        "human_pilot_labels_used_for_selection": False,
        "official_v2_test_rows_read": 0,
        "official_v2_test_predictions_run": False,
        "source_sha256": {
            "protocol": sha256_file(protocol_path),
            "neural_grid": sha256_file(grid_path),
            "neural_grid_lock": sha256_file(lock_path),
            "inner_selection_lock": sha256_file(selection_path),
            "representation_summary": sha256_file(representation_summary_path),
            "representation_probabilities": sha256_file(representation_probabilities_path),
        },
        "run_artifact_sha256": evidence,
        "artifact_sha256": {
            metrics_path.name: sha256_file(metrics_path),
            per_class_path.name: sha256_file(per_class_path),
            seed_path.name: sha256_file(seed_path),
            probabilities_path.name: sha256_file(probabilities_path),
            comparison_path.name: sha256_file(comparison_path),
            promotion_path.name: sha256_file(promotion_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
