"""Evaluate declared V-COCO v3 fusion families with fully nested grouped CV."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_analysis import per_class_metrics
from hac.vcoco_v3_models import (
    CLASS_NAMES,
    candidate_rank_key,
    cuda_svm_fit_audit,
    enumerate_candidates,
    evaluate_candidate_inner,
    fit_candidate,
    geometry_features,
    grouped_splits,
    holm_adjust,
    locomotion_f1,
    paired_cluster_bootstrap,
    reset_cuda_svm_fit_audit,
    restore_cuda_svm_fit_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", type=Path, default=Path("experiments/vcoco_v3_candidate_grid.json")
    )
    parser.add_argument(
        "--candidate-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/candidates/candidate_grid_lock.json"),
    )
    parser.add_argument(
        "--human-gate",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/final/summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(".runs/vcoco_v3/nested_stacks"))
    return parser.parse_args()


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_family_checkpoint(
    checkpoint_dir: Path,
    family: str,
    expected_sources: dict[str, str],
    *,
    rows: int,
) -> tuple[np.ndarray, list[int], list[dict], list[dict]] | None:
    metadata_path = checkpoint_dir / f"{family}.json"
    probabilities_path = checkpoint_dir / f"{family}.npz"
    if not metadata_path.is_file() or not probabilities_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("source_sha256") != expected_sources:
        return None
    if metadata.get("family") != family or int(metadata.get("rows", -1)) != rows:
        return None
    payload = np.load(probabilities_path)
    probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
    if probabilities.shape != (rows, len(CLASS_NAMES)):
        return None
    return (
        probabilities,
        list(map(int, metadata.get("completed_outer_folds", []))),
        list(metadata.get("selection_rows", [])),
        list(metadata.get("cuda_svm_fit_audit", [])),
    )


def save_family_checkpoint(
    checkpoint_dir: Path,
    family: str,
    probabilities: np.ndarray,
    completed_outer_folds: list[int],
    selection_rows: list[dict],
    expected_sources: dict[str, str],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    probabilities_path = checkpoint_dir / f"{family}.npz"
    np.savez_compressed(probabilities_path, probabilities=probabilities)
    write_json(
        checkpoint_dir / f"{family}.json",
        {
            "status": "VCOCO_V3_NESTED_FAMILY_CHECKPOINT",
            "family": family,
            "rows": len(probabilities),
            "completed_outer_folds": sorted(completed_outer_folds),
            "selection_rows": selection_rows,
            "cuda_svm_fit_audit": cuda_svm_fit_audit(),
            "source_sha256": expected_sources,
            "artifact_sha256": {probabilities_path.name: sha256_file(probabilities_path)},
        },
    )


def load_features(root: Path, grid: dict, lock: dict):
    features = {}
    reference_rows = None
    for name, declaration in grid["feature_caches"].items():
        cache_dir = (root / declaration["path"]).resolve()
        evidence = lock["feature_caches"][name]["source_sha256"]
        rows_path = cache_dir / "rows.csv"
        features_path = cache_dir / "features.npy"
        if sha256_file(rows_path) != evidence["rows.csv"]:
            raise RuntimeError(f"Locked row cache drift: {name}")
        if sha256_file(features_path) != evidence["features.npy"]:
            raise RuntimeError(f"Locked feature cache drift: {name}")
        rows = pd.read_csv(rows_path, dtype={"person_id": str, "image_id": str})
        if reference_rows is None:
            reference_rows = rows
        elif not rows["person_id"].equals(reference_rows["person_id"]):
            raise RuntimeError(f"Feature row order differs: {name}")
        features[name] = np.load(features_path, mmap_mode="r")
    return reference_rows, features


def label_indices(rows: pd.DataFrame) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = rows["label_3"].map(mapping)
    if labels.isna().any():
        raise RuntimeError("An unknown source activity tag entered development")
    return labels.to_numpy(dtype=int)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    root = Path.cwd().resolve()
    grid_path = args.grid.resolve()
    lock_path = args.candidate_lock.resolve()
    human_gate_path = args.human_gate.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_CANDIDATE_GRID_AND_CACHES_LOCKED_BEFORE_FIT":
        raise RuntimeError("The candidate grid and feature caches are not locked")
    if sha256_file(grid_path) != lock["source_sha256"]["candidate_grid"]:
        raise RuntimeError("The declared candidate grid changed after lock")
    locked_sources = {
        "models_source": root / "src/hac/vcoco_v3_models.py",
        "nested_evaluator_source": root / "experiments/evaluate_vcoco_v3_nested_stacks.py",
    }
    for name, source_path in locked_sources.items():
        if sha256_file(source_path) != lock["source_sha256"].get(name):
            raise RuntimeError(f"Locked nested-evaluation source drift: {name}")
    human_gate = json.loads(human_gate_path.read_text(encoding="utf-8"))
    if (
        human_gate.get("status") != "VCOCO_V3_HUMAN_PILOT_AUDIT_COMPLETE"
        or not human_gate.get("source_tag_development_model_fitting_permitted")
    ):
        raise RuntimeError("Nested model fitting is blocked by the human pilot audit")

    rows, features = load_features(root, grid, lock)
    labels = label_indices(rows)
    groups = rows["image_id"].astype(str).to_numpy(dtype=str)
    geometry = geometry_features(rows)
    cross_validation = grid["cross_validation"]
    outer_folds = int(cross_validation["outer_folds"])
    inner_folds = int(cross_validation["inner_folds"])
    stack_folds = int(cross_validation["stack_folds"])
    seed = int(cross_validation["random_seed"])
    outer_splits = grouped_splits(labels, groups, folds=outer_folds, seed=seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_sources = {
        "candidate_grid": sha256_file(grid_path),
        "candidate_lock": sha256_file(lock_path),
        "human_pilot_audit": sha256_file(human_gate_path),
    }

    family_probabilities = {}
    selection_rows = []
    reset_cuda_svm_fit_audit()
    for family_index, (family, declaration) in enumerate(grid["families"].items()):
        candidates = enumerate_candidates(grid, family)
        checkpoint = load_family_checkpoint(
            checkpoint_dir,
            family,
            checkpoint_sources,
            rows=len(rows),
        )
        if checkpoint is None:
            oof = np.zeros((len(rows), len(CLASS_NAMES)), dtype=np.float64)
            completed_outer_folds: list[int] = []
            family_selection_rows: list[dict] = []
        else:
            oof, completed_outer_folds, family_selection_rows, svm_audit = checkpoint
            restore_cuda_svm_fit_audit(svm_audit)
        for outer_fold, (outer_train, outer_held) in enumerate(outer_splits):
            if outer_fold in completed_outer_folds:
                continue
            train_features = {name: values[outer_train] for name, values in features.items()}
            train_geometry = geometry[outer_train]
            train_labels = labels[outer_train]
            train_groups = groups[outer_train]
            evaluated = []
            for candidate in candidates:
                _, metrics = evaluate_candidate_inner(
                    candidate,
                    declaration,
                    train_features,
                    train_geometry,
                    train_labels,
                    train_groups,
                    inner_folds=inner_folds,
                    stack_folds=stack_folds,
                    seed=seed + 1_000_000 * (family_index + 1) + 10_000 * (outer_fold + 1),
                )
                evaluated.append((candidate, metrics))
                family_selection_rows.append(
                    {
                        "family": family,
                        "outer_fold": outer_fold,
                        "candidate_id": candidate.candidate_id,
                        **metrics,
                        "selected": False,
                    }
                )
            selected, selected_metrics = min(
                evaluated, key=lambda value: candidate_rank_key(value[0], value[1])
            )
            for row in reversed(family_selection_rows):
                if row["family"] == family and row["outer_fold"] == outer_fold:
                    if row["candidate_id"] == selected.candidate_id:
                        row["selected"] = True
                    continue
                break
            oof[outer_held] = fit_candidate(
                selected,
                declaration,
                train_features,
                {name: values[outer_held] for name, values in features.items()},
                train_geometry,
                geometry[outer_held],
                train_labels,
                train_groups,
                stack_folds=stack_folds,
                seed=seed + 10_000_000 + 100_000 * family_index + outer_fold,
            )
            print(
                json.dumps(
                    {
                        "family": family,
                        "outer_fold": outer_fold,
                        "selected": selected.candidate_id,
                        "inner_macro_f1": selected_metrics["macro_f1"],
                    }
                ),
                flush=True,
            )
            completed_outer_folds.append(outer_fold)
            save_family_checkpoint(
                checkpoint_dir,
                family,
                oof,
                completed_outer_folds,
                family_selection_rows,
                checkpoint_sources,
            )
        if sorted(completed_outer_folds) != list(range(outer_folds)):
            raise RuntimeError(f"Nested family checkpoint is incomplete: {family}")
        selection_rows.extend(family_selection_rows)
        family_probabilities[family] = oof

    summary_rows = []
    per_class_rows = []
    for family, probabilities in family_probabilities.items():
        summary_rows.append(
            {
                "family": family,
                **classification_metrics(labels, probabilities),
                "locomotion_f1": locomotion_f1(labels, probabilities),
            }
        )
        per_class_rows.extend(
            {"family": family, **value}
            for value in per_class_metrics(labels, probabilities, CLASS_NAMES)
        )
    summary_frame = pd.DataFrame(summary_rows).sort_values(
        ["macro_f1", "locomotion_f1", "log_loss", "family"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    per_class_frame = pd.DataFrame(per_class_rows)

    baseline_name = grid["reporting"]["promotion_reference"]
    baseline = family_probabilities[baseline_name]
    uncertainty = {}
    macro_p = {}
    locomotion_p = {}
    for index, (family, probabilities) in enumerate(family_probabilities.items()):
        if family == baseline_name:
            continue
        comparison = paired_cluster_bootstrap(
            labels,
            probabilities,
            baseline,
            groups,
            resamples=int(grid["reporting"]["bootstrap_resamples"]),
            seed=seed + 50_000 + index,
        )
        uncertainty[family] = comparison
        macro_p[family] = comparison["macro_f1"]["two_sided_p"]
        locomotion_p[family] = comparison["per_class_f1"]["walking_running"]["two_sided_p"]
    macro_adjusted = holm_adjust(macro_p)
    locomotion_adjusted = holm_adjust(locomotion_p)
    promotions = {}
    for family, comparison in uncertainty.items():
        class_points = [comparison["per_class_f1"][name]["point_estimate"] for name in CLASS_NAMES]
        macro = comparison["macro_f1"]
        locomotion = comparison["per_class_f1"]["walking_running"]
        general = (
            macro["point_estimate"] >= 0.01
            and macro["ci_95_low"] > 0.0
            and macro_adjusted[family] <= 0.05
            and min(class_points) >= -0.01
        )
        specialist = (
            locomotion["point_estimate"] >= 0.02
            and locomotion["ci_95_low"] > 0.0
            and locomotion_adjusted[family] <= 0.05
            and macro["point_estimate"] >= -0.005
        )
        promotions[family] = {
            "general_candidate": general,
            "locomotion_specialist": specialist,
            "holm_adjusted_macro_p": macro_adjusted[family],
            "holm_adjusted_locomotion_p": locomotion_adjusted[family],
        }

    summary_path = output_dir / "nested_source_tag_metrics.csv"
    per_class_path = output_dir / "nested_source_tag_per_class.csv"
    selection_path = output_dir / "inner_candidate_selection.csv"
    probability_path = output_dir / "nested_oof_probabilities.npz"
    summary_frame.to_csv(summary_path, index=False)
    per_class_frame.to_csv(per_class_path, index=False)
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
    np.savez_compressed(
        probability_path,
        person_ids=rows["person_id"].astype(str).to_numpy(dtype=str),
        image_ids=groups,
        labels=labels,
        class_names=np.asarray(CLASS_NAMES),
        **family_probabilities,
    )
    uncertainty_path = output_dir / "nested_paired_uncertainty.json"
    promotion_path = output_dir / "promotion_decisions.json"
    svm_audit_path = output_dir / "cuda_svm_optimization.json"
    write_json(uncertainty_path, uncertainty)
    write_json(promotion_path, promotions)
    svm_records = cuda_svm_fit_audit()
    write_json(
        svm_audit_path,
        {
            "status": "VCOCO_V3_CUDA_LINEAR_SVM_OPTIMIZATION_AUDITED",
            "solver": "pytorch_cuda_lbfgs_ovr_squared_hinge",
            "fits": len(svm_records),
            "iteration_limit_reached_fits": sum(
                bool(record["iteration_limit_reached"]) for record in svm_records
            ),
            "records": svm_records,
        },
    )
    result = {
        "status": "VCOCO_V3_NESTED_CACHED_FUSION_DEVELOPMENT_COMPLETE",
        "endpoint": "source_tag_macro_f1",
        "human_pilot_labels_used_for_selection": False,
        "official_v2_test_rows_read": 0,
        "official_v2_test_predictions_run": False,
        "people": len(rows),
        "source_images": int(rows["image_id"].nunique()),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "stack_folds": stack_folds,
        "best_family": summary_frame.iloc[0]["family"],
        "best_macro_f1": float(summary_frame.iloc[0]["macro_f1"]),
        "runtime_seconds": time.perf_counter() - started,
        "source_sha256": {
            "candidate_grid": sha256_file(grid_path),
            "candidate_lock": sha256_file(lock_path),
            "human_pilot_audit": sha256_file(human_gate_path),
        },
        "artifact_sha256": {
            summary_path.name: sha256_file(summary_path),
            per_class_path.name: sha256_file(per_class_path),
            selection_path.name: sha256_file(selection_path),
            probability_path.name: sha256_file(probability_path),
            uncertainty_path.name: sha256_file(uncertainty_path),
            promotion_path.name: sha256_file(promotion_path),
            svm_audit_path.name: sha256_file(svm_audit_path),
        },
    }
    write_json(output_dir / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
