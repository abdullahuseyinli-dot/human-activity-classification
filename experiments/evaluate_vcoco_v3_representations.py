"""Compare frozen DINOv2, DINOv3, and SigLIP2 under matched nested grouped CV."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_analysis import per_class_metrics
from hac.vcoco_v3_cuda_heads import (
    cuda_logistic_fit_audit,
    evaluate_candidate_inner_cuda,
    fit_candidate_cuda,
    reset_cuda_logistic_fit_audit,
    restore_cuda_logistic_fit_audit,
)
from hac.vcoco_v3_models import (
    CLASS_NAMES,
    candidate_rank_key,
    enumerate_candidates,
    geometry_features,
    grouped_splits,
    holm_adjust,
    locomotion_f1,
    paired_cluster_bootstrap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path("experiments/vcoco_v3_representation_grid.json"),
    )
    parser.add_argument(
        "--representation-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/representations/representation_grid_lock.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".runs/vcoco_v3/representations/evaluation"),
    )
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
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if metadata.get("source_sha256") != expected_sources:
        return None
    if metadata.get("family") != family or int(metadata.get("rows", -1)) != rows:
        return None
    if sha256_file(probabilities_path) != metadata.get("artifact_sha256"):
        return None
    with np.load(probabilities_path) as payload:
        probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
    if probabilities.shape != (rows, len(CLASS_NAMES)):
        return None
    completed = list(map(int, metadata.get("completed_outer_folds", [])))
    if len(completed) != len(set(completed)):
        return None
    return (
        probabilities,
        completed,
        list(metadata.get("selection_rows", [])),
        list(metadata.get("cuda_logistic_fit_audit", [])),
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
            "status": "VCOCO_V3_REPRESENTATION_FAMILY_CHECKPOINT",
            "family": family,
            "rows": len(probabilities),
            "completed_outer_folds": sorted(completed_outer_folds),
            "selection_rows": selection_rows,
            "cuda_logistic_fit_audit": cuda_logistic_fit_audit(),
            "source_sha256": expected_sources,
            "artifact_sha256": sha256_file(probabilities_path),
        },
    )


def load_locked_features(root: Path, grid: dict, lock: dict):
    reference = None
    features = {}
    for name, declaration in grid["feature_caches"].items():
        evidence = lock["feature_caches"][name]["source_sha256"]
        cache_dir = (root / declaration["path"]).resolve()
        rows_path = cache_dir / "rows.csv"
        features_path = cache_dir / "features.npy"
        if sha256_file(rows_path) != evidence["rows.csv"]:
            raise RuntimeError(f"Locked row cache drift: {name}")
        if sha256_file(features_path) != evidence["features.npy"]:
            raise RuntimeError(f"Locked feature cache drift: {name}")
        rows = pd.read_csv(rows_path, dtype={"person_id": str, "image_id": str})
        if reference is None:
            reference = rows
        elif not rows["person_id"].equals(reference["person_id"]):
            raise RuntimeError(f"Feature row order differs: {name}")
        features[name] = np.load(features_path, mmap_mode="r")
    if reference is None:
        raise RuntimeError("The representation grid has no feature caches")
    return reference, features


def source_labels(rows: pd.DataFrame) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = rows["label_3"].map(mapping)
    if labels.isna().any():
        raise RuntimeError("An unknown source tag entered representation development")
    return labels.to_numpy(dtype=int)


def promotion_decisions(
    grid: dict,
    probabilities: dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict, dict]:
    reporting = grid["reporting"]
    reference_name = reporting["promotion_reference"]
    baseline = probabilities[reference_name]
    comparisons = {}
    macro_p = {}
    locomotion_p = {}
    for index, (family, values) in enumerate(probabilities.items()):
        if family == reference_name:
            continue
        comparison = paired_cluster_bootstrap(
            labels,
            values,
            baseline,
            groups,
            resamples=int(reporting["bootstrap_resamples"]),
            seed=int(grid["cross_validation"]["random_seed"]) + 80_000 + index,
        )
        comparisons[family] = comparison
        macro_p[family] = comparison["macro_f1"]["two_sided_p"]
        locomotion_p[family] = comparison["per_class_f1"]["walking_running"]["two_sided_p"]
    macro_adjusted = holm_adjust(macro_p)
    locomotion_adjusted = holm_adjust(locomotion_p)
    decisions = {
        reference_name: {
            "role": "reference",
            "general_candidate": False,
            "locomotion_specialist": False,
        }
    }
    rules = reporting["promotion_rules"]
    for family, comparison in comparisons.items():
        macro = comparison["macro_f1"]
        motion = comparison["per_class_f1"]["walking_running"]
        class_points = [comparison["per_class_f1"][name]["point_estimate"] for name in CLASS_NAMES]
        decisions[family] = {
            "role": "challenger",
            "general_candidate": bool(
                macro["point_estimate"] >= float(rules["general_macro_f1_gain"])
                and macro["ci_95_low"] > 0.0
                and macro_adjusted[family] <= 0.05
                and min(class_points) >= -float(rules["maximum_class_f1_regression"])
            ),
            "locomotion_specialist": bool(
                motion["point_estimate"] >= float(rules["locomotion_specialist_f1_gain"])
                and motion["ci_95_low"] > 0.0
                and locomotion_adjusted[family] <= 0.05
                and macro["point_estimate"]
                >= -float(rules["locomotion_macro_noninferiority_margin"])
            ),
            "holm_adjusted_macro_p": float(macro_adjusted[family]),
            "holm_adjusted_locomotion_p": float(locomotion_adjusted[family]),
        }
    return comparisons, decisions


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    root = Path.cwd().resolve()
    grid_path = args.grid.resolve()
    lock_path = args.representation_lock.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_REPRESENTATION_GRID_AND_CACHES_LOCKED_BEFORE_FIT":
        raise RuntimeError("The matched representation grid and caches are not locked")
    if sha256_file(grid_path) != lock["source_sha256"]["representation_grid"]:
        raise RuntimeError("The matched representation grid changed after locking")
    locked_sources = {
        "representation_evaluator_source": Path(__file__).resolve(),
        "cuda_heads_source": root / "src/hac/vcoco_v3_cuda_heads.py",
    }
    for name, source_path in locked_sources.items():
        if sha256_file(source_path) != lock["source_sha256"].get(name):
            raise RuntimeError(f"Locked representation source drift: {name}")
    execution = grid.get("execution_backend", {})
    if execution.get("solver") != "pytorch_cuda_lbfgs_logistic":
        raise RuntimeError("The representation CUDA logistic solver is not declared")
    if not execution.get("cuda_required") or not torch.cuda.is_available():
        raise RuntimeError("Matched representation fitting requires CUDA")
    maximum_iterations = int(execution["maximum_iterations"])
    tolerance = float(execution["gradient_tolerance"])

    rows, features = load_locked_features(root, grid, lock)
    labels = source_labels(rows)
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
        "representation_grid": sha256_file(grid_path),
        "representation_lock": sha256_file(lock_path),
        "representation_evaluator_source": sha256_file(Path(__file__).resolve()),
        "cuda_heads_source": sha256_file(root / "src/hac/vcoco_v3_cuda_heads.py"),
    }
    family_probabilities = {}
    selection_rows = []
    all_cuda_audit = []
    for family_index, (family, declaration) in enumerate(grid["families"].items()):
        component_names = set(declaration["components"])
        family_features = {name: features[name] for name in component_names}
        candidates = enumerate_candidates(grid, family)
        reset_cuda_logistic_fit_audit()
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
            oof, completed_outer_folds, family_selection_rows, family_audit = checkpoint
            restore_cuda_logistic_fit_audit(family_audit)
        for outer_fold, (outer_train, outer_held) in enumerate(outer_splits):
            if outer_fold in completed_outer_folds:
                continue
            train_features = {name: values[outer_train] for name, values in family_features.items()}
            evaluated = []
            for candidate in candidates:
                _, metrics = evaluate_candidate_inner_cuda(
                    candidate,
                    declaration,
                    train_features,
                    geometry[outer_train],
                    labels[outer_train],
                    groups[outer_train],
                    inner_folds=inner_folds,
                    stack_folds=stack_folds,
                    seed=seed + 1_000_000 * (family_index + 1) + 10_000 * outer_fold,
                    maximum_iterations=maximum_iterations,
                    tolerance=tolerance,
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
                if row["family"] != family or row["outer_fold"] != outer_fold:
                    break
                if row["candidate_id"] == selected.candidate_id:
                    row["selected"] = True
            oof[outer_held] = fit_candidate_cuda(
                selected,
                declaration,
                train_features,
                {name: values[outer_held] for name, values in family_features.items()},
                geometry[outer_train],
                geometry[outer_held],
                labels[outer_train],
                groups[outer_train],
                stack_folds=stack_folds,
                seed=seed + 10_000_000 + 100_000 * family_index + outer_fold,
                maximum_iterations=maximum_iterations,
                tolerance=tolerance,
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
            raise RuntimeError(f"Representation family checkpoint is incomplete: {family}")
        selection_rows.extend(family_selection_rows)
        all_cuda_audit.extend(cuda_logistic_fit_audit())
        family_probabilities[family] = oof

    metric_rows = []
    class_rows = []
    for family, probabilities in family_probabilities.items():
        metric_rows.append(
            {
                "family": family,
                **classification_metrics(labels, probabilities),
                "locomotion_f1": locomotion_f1(labels, probabilities),
            }
        )
        class_rows.extend(
            {"family": family, **row}
            for row in per_class_metrics(labels, probabilities, CLASS_NAMES)
        )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["macro_f1", "locomotion_f1", "log_loss", "family"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    uncertainty, promotions = promotion_decisions(grid, family_probabilities, labels, groups)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "nested_source_tag_metrics.csv"
    classes_path = output_dir / "nested_source_tag_per_class.csv"
    selection_path = output_dir / "inner_candidate_selection.csv"
    probabilities_path = output_dir / "nested_oof_probabilities.npz"
    uncertainty_path = output_dir / "paired_uncertainty.json"
    promotions_path = output_dir / "promotion_decisions.json"
    optimization_path = output_dir / "cuda_logistic_optimization.json"
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(class_rows).to_csv(classes_path, index=False)
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
    np.savez_compressed(
        probabilities_path,
        person_ids=rows["person_id"].astype(str).to_numpy(dtype=str),
        image_ids=groups,
        labels=labels,
        class_names=np.asarray(CLASS_NAMES),
        **family_probabilities,
    )
    write_json(uncertainty_path, uncertainty)
    write_json(promotions_path, promotions)
    write_json(
        optimization_path,
        {
            "status": "VCOCO_V3_CUDA_LOGISTIC_OPTIMIZATION_AUDITED",
            "solver": execution["solver"],
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "fits": len(all_cuda_audit),
            "iteration_limit_reached_fits": sum(
                bool(record["iteration_limit_reached"]) for record in all_cuda_audit
            ),
            "records": all_cuda_audit,
        },
    )
    summary = {
        "status": "VCOCO_V3_MATCHED_REPRESENTATION_DEVELOPMENT_COMPLETE",
        "endpoint": "source_tag_macro_f1",
        "best_family": str(metrics.iloc[0]["family"]),
        "best_macro_f1": float(metrics.iloc[0]["macro_f1"]),
        "development_people": len(rows),
        "development_images": int(rows["image_id"].nunique()),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "stack_folds": stack_folds,
        "human_pilot_labels_used_for_selection": False,
        "official_v2_test_rows_read": 0,
        "official_v2_test_predictions_run": False,
        "training_backend": execution["solver"],
        "training_device": torch.cuda.get_device_name(0),
        "cuda_logistic_fits": len(all_cuda_audit),
        "cuda_logistic_iteration_limit_reached_fits": sum(
            bool(record["iteration_limit_reached"]) for record in all_cuda_audit
        ),
        "runtime_seconds": time.perf_counter() - started,
        "source_sha256": {
            "representation_grid": sha256_file(grid_path),
            "representation_lock": sha256_file(lock_path),
            "representation_evaluator_source": sha256_file(Path(__file__).resolve()),
            "cuda_heads_source": sha256_file(root / "src/hac/vcoco_v3_cuda_heads.py"),
        },
        "artifact_sha256": {
            metrics_path.name: sha256_file(metrics_path),
            classes_path.name: sha256_file(classes_path),
            selection_path.name: sha256_file(selection_path),
            probabilities_path.name: sha256_file(probabilities_path),
            uncertainty_path.name: sha256_file(uncertainty_path),
            promotions_path.name: sha256_file(promotions_path),
            optimization_path.name: sha256_file(optimization_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
