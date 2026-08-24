"""Run the declared nested geometry, context, resolution, and box study."""

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
    fit_box_augmented_factorized_cuda,
    fit_candidate_cuda,
    fit_candidate_cuda_many,
    reset_cuda_logistic_fit_audit,
    restore_cuda_logistic_fit_audit,
)
from hac.vcoco_v3_models import (
    CLASS_NAMES,
    Candidate,
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
    parser.add_argument("--grid", type=Path, default=Path("experiments/vcoco_v3_spatial_grid.json"))
    parser.add_argument(
        "--nested-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/nested_stacks/summary.json"),
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
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".runs/vcoco_v3/spatial")
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
    condition_names: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[int], list[dict], list[dict]] | None:
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
        if set(payload.files) != {"probabilities", *condition_names}:
            return None
        probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
        conditions = {
            name: np.asarray(payload[name], dtype=np.float64) for name in condition_names
        }
    expected_shape = (rows, len(CLASS_NAMES))
    if probabilities.shape != expected_shape or any(
        values.shape != expected_shape for values in conditions.values()
    ):
        return None
    completed = list(map(int, metadata.get("completed_outer_folds", [])))
    if len(completed) != len(set(completed)):
        return None
    return (
        probabilities,
        conditions,
        completed,
        list(metadata.get("selection_rows", [])),
        list(metadata.get("cuda_logistic_fit_audit", [])),
    )


def save_family_checkpoint(
    checkpoint_dir: Path,
    family: str,
    probabilities: np.ndarray,
    condition_probabilities: dict[str, np.ndarray],
    completed_outer_folds: list[int],
    selection_rows: list[dict],
    expected_sources: dict[str, str],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    probabilities_path = checkpoint_dir / f"{family}.npz"
    np.savez_compressed(
        probabilities_path,
        probabilities=probabilities,
        **condition_probabilities,
    )
    write_json(
        checkpoint_dir / f"{family}.json",
        {
            "status": "VCOCO_V3_SPATIAL_FAMILY_CHECKPOINT",
            "family": family,
            "rows": len(probabilities),
            "completed_outer_folds": sorted(completed_outer_folds),
            "selection_rows": selection_rows,
            "cuda_logistic_fit_audit": cuda_logistic_fit_audit(),
            "source_sha256": expected_sources,
            "artifact_sha256": sha256_file(probabilities_path),
        },
    )


def load_spatial_caches(root: Path, grid: dict):
    reference_rows = None
    features = {}
    evidence = {}
    for name, relative in grid["feature_caches"].items():
        cache_dir = (root / relative).resolve()
        provenance_path = cache_dir / "provenance.json"
        rows_path = cache_dir / "rows.csv"
        features_path = cache_dir / "features.npy"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("status") not in {
            "VCOCO_V2_DEVELOPMENT_FEATURE_CACHE_COMPLETE",
            "VCOCO_V3_GATED_DEVELOPMENT_FEATURE_CACHE_COMPLETE",
        }:
            raise RuntimeError(f"Incomplete spatial cache: {name}")
        if provenance.get("test_rows_read", provenance.get("official_v2_test_rows_read", 0)) != 0:
            raise RuntimeError(f"Cache {name} crossed the consumed-test boundary")
        if provenance.get(
            "test_predictions_run", provenance.get("official_v2_test_predictions_run", False)
        ):
            raise RuntimeError(f"Cache {name} made consumed-test predictions")
        if sha256_file(rows_path) != provenance["artifact_sha256"]["rows.csv"]:
            raise RuntimeError(f"Spatial row cache drift: {name}")
        if sha256_file(features_path) != provenance["artifact_sha256"]["features.npy"]:
            raise RuntimeError(f"Spatial feature cache drift: {name}")
        rows = pd.read_csv(rows_path, dtype={"person_id": str, "image_id": str})
        if reference_rows is None:
            reference_rows = rows
        elif not rows["person_id"].equals(reference_rows["person_id"]):
            raise RuntimeError(f"Spatial cache row order differs: {name}")
        features[name] = np.load(features_path, mmap_mode="r")
        evidence[name] = {
            "path": relative,
            "provenance_sha256": sha256_file(provenance_path),
            "features_sha256": sha256_file(features_path),
        }
    return reference_rows, features, evidence


def full_manifest(args: argparse.Namespace, rows: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for split, path in (("train", args.train_manifest), ("val", args.val_manifest)):
        frame = pd.read_csv(path.resolve(), dtype={"person_id": str, "image_id": str})
        frame["split"] = split
        frames.append(frame)
    manifest = pd.concat(frames, ignore_index=True).set_index("person_id", drop=False)
    try:
        aligned = manifest.loc[rows["person_id"].astype(str)].reset_index(drop=True)
    except KeyError as error:
        raise RuntimeError("A spatial cache person is absent from the locked manifest") from error
    if not aligned["image_id"].astype(str).equals(rows["image_id"].astype(str)):
        raise RuntimeError("Spatial cache and locked manifest image IDs differ")
    return aligned


def label_indices(rows: pd.DataFrame) -> np.ndarray:
    labels = rows["label_3"].map({name: index for index, name in enumerate(CLASS_NAMES)})
    if labels.isna().any():
        raise RuntimeError("Unknown source tag in the spatial cohort")
    return labels.to_numpy(dtype=int)


def select_geometry(all_geometry: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.empty((len(all_geometry), 0), dtype=np.float32)
    if mode == "basic":
        return all_geometry[:, :5]
    if mode == "boundary":
        return all_geometry
    raise ValueError(f"Unknown geometry mode: {mode}")


def box_candidates(grid: dict, family: str) -> list[Candidate]:
    hyperparameters = grid["hyperparameters"]["probability_stacks"]
    output = []
    for c_value in hyperparameters["component_C"]:
        for class_weight in hyperparameters["class_weight"]:
            output.append(
                Candidate(
                    candidate_id=f"{family}__c-{float(c_value):g}__cw-{class_weight}",
                    family=family,
                    component_c=float(c_value),
                    meta_c=None,
                    svm_c=None,
                    class_weight=str(class_weight),
                )
            )
    return output


def fit_box_candidate(
    candidate: Candidate,
    declaration: dict,
    features: dict[str, np.ndarray],
    geometry: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    target_index: np.ndarray,
    *,
    seed: int,
    maximum_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    context_names = tuple(map(str, declaration["context_components"]))
    return fit_box_augmented_factorized_cuda(
        features[declaration["tight_component"]][train_index],
        {name: features[name][train_index] for name in context_names},
        features[declaration["tight_component"]][target_index],
        {name: features[name][target_index] for name in context_names},
        geometry[train_index],
        geometry[target_index],
        labels[train_index],
        c_value=float(candidate.component_c),
        class_weight=candidate.class_weight,
        seed=seed,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
    )


def evaluate_box_inner(
    candidate: Candidate,
    declaration: dict,
    features: dict[str, np.ndarray],
    geometry: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    inner_folds: int,
    seed: int,
    maximum_iterations: int,
    tolerance: float,
) -> dict[str, float]:
    aggregate = np.zeros((len(labels), len(CLASS_NAMES)), dtype=float)
    conditions = {name: np.zeros_like(aggregate) for name in declaration["context_components"]}
    for fold, (fit_index, held_index) in enumerate(
        grouped_splits(labels, groups, folds=inner_folds, seed=seed)
    ):
        predicted, by_condition = fit_box_candidate(
            candidate,
            declaration,
            features,
            geometry,
            labels,
            fit_index,
            held_index,
            seed=seed + fold,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
        )
        aggregate[held_index] = predicted
        for name, probabilities in by_condition.items():
            conditions[name][held_index] = probabilities
    metrics = classification_metrics(labels, aggregate)
    metrics["locomotion_f1"] = locomotion_f1(labels, aggregate)
    condition_macro = [
        classification_metrics(labels, value)["macro_f1"] for value in conditions.values()
    ]
    condition_locomotion = [locomotion_f1(labels, value) for value in conditions.values()]
    metrics["worst_box_macro_f1"] = min(condition_macro)
    metrics["worst_box_locomotion_f1"] = min(condition_locomotion)
    return metrics


def spatial_rank(candidate: Candidate, metrics: dict[str, float]) -> tuple:
    if "worst_box_macro_f1" in metrics:
        return (
            -metrics["macro_f1"],
            -metrics["worst_box_macro_f1"],
            -metrics["worst_box_locomotion_f1"],
            candidate.candidate_id,
        )
    return candidate_rank_key(candidate, metrics)


def add_subgroups(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["bbox_area_quartile"] = pd.qcut(
        output["bbox_area_fraction"], 4, labels=["Q1_small", "Q2", "Q3", "Q4_large"]
    ).astype(str)
    output["person_height_quartile"] = pd.qcut(
        output["person_pixel_height"], 4, labels=["Q1_short", "Q2", "Q3", "Q4_tall"]
    ).astype(str)
    output["bbox_aspect_quartile"] = pd.qcut(
        output["bbox_aspect_ratio"], 4, labels=["Q1_narrow", "Q2", "Q3", "Q4_wide"]
    ).astype(str)
    output["boundary_contact"] = np.where(
        output["bbox_xmin"].le(0.5)
        | output["bbox_ymin"].le(0.5)
        | output["bbox_xmax"].ge(output["image_width"] - 0.5)
        | output["bbox_ymax"].ge(output["image_height"] - 0.5),
        "boundary",
        "interior",
    )
    occupancy = output.groupby("image_id")["person_id"].transform("size")
    output["scene_occupancy"] = np.where(occupancy.eq(1), "single", "multiple")
    return output


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    root = Path.cwd().resolve()
    grid_path = args.grid.resolve()
    nested_path = args.nested_summary.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    if grid.get("status") != "DECLARED_BEFORE_SPATIAL_FITTING":
        raise RuntimeError("The spatial grid is not declared")
    execution = grid.get("execution_backend", {})
    if execution.get("solver") != "pytorch_cuda_lbfgs_logistic":
        raise RuntimeError("The spatial CUDA logistic solver is not declared")
    if not execution.get("cuda_required") or not torch.cuda.is_available():
        raise RuntimeError("The spatial model-selection stage requires CUDA")
    maximum_iterations = int(execution["maximum_iterations"])
    tolerance = float(execution["gradient_tolerance"])
    if nested.get("status") != grid["gate"]:
        raise RuntimeError("The cached-fusion stage has not completed")
    if nested.get("official_v2_test_rows_read") != 0:
        raise RuntimeError("The cached-fusion stage crossed the test boundary")

    cache_rows, features, cache_evidence = load_spatial_caches(root, grid)
    manifest = full_manifest(args, cache_rows)
    labels = label_indices(cache_rows)
    groups = cache_rows["image_id"].astype(str).to_numpy(dtype=str)
    full_geometry = geometry_features(cache_rows)
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
        "spatial_grid": sha256_file(grid_path),
        "nested_stage": sha256_file(nested_path),
        "train_manifest": sha256_file(args.train_manifest.resolve()),
        "val_manifest": sha256_file(args.val_manifest.resolve()),
        "spatial_evaluator_source": sha256_file(Path(__file__).resolve()),
        "cuda_heads_source": sha256_file(root / "src/hac/vcoco_v3_cuda_heads.py"),
        **{
            f"cache_{name}": evidence["features_sha256"]
            for name, evidence in cache_evidence.items()
        },
    }
    family_probabilities = {}
    selection_rows = []
    all_cuda_audit = []
    condition_probabilities: dict[str, dict[str, np.ndarray]] = {}
    robustness_reference = grid["robustness"]["reference_family"]
    nominal_component = grid["robustness"]["nominal_component"]
    perturbation_components = tuple(grid["robustness"]["perturbation_components"])
    for family_index, (family, declaration) in enumerate(grid["families"].items()):
        geometry = select_geometry(full_geometry, declaration["geometry"])
        is_box = declaration.get("training_mode") == "augmentation_invariant_early_fusion"
        candidates = box_candidates(grid, family) if is_box else enumerate_candidates(grid, family)
        condition_names: tuple[str, ...] = ()
        if is_box:
            condition_names = tuple(map(str, declaration["context_components"]))
        elif family == robustness_reference:
            condition_names = (nominal_component, *perturbation_components)
        reset_cuda_logistic_fit_audit()
        checkpoint = load_family_checkpoint(
            checkpoint_dir,
            family,
            checkpoint_sources,
            rows=len(labels),
            condition_names=condition_names,
        )
        if checkpoint is None:
            oof = np.zeros((len(labels), len(CLASS_NAMES)), dtype=float)
            family_conditions = {name: np.zeros_like(oof) for name in condition_names}
            completed_outer_folds: list[int] = []
            family_selection_rows: list[dict] = []
        else:
            (
                oof,
                family_conditions,
                completed_outer_folds,
                family_selection_rows,
                family_cuda_audit,
            ) = checkpoint
            restore_cuda_logistic_fit_audit(family_cuda_audit)
        if family_conditions:
            condition_probabilities[family] = family_conditions
        for outer_fold, (outer_train, outer_held) in enumerate(outer_splits):
            if outer_fold in completed_outer_folds:
                continue
            evaluated = []
            for candidate in candidates:
                if is_box:
                    metrics = evaluate_box_inner(
                        candidate,
                        declaration,
                        {name: value[outer_train] for name, value in features.items()},
                        geometry[outer_train],
                        labels[outer_train],
                        groups[outer_train],
                        inner_folds=inner_folds,
                        seed=seed + 100_000 * family_index + 1_000 * outer_fold,
                        maximum_iterations=maximum_iterations,
                        tolerance=tolerance,
                    )
                else:
                    _, metrics = evaluate_candidate_inner_cuda(
                        candidate,
                        declaration,
                        {name: value[outer_train] for name, value in features.items()},
                        geometry[outer_train],
                        labels[outer_train],
                        groups[outer_train],
                        inner_folds=inner_folds,
                        stack_folds=stack_folds,
                        seed=seed + 100_000 * family_index + 1_000 * outer_fold,
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
                evaluated, key=lambda value: spatial_rank(value[0], value[1])
            )
            for row in reversed(family_selection_rows):
                if row["family"] == family and row["outer_fold"] == outer_fold:
                    row["selected"] = row["candidate_id"] == selected.candidate_id
                    continue
                break
            if is_box:
                predicted, by_condition = fit_box_candidate(
                    selected,
                    declaration,
                    features,
                    geometry,
                    labels,
                    outer_train,
                    outer_held,
                    seed=seed + 10_000_000 + 100_000 * family_index + outer_fold,
                    maximum_iterations=maximum_iterations,
                    tolerance=tolerance,
                )
                oof[outer_held] = predicted
                for name, probabilities in by_condition.items():
                    family_conditions[name][outer_held] = probabilities
            else:
                train_features = {name: value[outer_train] for name, value in features.items()}
                target_features = {name: value[outer_held] for name, value in features.items()}
                fit_seed = seed + 10_000_000 + 100_000 * family_index + outer_fold
                if family == robustness_reference:
                    target_feature_sets = {nominal_component: target_features}
                    for perturbation in perturbation_components:
                        perturbed_target = dict(target_features)
                        perturbed_target[nominal_component] = features[perturbation][outer_held]
                        target_feature_sets[perturbation] = perturbed_target
                    by_condition = fit_candidate_cuda_many(
                        selected,
                        declaration,
                        train_features,
                        target_feature_sets,
                        geometry[outer_train],
                        geometry[outer_held],
                        labels[outer_train],
                        groups[outer_train],
                        stack_folds=stack_folds,
                        seed=fit_seed,
                        maximum_iterations=maximum_iterations,
                        tolerance=tolerance,
                    )
                    oof[outer_held] = by_condition[nominal_component]
                    for condition, probabilities in by_condition.items():
                        family_conditions[condition][outer_held] = probabilities
                else:
                    oof[outer_held] = fit_candidate_cuda(
                        selected,
                        declaration,
                        train_features,
                        target_features,
                        geometry[outer_train],
                        geometry[outer_held],
                        labels[outer_train],
                        groups[outer_train],
                        stack_folds=stack_folds,
                        seed=fit_seed,
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
                family_conditions,
                completed_outer_folds,
                family_selection_rows,
                checkpoint_sources,
            )
        if sorted(completed_outer_folds) != list(range(outer_folds)):
            raise RuntimeError(f"Spatial family checkpoint is incomplete: {family}")
        selection_rows.extend(family_selection_rows)
        all_cuda_audit.extend(cuda_logistic_fit_audit())
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
            {"family": family, **row}
            for row in per_class_metrics(labels, probabilities, CLASS_NAMES)
        )
    summary_frame = pd.DataFrame(summary_rows).sort_values(
        ["macro_f1", "locomotion_f1", "log_loss", "family"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    per_class_frame = pd.DataFrame(per_class_rows)

    subgroup_rows = []
    subgroup_frame = add_subgroups(manifest)
    for family, probabilities in family_probabilities.items():
        for dimension in grid["reporting"]["subgroups"]:
            for value, index in subgroup_frame.groupby(dimension, sort=True).groups.items():
                subset = np.asarray(list(index), dtype=int)
                metrics = classification_metrics(labels[subset], probabilities[subset])
                subgroup_rows.append(
                    {
                        "family": family,
                        "dimension": dimension,
                        "value": value,
                        "people": len(subset),
                        "macro_f1": metrics["macro_f1"],
                        "locomotion_f1": locomotion_f1(labels[subset], probabilities[subset]),
                    }
                )
    subgroup_results = pd.DataFrame(subgroup_rows)

    robustness_rows = []
    for family, conditions in condition_probabilities.items():
        for condition, probabilities in conditions.items():
            robustness_rows.append(
                {
                    "family": family,
                    "condition": condition,
                    **classification_metrics(labels, probabilities),
                    "locomotion_f1": locomotion_f1(labels, probabilities),
                }
            )
    robustness_frame = pd.DataFrame(robustness_rows)

    baseline_name = grid["reporting"]["promotion_reference"]
    baseline = family_probabilities[baseline_name]
    uncertainty = {}
    p_values = {}
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
        p_values[family] = comparison["macro_f1"]["two_sided_p"]
    adjusted = holm_adjust(p_values)
    promotions = {}
    for family, comparison in uncertainty.items():
        macro = comparison["macro_f1"]
        class_deltas = [comparison["per_class_f1"][name]["point_estimate"] for name in CLASS_NAMES]
        promotions[family] = {
            "holm_adjusted_macro_p": adjusted[family],
            "general_candidate": (
                macro["point_estimate"] >= 0.01
                and macro["ci_95_low"] > 0.0
                and adjusted[family] <= 0.05
                and min(class_deltas) >= -0.01
            ),
            "locomotion_specialist": (
                comparison["per_class_f1"]["walking_running"]["point_estimate"] >= 0.02
                and comparison["per_class_f1"]["walking_running"]["ci_95_low"] > 0.0
                and macro["point_estimate"] >= -0.005
            ),
        }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "spatial_metrics.csv": summary_frame,
        "spatial_per_class.csv": per_class_frame,
        "spatial_subgroups.csv": subgroup_results,
        "box_robustness.csv": robustness_frame,
        "inner_candidate_selection.csv": pd.DataFrame(selection_rows),
    }
    artifact_hashes = {}
    for name, frame in artifacts.items():
        path = output_dir / name
        frame.to_csv(path, index=False)
        artifact_hashes[name] = sha256_file(path)
    probability_path = output_dir / "spatial_oof_probabilities.npz"
    np.savez_compressed(
        probability_path,
        person_ids=manifest["person_id"].astype(str).to_numpy(dtype=str),
        image_ids=groups,
        labels=labels,
        **family_probabilities,
    )
    artifact_hashes[probability_path.name] = sha256_file(probability_path)
    uncertainty_path = output_dir / "spatial_uncertainty.json"
    promotions_path = output_dir / "spatial_promotions.json"
    write_json(uncertainty_path, uncertainty)
    write_json(promotions_path, promotions)
    artifact_hashes[uncertainty_path.name] = sha256_file(uncertainty_path)
    artifact_hashes[promotions_path.name] = sha256_file(promotions_path)
    optimization_path = output_dir / "cuda_logistic_optimization.json"
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
    artifact_hashes[optimization_path.name] = sha256_file(optimization_path)
    result = {
        "status": "VCOCO_V3_SPATIAL_DEVELOPMENT_COMPLETE",
        "endpoint": "source_tag_macro_f1",
        "official_v2_test_rows_read": 0,
        "official_v2_test_predictions_run": False,
        "best_family": str(summary_frame.iloc[0]["family"]),
        "best_macro_f1": float(summary_frame.iloc[0]["macro_f1"]),
        "training_backend": execution["solver"],
        "training_device": torch.cuda.get_device_name(0),
        "cuda_logistic_fits": len(all_cuda_audit),
        "cuda_logistic_iteration_limit_reached_fits": sum(
            bool(record["iteration_limit_reached"]) for record in all_cuda_audit
        ),
        "runtime_seconds": time.perf_counter() - started,
        "source_sha256": {
            "spatial_grid": sha256_file(grid_path),
            "nested_stage": sha256_file(nested_path),
            "train_manifest": sha256_file(args.train_manifest.resolve()),
            "val_manifest": sha256_file(args.val_manifest.resolve()),
            "spatial_evaluator_source": sha256_file(Path(__file__).resolve()),
            "cuda_heads_source": sha256_file(root / "src/hac/vcoco_v3_cuda_heads.py"),
        },
        "cache_evidence": cache_evidence,
        "artifact_sha256": artifact_hashes,
    }
    write_json(output_dir / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
