"""Fit a source-only V-COCO head on CUDA and score the Okutama feature store."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hac.polar import sha256_file
from hac.vcoco_v3_cuda_heads import (
    cuda_logistic_fit_audit,
    evaluate_candidate_inner_cuda,
    fit_candidate_cuda,
    reset_cuda_logistic_fit_audit,
)
from hac.vcoco_v3_models import (
    CLASS_NAMES,
    candidate_rank_key,
    enumerate_candidates,
    geometry_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representation-grid",
        type=Path,
        default=Path("experiments/vcoco_v3_representation_grid.json"),
    )
    parser.add_argument(
        "--representation-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/representations/representation_grid_lock.json"),
    )
    parser.add_argument(
        "--representation-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/representations/evaluation/summary.json"),
    )
    parser.add_argument(
        "--representation-metrics",
        type=Path,
        default=Path(
            ".runs/vcoco_v3/representations/evaluation/nested_source_tag_metrics.csv"
        ),
    )
    parser.add_argument("--target-store", type=Path, required=True)
    parser.add_argument(
        "--target-partition",
        choices=["development", "confirmation"],
        default="development",
    )
    parser.add_argument("--pipeline-lock", type=Path)
    parser.add_argument("--target-cache-summary", type=Path)
    parser.add_argument(
        "--protocol-amendment",
        type=Path,
        default=Path(".runs/vcoco_v3/protocol/external_cuda_amendment_lock.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".runs/vcoco_v3/okutama/source_only"),
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_cache(root: Path, declaration: dict) -> Path:
    path = Path(str(declaration["path"]))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_source_features(
    root: Path,
    grid: dict,
    lock: dict,
    component_names: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    reference = None
    features = {}
    for name in component_names:
        declaration = grid["feature_caches"][name]
        evidence = lock["feature_caches"][name]["source_sha256"]
        cache = resolve_cache(root, declaration)
        rows_path = cache / "rows.csv"
        features_path = cache / "features.npy"
        if sha256_file(rows_path) != evidence["rows.csv"]:
            raise RuntimeError(f"Locked source rows changed: {name}")
        if sha256_file(features_path) != evidence["features.npy"]:
            raise RuntimeError(f"Locked source features changed: {name}")
        rows = pd.read_csv(rows_path, dtype={"person_id": str, "image_id": str})
        if reference is None:
            reference = rows
        elif not rows.equals(reference):
            raise RuntimeError("Source representation rows are not aligned")
        features[name] = np.load(features_path, mmap_mode="r")
    if reference is None:
        raise RuntimeError("No source representation features were loaded")
    return reference, features


def load_target_store(
    path: Path,
    *,
    expected_model_kind: str,
    confirmation_archive_opened: bool,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    store = json.loads(path.read_text(encoding="utf-8"))
    if store.get("status") != "VCOCO_V3_PACKED_TEMPORAL_FEATURE_STORE_COMPLETE":
        raise RuntimeError("The Okutama target feature store is incomplete")
    if store.get("model_kind") != expected_model_kind:
        raise RuntimeError("The target store uses a different frozen representation")
    if store.get("confirmation_archive_opened") is not confirmation_archive_opened:
        raise RuntimeError("The target store belongs to a different provider partition")
    arrays = []
    for name in ("tight", "context", "geometry"):
        declaration = store["arrays"][name]
        array_path = (path.parent / str(declaration["path"])).resolve()
        if sha256_file(array_path) != declaration["sha256"]:
            raise RuntimeError(f"Okutama target {name} features changed")
        arrays.append(np.load(array_path, mmap_mode="r"))
    tight, context, geometry = arrays
    expected = (int(store["samples"]), int(store["frames_per_sample"]))
    if tight.shape[:2] != expected or context.shape != tight.shape:
        raise RuntimeError("Okutama visual feature shapes differ from the store")
    if geometry.shape != (*expected, 6):
        raise RuntimeError("Okutama geometry shape differs from the store")
    return store, tight, context, geometry


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Source-only transfer fitting requires CUDA")
    root = Path.cwd().resolve()
    grid_path = args.representation_grid.resolve()
    lock_path = args.representation_lock.resolve()
    summary_path = args.representation_summary.resolve()
    metrics_path = args.representation_metrics.resolve()
    target_path = args.target_store.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    representation = json.loads(summary_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_REPRESENTATION_GRID_AND_CACHES_LOCKED_BEFORE_FIT":
        raise RuntimeError("The matched representation grid is not locked")
    if representation.get("status") != "VCOCO_V3_MATCHED_REPRESENTATION_DEVELOPMENT_COMPLETE":
        raise RuntimeError("The matched representation screen is incomplete")
    if sha256_file(metrics_path) != representation["artifact_sha256"][metrics_path.name]:
        raise RuntimeError("The matched representation metrics changed")
    execution = grid.get("execution_backend", {})
    if execution.get("cuda_required") is not True:
        raise RuntimeError("The source-only solver was not declared as CUDA-only")

    metrics = pd.read_csv(metrics_path)
    dino = metrics[metrics["family"].isin({"dinov2_base", "dinov3_base"})].sort_values(
        ["macro_f1", "locomotion_f1", "log_loss", "family"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    if len(dino) != 2:
        raise RuntimeError("The representation screen lacks one of the matched DINO families")
    family = str(dino.iloc[0]["family"])
    declaration = grid["families"][family]
    component_names = tuple(map(str, declaration["components"]))
    rows, source_features = load_source_features(root, grid, lock, component_names)
    source_geometry = geometry_features(rows)
    label_map = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = rows["label_3"].map(label_map).to_numpy(dtype=int)
    groups = rows["image_id"].astype(str).to_numpy(dtype=str)
    if args.target_cache_summary is None:
        raise ValueError("Source-only scoring requires the target cache summary")
    cache_path = args.target_cache_summary.resolve()
    amendment_path = args.protocol_amendment.resolve()
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if (
        amendment.get("status")
        != "VCOCO_V3_EXTERNAL_CUDA_AMENDMENT_LOCKED_BEFORE_TARGET_FITTING"
    ):
        raise RuntimeError("The external CUDA protocol amendment is not locked")
    if amendment["source_sha256"].get("source_transfer_source") != sha256_file(
        Path(__file__).resolve()
    ):
        raise RuntimeError("The amended source-transfer implementation changed")
    expected_cache_status = (
        "VCOCO_V3_OKUTAMA_CONFIRMATION_FEATURE_STORE_COMPLETE"
        if args.target_partition == "confirmation"
        else "VCOCO_V3_OKUTAMA_DEVELOPMENT_FEATURE_STORE_COMPLETE"
    )
    if cache.get("status") != expected_cache_status:
        raise RuntimeError("The target feature cache belongs to a different partition")
    if cache.get("model_kind") != family:
        raise RuntimeError("The target feature cache uses a different DINO family")
    if cache["source_sha256"].get("external_cuda_amendment") != sha256_file(
        amendment_path
    ):
        raise RuntimeError("The target feature cache belongs to a different amendment")
    target_store_sha256 = sha256_file(target_path)
    if cache["artifact_sha256"].get(target_path.name) != target_store_sha256:
        raise RuntimeError("The target feature store changed after caching")
    source_sha256 = {
        "representation_grid": sha256_file(grid_path),
        "representation_lock": sha256_file(lock_path),
        "representation_summary": sha256_file(summary_path),
        "representation_metrics": sha256_file(metrics_path),
        "external_cuda_amendment": sha256_file(amendment_path),
        "target_cache_summary": sha256_file(cache_path),
        "target_store": target_store_sha256,
        "runner": sha256_file(Path(__file__).resolve()),
    }
    if args.target_partition == "confirmation":
        if args.pipeline_lock is None:
            raise ValueError("Confirmation source-only scoring requires the pipeline lock")
        pipeline_path = args.pipeline_lock.resolve()
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        if pipeline.get("status") != "VCOCO_V3_TEMPORAL_PIPELINE_LOCKED_BEFORE_CONFIRMATION":
            raise RuntimeError("The temporal pipeline is not locked for confirmation")
        if cache["source_sha256"].get("pipeline_lock") != sha256_file(pipeline_path):
            raise RuntimeError("The confirmation cache belongs to a different pipeline")
        source_sha256["pipeline_lock"] = sha256_file(pipeline_path)

    # The confirmation branch must establish provenance before any feature array is opened.
    store, target_tight, target_context, target_geometry = load_target_store(
        target_path,
        expected_model_kind=family,
        confirmation_archive_opened=args.target_partition == "confirmation",
    )
    center = int(store["center_frame_index"])
    target_features = {
        component_names[0]: target_tight[:, center],
        component_names[1]: target_context[:, center],
    }
    target_geometry_center = target_geometry[:, center]
    request_core = {
        "status": "OKUTAMA_SOURCE_ONLY_TRANSFER_REQUEST",
        "family": family,
        "source_samples": len(rows),
        "source_groups": int(pd.Series(groups).nunique()),
        "target_feature_rows": int(store["samples"]),
        "target_partition": args.target_partition,
        "target_labels_read": 0,
        "solver": execution["solver"],
        "source_sha256": source_sha256,
    }
    request_hash = hashlib.sha256(
        json.dumps(request_core, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "request.json", {**request_core, "request_sha256": request_hash})

    started = time.perf_counter()
    reset_cuda_logistic_fit_audit()
    candidates = enumerate_candidates(grid, family)
    evaluated = []
    selection_rows = []
    cross_validation = grid["cross_validation"]
    maximum_iterations = int(execution["maximum_iterations"])
    tolerance = float(execution["gradient_tolerance"])
    for candidate in candidates:
        _, candidate_metrics = evaluate_candidate_inner_cuda(
            candidate,
            declaration,
            source_features,
            source_geometry,
            labels,
            groups,
            inner_folds=int(cross_validation["inner_folds"]),
            stack_folds=int(cross_validation["stack_folds"]),
            seed=int(cross_validation["random_seed"]),
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
        )
        evaluated.append((candidate, candidate_metrics))
        selection_rows.append({**asdict(candidate), **candidate_metrics})
    selected, selected_metrics = min(
        evaluated,
        key=lambda item: candidate_rank_key(item[0], item[1]),
    )
    probabilities = fit_candidate_cuda(
        selected,
        declaration,
        source_features,
        target_features,
        source_geometry,
        target_geometry_center,
        labels,
        groups,
        stack_folds=int(cross_validation["stack_folds"]),
        seed=int(cross_validation["random_seed"]) + 100_000,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
    )
    predictions_path = output_dir / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        target_feature_indices=np.arange(len(probabilities), dtype=np.int64),
        probabilities=probabilities,
        class_names=np.asarray(CLASS_NAMES),
    )
    selection_path = output_dir / "source_candidate_selection.csv"
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
    optimization_path = output_dir / "cuda_optimization.json"
    audit = cuda_logistic_fit_audit()
    write_json(
        optimization_path,
        {
            "device": torch.cuda.get_device_name(0),
            "fits": len(audit),
            "iteration_limit_reached_fits": sum(
                bool(record["iteration_limit_reached"]) for record in audit
            ),
            "records": audit,
        },
    )
    summary = {
        "status": "OKUTAMA_SOURCE_ONLY_TRANSFER_PREDICTIONS_COMPLETE",
        "family": family,
        "selected_candidate": asdict(selected),
        "source_inner_metrics": selected_metrics,
        "source_samples": len(rows),
        "source_groups": int(pd.Series(groups).nunique()),
        "target_feature_rows": len(probabilities),
        "target_partition": args.target_partition,
        "target_labels_read": 0,
        "training_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "request_sha256": request_hash,
        "source_sha256": source_sha256,
        "runtime_seconds": time.perf_counter() - started,
        "artifact_sha256": {
            predictions_path.name: sha256_file(predictions_path),
            selection_path.name: sha256_file(selection_path),
            optimization_path.name: sha256_file(optimization_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
