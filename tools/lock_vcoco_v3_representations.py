"""Lock the matched frozen-representation grid and aligned development caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file

EXPECTED_FAMILIES = {"dinov2_base", "dinov3_base", "siglip2_base"}
ALLOWED_CACHE_STATES = {
    "VCOCO_V2_DEVELOPMENT_FEATURE_CACHE_COMPLETE",
    "VCOCO_V3_GATED_DEVELOPMENT_FEATURE_CACHE_COMPLETE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path("experiments/vcoco_v3_representation_grid.json"),
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/protocol/vcoco_v3_lock.json"),
    )
    parser.add_argument(
        "--spatial-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/spatial/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runs/vcoco_v3/representations/representation_grid_lock.json"),
    )
    return parser.parse_args()


def validate_grid(grid: dict) -> None:
    if grid.get("status") != "DECLARED_BEFORE_REPRESENTATION_FITTING":
        raise ValueError("Representation grid is not in its declared pre-fit state")
    if set(grid.get("families", {})) != EXPECTED_FAMILIES:
        raise ValueError("The matched representation families changed")
    if grid.get("development_data", {}).get("official_v2_test_used"):
        raise ValueError("The consumed V-COCO test cannot enter representation selection")
    if grid.get("development_data", {}).get("human_pilot_labels_used_for_selection"):
        raise ValueError("Human pilot labels cannot select the representation")
    cross_validation = grid.get("cross_validation", {})
    if cross_validation.get("splitter") != "StratifiedGroupKFold":
        raise ValueError("Representation selection must remain grouped")
    for name in ("outer_folds", "inner_folds", "stack_folds"):
        if int(cross_validation.get(name, 0)) < 3:
            raise ValueError(f"{name} must be at least three")
    execution = grid.get("execution_backend", {})
    if execution.get("solver") != "pytorch_cuda_lbfgs_logistic":
        raise ValueError("Representation fitting must use the declared CUDA solver")
    if not execution.get("cuda_required"):
        raise ValueError("Representation fitting must require CUDA")


def _cache_test_boundary(provenance: dict) -> tuple[int, bool]:
    if provenance["status"] == "VCOCO_V2_DEVELOPMENT_FEATURE_CACHE_COMPLETE":
        return int(provenance.get("test_rows_read", -1)), bool(
            provenance.get("test_predictions_run", True)
        )
    return int(provenance.get("official_v2_test_rows_read", -1)), bool(
        provenance.get("official_v2_test_predictions_run", True)
    )


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    grid_path = args.grid.resolve()
    protocol_lock_path = args.protocol_lock.resolve()
    spatial_path = args.spatial_summary.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    validate_grid(grid)
    protocol = json.loads(protocol_lock_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "VCOCO_V3_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("The v3 protocol lock is invalid")
    spatial = json.loads(spatial_path.read_text(encoding="utf-8"))
    if spatial.get("status") != "VCOCO_V3_SPATIAL_DEVELOPMENT_COMPLETE":
        raise RuntimeError("The representation lock requires the completed spatial stage")
    if spatial.get("official_v2_test_rows_read") != 0:
        raise RuntimeError("The spatial stage crossed the consumed-test boundary")

    cache_evidence = {}
    reference_rows = None
    for cache_name, declaration in grid["feature_caches"].items():
        cache_dir = (root / declaration["path"]).resolve()
        provenance_path = cache_dir / "provenance.json"
        rows_path = cache_dir / "rows.csv"
        features_path = cache_dir / "features.npy"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("status") not in ALLOWED_CACHE_STATES:
            raise RuntimeError(f"Incomplete representation cache: {cache_name}")
        for field in ("model_kind", "view", "preprocess", "image_size"):
            if provenance.get(field) != declaration[field]:
                raise RuntimeError(f"Declared {field} does not match cache {cache_name}")
        if provenance["status"] == "VCOCO_V3_GATED_DEVELOPMENT_FEATURE_CACHE_COMPLETE":
            if provenance.get("stage") != "representation":
                raise RuntimeError(f"A non-representation v3 cache entered {cache_name}")
            if provenance.get("source_sha256", {}).get("spatial_stage") != sha256_file(
                spatial_path
            ):
                raise RuntimeError(f"Spatial-stage drift in cache {cache_name}")
            model = provenance.get("model", {})
            declared_model = grid["models"][declaration["model_kind"]]
            for field in ("model_id", "revision"):
                if model.get(field) != declared_model[field]:
                    raise RuntimeError(f"Pinned {field} drift in cache {cache_name}")
            if not provenance.get("checkpoint", {}).get("files"):
                raise RuntimeError(f"Checkpoint evidence is missing from cache {cache_name}")
        test_rows, test_predictions = _cache_test_boundary(provenance)
        if test_rows != 0 or test_predictions:
            raise RuntimeError(f"Cache {cache_name} crossed the consumed-test boundary")
        if sha256_file(rows_path) != provenance["artifact_sha256"]["rows.csv"]:
            raise RuntimeError(f"Row hash drift in cache {cache_name}")
        if sha256_file(features_path) != provenance["artifact_sha256"]["features.npy"]:
            raise RuntimeError(f"Feature hash drift in cache {cache_name}")
        rows = pd.read_csv(rows_path, dtype={"person_id": str, "image_id": str})
        required = ["person_id", "image_id", "split", "label_3"]
        if any(column not in rows for column in required):
            raise RuntimeError(f"Cache {cache_name} has an incomplete row manifest")
        aligned = rows[required].astype(str).reset_index(drop=True)
        if reference_rows is None:
            reference_rows = aligned
        elif not aligned.equals(reference_rows):
            raise RuntimeError(f"Representation cache rows do not align: {cache_name}")
        cache_evidence[cache_name] = {
            "path": declaration["path"],
            "rows": len(rows),
            "feature_dimensions": int(provenance["feature_dimensions"]),
            "cache_status": provenance["status"],
            "source_sha256": {
                "provenance.json": sha256_file(provenance_path),
                "rows.csv": sha256_file(rows_path),
                "features.npy": sha256_file(features_path),
            },
        }
    if reference_rows is None:
        raise RuntimeError("No representation caches were declared")

    hyperparameters = grid["hyperparameters"]["probability_stacks"]
    candidates_per_family = (
        len(hyperparameters["component_C"])
        * len(hyperparameters["meta_C"])
        * len(hyperparameters["class_weight"])
    )
    result = {
        "status": "VCOCO_V3_REPRESENTATION_GRID_AND_CACHES_LOCKED_BEFORE_FIT",
        "grid_version": grid["grid_version"],
        "candidate_count": len(grid["families"]) * candidates_per_family,
        "development_people": len(reference_rows),
        "development_images": int(reference_rows["image_id"].nunique()),
        "official_v2_test_rows_read": 0,
        "official_v2_test_predictions_run": False,
        "human_pilot_labels_used_for_selection": False,
        "source_sha256": {
            "representation_grid": sha256_file(grid_path),
            "v3_protocol_lock": sha256_file(protocol_lock_path),
            "spatial_stage": sha256_file(spatial_path),
            "representation_evaluator_source": sha256_file(
                root / "experiments/evaluate_vcoco_v3_representations.py"
            ),
            "cuda_heads_source": sha256_file(root / "src/hac/vcoco_v3_cuda_heads.py"),
        },
        "feature_caches": cache_evidence,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
