"""Lock the V-COCO v3 nested-CV grid and reusable development feature caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file

EXPECTED_FAMILIES = {
    "dino_flat_probability_stack",
    "dino_factorized_probability_stack",
    "dino_siglip_factorized_reliability_stack",
    "dino_siglip_linear_svm_control",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", type=Path, default=Path("experiments/vcoco_v3_candidate_grid.json")
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/protocol/vcoco_v3_lock.json"),
    )
    parser.add_argument(
        "--v2-lock",
        type=Path,
        default=Path(".runs/polar_v2/locked_protocol/vcoco_v2_protocol_lock.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runs/vcoco_v3/candidates/candidate_grid_lock.json"),
    )
    return parser.parse_args()


def validate_grid(grid: dict) -> None:
    if grid.get("status") != "DECLARED_BEFORE_CANDIDATE_FITTING":
        raise ValueError("Candidate grid is not in its pre-fit state")
    cross_validation = grid.get("cross_validation", {})
    for key in ("outer_folds", "inner_folds", "stack_folds"):
        if int(cross_validation.get(key, 0)) < 3:
            raise ValueError(f"{key} must be at least three")
    if cross_validation.get("splitter") != "StratifiedGroupKFold":
        raise ValueError("Candidate selection must remain image-grouped and stratified")
    if set(grid.get("families", {})) != EXPECTED_FAMILIES:
        raise ValueError("The declared candidate families changed")
    svm = grid["families"]["dino_siglip_linear_svm_control"]
    if svm.get("solver") != "pytorch_cuda_lbfgs_ovr_squared_hinge":
        raise ValueError("The linear-SVM CUDA solver changed")
    if not svm.get("cuda_required"):
        raise ValueError("The linear-SVM control must require CUDA")
    if grid.get("development_data", {}).get("official_v2_test_used"):
        raise ValueError("The consumed v2 test cannot enter v3 development")
    selection = grid.get("selection", {})
    if selection.get("primary") != "source_tag_macro_f1":
        raise ValueError("The development endpoint must be named explicitly")
    if selection.get("human_harmonized_labels_used_for_selection"):
        raise ValueError("The human pilot cannot select development candidates")


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    grid_path = args.grid.resolve()
    protocol_lock_path = args.protocol_lock.resolve()
    v2_lock_path = args.v2_lock.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    validate_grid(grid)
    protocol_lock = json.loads(protocol_lock_path.read_text(encoding="utf-8"))
    if protocol_lock.get("status") != "VCOCO_V3_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("The v3 research protocol is not locked")
    v2_lock = json.loads(v2_lock_path.read_text(encoding="utf-8"))
    if v2_lock.get("protocol_version") != "2.0.0":
        raise RuntimeError("The referenced v2 development lock is invalid")
    v2_lock_hash = sha256_file(v2_lock_path)

    cache_evidence = {}
    reference_ids = None
    reference_labels = None
    for name, declaration in grid["feature_caches"].items():
        cache_dir = (root / declaration["path"]).resolve()
        provenance_path = cache_dir / "provenance.json"
        rows_path = cache_dir / "rows.csv"
        features_path = cache_dir / "features.npy"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("status") != "VCOCO_V2_DEVELOPMENT_FEATURE_CACHE_COMPLETE":
            raise RuntimeError(f"Incomplete development cache: {name}")
        if provenance.get("protocol_lock_sha256") != v2_lock_hash:
            raise RuntimeError(f"Development lock drift in cache: {name}")
        for field in ("model_kind", "view", "image_size"):
            if provenance.get(field) != declaration[field]:
                raise RuntimeError(f"Declared {field} does not match cache {name}")
        if provenance.get("test_rows_read") != 0 or provenance.get("test_predictions_run"):
            raise RuntimeError(f"Cache {name} crossed the test boundary")
        if sha256_file(rows_path) != provenance["artifact_sha256"]["rows.csv"]:
            raise RuntimeError(f"Row hash drift in cache: {name}")
        if sha256_file(features_path) != provenance["artifact_sha256"]["features.npy"]:
            raise RuntimeError(f"Feature hash drift in cache: {name}")
        rows = pd.read_csv(rows_path, dtype={"person_id": str, "image_id": str})
        ids = tuple(rows["person_id"].astype(str))
        labels = tuple(rows["label_3"].astype(str))
        if reference_ids is None:
            reference_ids, reference_labels = ids, labels
        elif ids != reference_ids or labels != reference_labels:
            raise RuntimeError(f"Cache rows do not align: {name}")
        if set(rows["split"].astype(str)) != {"train", "val"}:
            raise RuntimeError(f"Cache {name} contains an unexpected split")
        cache_evidence[name] = {
            "path": declaration["path"],
            "rows": len(rows),
            "people": int(rows["person_id"].nunique()),
            "images": int(rows["image_id"].nunique()),
            "feature_dimensions": int(provenance["feature_dimensions"]),
            "source_sha256": {
                "provenance.json": sha256_file(provenance_path),
                "rows.csv": sha256_file(rows_path),
                "features.npy": sha256_file(features_path),
            },
        }

    stack_grid = grid["hyperparameters"]["probability_stacks"]
    svm_grid = grid["hyperparameters"]["linear_svm_control"]
    stack_candidates_per_family = (
        len(stack_grid["component_C"]) * len(stack_grid["meta_C"]) * len(stack_grid["class_weight"])
    )
    candidate_count = 3 * stack_candidates_per_family + len(svm_grid["C"]) * len(
        svm_grid["class_weight"]
    )
    output = {
        "status": "VCOCO_V3_CANDIDATE_GRID_AND_CACHES_LOCKED_BEFORE_FIT",
        "grid_version": grid["grid_version"],
        "candidate_count": candidate_count,
        "development_people": len(reference_ids or ()),
        "official_v2_test_rows_read": 0,
        "official_v2_test_predictions_run": False,
        "human_pilot_labels_used_for_selection": False,
        "source_sha256": {
            "candidate_grid": sha256_file(grid_path),
            "v3_protocol_lock": sha256_file(protocol_lock_path),
            "v2_development_lock": v2_lock_hash,
            "models_source": sha256_file(root / "src/hac/vcoco_v3_models.py"),
            "nested_evaluator_source": sha256_file(
                root / "experiments/evaluate_vcoco_v3_nested_stacks.py"
            ),
        },
        "feature_caches": cache_evidence,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
