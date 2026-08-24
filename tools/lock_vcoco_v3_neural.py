"""Resolve conditional neural eligibility and lock the exact multiview training grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=Path("experiments/vcoco_v3_neural_grid.json"))
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/protocol/vcoco_v3_lock.json"),
    )
    parser.add_argument(
        "--human-gate",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/final/summary.json"),
    )
    parser.add_argument(
        "--spatial-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/spatial/summary.json"),
    )
    parser.add_argument(
        "--spatial-promotions",
        type=Path,
        default=Path(".runs/vcoco_v3/spatial/spatial_promotions.json"),
    )
    parser.add_argument(
        "--representation-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/representations/evaluation/summary.json"),
    )
    parser.add_argument(
        "--representation-metrics",
        type=Path,
        default=Path(".runs/vcoco_v3/representations/evaluation/nested_source_tag_metrics.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runs/vcoco_v3/neural/neural_grid_lock.json"),
    )
    return parser.parse_args()


def validate_grid(grid: dict) -> None:
    if grid.get("status") != "DECLARED_BEFORE_NEURAL_FITTING":
        raise ValueError("Neural grid is not in its declared pre-fit state")
    if grid.get("development_data", {}).get("official_v2_test_used"):
        raise ValueError("The consumed V-COCO test cannot enter neural development")
    if grid.get("development_data", {}).get("human_pilot_labels_used_for_selection"):
        raise ValueError("Human pilot labels cannot select a neural candidate")
    cross_validation = grid.get("cross_validation", {})
    if min(int(cross_validation.get(name, 0)) for name in ("outer_folds", "inner_folds")) < 3:
        raise ValueError("Neural development requires at least three grouped folds")
    if len(set(cross_validation.get("screening_seeds", ()))) < 3:
        raise ValueError("Neural screening requires at least three distinct seeds")
    if len(set(cross_validation.get("outer_fit_seeds", ()))) < 5:
        raise ValueError("Outer neural estimation requires at least five distinct seeds")
    candidates = grid.get("candidate_templates", ())
    identifiers = [candidate.get("candidate_id") for candidate in candidates]
    if len(candidates) < 2 or len(identifiers) != len(set(identifiers)):
        raise ValueError("Neural candidate identifiers must be unique")
    if {float(candidate["dropout"]) for candidate in candidates} != {0.1, 0.2, 0.3}:
        raise ValueError("The declared dropout screen must remain 0.10/0.20/0.30")
    if not any(float(candidate["random_erasing_probability"]) == 0.0 for candidate in candidates):
        raise ValueError("The no-erasing augmentation control is missing")
    for candidate in candidates:
        if candidate["strategy"] == "adapter_only":
            if int(candidate["top_blocks"]) or int(candidate["lora_rank"]):
                raise ValueError("Adapter-only candidates cannot declare LoRA blocks")
        elif candidate["strategy"] == "lora_top_blocks":
            if int(candidate["top_blocks"]) < 1 or int(candidate["lora_rank"]) < 1:
                raise ValueError("LoRA candidates require a positive depth and rank")
        else:
            raise ValueError(f"Unknown neural strategy: {candidate['strategy']}")
    training = grid.get("training", {})
    if training.get("schedule") != "linear_probe_then_parameter_efficient_adaptation":
        raise ValueError("Neural training must retain the probe-then-adapt schedule")
    if int(training.get("probe_epochs", 0)) >= int(training.get("maximum_epochs", 0)):
        raise ValueError("The probe phase must be shorter than the full schedule")
    execution = grid.get("execution_backend", {})
    if not execution.get("cuda_required"):
        raise ValueError("Neural fitting must require CUDA")
    if not execution.get("automatic_mixed_precision"):
        raise ValueError("The declared neural screen requires CUDA mixed precision")


def _validate_summary(path: Path, expected_status: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != expected_status:
        raise RuntimeError(f"Required gate has not passed: {expected_status}")
    if payload.get("official_v2_test_rows_read", 0) != 0:
        raise RuntimeError("An upstream neural gate crossed the consumed-test boundary")
    return payload


def choose_backbone(grid: dict, metrics: pd.DataFrame) -> dict:
    eligible = set(grid["eligibility"]["backbone_pool"])
    candidates = metrics[metrics["family"].isin(eligible)].copy()
    if set(candidates["family"]) != eligible:
        raise RuntimeError("The matched representation screen omitted an eligible DINO family")
    candidates = candidates.sort_values(
        ["macro_f1", "locomotion_f1", "log_loss", "family"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    return {
        "model_kind": str(candidates.iloc[0]["family"]),
        "macro_f1": float(candidates.iloc[0]["macro_f1"]),
        "locomotion_f1": float(candidates.iloc[0]["locomotion_f1"]),
        "log_loss": float(candidates.iloc[0]["log_loss"]),
        "selection_rule": grid["eligibility"]["backbone_rule"],
    }


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    grid_path = args.grid.resolve()
    protocol_path = args.protocol_lock.resolve()
    human_path = args.human_gate.resolve()
    spatial_path = args.spatial_summary.resolve()
    promotions_path = args.spatial_promotions.resolve()
    representation_path = args.representation_summary.resolve()
    metrics_path = args.representation_metrics.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    validate_grid(grid)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "VCOCO_V3_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("The v3 protocol lock is invalid")
    _validate_summary(human_path, grid["required_gates"]["human_annotation"])
    spatial = _validate_summary(spatial_path, grid["required_gates"]["spatial"])
    representation = _validate_summary(
        representation_path, grid["required_gates"]["representations"]
    )
    if sha256_file(promotions_path) != spatial["artifact_sha256"][promotions_path.name]:
        raise RuntimeError("Spatial promotion evidence changed after completion")
    if sha256_file(metrics_path) != representation["artifact_sha256"][metrics_path.name]:
        raise RuntimeError("Representation metrics changed after completion")
    promotions = json.loads(promotions_path.read_text(encoding="utf-8"))
    promoted = sorted(
        family
        for family, decision in promotions.items()
        if decision.get("general_candidate") or decision.get("locomotion_specialist")
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if grid["eligibility"]["spatial_promotion_required"] and not promoted:
        result = {
            "status": "VCOCO_V3_NEURAL_STAGE_NOT_ELIGIBLE",
            "reason": "No declared spatial multiview family passed the promotion gate",
            "official_v2_test_rows_read": 0,
            "source_sha256": {
                "neural_grid": sha256_file(grid_path),
                "protocol_lock": sha256_file(protocol_path),
                "human_gate": sha256_file(human_path),
                "spatial_summary": sha256_file(spatial_path),
                "spatial_promotions": sha256_file(promotions_path),
                "representation_summary": sha256_file(representation_path),
                "representation_metrics": sha256_file(metrics_path),
            },
        }
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return

    metrics = pd.read_csv(metrics_path)
    backbone = choose_backbone(grid, metrics)
    cross_validation = grid["cross_validation"]
    candidates = [
        {**candidate, "model_kind": backbone["model_kind"]}
        for candidate in grid["candidate_templates"]
    ]
    result = {
        "status": "VCOCO_V3_NEURAL_GRID_LOCKED_BEFORE_FIT",
        "grid_version": grid["grid_version"],
        "selected_backbone": backbone,
        "promoted_spatial_families": promoted,
        "candidates": candidates,
        "inner_screen_run_count": (
            int(cross_validation["outer_folds"])
            * int(cross_validation["inner_folds"])
            * len(cross_validation["screening_seeds"])
            * len(candidates)
        ),
        "outer_fit_run_count": (
            int(cross_validation["outer_folds"]) * len(cross_validation["outer_fit_seeds"])
        ),
        "human_pilot_labels_used_for_selection": False,
        "official_v2_test_rows_read": 0,
        "official_v2_test_predictions_run": False,
        "source_sha256": {
            "neural_grid": sha256_file(grid_path),
            "protocol_lock": sha256_file(protocol_path),
            "human_gate": sha256_file(human_path),
            "spatial_summary": sha256_file(spatial_path),
            "spatial_promotions": sha256_file(promotions_path),
            "representation_summary": sha256_file(representation_path),
            "representation_metrics": sha256_file(metrics_path),
            "neural_runner_source": sha256_file(
                root / "experiments/train_vcoco_v3_neural.py"
            ),
            "neural_queue_source": sha256_file(
                root / "experiments/run_vcoco_v3_neural_queue.py"
            ),
            "neural_module_source": sha256_file(root / "src/hac/vcoco_v3_neural.py"),
        },
    }
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
