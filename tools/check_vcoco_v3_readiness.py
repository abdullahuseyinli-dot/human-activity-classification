"""Audit V-COCO v3 execution gates without fitting models or opening holdouts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch
from huggingface_hub import get_hf_file_metadata, hf_hub_url, try_to_load_from_cache

from hac.polar import sha256_file

DINOV3_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINOV3_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--probe-dinov3", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runs/vcoco_v3/readiness/summary.json"),
    )
    return parser.parse_args()


def read_status(
    path: Path,
    expected: set[str],
    source_paths: dict[str, Path] | None = None,
) -> dict:
    if not path.is_file():
        return {"exists": False, "status": None, "accepted": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return {
            "exists": True,
            "status": "INVALID_JSON",
            "accepted": False,
            "error": str(error),
        }
    status = str(payload.get("status", "MISSING_STATUS"))
    result = {
        "exists": True,
        "status": status,
        "accepted": status in expected,
        "sha256": sha256_file(path),
    }
    if source_paths:
        stored_hashes = payload.get("source_sha256", {})
        entries = {}
        for name, source_path in source_paths.items():
            current_hash = sha256_file(source_path) if source_path.is_file() else None
            expected_hash = stored_hashes.get(name)
            entries[name] = {
                "expected_sha256": expected_hash,
                "current_sha256": current_hash,
                "matches": bool(expected_hash and current_hash == expected_hash),
            }
        integrity_passed = all(value["matches"] for value in entries.values())
        result["source_integrity"] = {
            "passed": integrity_passed,
            "entries": entries,
        }
        result["accepted"] = bool(result["accepted"] and integrity_passed)
    return result


def effective_protocol_status(root: Path, base_path: Path) -> dict:
    """Validate the immutable base lock and its explicit external/CUDA amendment."""

    base = read_status(
        base_path,
        {"VCOCO_V3_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING"},
        {
            "protocol_spec": root / "experiments/vcoco_v3_protocol.json",
            "protocol_locker": root / "tools/lock_vcoco_v3_protocol.py",
            "v2_official_test_summary": root / "results/vcoco_v2/official_test_summary.json",
            "v2_final_selection_lock": root / "results/vcoco_v2/final_selection_lock.json",
        },
    )
    amendment_path = root / ".runs/vcoco_v3/protocol/external_cuda_amendment_lock.json"
    amendment = read_status(
        amendment_path,
        {"VCOCO_V3_EXTERNAL_CUDA_AMENDMENT_LOCKED_BEFORE_TARGET_FITTING"},
        {
            "amendment_spec": root / "experiments/vcoco_v3_external_cuda_amendment.json",
            "amendment_document": root / "docs/VCOCO_V3_EXTERNAL_CUDA_AMENDMENT.md",
            "amended_protocol_document": root / "docs/VCOCO_V3_RESEARCH_PROTOCOL.md",
            "amendment_locker": root / "tools/lock_vcoco_v3_external_cuda_amendment.py",
            "base_protocol_lock": base_path,
            "base_protocol_spec": root / "experiments/vcoco_v3_protocol.json",
            "human_pilot_audit": root / ".runs/vcoco_v3/annotation/final/summary.json",
            "okutama_protocol": root / "experiments/okutama_action_protocol.json",
            "okutama_temporal_grid": root / "experiments/okutama_temporal_grid.json",
            "okutama_development_audit": root
            / ".runs/vcoco_v3/okutama/development_audit/summary.json",
            "spatial_grid": root / "experiments/vcoco_v3_spatial_grid.json",
            "representation_grid": root / "experiments/vcoco_v3_representation_grid.json",
            "neural_grid": root / "experiments/vcoco_v3_neural_grid.json",
            "okutama_cache_source": root / "experiments/cache_okutama_temporal_features.py",
            "source_transfer_source": root
            / "experiments/fit_okutama_source_only_transfer.py",
            "fewshot_transfer_source": root
            / "experiments/evaluate_okutama_fewshot_transfer.py",
            "temporal_locker_source": root / "tools/lock_vcoco_v3_temporal.py",
            "temporal_calibration_source": root
            / "tools/calibrate_vcoco_v3_temporal_pipeline.py",
            "confirmation_evaluator_source": root
            / "experiments/evaluate_vcoco_v3_temporal_confirmation.py",
        },
    )
    base_payload = json.loads(base_path.read_text(encoding="utf-8"))
    amendment_payload = (
        json.loads(amendment_path.read_text(encoding="utf-8"))
        if amendment_path.is_file()
        else {}
    )
    original_document_sha256 = base_payload.get("source_sha256", {}).get(
        "protocol_document"
    )
    current_document_path = root / "docs/VCOCO_V3_RESEARCH_PROTOCOL.md"
    current_document_sha256 = (
        sha256_file(current_document_path) if current_document_path.is_file() else None
    )
    document_transition = {
        "base_sha256": original_document_sha256,
        "current_sha256": current_document_sha256,
        "unchanged": current_document_sha256 == original_document_sha256,
        "superseded_by_amendment": bool(
            amendment.get("accepted")
            and amendment_payload.get("base_protocol_document_sha256")
            == original_document_sha256
            and amendment_payload.get("source_sha256", {}).get(
                "amended_protocol_document"
            )
            == current_document_sha256
        ),
    }
    accepted = bool(
        base.get("accepted")
        and amendment.get("accepted")
        and (document_transition["unchanged"] or document_transition["superseded_by_amendment"])
    )
    return {
        "exists": base.get("exists", False) and amendment.get("exists", False),
        "status": (
            "VCOCO_V3_EFFECTIVE_PROTOCOL_WITH_EXTERNAL_CUDA_AMENDMENT"
            if accepted
            else "VCOCO_V3_PROTOCOL_OR_AMENDMENT_INVALID"
        ),
        "accepted": accepted,
        "base_lock": base,
        "external_cuda_amendment": amendment,
        "protocol_document_transition": document_transition,
    }


def annotation_progress(root: Path, primary_tasks: int = 130) -> dict:
    manifest_path = root / ".runs/vcoco_v3/annotation/pilot/blind_tasks.csv"
    annotation_dir = root / ".runs/vcoco_v3/annotation/pilot/annotations"
    expected = 0
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as source:
            expected = max(0, sum(1 for _ in source) - 1)
    raters = []
    for path in sorted(annotation_dir.glob("*.json")) if annotation_dir.is_dir() else ():
        payload = json.loads(path.read_text(encoding="utf-8"))
        completed = int(payload.get("completed_rows", 0))
        raters.append(
            {
                "annotator_id": str(payload.get("annotator_id", path.stem)),
                "completed": completed,
                "expected": expected,
                "primary_complete": completed >= primary_tasks,
                "full_pass_complete": bool(expected and completed == expected),
            }
        )
    return {
        "task_manifest_exists": manifest_path.is_file(),
        "tasks_per_rater": expected,
        "raters": raters,
        "primary_tasks": primary_tasks,
        "primary_complete_raters": sum(bool(row["primary_complete"]) for row in raters),
        "full_pass_complete_raters": sum(bool(row["full_pass_complete"]) for row in raters),
        "server_url": "http://127.0.0.1:8765",
    }


def dino_readiness(probe_remote: bool) -> dict:
    config = try_to_load_from_cache(DINOV3_REPO, "config.json", revision=DINOV3_REVISION)
    weights = try_to_load_from_cache(DINOV3_REPO, "model.safetensors", revision=DINOV3_REVISION)
    result = {
        "model_id": DINOV3_REPO,
        "revision": DINOV3_REVISION,
        "config_cached": isinstance(config, str) and Path(config).is_file(),
        "weights_cached": isinstance(weights, str) and Path(weights).is_file(),
        "remote_authorization_probe": "not_requested",
    }
    if result["config_cached"]:
        result["config_sha256"] = sha256_file(Path(str(config)))
    if result["weights_cached"]:
        result["weights_sha256"] = sha256_file(Path(str(weights)))
    if probe_remote:
        metadata = get_hf_file_metadata(
            hf_hub_url(DINOV3_REPO, "model.safetensors", revision=DINOV3_REVISION),
            token=True,
        )
        result["remote_authorization_probe"] = "authorized"
        result["remote_checkpoint"] = {
            "commit_hash": metadata.commit_hash,
            "etag": metadata.etag,
            "size_bytes": metadata.size,
        }
    return result


def git_state(root: Path) -> dict:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "worktree_changes": len(run("status", "--porcelain").splitlines()),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    protocol_lock_path = root / ".runs/vcoco_v3/protocol/vcoco_v3_lock.json"
    candidate_lock_path = root / ".runs/vcoco_v3/candidates/candidate_grid_lock.json"
    v2_development_lock_path = (
        root / ".runs/polar_v2/locked_protocol/vcoco_v2_protocol_lock.json"
    )
    stages = {
        "protocol": effective_protocol_status(root, protocol_lock_path),
        "candidate_grid": read_status(
            candidate_lock_path,
            {"VCOCO_V3_CANDIDATE_GRID_AND_CACHES_LOCKED_BEFORE_FIT"},
            {
                "candidate_grid": root / "experiments/vcoco_v3_candidate_grid.json",
                "v3_protocol_lock": protocol_lock_path,
                "v2_development_lock": v2_development_lock_path,
                "models_source": root / "src/hac/vcoco_v3_models.py",
                "nested_evaluator_source": root
                / "experiments/evaluate_vcoco_v3_nested_stacks.py",
            },
        ),
        "human_pilot": read_status(
            root / ".runs/vcoco_v3/annotation/final/summary.json",
            {"VCOCO_V3_HUMAN_PILOT_AUDIT_COMPLETE"},
        ),
        "nested_stacks": read_status(
            root / ".runs/vcoco_v3/nested_stacks/summary.json",
            {"VCOCO_V3_NESTED_CACHED_FUSION_DEVELOPMENT_COMPLETE"},
        ),
        "spatial": read_status(
            root / ".runs/vcoco_v3/spatial/summary.json",
            {"VCOCO_V3_SPATIAL_DEVELOPMENT_COMPLETE"},
        ),
        "representations": read_status(
            root / ".runs/vcoco_v3/representations/evaluation/summary.json",
            {"VCOCO_V3_MATCHED_REPRESENTATION_DEVELOPMENT_COMPLETE"},
        ),
        "okutama_development_audit": read_status(
            root / ".runs/vcoco_v3/okutama/development_audit/summary.json",
            {"OKUTAMA_DEVELOPMENT_ARCHIVE_AND_CENTRES_AUDITED"},
        ),
        "okutama_source_only": read_status(
            root / ".runs/vcoco_v3/okutama/source_only/evaluation/summary.json",
            {"OKUTAMA_SOURCE_ONLY_TRANSFER_EVALUATED"},
        ),
        "okutama_fewshot": read_status(
            root / ".runs/vcoco_v3/okutama/fewshot/summary.json",
            {"OKUTAMA_FEWSHOT_TRANSFER_DEVELOPMENT_COMPLETE"},
        ),
        "neural_grid": read_status(
            root / ".runs/vcoco_v3/neural/neural_grid_lock.json",
            {
                "VCOCO_V3_NEURAL_GRID_LOCKED_BEFORE_FIT",
                "VCOCO_V3_NEURAL_STAGE_NOT_ELIGIBLE",
            },
        ),
        "temporal_manifest": read_status(
            root / ".runs/vcoco_v3/temporal/temporal_manifest_lock.json",
            {"VCOCO_V3_TEMPORAL_MANIFEST_LOCKED"},
        ),
        "temporal_pipeline": read_status(
            root / ".runs/vcoco_v3/temporal/pipeline_lock/summary.json",
            {"VCOCO_V3_TEMPORAL_PIPELINE_LOCKED_BEFORE_CONFIRMATION"},
        ),
        "okutama_confirmation_audit": read_status(
            root / ".runs/vcoco_v3/okutama/confirmation_audit/summary.json",
            {"OKUTAMA_CONFIRMATION_ARCHIVE_AND_CENTRES_AUDITED"},
        ),
        "temporal_confirmation": read_status(
            root / ".runs/vcoco_v3/temporal/confirmation/summary.json",
            {"VCOCO_V3_TEMPORAL_CONFIRMATION_COMPLETE"},
        ),
    }
    ordered = [
        "protocol",
        "candidate_grid",
        "human_pilot",
        "nested_stacks",
        "spatial",
        "representations",
        "okutama_development_audit",
        "okutama_source_only",
        "okutama_fewshot",
        "neural_grid",
        "temporal_manifest",
        "temporal_pipeline",
        "okutama_confirmation_audit",
        "temporal_confirmation",
    ]
    next_gate = next((name for name in ordered if not stages[name]["accepted"]), "complete")
    disk = shutil.disk_usage(root)
    result = {
        "status": "VCOCO_V3_READINESS_AUDIT_COMPLETE",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "git": git_state(root),
        "stages": stages,
        "annotation": annotation_progress(root),
        "dinov3": dino_readiness(args.probe_dinov3),
        "runtime": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "storage": {
            "free_gib": round(disk.free / 2**30, 2),
            "total_gib": round(disk.total / 2**30, 2),
            "okutama_temporal_workspace_fits": disk.free >= 5 * 2**30,
        },
        "next_gate": next_gate,
        "model_fitting_permitted": bool(
            stages["human_pilot"]["accepted"]
            and stages["human_pilot"].get("status")
            == "VCOCO_V3_HUMAN_PILOT_AUDIT_COMPLETE"
        ),
        "confirmation_open_permitted": stages["temporal_pipeline"]["accepted"],
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
