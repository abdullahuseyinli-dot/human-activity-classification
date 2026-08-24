"""Lock the Okutama substitution and CUDA-only execution amendment."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

from hac.polar import sha256_file

STATUS = "VCOCO_V3_EXTERNAL_CUDA_AMENDMENT_LOCKED_BEFORE_TARGET_FITTING"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--amendment",
        type=Path,
        default=Path("experiments/vcoco_v3_external_cuda_amendment.json"),
    )
    parser.add_argument(
        "--base-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/protocol/vcoco_v3_lock.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runs/vcoco_v3/protocol/external_cuda_amendment_lock.json"),
    )
    return parser.parse_args()


def git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    amendment_path = args.amendment.resolve()
    base_path = args.base_lock.resolve()
    output_path = args.output.resolve()
    amendment = load_json(amendment_path)
    base = load_json(base_path)
    if amendment.get("status") != "DECLARED_BEFORE_OKUTAMA_TARGET_FITTING_AND_CONFIRMATION_OPEN":
        raise RuntimeError("The external amendment is not in its declared pre-fit state")
    base_sha256 = sha256_file(base_path)
    if base_sha256 != amendment.get("base_protocol_lock_sha256"):
        raise RuntimeError("The external amendment references a different base protocol lock")
    if base.get("status") != "VCOCO_V3_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("The base V-COCO v3 protocol lock is invalid")

    protocol_path = root / "experiments/vcoco_v3_protocol.json"
    protocol_document_path = root / "docs/VCOCO_V3_RESEARCH_PROTOCOL.md"
    amendment_document_path = root / "docs/VCOCO_V3_EXTERNAL_CUDA_AMENDMENT.md"
    okutama_protocol_path = root / "experiments/okutama_action_protocol.json"
    temporal_grid_path = root / "experiments/okutama_temporal_grid.json"
    spatial_grid_path = root / "experiments/vcoco_v3_spatial_grid.json"
    representation_grid_path = root / "experiments/vcoco_v3_representation_grid.json"
    neural_grid_path = root / "experiments/vcoco_v3_neural_grid.json"
    okutama_cache_path = root / "experiments/cache_okutama_temporal_features.py"
    source_transfer_path = root / "experiments/fit_okutama_source_only_transfer.py"
    fewshot_path = root / "experiments/evaluate_okutama_fewshot_transfer.py"
    temporal_locker_path = root / "tools/lock_vcoco_v3_temporal.py"
    temporal_calibration_path = root / "tools/calibrate_vcoco_v3_temporal_pipeline.py"
    confirmation_evaluator_path = (
        root / "experiments/evaluate_vcoco_v3_temporal_confirmation.py"
    )
    human_path = root / ".runs/vcoco_v3/annotation/final/summary.json"
    development_audit_path = (
        root / ".runs/vcoco_v3/okutama/development_audit/summary.json"
    )
    if sha256_file(protocol_path) != base["source_sha256"]["protocol_spec"]:
        raise RuntimeError("The immutable base protocol specification changed")

    human = load_json(human_path)
    if human.get("status") != "VCOCO_V3_HUMAN_PILOT_AUDIT_COMPLETE":
        raise RuntimeError("The fixed human pilot has not passed its descriptive audit")
    if human.get("primary_task_presentations") != 130:
        raise RuntimeError("The external amendment requires the fixed 130-presentation pilot")
    if human.get("human_pilot_labels_used_for_candidate_selection"):
        raise RuntimeError("Human pilot labels entered source model selection")

    okutama_protocol = load_json(okutama_protocol_path)
    if (
        okutama_protocol.get("status")
        != "DECLARED_BEFORE_OKUTAMA_AGGREGATE_LABEL_AUDIT_OR_MODEL_FITTING"
    ):
        raise RuntimeError("The Okutama ontology protocol is not in its declared state")
    backend = okutama_protocol.get("execution_backend", {})
    if backend.get("model_training") != "cuda" or backend.get("cpu_fallback_permitted"):
        raise RuntimeError("The Okutama protocol must require CUDA without CPU fallback")

    temporal_grid = load_json(temporal_grid_path)
    if temporal_grid.get("status") != "DECLARED_BEFORE_TEMPORAL_FITTING":
        raise RuntimeError("The Okutama temporal grid is not in its pre-fit state")
    if temporal_grid["training"].get("execution_backend") != "cuda":
        raise RuntimeError("Temporal fitting is not declared as CUDA")
    if temporal_grid["training"].get("cpu_fallback_permitted") is not False:
        raise RuntimeError("Temporal CPU fallback must remain disabled")
    if not temporal_grid["dataset"].get("confirmation_archive_sealed_until_pipeline_lock"):
        raise RuntimeError("The provider test is not declared sealed")

    for name, path in (
        ("spatial", spatial_grid_path),
        ("representation", representation_grid_path),
        ("neural", neural_grid_path),
    ):
        grid = load_json(path)
        if not grid.get("execution_backend", {}).get("cuda_required"):
            raise RuntimeError(f"The {name} grid does not require CUDA")

    development = load_json(development_audit_path)
    if development.get("status") != "OKUTAMA_DEVELOPMENT_ARCHIVE_AND_CENTRES_AUDITED":
        raise RuntimeError("The Okutama development audit is incomplete")
    if development.get("confirmation_archive_opened"):
        raise RuntimeError("The provider test was opened before the external amendment lock")

    premature = [
        root / ".runs/vcoco_v3/okutama/source_only/summary.json",
        root / ".runs/vcoco_v3/okutama/fewshot/summary.json",
        root / ".runs/vcoco_v3/temporal/temporal_grid_lock.json",
        root / ".runs/vcoco_v3/okutama/confirmation_audit/summary.json",
    ]
    existing = [path for path in premature if path.exists()]
    if existing:
        raise RuntimeError(
            "The external amendment must precede target fitting and confirmation: "
            + ", ".join(str(path) for path in existing)
        )
    if not torch.cuda.is_available():
        raise RuntimeError("The external amendment requires an available CUDA runtime")

    payload = {
        "status": STATUS,
        "amendment_version": amendment["amendment_version"],
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "repository_revision_at_lock": git_output(root, "rev-parse", "HEAD"),
        "base_protocol_lock_sha256": base_sha256,
        "base_protocol_document_sha256": base["source_sha256"]["protocol_document"],
        "base_protocol_document_superseded": True,
        "okutama_target_models_fitted_before_lock": 0,
        "confirmation_archive_opened_before_lock": False,
        "cuda": {
            "available": True,
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cpu_fallback_permitted": False,
        },
        "source_sha256": {
            "amendment_spec": sha256_file(amendment_path),
            "amendment_document": sha256_file(amendment_document_path),
            "amended_protocol_document": sha256_file(protocol_document_path),
            "amendment_locker": sha256_file(Path(__file__).resolve()),
            "base_protocol_lock": base_sha256,
            "base_protocol_spec": sha256_file(protocol_path),
            "human_pilot_audit": sha256_file(human_path),
            "okutama_protocol": sha256_file(okutama_protocol_path),
            "okutama_temporal_grid": sha256_file(temporal_grid_path),
            "okutama_development_audit": sha256_file(development_audit_path),
            "spatial_grid": sha256_file(spatial_grid_path),
            "representation_grid": sha256_file(representation_grid_path),
            "neural_grid": sha256_file(neural_grid_path),
            "okutama_cache_source": sha256_file(okutama_cache_path),
            "source_transfer_source": sha256_file(source_transfer_path),
            "fewshot_transfer_source": sha256_file(fewshot_path),
            "temporal_locker_source": sha256_file(temporal_locker_path),
            "temporal_calibration_source": sha256_file(temporal_calibration_path),
            "confirmation_evaluator_source": sha256_file(confirmation_evaluator_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
