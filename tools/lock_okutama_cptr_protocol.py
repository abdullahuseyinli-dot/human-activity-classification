"""Lock the CPTR development protocol before any new candidate is fitted."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file
from hac.vcoco_v3_temporal import validate_temporal_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/okutama_cptr_protocol.json"),
    )
    parser.add_argument(
        "--v3-audit",
        type=Path,
        default=Path(".runs/vcoco_v3/okutama/development_audit/summary.json"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help=(
            "Path to the provider-supplied Okutama-Action training-frame archive; "
            "overrides the portable location recorded in the protocol"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runs/cptr/protocol_lock.json"),
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_declared(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    repository = protocol_path.parents[1]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "DECLARED_BEFORE_CPTR_DEVELOPMENT_FITTING":
        raise RuntimeError("The CPTR protocol is not in its pre-fit state")
    expected_order = [
        "baseline_replay",
        "temporal_faithfulness_diagnostics",
        "compact_raw_kinematics",
        "camera_compensated_kinematics",
        "centre_conditioned_short_residual",
        "dual_clock_residual",
        "confidence_masked_part_articulation",
        "motion_null_and_invariance_training",
        "cross_fitted_continuous_utility_router",
        "target_video_masked_adaptation",
        "top_block_lora_and_centre_siglip_specialists",
        "passing_component_integration",
        "five_seed_and_grouped_crossfit_validation",
        "development_lock",
    ]
    if protocol.get("execution_order") != expected_order:
        raise RuntimeError("The CPTR execution order changed")
    if protocol["final_evaluation"].get("existing_okutama_confirmation_must_not_open") is not True:
        raise RuntimeError("The consumed confirmation partition must remain unavailable")
    if protocol["training"].get("device") != "cuda" or protocol["training"].get(
        "cpu_fallback"
    ) is not False:
        raise RuntimeError("CPTR fitting must remain CUDA-only")

    input_spec = protocol["input"]
    manifest_path = resolve_declared(repository, input_spec["development_manifest"])
    centres_path = resolve_declared(repository, input_spec["development_centres"])
    base_store_path = resolve_declared(repository, input_spec["base_store"])
    archive_path = (
        args.archive.resolve()
        if args.archive is not None
        else resolve_declared(repository, input_spec["raw_development_archive"])
    )
    for path in (manifest_path, centres_path, base_store_path, archive_path, args.v3_audit.resolve()):
        if not path.is_file():
            raise FileNotFoundError(path)

    frame = validate_temporal_manifest(
        pd.read_csv(
            manifest_path,
            dtype={"sample_id": str, "recording_id": str, "track_id": str},
        )
    )
    if "confirmation" in set(frame["split"].astype(str)):
        raise RuntimeError("The CPTR development manifest unexpectedly contains confirmation rows")
    if set(frame["split"].astype(str)) != {"train", "validation", "calibration"}:
        raise RuntimeError("The CPTR development split set changed")
    centres = pd.read_csv(centres_path, dtype={"sample_id": str})
    if len(centres) != len(frame):
        raise RuntimeError("The provider centres and temporal manifest differ in length")
    if not centres["sample_id"].astype(str).equals(frame["sample_id"].astype(str)):
        raise RuntimeError("The provider centres and temporal manifest row orders differ")
    base_store = json.loads(base_store_path.read_text(encoding="utf-8"))
    if base_store.get("status") != "VCOCO_V3_PACKED_TEMPORAL_FEATURE_STORE_COMPLETE":
        raise RuntimeError("The frozen DINOv2 feature store is incomplete")
    if base_store.get("confirmation_archive_opened") is not False:
        raise RuntimeError("The development feature store reports confirmation access")
    v3_audit = json.loads(args.v3_audit.resolve().read_text(encoding="utf-8"))
    if v3_audit.get("confirmation_archive_opened") is not False:
        raise RuntimeError("The provider development audit reports confirmation access")
    archive_evidence = v3_audit["development_archive"]
    if archive_path.stat().st_size != int(archive_evidence["bytes"]):
        raise RuntimeError("The declared development archive byte count changed")

    split_counts = (
        frame.groupby(["split", "label"], observed=True)
        .size()
        .rename("samples")
        .reset_index()
        .to_dict(orient="records")
    )
    payload = {
        "status": "OKUTAMA_CPTR_PROTOCOL_LOCKED_BEFORE_FIT",
        "protocol_version": protocol["protocol_version"],
        "study_name": protocol["study_name"],
        "execution_order": expected_order,
        "development_samples": int(len(frame)),
        "development_scenarios": int(frame["recording_id"].nunique()),
        "split_counts": split_counts,
        "calibration_model_outcomes_read": 0,
        "confirmation_samples_read": 0,
        "confirmation_model_outcomes_read": 0,
        "archive_evidence": {
            "bytes": int(archive_evidence["bytes"]),
            "sha256": str(archive_evidence["sha256"]),
        },
        "source_sha256": {
            "protocol": sha256_file(protocol_path),
            "development_manifest": sha256_file(manifest_path),
            "development_centres": sha256_file(centres_path),
            "base_store": sha256_file(base_store_path),
            "v3_development_audit": sha256_file(args.v3_audit.resolve()),
            "locker": sha256_file(Path(__file__).resolve()),
        },
    }
    output = args.output.resolve()
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous != payload:
            raise RuntimeError("An incompatible CPTR protocol lock already exists")
    else:
        write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
