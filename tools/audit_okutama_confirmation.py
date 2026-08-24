"""Open and audit the Okutama provider-test archive after the pipeline is locked."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from hac.okutama import select_temporal_centres
from hac.polar import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiments/okutama_action_protocol.json")
    )
    parser.add_argument("--pipeline-lock", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".runs/vcoco_v3/okutama/confirmation_audit")
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


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    archive_path = args.archive.resolve()
    protocol_path = args.protocol.resolve()
    pipeline_path = args.pipeline_lock.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("status") == "OKUTAMA_CONFIRMATION_ARCHIVE_AND_CENTRES_AUDITED":
            print(json.dumps(previous, indent=2, sort_keys=True), flush=True)
            return

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != (
        "DECLARED_BEFORE_OKUTAMA_AGGREGATE_LABEL_AUDIT_OR_MODEL_FITTING"
    ):
        raise RuntimeError("The Okutama protocol is not the declared version")
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    if pipeline.get("status") != "VCOCO_V3_TEMPORAL_PIPELINE_LOCKED_BEFORE_CONFIRMATION":
        raise RuntimeError("The temporal pipeline is not locked for confirmation")
    if pipeline.get("confirmation_feature_arrays_opened") != 0:
        raise RuntimeError("Confirmation features were opened before the pipeline lock")
    if pipeline.get("confirmation_evaluations_run") != 0:
        raise RuntimeError("A confirmation evaluation was already recorded")
    declaration = protocol["archives"]["confirmation"]
    if archive_path.name != declaration["file_name"]:
        raise RuntimeError("The confirmation archive name differs from the declaration")
    if archive_path.stat().st_size != int(declaration["expected_bytes"]):
        raise RuntimeError("The confirmation archive byte count differs from the declaration")

    pipeline_hash = sha256_file(pipeline_path)
    ledger_path = output_dir / "confirmation_open.json"
    opened_at = datetime.now(UTC).isoformat()
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("pipeline_lock_sha256") != pipeline_hash:
            raise RuntimeError("The confirmation archive was opened for a different pipeline")
        if ledger.get("status") == "COMPLETE":
            raise RuntimeError("The confirmation ledger is complete but its summary is missing")
        open_attempt = int(ledger.get("open_attempt", 1)) + 1
    else:
        open_attempt = 1
    write_json(
        ledger_path,
        {
            "status": "STARTED",
            "opened_at_utc": opened_at,
            "open_attempt": open_attempt,
            "conceptual_confirmation_open_number": 1,
            "pipeline_lock_sha256": pipeline_hash,
            "declared_archive_sha256": declaration["sha256"],
        },
    )

    try:
        archive_hash = sha256_file(archive_path)
        if archive_hash != declaration["sha256"]:
            raise RuntimeError("The confirmation archive hash differs from the declaration")
        centres, evidence = select_temporal_centres(
            archive_path,
            provider_partition="test",
        )
        centres_path = output_dir / "confirmation_centres.csv"
        centres.to_csv(centres_path, index=False)
        summary = {
            "status": "OKUTAMA_CONFIRMATION_ARCHIVE_AND_CENTRES_AUDITED",
            "confirmation_archive_opened": True,
            "confirmation_open_number": 1,
            "open_attempt": open_attempt,
            "confirmation_archive": {
                "path": str(archive_path),
                "bytes": archive_path.stat().st_size,
                "sha256": archive_hash,
            },
            **evidence,
            "runtime_seconds": time.perf_counter() - started,
            "source_sha256": {
                "okutama_protocol": sha256_file(protocol_path),
                "pipeline_lock": pipeline_hash,
                "auditor": sha256_file(Path(__file__).resolve()),
                "okutama_module": sha256_file(
                    Path(__file__).resolve().parents[1] / "src/hac/okutama.py"
                ),
            },
            "artifact_sha256": {centres_path.name: sha256_file(centres_path)},
        }
        write_json(summary_path, summary)
        write_json(
            ledger_path,
            {
                "status": "COMPLETE",
                "opened_at_utc": opened_at,
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "open_attempt": open_attempt,
                "conceptual_confirmation_open_number": 1,
                "pipeline_lock_sha256": pipeline_hash,
                "archive_sha256": archive_hash,
                "summary_sha256": sha256_file(summary_path),
            },
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except Exception as error:
        failure_dir = output_dir / "failed_runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        write_json(
            failure_dir / "failure.json",
            {
                "status": "OKUTAMA_CONFIRMATION_AUDIT_FAILED",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "open_attempt": open_attempt,
                "pipeline_lock_sha256": pipeline_hash,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        write_json(
            ledger_path,
            {
                "status": "FAILED",
                "opened_at_utc": opened_at,
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "open_attempt": open_attempt,
                "conceptual_confirmation_open_number": 1,
                "pipeline_lock_sha256": pipeline_hash,
                "failure_evidence": str(failure_dir / "failure.json"),
            },
        )
        raise


if __name__ == "__main__":
    main()
