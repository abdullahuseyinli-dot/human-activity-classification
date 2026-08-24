"""Bind the V-COCO v3 protocol to the immutable v2 evidence and current sources."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from hac.polar import sha256_file
from hac.vcoco_v3 import load_protocol_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("experiments/vcoco_v3_protocol.json"))
    parser.add_argument(
        "--v2-summary", type=Path, default=Path("results/vcoco_v2/official_test_summary.json")
    )
    parser.add_argument(
        "--v2-selection", type=Path, default=Path("results/vcoco_v2/final_selection_lock.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path(".runs/vcoco_v3/protocol/vcoco_v3_lock.json")
    )
    return parser.parse_args()


def git_output(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    spec_path = args.spec.resolve()
    summary_path = args.v2_summary.resolve()
    selection_path = args.v2_selection.resolve()
    output_path = args.output.resolve()

    spec = load_protocol_spec(spec_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    reference = spec["frozen_reference"]
    previous_lock_sha256 = sha256_file(output_path) if output_path.is_file() else None
    previous_revision = 0
    if output_path.is_file():
        previous_payload = json.loads(output_path.read_text(encoding="utf-8"))
        previous_revision = int(previous_payload.get("lock_revision", 1))

    tag_commit = git_output(root, "rev-list", "-n", "1", reference["git_tag"])
    if tag_commit != reference["git_commit"]:
        raise RuntimeError("The frozen v2 tag no longer resolves to the declared commit")
    if summary.get("official_test_label_open_count") != 1:
        raise RuntimeError("The v2 official-test access evidence is incomplete")
    if summary.get("status") != "VCOCO_V2_OFFICIAL_TEST_EVALUATION_COMPLETE":
        raise RuntimeError("The v2 official result is not complete")
    if selection.get("status") != "VCOCO_V2_FINAL_SELECTION_LOCKED_PRE_TEST":
        raise RuntimeError("The v2 selection lock is invalid")

    payload = {
        "status": "VCOCO_V3_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING",
        "lock_revision": previous_revision + 1,
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "protocol_version": spec["protocol_version"],
        "repository_revision_at_lock": git_output(root, "rev-parse", "HEAD"),
        "frozen_v2_tag_commit": tag_commit,
        "v2_test_status": "CONSUMED_EXPLORATORY_ONLY",
        "human_pilot_status": spec["ontology"]["development_pilot"]["status"],
        "human_annotation_status": spec["ontology"]["annotation_gate"]["status"],
        "external_confirmation_status": spec["confirmation"]["status"],
        "source_sha256": {
            "protocol_spec": sha256_file(spec_path),
            "protocol_document": sha256_file(root / "docs" / "VCOCO_V3_RESEARCH_PROTOCOL.md"),
            "protocol_locker": sha256_file(root / "tools" / "lock_vcoco_v3_protocol.py"),
            "v2_official_test_summary": sha256_file(summary_path),
            "v2_final_selection_lock": sha256_file(selection_path),
        },
        "blocked_gates": [
            "human_harmonized_endpoint_annotation_and_adjudication",
            "new_non_coco_confirmation_data",
        ],
    }
    if previous_lock_sha256 is not None:
        payload["supersedes_lock_sha256"] = previous_lock_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
