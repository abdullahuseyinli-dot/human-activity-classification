"""Summarize one incomplete V-COCO v3 annotation pass as descriptive pilot evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file
from hac.vcoco_v3_audit import load_annotation_snapshot, single_rater_pilot_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--private-manifest",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/pilot/private_sampling_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/single_rater_pilot"),
    )
    parser.add_argument("--expected-responses", type=int)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    snapshot_path = args.snapshot.resolve()
    private_path = args.private_manifest.resolve()
    private = pd.read_csv(
        private_path,
        dtype={
            "task_id": str,
            "repeat_of_task_id": str,
            "person_id": str,
            "image_id": str,
        },
        keep_default_na=False,
    )
    snapshot, ratings = load_annotation_snapshot(snapshot_path, private["task_id"])
    if args.expected_responses is not None and len(ratings) != args.expected_responses:
        raise RuntimeError(
            f"Expected {args.expected_responses} responses, found {len(ratings)}"
        )
    if int(snapshot.get("completed_rows", -1)) != len(ratings):
        raise RuntimeError("Snapshot completion metadata does not match its responses")

    summary, tables = single_rater_pilot_audit(private, ratings)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_hashes = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        artifact_hashes[path.name] = sha256_file(path)

    summary.update(
        {
            "annotator_id": str(snapshot.get("annotator_id", "")),
            "guide_versions": sorted(ratings["guide_version"].astype(str).unique()),
            "evidence_scope": "descriptive_single_rater_partial_pilot",
            "model_fitting_permitted_by_annotation_gate": False,
            "source_sha256": {
                "annotation_snapshot": sha256_file(snapshot_path),
                "private_manifest": sha256_file(private_path),
            },
            "artifact_sha256": artifact_hashes,
        }
    )
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
