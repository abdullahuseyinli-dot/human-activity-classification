"""Publish compact, path-free evidence from the locked POLAR evaluations."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from hac.polar import sha256_file

GROUPS = {
    "test": (
        "summary.json",
        "test_access_gate.json",
        "test_metrics.csv",
        "test_per_class.csv",
        "test_seed_metrics.csv",
        "test_secondary_metrics.csv",
        "test_confusions.json",
        "test_uncertainty.json",
    ),
    "external": (
        "summary.json",
        "external_person_metrics.csv",
        "external_image_metrics.csv",
        "external_per_class.csv",
        "external_confusions.json",
        "external_image_uncertainty.json",
    ),
    "faithfulness": (
        "summary.json",
        "faithfulness_per_image.csv",
        "faithfulness_curves.csv",
        "faithfulness_randomization.csv",
        "faithfulness_strata.csv",
    ),
    "fault": (
        "summary.json",
        "fault_robustness_metrics.csv",
    ),
}

EXPECTED_STATUS = {
    "test": "LOCKED_FINAL_TEST_COMPLETE",
    "external": "LOCKED_EXTERNAL_EVALUATION_COMPLETE",
    "faithfulness": "LOCKED_POLAR_FAITHFULNESS_COMPLETE",
    "fault": "LOCKED_POLAR_FAULT_ROBUSTNESS_COMPLETE",
}


def published_name(group: str, source_name: str) -> str:
    stem = source_name
    repeated_prefix = f"{group}_"
    if stem.startswith(repeated_prefix):
        stem = stem[len(repeated_prefix) :]
    return f"polar_{group}_{stem}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--faithfulness-dir", type=Path, required=True)
    parser.add_argument("--fault-dir", type=Path, required=True)
    parser.add_argument("--overlap-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock_path = args.selection_lock.resolve()
    lock_hash = sha256_file(lock_path)
    source_dirs = {
        "test": args.test_dir.resolve(),
        "external": args.external_dir.resolve(),
        "faithfulness": args.faithfulness_dir.resolve(),
        "fault": args.fault_dir.resolve(),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = {}
    summaries = {}
    for group, names in GROUPS.items():
        source_dir = source_dirs[group]
        summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
        summaries[group] = summary
        if summary.get("status") != EXPECTED_STATUS[group]:
            raise RuntimeError(f"Incomplete {group} evidence: {source_dir}")
        if summary.get("selection_lock_sha256") != lock_hash:
            raise RuntimeError(f"Selection-lock drift in {group} evidence")
        for name in names:
            source = source_dir / name
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = output_dir / published_name(group, name)
            shutil.copyfile(source, destination)
            exported[destination.name] = sha256_file(destination)

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    overlap_path = args.overlap_audit.resolve()
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    if (
        overlap.get("status") != "POLAR_VCOCO_CROSS_DATASET_OVERLAP_AUDITED"
        or overlap.get("confirmed_source_related_pairs") != 0
        or overlap.get("source_sha256", {}).get("vcoco_person_manifest")
        != lock["external_validation"]["manifest_sha256"]
    ):
        raise RuntimeError("Cross-dataset overlap evidence is incomplete")
    overlap_destination = output_dir / "polar_external_overlap_audit.json"
    if summaries["external"].get("overlap_audit_sha256") != sha256_file(overlap_path):
        raise RuntimeError("External evaluation used a different overlap audit")
    shutil.copyfile(overlap_path, overlap_destination)
    exported[overlap_destination.name] = sha256_file(overlap_destination)

    manifest = {
        "status": "LOCKED_POLAR_PORTFOLIO_EVIDENCE",
        "selection_lock_sha256": lock_hash,
        "exported_files": exported,
        "excluded_local_artifacts": [
            "opened test manifest with machine-local image paths",
            "model checkpoints and fitted classifier binaries",
            "per-example probability arrays",
            "full-resolution attribution maps",
        ],
        "test_used_for_selection": False,
    }
    manifest_path = output_dir / "polar_final_evidence_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
