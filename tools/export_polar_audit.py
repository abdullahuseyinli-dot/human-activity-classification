"""Export path-sanitized POLAR audit evidence and the pre-fit data lock."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    embedding_dir = args.embedding_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = json.loads((data_dir / "polar_data_audit.json").read_text(encoding="utf-8"))
    public_audit = {
        **audit,
        "status": "PRE_SUPERVISED_FIT_DATA_LOCK",
        "protocol_version": "1.1.0",
        "dataset_doi": "10.17632/hvnsh7rwz7.1",
        "raw_images_redistributed": False,
        "test_model_evaluated": False,
    }
    write_json(output_dir / "polar_data_audit.json", public_audit)

    quarantine = pd.read_csv(data_dir / "quarantine.csv", dtype=str)
    quarantine_columns = [
        "quarantine_group",
        "image_id",
        "split",
        "label_4",
        "label_3",
        "sha256",
        "phash",
        "exclusion_reason",
    ]
    public_quarantine = quarantine[quarantine_columns].sort_values(
        ["quarantine_group", "image_id"]
    )
    quarantine_path = output_dir / "polar_quarantine.csv"
    public_quarantine.to_csv(quarantine_path, index=False)

    embedding_audit = json.loads(
        (embedding_dir / "dinov2_embedding_audit.json").read_text(encoding="utf-8")
    )
    embedding_audit["manifest"] = Path(embedding_audit["manifest"]).name
    embedding_audit["model_snapshot"] = Path(embedding_audit["model_snapshot"]).name
    write_json(output_dir / "polar_embedding_audit.json", embedding_audit)

    candidates = pd.read_csv(embedding_dir / "dinov2_cross_split_candidates.csv")
    quarantined_ids = set(public_quarantine["image_id"].astype(str))
    candidates["decision"] = [
        "quarantined_source_related"
        if str(left) in quarantined_ids and str(right) in quarantined_ids
        else "retained_embedding_similarity_only"
        for left, right in zip(
            candidates["left_image_id"], candidates["right_image_id"], strict=True
        )
    ]
    candidate_columns = [
        "left_image_id",
        "left_split",
        "left_label",
        "right_image_id",
        "right_split",
        "right_label",
        "embedding_cosine",
        "phash_distance",
        "normalized_mae",
        "normalized_correlation",
        "decision",
    ]
    candidate_path = output_dir / "polar_embedding_audit_candidates.csv"
    candidates[candidate_columns].to_csv(candidate_path, index=False)

    data_lock = {
        "status": "LOCKED_BEFORE_SUPERVISED_FITTING",
        "protocol_version": "1.1.0",
        "repository_head": git_head(),
        "clean_manifest_sha256": audit["clean_manifest_sha256"],
        "full_manifest_sha256": audit["manifest_sha256"],
        "quarantine_sha256": sha256_file(quarantine_path),
        "embedding_audit_candidates_sha256": sha256_file(candidate_path),
        "clean_rows": audit["clean_rows"],
        "clean_target_counts": audit["clean_target_counts"],
        "test_model_evaluated": False,
        "test_used_for_selection": False,
    }
    write_json(output_dir / "polar_data_lock.json", data_lock)
    print(json.dumps(data_lock, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
