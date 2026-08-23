"""Build a person-indexed neural-training manifest from locked V-COCO development splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock_path = args.protocol_lock.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("Neural training requires the locked V-COCO v2 protocol")
    paths = {
        "train": args.train_manifest.resolve(),
        "val": args.val_manifest.resolve(),
    }
    frames = []
    for split, path in paths.items():
        if sha256_file(path) != lock["artifact_sha256"][f"vcoco_{split}_clean.csv"]:
            raise RuntimeError(f"Locked {split} manifest drift")
        frame = pd.read_csv(path, dtype={"person_id": str, "image_id": str})
        frame["split"] = split
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.rename(columns={"image_id": "coco_image_id"})
    rows.insert(1, "image_id", rows["person_id"].astype(str))
    rows["source_image_group"] = rows["coco_image_id"].astype(str)
    if rows["image_id"].duplicated().any():
        raise RuntimeError("Person-indexed training identifiers are not unique")
    if set(rows["split"]) != {"train", "val"}:
        raise RuntimeError("Training manifest must contain exactly train and validation")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "vcoco_v2_person_training.csv"
    rows.to_csv(manifest_path, index=False)
    provenance = {
        "status": "VCOCO_V2_PERSON_TRAINING_MANIFEST_COMPLETE",
        "rows": len(rows),
        "train_rows": int(rows["split"].eq("train").sum()),
        "val_rows": int(rows["split"].eq("val").sum()),
        "train_source_images": int(
            rows.loc[rows["split"].eq("train"), "source_image_group"].nunique()
        ),
        "val_source_images": int(
            rows.loc[rows["split"].eq("val"), "source_image_group"].nunique()
        ),
        "identifier_unit": "person",
        "grouping_unit": "source_coco_image",
        "protocol_lock_sha256": sha256_file(lock_path),
        "source_sha256": {split: sha256_file(path) for split, path in paths.items()},
        "manifest_sha256": sha256_file(manifest_path),
        "test_rows_read": 0,
        "test_predictions_run": False,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
