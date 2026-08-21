"""Convert the experiment manifest to a portable, repository-relative form."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

import pandas as pd

REQUIRED = {
    "image_id",
    "image_path",
    "label",
    "split",
    "width",
    "height",
    "aspect_ratio",
    "image_url",
    "file_name",
    "sha256",
    "phash",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.source)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing manifest fields: {sorted(missing)}")
    if len(frame) != 285 or frame["image_id"].astype(str).duplicated().any():
        raise RuntimeError("Unexpected dataset cardinality or duplicate image IDs.")
    if set(frame["split"].astype(str)) != {"train", "val", "test"}:
        raise RuntimeError("Unexpected fixed split labels.")

    portable = frame[
        [
            "image_id",
            "label",
            "split",
            "width",
            "height",
            "aspect_ratio",
            "image_url",
            "file_name",
            "sha256",
            "phash",
        ]
    ].copy()
    portable.insert(
        1,
        "image_path",
        [str(PurePosixPath("data", "images", name)) for name in portable["file_name"]],
    )
    portable["original_split"] = portable["split"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    portable.to_csv(args.output, index=False, lineterminator="\n")
    print(f"Wrote {len(portable)} records to {args.output.resolve()}")


if __name__ == "__main__":
    main()
