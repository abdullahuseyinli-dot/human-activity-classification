"""Manifest validation and fixed-test split handling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "image_id",
    "image_path",
    "label",
    "split",
    "sha256",
    "phash",
}
EXPECTED_CLASSES = {"sitting", "standing", "walking_running"}


@dataclass(frozen=True, slots=True)
class FixedTestProtocol:
    development_rows: int
    test_rows: int
    test_image_ids_sha256: str
    manifest_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_image_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    repo_root = manifest_path.parent.parent
    return (repo_root / path).resolve()


def load_and_validate_manifest(
    manifest_path: str | Path,
    *,
    require_images: bool = True,
    expected_total: int = 285,
    expected_test: int = 43,
) -> tuple[pd.DataFrame, FixedTestProtocol]:
    path = Path(manifest_path).resolve()
    frame = pd.read_csv(path, dtype={"image_id": str, "sha256": str, "phash": str})
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if len(frame) != expected_total:
        raise ValueError(f"Expected {expected_total} rows, found {len(frame)}")
    if frame["image_id"].duplicated().any():
        raise ValueError("image_id must be unique")
    if frame["sha256"].isna().any() or frame["sha256"].duplicated().any():
        raise ValueError("sha256 must be complete and unique")
    if set(frame["label"].astype(str)) != EXPECTED_CLASSES:
        raise ValueError("Unexpected activity labels")

    normalized_split = (
        frame["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"valid": "val", "validation": "val"})
    )
    if set(normalized_split) != {"train", "val", "test"}:
        raise ValueError("Expected train, val, and test split labels")
    frame["original_split"] = normalized_split
    frame["protocol_split"] = normalized_split.map(
        {"train": "development", "val": "development", "test": "test"}
    )
    frame["resolved_image_path"] = [
        str(_resolve_image_path(path, value)) for value in frame["image_path"].astype(str)
    ]
    if require_images:
        missing_images = [
            value for value in frame["resolved_image_path"] if not Path(value).is_file()
        ]
        if missing_images:
            raise FileNotFoundError(f"{len(missing_images)} manifest images are unavailable")

    test = frame[frame["protocol_split"] == "test"]
    development = frame[frame["protocol_split"] == "development"]
    if len(test) != expected_test or len(development) != expected_total - expected_test:
        raise ValueError("Fixed split cardinality changed")

    for duplicate_key in ("sha256", "phash"):
        crossing = (
            frame.groupby(duplicate_key)["protocol_split"].nunique().loc[lambda values: values > 1]
        )
        if len(crossing):
            raise ValueError(f"{duplicate_key} duplicate groups cross the fixed test boundary")

    phash_values = [int(value, 16) for value in frame["phash"].astype(str)]
    split_values = frame["protocol_split"].astype(str).tolist()
    for left in range(len(frame)):
        for right in range(left + 1, len(frame)):
            if split_values[left] == split_values[right]:
                continue
            if (phash_values[left] ^ phash_values[right]).bit_count() <= 6:
                raise ValueError("Near-duplicate perceptual hashes cross the fixed test boundary")

    test_hash = hashlib.sha256(
        "\n".join(sorted(test["image_id"].astype(str))).encode("utf-8")
    ).hexdigest()
    protocol = FixedTestProtocol(
        development_rows=len(development),
        test_rows=len(test),
        test_image_ids_sha256=test_hash,
        manifest_sha256=sha256_file(path),
    )
    return frame, protocol
