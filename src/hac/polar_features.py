"""Feature-cache contracts shared by the frozen POLAR experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_aligned_feature_view(
    root: Path,
    model_kind: str,
    view: str,
    manifest: pd.DataFrame,
    manifest_hash: str,
) -> tuple[np.ndarray, dict]:
    """Load one cache and align its rows to an explicitly ordered manifest."""

    cache_dir = root / model_kind / view
    provenance_path = cache_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("manifest_sha256") != manifest_hash:
        raise RuntimeError(f"Feature cache manifest drift: {provenance_path}")
    test_rows_read = provenance.get("test_rows_read", provenance.get("test_rows"))
    if test_rows_read != 0 or provenance.get("test_labels_read") is True:
        raise RuntimeError(f"Development feature cache violates the test gate: {provenance_path}")
    rows = pd.read_csv(cache_dir / "rows.csv", dtype={"image_id": str})
    features = np.load(cache_dir / "features.npy", mmap_mode="r")
    if len(rows) != len(features):
        raise RuntimeError(f"Feature and metadata rows differ: {cache_dir}")
    if rows["image_id"].astype(str).duplicated().any():
        raise RuntimeError(f"Feature cache identifiers are not unique: {cache_dir}")
    index_by_id = {value: index for index, value in enumerate(rows["image_id"].astype(str))}
    try:
        order = np.asarray(
            [index_by_id[value] for value in manifest["image_id"].astype(str)], dtype=int
        )
    except KeyError as error:
        raise RuntimeError(f"Feature cache is missing manifest image {error}") from error
    aligned = np.asarray(features[order], dtype=np.float32)
    return aligned, provenance
