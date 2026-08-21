from pathlib import Path

import pandas as pd
import pytest

from hac.protocol import load_and_validate_manifest


def test_tracked_manifest_preserves_the_fixed_test_contract():
    repository = Path(__file__).resolve().parents[1]
    frame, protocol = load_and_validate_manifest(
        repository / "data" / "manifest.csv", require_images=False
    )
    assert len(frame) == 285
    assert protocol.development_rows == 242
    assert protocol.test_rows == 43
    assert len(protocol.test_image_ids_sha256) == 64


def test_near_duplicate_crossing_the_test_boundary_is_rejected(tmp_path):
    labels = ["sitting", "standing", "walking_running"] * 2
    frame = pd.DataFrame(
        {
            "image_id": [str(index) for index in range(6)],
            "image_path": [f"data/images/{index}.jpg" for index in range(6)],
            "label": labels,
            "split": ["train", "train", "val", "val", "test", "test"],
            "sha256": [f"{index:064x}" for index in range(6)],
            "phash": [
                "0000000000000000",
                "1111111111111111",
                "2222222222222222",
                "3333333333333333",
                "0000000000000001",
                "5555555555555555",
            ],
        }
    )
    manifest = tmp_path / "data" / "manifest.csv"
    manifest.parent.mkdir()
    frame.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="Near-duplicate"):
        load_and_validate_manifest(
            manifest,
            require_images=False,
            expected_total=6,
            expected_test=2,
        )
