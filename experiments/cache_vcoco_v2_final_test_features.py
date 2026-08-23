"""Extract the two locked DINO views from official V-COCO test after selection lock."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cache_vcoco_v2_features import FeatureDataset, extract
from torch.utils.data import DataLoader

from hac.augmentations import build_aspect_preserving_eval_transform
from hac.polar import sha256_file
from hac.polar_features import PinnedDinoFeatureModel
from hac.polar_models import DINO_MODEL_SPECS

TEST_COLUMNS = [
    "person_id",
    "image_id",
    "image_path",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "bbox_area_fraction",
    "bbox_aspect_ratio",
    "bbox_center_x_fraction",
    "bbox_center_y_fraction",
    "person_pixel_height",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid loader settings")
    protocol_path = args.protocol_lock.resolve()
    selection_path = args.selection_lock.resolve()
    test_path = args.test_manifest.resolve()
    protocol_hash = sha256_file(protocol_path)
    selection_hash = sha256_file(selection_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("Official-test extraction requires the v2 protocol lock")
    if (
        selection.get("status") != "VCOCO_V2_FINAL_SELECTION_LOCKED_PRE_TEST"
        or selection.get("protocol_lock_sha256") != protocol_hash
    ):
        raise RuntimeError("Official-test extraction requires the matching final selection lock")
    if sha256_file(test_path) != selection["final_test"]["manifest_sha256"]:
        raise RuntimeError("Official-test manifest drift")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "VCOCO_V2_LOCKED_TEST_FEATURES_COMPLETE"
            and existing.get("selection_lock_sha256") == selection_hash
        ):
            print(json.dumps(existing, indent=2, sort_keys=True), flush=True)
            return
        raise RuntimeError("Existing official-test feature output belongs to another selection")

    rows = pd.read_csv(
        test_path,
        dtype={"person_id": str, "image_id": str},
        usecols=TEST_COLUMNS,
    )
    if len(rows) != selection["final_test"]["expected_people"]:
        raise RuntimeError("Official-test person count drift")
    if rows["image_id"].nunique() != selection["final_test"]["expected_images"]:
        raise RuntimeError("Official-test image count drift")
    if rows["person_id"].duplicated().any():
        raise RuntimeError("Official-test person identifiers are not unique")
    if not rows["image_path"].map(lambda value: Path(str(value)).is_file()).all():
        raise RuntimeError("Official-test images are incomplete")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    model = PinnedDinoFeatureModel("dinov2_base", "final_cls").to(device)
    transform = build_aspect_preserving_eval_transform(224)
    started = time.perf_counter()
    features = {}
    for name, view in (("tight", "person_tight"), ("context", "person_context_25")):
        loader = DataLoader(
            FeatureDataset(rows, view=view, transform=transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        features[name] = extract(model, loader, device)

    rows_path = output_dir / "rows.csv"
    tight_path = output_dir / "tight_features.npy"
    context_path = output_dir / "context_features.npy"
    rows.to_csv(rows_path, index=False)
    np.save(tight_path, features["tight"])
    np.save(context_path, features["context"])
    result = {
        "status": "VCOCO_V2_LOCKED_TEST_FEATURES_COMPLETE",
        "selection_lock_sha256": selection_hash,
        "protocol_lock_sha256": protocol_hash,
        "model": DINO_MODEL_SPECS["dinov2_base"],
        "views": ["person_tight", "person_context_25"],
        "preprocess": "aspect_preserving_pad_224",
        "rows": len(rows),
        "images": int(rows["image_id"].nunique()),
        "feature_dimensions": int(features["tight"].shape[1]),
        "runtime_seconds": time.perf_counter() - started,
        "test_metadata_rows_read": len(rows),
        "test_label_columns_read": 0,
        "official_test_feature_extraction_count": 1,
        "artifact_sha256": {
            rows_path.name: sha256_file(rows_path),
            tight_path.name: sha256_file(tight_path),
            context_path.name: sha256_file(context_path),
        },
    }
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
