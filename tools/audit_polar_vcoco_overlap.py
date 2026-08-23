"""Audit exact and perceptual image overlap between clean POLAR and V-COCO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hac.polar import (
    enrich_near_pairs,
    exact_cross_split_duplicates,
    near_phash_cross_split_pairs,
    sha256_file,
    source_related_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polar-clean-manifest", type=Path, required=True)
    parser.add_argument("--vcoco-image-audit", type=Path, required=True)
    parser.add_argument("--vcoco-person-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--near-distance", type=int, default=6)
    parser.add_argument("--minimum-correlation", type=float, default=0.90)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def normalized_frame(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    required = {"image_id", "image_path", "sha256", "phash"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} image table is missing columns: {sorted(missing)}")
    output = frame[list(required)].copy()
    output["image_id"] = source + ":" + output["image_id"].astype(str)
    output["split"] = source
    output["label_4"] = ""
    if output["image_id"].duplicated().any():
        raise ValueError(f"{source} image identifiers must be unique")
    return output


def main() -> None:
    args = parse_args()
    polar_path = args.polar_clean_manifest.resolve()
    vcoco_audit_path = args.vcoco_image_audit.resolve()
    vcoco_person_path = args.vcoco_person_manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    polar_raw = pd.read_csv(
        polar_path,
        usecols=["image_id", "image_path", "sha256", "phash", "split"],
        dtype={"image_id": str, "sha256": str, "phash": str},
    )
    vcoco_raw = pd.read_csv(
        vcoco_audit_path,
        dtype={"image_id": str, "sha256": str, "phash": str},
    )
    if "eligible_image" not in vcoco_raw:
        raise ValueError("V-COCO image audit is missing eligible_image")
    vcoco_raw = vcoco_raw[
        vcoco_raw["eligible_image"].astype(str).str.casefold().eq("true")
    ].copy()
    person_images = set(
        pd.read_csv(vcoco_person_path, usecols=["image_id"], dtype={"image_id": str})[
            "image_id"
        ]
    )
    if set(vcoco_raw["image_id"]) != person_images:
        raise RuntimeError("V-COCO image audit and clean person manifest cover different images")

    polar = normalized_frame(polar_raw, source="polar")
    vcoco = normalized_frame(vcoco_raw, source="vcoco")
    combined = pd.concat([polar, vcoco], ignore_index=True)
    exact = exact_cross_split_duplicates(combined)
    if exact.empty:
        exact = pd.DataFrame(
            columns=[
                "sha256",
                "left_image_id",
                "left_split",
                "left_label",
                "right_image_id",
                "right_split",
                "right_label",
            ]
        )
    near = near_phash_cross_split_pairs(combined, max_distance=args.near_distance)
    enriched = enrich_near_pairs(near, workers=args.workers)
    confirmed = source_related_pairs(
        enriched, minimum_correlation=args.minimum_correlation
    )

    exact_path = output_dir / "exact_overlap_pairs.csv"
    near_path = output_dir / "perceptual_overlap_candidates.csv"
    confirmed_path = output_dir / "confirmed_source_related_pairs.csv"
    exact.to_csv(exact_path, index=False)
    enriched.to_csv(near_path, index=False)
    confirmed.to_csv(confirmed_path, index=False)
    summary = {
        "status": "POLAR_VCOCO_CROSS_DATASET_OVERLAP_AUDITED",
        "selection_role": "none",
        "polar_clean_rows": len(polar),
        "polar_test_images_compared": int(polar_raw["split"].astype(str).eq("test").sum()),
        "vcoco_unique_images": len(vcoco),
        "exact_overlap_pairs": len(exact),
        "perceptual_candidates": len(enriched),
        "confirmed_source_related_pairs": len(confirmed),
        "near_phash_maximum_distance": int(args.near_distance),
        "source_related_minimum_correlation": float(args.minimum_correlation),
        "polar_test_labels_read": False,
        "model_predictions_read": 0,
        "test_used_for_selection": False,
        "source_sha256": {
            "polar_clean_manifest": sha256_file(polar_path),
            "vcoco_image_audit": sha256_file(vcoco_audit_path),
            "vcoco_person_manifest": sha256_file(vcoco_person_path),
        },
        "artifact_sha256": {
            exact_path.name: sha256_file(exact_path),
            near_path.name: sha256_file(near_path),
            confirmed_path.name: sha256_file(confirmed_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
