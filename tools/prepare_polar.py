"""Build and audit the local POLAR target-task manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hac.polar import (
    apply_quarantine,
    build_manifest,
    embedding_confirmed_source_pairs,
    enrich_near_pairs,
    exact_cross_split_duplicates,
    legacy_overlap,
    manifest_summary,
    near_phash_cross_split_pairs,
    quarantine_components,
    sha256_file,
    source_related_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--image-sets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--near-distance", type=int, default=6)
    parser.add_argument("--source-correlation", type=float, default=0.90)
    parser.add_argument("--embedding-candidates", type=Path)
    parser.add_argument("--embedding-minimum-cosine", type=float, default=0.985)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, base_audit = build_manifest(
        args.annotations_dir,
        args.images_dir,
        args.image_sets_dir,
        workers=args.workers,
        show_progress=True,
    )
    exact = exact_cross_split_duplicates(frame)
    exact.to_csv(output_dir / "exact_cross_split_duplicates.csv", index=False)
    near = near_phash_cross_split_pairs(frame, max_distance=args.near_distance)
    near = enrich_near_pairs(near, workers=args.workers)
    near.to_csv(output_dir / "near_cross_split_candidates.csv", index=False)
    phash_confirmed = source_related_pairs(near, minimum_correlation=args.source_correlation)
    phash_confirmed["audit_source"] = "phash_normalized"
    embedding_confirmed = pd.DataFrame()
    if args.embedding_candidates:
        embedding_pairs = pd.read_csv(args.embedding_candidates)
        embedding_confirmed = embedding_confirmed_source_pairs(
            embedding_pairs,
            minimum_cosine=args.embedding_minimum_cosine,
            minimum_correlation=args.source_correlation,
        )
        embedding_confirmed["audit_source"] = "dinov2_normalized"
        embedding_confirmed.to_csv(
            output_dir / "embedding_confirmed_source_related_pairs.csv", index=False
        )
    confirmed = pd.concat([phash_confirmed, embedding_confirmed], ignore_index=True)
    confirmed["pair_key"] = confirmed.apply(
        lambda row: "::".join(
            sorted((str(row["left_image_id"]), str(row["right_image_id"])))
        ),
        axis=1,
    )
    confirmed = confirmed.sort_values(["pair_key", "audit_source"]).drop_duplicates(
        "pair_key", keep="first"
    )
    confirmed = confirmed.drop(columns="pair_key").reset_index(drop=True)
    confirmed.to_csv(output_dir / "confirmed_source_related_pairs.csv", index=False)
    quarantine = quarantine_components(confirmed)
    full_frame, clean_frame = apply_quarantine(frame, quarantine)
    exclusion_details = quarantine.merge(
        full_frame[
            ["image_id", "split", "label_4", "label_3", "sha256", "phash", "original_name"]
        ],
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    exclusion_details["exclusion_reason"] = "cross_split_source_related"
    exclusion_details.to_csv(output_dir / "quarantine.csv", index=False)

    manifest_path = output_dir / "polar_target_manifest.csv"
    clean_manifest_path = output_dir / "polar_clean_manifest.csv"
    development_manifest_path = output_dir / "polar_development_manifest.csv"
    test_manifest_path = output_dir / "polar_test_manifest.csv"
    full_frame.to_csv(manifest_path, index=False)
    clean_frame.to_csv(clean_manifest_path, index=False)
    clean_frame[clean_frame["split"].isin(["train", "val"])].to_csv(
        development_manifest_path, index=False
    )
    clean_frame[clean_frame["split"] == "test"].to_csv(test_manifest_path, index=False)

    overlaps = None
    if args.legacy_manifest:
        overlaps = legacy_overlap(full_frame, args.legacy_manifest)
        overlaps.to_csv(output_dir / "legacy_overlap_candidates.csv", index=False)

    clean_counts = (
        clean_frame.groupby(["split", "label_4"], observed=True).size().unstack(fill_value=0)
    )
    audit = manifest_summary(full_frame, base_audit)
    audit.update(
        {
            "manifest": manifest_path.name,
            "manifest_sha256": sha256_file(manifest_path),
            "clean_manifest": clean_manifest_path.name,
            "clean_manifest_sha256": sha256_file(clean_manifest_path),
            "development_manifest": development_manifest_path.name,
            "development_manifest_sha256": sha256_file(development_manifest_path),
            "test_manifest": test_manifest_path.name,
            "test_manifest_sha256": sha256_file(test_manifest_path),
            "exact_cross_split_pairs": len(exact),
            "near_cross_split_candidates": len(near),
            "confirmed_source_related_pairs": len(confirmed),
            "phash_confirmed_source_related_pairs": len(phash_confirmed),
            "embedding_confirmed_source_related_pairs": len(embedding_confirmed),
            "embedding_minimum_cosine": (
                args.embedding_minimum_cosine if args.embedding_candidates else None
            ),
            "source_related_minimum_correlation": args.source_correlation,
            "quarantine_components": int(quarantine["quarantine_group"].nunique()),
            "quarantine_images": len(quarantine),
            "clean_rows": len(clean_frame),
            "clean_target_counts": {
                split: {
                    label: int(clean_counts.loc[split, label])
                    for label in ("sitting", "standing", "walking", "running")
                }
                for split in ("train", "val", "test")
            },
            "legacy_overlap_candidates": len(overlaps) if overlaps is not None else None,
            "near_hash_review_threshold": args.near_distance,
            "quarantine_applied": True,
            "test_used_for_model_selection": False,
        }
    )
    write_json(output_dir / "polar_data_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
