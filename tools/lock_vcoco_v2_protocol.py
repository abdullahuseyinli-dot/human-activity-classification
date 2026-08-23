"""Create the immutable, eligibility-audited V-COCO v2 protocol artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest-dir", type=Path, required=True)
    parser.add_argument("--development-clean-manifest", type=Path, required=True)
    parser.add_argument("--development-overlap-summary", type=Path, required=True)
    parser.add_argument("--test-clean-manifest", type=Path, required=True)
    parser.add_argument("--test-overlap-summary", type=Path, required=True)
    parser.add_argument("--polar-clean-manifest", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def load_overlap_summary(path: Path, expected_manifest_hash: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "POLAR_VCOCO_CROSS_DATASET_OVERLAP_AUDITED":
        raise RuntimeError(f"Overlap audit is incomplete: {path}")
    if int(payload.get("confirmed_source_related_pairs", -1)) != 0:
        raise RuntimeError(f"Confirmed POLAR overlap remains: {path}")
    if payload.get("source_sha256", {}).get("vcoco_person_manifest") != expected_manifest_hash:
        raise RuntimeError(f"Overlap audit manifest drift: {path}")
    return payload


def attach_factorized_columns(source: pd.DataFrame, audited: pd.DataFrame) -> pd.DataFrame:
    factor_columns = [
        "person_id",
        "external_split",
        "ontology_source",
        "posture_label",
        "motion_label",
        "gait_label",
        "legacy_eligible",
        "factorized_clear",
        "bbox_aspect_ratio",
        "bbox_center_x_fraction",
        "bbox_center_y_fraction",
        "person_pixel_height",
    ]
    missing = set(factor_columns) - set(source)
    if missing:
        raise ValueError(f"Source manifest lacks factor columns: {sorted(missing)}")
    factors = source[factor_columns].copy()
    factors["person_id"] = factors["person_id"].astype(str)
    output = audited.copy()
    output["person_id"] = output["person_id"].astype(str)
    overlapping = (set(factor_columns) - {"person_id"}) & set(output)
    output = output.drop(columns=sorted(overlapping))
    output = output.merge(factors, on="person_id", how="left", validate="one_to_one")
    if output["external_split"].isna().any():
        raise RuntimeError("Audited manifest contains people absent from the source split")
    return output


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    source_dir = args.source_manifest_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        split: pd.read_csv(
            source_dir / f"vcoco_{split}_persons.csv",
            dtype={"person_id": str, "image_id": str, "annotation_id": str},
        )
        for split in ("train", "val", "test")
    }
    development_path = args.development_clean_manifest.resolve()
    development = pd.read_csv(
        development_path,
        dtype={"person_id": str, "image_id": str, "annotation_id": str},
    )
    test_path = args.test_clean_manifest.resolve()
    test = pd.read_csv(
        test_path,
        dtype={"person_id": str, "image_id": str, "annotation_id": str},
    )

    development_hash = sha256_file(development_path)
    test_hash = sha256_file(test_path)
    development_overlap = load_overlap_summary(
        args.development_overlap_summary.resolve(), development_hash
    )
    test_overlap = load_overlap_summary(args.test_overlap_summary.resolve(), test_hash)

    development_by_id = set(development["person_id"])
    clean_frames = {}
    for split in ("train", "val"):
        source = sources[split]
        audited = development[development["person_id"].isin(set(source["person_id"]))].copy()
        expected = set(source.loc[source["legacy_eligible"].astype(bool), "person_id"])
        if set(audited["person_id"]) != expected:
            missing = expected - development_by_id
            raise RuntimeError(
                f"Development clean cohort drift for {split}: {len(missing)} missing people"
            )
        clean_frames[split] = attach_factorized_columns(source, audited)

    test_factorized = attach_factorized_columns(sources["test"], test)
    clean_frames["test"] = test_factorized[test_factorized["legacy_eligible"].astype(bool)].copy()

    split_images = {
        split: set(frame["image_id"].astype(str)) for split, frame in clean_frames.items()
    }
    for left_index, left in enumerate(("train", "val", "test")):
        for right in ("train", "val", "test")[left_index + 1 :]:
            overlap = split_images[left] & split_images[right]
            if overlap:
                raise RuntimeError(f"Clean V-COCO {left}/{right} image overlap: {len(overlap)}")

    artifact_hashes = {}
    summaries = {}
    for split, frame in clean_frames.items():
        frame = frame.sort_values(["image_id", "annotation_id"], ignore_index=True)
        frame["selection_role"] = {
            "train": "adaptation",
            "val": "selection_and_calibration",
            "test": "confirmatory_only",
        }[split]
        path = output_dir / f"vcoco_{split}_clean.csv"
        frame.to_csv(path, index=False)
        artifact_hashes[path.name] = sha256_file(path)
        summaries[split] = {
            "people": len(frame),
            "images": int(frame["image_id"].nunique()),
            "class_counts": {
                str(key): int(value)
                for key, value in frame["label_3"].value_counts().sort_index().items()
            },
            "image_files_present": int(
                frame.drop_duplicates("image_id")["image_path"]
                .map(lambda value: Path(str(value)).is_file())
                .sum()
            ),
        }

    policy = {
        "status": "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING",
        "protocol_version": "2.0.0",
        "v1_baseline": {
            "git_tag": "polar-study-v1.0.0",
            "external_image_macro_f1": 0.6669343425821733,
            "external_person_macro_f1": 0.6840393638307938,
            "selection_role": "frozen_historical_baseline",
        },
        "primary_endpoint": "person_level_macro_f1_on_legacy_eligible_vcoco",
        "secondary_endpoints": [
            "per_class_precision_recall_f1",
            "balanced_accuracy",
            "log_loss",
            "brier_score",
            "classwise_and_adaptive_ece",
            "risk_coverage_auc",
        ],
        "selection_policy": {
            "train": "adaptation_and_inner_cross_validation_grouped_by_image",
            "val": "external_model_selection_and_calibration",
            "test": "one_final_run_after_champion_lock",
        },
        "promotion_rule": {
            "minimum_macro_f1_gain": 0.01,
            "paired_image_cluster_bootstrap_interval_must_exclude_zero": True,
            "standing_or_locomotion_collapse_forbidden": True,
        },
        "ontology": {
            "current_labels": "source_tags_factorized_not_human_harmonized",
            "manual_annotation_status": "NOT_PERFORMED_REQUIRES_INDEPENDENT_HUMAN_ANNOTATORS",
            "ambiguous_source_tags_preserved": True,
            "test_source_class_counts_forbidden_for_selection": True,
        },
        "known_pretraining_caveat": (
            "V-COCO uses COCO images; COCO-fine-tuned detector or pose auxiliaries are "
            "reported separately and cannot establish clean cross-dataset transfer."
        ),
        "split_summary": summaries,
        "quarantine": {
            "legacy_overlap_test_images": 60,
            "polar_overlap_development_confirmed_pairs": int(
                development_overlap["confirmed_source_related_pairs"]
            ),
            "polar_overlap_test_confirmed_pairs": int(
                test_overlap["confirmed_source_related_pairs"]
            ),
        },
        "test_access": {
            "annotation_counts_inspected": True,
            "images_hashed_for_overlap": True,
            "model_predictions_run": False,
            "labels_used_for_model_or_prior_selection": False,
        },
        "repository_revision_at_lock": git_revision(root),
        "source_sha256": {
            "polar_clean_manifest": sha256_file(args.polar_clean_manifest.resolve()),
            "legacy_manifest": sha256_file(args.legacy_manifest.resolve()),
            "development_clean_manifest": development_hash,
            "test_clean_manifest": test_hash,
            "development_overlap_summary": sha256_file(args.development_overlap_summary.resolve()),
            "test_overlap_summary": sha256_file(args.test_overlap_summary.resolve()),
            "manifest_builder": sha256_file(root / "tools" / "build_vcoco_v2_manifests.py"),
            "ontology_module": sha256_file(root / "src" / "hac" / "vcoco.py"),
            "protocol_locker": sha256_file(root / "tools" / "lock_vcoco_v2_protocol.py"),
        },
        "artifact_sha256": artifact_hashes,
    }
    lock_path = output_dir / "vcoco_v2_protocol_lock.json"
    lock_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(policy, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
