"""Download, verify, and legacy-deduplicate images for the V-COCO external cohort."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from hac.polar import normalized_pair_similarity, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def inspect_encoded(encoded: bytes) -> dict:
    digest = hashlib.sha256(encoded).hexdigest()
    with Image.open(io.BytesIO(encoded)) as image:
        image.load()
        return {
            "sha256": digest,
            "phash": str(imagehash.phash(image.convert("RGB"), hash_size=8)),
            "actual_width": int(image.width),
            "actual_height": int(image.height),
            "image_format": str(image.format),
        }


def download_one(record: dict, retries: int, timeout: int) -> dict:
    path = Path(record["image_path"])
    try:
        if path.is_file():
            inspection = inspect_encoded(path.read_bytes())
            return {**record, **inspection, "download_status": "existing_verified", "error": ""}
        path.parent.mkdir(parents=True, exist_ok=True)
        last_error = None
        for attempt in range(1, retries + 1):
            temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
            try:
                request = urllib.request.Request(
                    str(record["image_url"]),
                    headers={"User-Agent": "hac-research-reproduction/1.0"},
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    encoded = response.read()
                inspection = inspect_encoded(encoded)
                temporary.write_bytes(encoded)
                temporary.replace(path)
                return {
                    **record,
                    **inspection,
                    "download_status": "downloaded_verified",
                    "error": "",
                }
            except Exception as error:  # pragma: no cover - depends on external network
                last_error = error
                if temporary.is_file():
                    temporary.rename(
                        temporary.with_suffix(temporary.suffix + f".failed_attempt_{attempt}")
                    )
                if attempt < retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"download failed after {retries} attempts: {last_error}")
    except Exception as error:  # pragma: no cover - depends on external network/data
        return {
            **record,
            "sha256": "",
            "phash": "",
            "actual_width": -1,
            "actual_height": -1,
            "image_format": "",
            "download_status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }


def legacy_overlap_audit(
    images: pd.DataFrame, legacy_manifest_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    legacy = pd.read_csv(legacy_manifest_path, dtype=str)
    legacy_root = legacy_manifest_path.resolve().parent.parent
    legacy_by_sha = {
        str(record["sha256"]): record for record in legacy.to_dict("records")
    }
    legacy_records = legacy.to_dict("records")
    audit_rows = []
    overlap_by_image = {}
    for record in images.to_dict("records"):
        image_id = str(record["image_id"])
        matches = []
        exact = legacy_by_sha.get(str(record["sha256"]))
        if exact is not None:
            matches.append((exact, "sha256", 0, 1.0))
        external_hash = str(record["phash"])
        if len(external_hash) == 16:
            external_value = int(external_hash, 16)
            for candidate in legacy_records:
                legacy_hash = str(candidate.get("phash", ""))
                if len(legacy_hash) != 16:
                    continue
                distance = (external_value ^ int(legacy_hash, 16)).bit_count()
                if distance > 6:
                    continue
                legacy_path = legacy_root / str(candidate["image_path"])
                if not legacy_path.is_file():
                    continue
                similarity = normalized_pair_similarity(record["image_path"], legacy_path)
                if similarity["normalized_correlation"] >= 0.90:
                    matches.append(
                        (
                            candidate,
                            "phash_and_normalized_correlation",
                            distance,
                            similarity["normalized_correlation"],
                        )
                    )
        direct = [candidate for candidate in legacy_records if str(candidate["image_id"]) == image_id]
        matches.extend((candidate, "coco_image_id", -1, 1.0) for candidate in direct)
        if matches:
            matches.sort(key=lambda value: (value[1], str(value[0]["image_id"])))
            candidate, rule, distance, correlation = matches[0]
            overlap_by_image[image_id] = rule
            audit_rows.append(
                {
                    "external_image_id": image_id,
                    "legacy_image_id": str(candidate["image_id"]),
                    "confirmation_rule": rule,
                    "phash_distance": distance,
                    "normalized_correlation": correlation,
                }
            )
    output = images.copy()
    output["legacy_overlap_rule"] = output["image_id"].astype(str).map(overlap_by_image).fillna("")
    output["legacy_overlap"] = output["legacy_overlap_rule"].astype(str).str.len() > 0
    return output, pd.DataFrame(audit_rows)


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.retries < 1 or args.timeout < 1:
        raise ValueError("workers, retries, and timeout must be positive")
    manifest_path = args.manifest.resolve()
    frame = pd.read_csv(manifest_path, dtype={"image_id": str, "annotation_id": str})
    required = {"image_id", "image_path", "image_url", "file_name", "label_3"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"V-COCO manifest is missing columns: {sorted(missing)}")
    images = frame.drop_duplicates("image_id").sort_values("image_id").reset_index(drop=True)
    records = images[["image_id", "image_path", "image_url", "file_name"]].to_dict("records")

    def fetch(record: dict) -> dict:
        return download_one(record, args.retries, args.timeout)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        inspections = list(
            tqdm(
                executor.map(fetch, records),
                total=len(records),
                desc="downloading V-COCO images",
                unit="image",
            )
        )
    inspected = pd.DataFrame(inspections)
    inspected["dimension_match"] = (
        (inspected["actual_width"] == images["image_width"].to_numpy())
        & (inspected["actual_height"] == images["image_height"].to_numpy())
    )
    inspected, overlaps = legacy_overlap_audit(inspected, args.legacy_manifest.resolve())
    inspected["eligible_image"] = (
        inspected["download_status"].ne("failed")
        & inspected["dimension_match"]
        & ~inspected["legacy_overlap"]
    )
    audit_columns = [
        "image_id",
        "sha256",
        "phash",
        "actual_width",
        "actual_height",
        "image_format",
        "download_status",
        "error",
        "dimension_match",
        "legacy_overlap",
        "legacy_overlap_rule",
        "eligible_image",
    ]
    audited = frame.merge(inspected[audit_columns], on="image_id", how="left", validate="many_to_one")
    audited["eligible_person"] = audited["eligible_image"].astype(bool)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audited.to_csv(output_dir / "vcoco_person_manifest_audited.csv", index=False)
    audited[audited["eligible_person"]].to_csv(
        output_dir / "vcoco_person_manifest_clean.csv", index=False
    )
    inspected.to_csv(output_dir / "vcoco_image_audit.csv", index=False)
    overlaps.to_csv(output_dir / "vcoco_legacy_overlap.csv", index=False)
    failures = inspected[inspected["download_status"].eq("failed")][
        ["image_id", "image_url", "error"]
    ]
    failures.to_csv(output_dir / "vcoco_download_failures.csv", index=False)
    clean = audited[audited["eligible_person"]]
    provenance = {
        "status": "VCOCO_EXTERNAL_COHORT_AUDITED",
        "selection_role": "none",
        "source_manifest_sha256": sha256_file(manifest_path),
        "legacy_manifest_sha256": sha256_file(args.legacy_manifest),
        "person_rows_source": len(frame),
        "unique_images_source": len(images),
        "downloaded_or_verified_images": int(inspected["download_status"].ne("failed").sum()),
        "download_failures": len(failures),
        "dimension_mismatches": int((~inspected["dimension_match"]).sum()),
        "legacy_overlap_images": int(inspected["legacy_overlap"].sum()),
        "clean_person_rows": len(clean),
        "clean_unique_images": clean["image_id"].nunique(),
        "clean_image_level_unambiguous_images": clean.loc[
            clean["image_level_unambiguous"].astype(bool), "image_id"
        ].nunique(),
        "clean_person_class_counts": clean["label_3"].value_counts().sort_index().to_dict(),
        "clean_manifest_sha256": sha256_file(
            output_dir / "vcoco_person_manifest_clean.csv"
        ),
        "workers": args.workers,
        "retries": args.retries,
        "timeout_seconds": args.timeout,
        "model_predictions_read": 0,
        "polar_test_rows_read": 0,
        "test_used_for_selection": False,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
