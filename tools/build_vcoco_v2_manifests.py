"""Build split-preserving V-COCO manifests with factorized activity targets."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file
from hac.vcoco import factorize_actions

SPLITS = ("train", "val", "test")
TARGET_ACTIONS = ("sit", "stand", "walk", "run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcoco-root", type=Path, required=True)
    parser.add_argument("--coco-annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
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


def load_targets(path: Path) -> tuple[dict[int, set[str]], dict[int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = [row for row in payload if str(row["action_name"]) in TARGET_ACTIONS]
    if {str(row["action_name"]) for row in selected} != set(TARGET_ACTIONS):
        raise RuntimeError(f"V-COCO target actions are incomplete in {path}")

    positive: dict[int, set[str]] = defaultdict(set)
    image_by_annotation: dict[int, int] = {}
    reference_ids: list[int] | None = None
    for action in selected:
        annotation_ids = [int(value) for value in action["ann_id"]]
        image_ids = [int(value) for value in action["image_id"]]
        labels = [int(value) for value in action["label"]]
        if not (len(annotation_ids) == len(image_ids) == len(labels)):
            raise RuntimeError(f"V-COCO arrays differ for {action['action_name']}")
        if reference_ids is None:
            reference_ids = annotation_ids
        elif annotation_ids != reference_ids:
            raise RuntimeError("V-COCO actions do not share a person ordering")
        for annotation_id, image_id, label in zip(annotation_ids, image_ids, labels, strict=True):
            previous = image_by_annotation.setdefault(annotation_id, image_id)
            if previous != image_id:
                raise RuntimeError(f"Image drift for annotation {annotation_id}")
            if label == 1:
                positive[annotation_id].add(str(action["action_name"]))
            elif label != 0:
                raise RuntimeError(f"Unexpected V-COCO label {label}")
    return dict(positive), image_by_annotation


def load_coco_records(
    annotations_root: Path, requested_annotation_ids: set[int]
) -> tuple[dict[int, dict], dict[int, tuple[dict, str]], dict[str, str]]:
    annotations: dict[int, dict] = {}
    images: dict[int, tuple[dict, str]] = {}
    hashes = {}
    for partition in ("train2014", "val2014"):
        path = annotations_root / f"instances_{partition}.json"
        hashes[path.name] = sha256_file(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        partition_images = {int(row["id"]): row for row in payload["images"]}
        relevant_image_ids = set()
        for row in payload["annotations"]:
            annotation_id = int(row["id"])
            if annotation_id not in requested_annotation_ids:
                continue
            if annotation_id in annotations:
                raise RuntimeError(f"Duplicate COCO annotation id: {annotation_id}")
            annotations[annotation_id] = row
            relevant_image_ids.add(int(row["image_id"]))
        for image_id in relevant_image_ids:
            if image_id in images:
                raise RuntimeError(f"Duplicate COCO image id: {image_id}")
            images[image_id] = (partition_images[image_id], partition)
    missing = requested_annotation_ids - set(annotations)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} COCO person annotations")
    return annotations, images, hashes


def build_rows(
    split: str,
    targets: dict[int, set[str]],
    image_by_annotation: dict[int, int],
    annotations: dict[int, dict],
    images: dict[int, tuple[dict, str]],
    image_root: Path,
) -> list[dict]:
    labels_by_image: dict[int, set[str]] = defaultdict(set)
    for annotation_id, actions in targets.items():
        factor = factorize_actions(actions)
        if factor.legacy_eligible:
            labels_by_image[image_by_annotation[annotation_id]].add(factor.label_3)
        else:
            labels_by_image[image_by_annotation[annotation_id]].update({"sitting", "standing"})

    rows = []
    for annotation_id, actions in sorted(targets.items()):
        annotation = annotations[annotation_id]
        image_id = int(annotation["image_id"])
        if image_id != image_by_annotation[annotation_id]:
            raise RuntimeError(f"V-COCO/COCO image mismatch for {annotation_id}")
        if int(annotation["category_id"]) != 1:
            raise RuntimeError(f"V-COCO agent is not a COCO person: {annotation_id}")

        image, partition = images[image_id]
        x, y, width, height = (float(value) for value in annotation["bbox"])
        xmin = max(0.0, x)
        ymin = max(0.0, y)
        xmax = min(float(image["width"]), x + width)
        ymax = min(float(image["height"]), y + height)
        if xmax <= xmin or ymax <= ymin:
            raise RuntimeError(f"Invalid person box for annotation {annotation_id}")

        factor = factorize_actions(actions)
        file_name = str(image["file_name"])
        image_path = (image_root / partition / file_name).resolve()
        rows.append(
            {
                "person_id": f"{image_id}_{annotation_id}",
                "image_id": str(image_id),
                "annotation_id": str(annotation_id),
                "external_split": split,
                "coco_partition": partition,
                "file_name": file_name,
                "image_path": str(image_path),
                "image_url": f"http://images.cocodataset.org/{partition}/{file_name}",
                "image_present": image_path.is_file(),
                "source_actions": "|".join(sorted(actions)),
                "ontology_source": "vcoco_tags_factorized_not_human_harmonized",
                "posture_label": factor.posture,
                "motion_label": factor.motion,
                "gait_label": factor.gait,
                "label_4": factor.label_4,
                "label_3": factor.label_3,
                "legacy_eligible": factor.legacy_eligible,
                "factorized_clear": factor.factorized_clear,
                "image_level_unambiguous": len(labels_by_image[image_id]) == 1,
                "bbox_xmin": xmin,
                "bbox_ymin": ymin,
                "bbox_xmax": xmax,
                "bbox_ymax": ymax,
                "bbox_area_fraction": ((xmax - xmin) * (ymax - ymin))
                / (float(image["width"]) * float(image["height"])),
                "bbox_aspect_ratio": (xmax - xmin) / (ymax - ymin),
                "bbox_center_x_fraction": ((xmin + xmax) / 2.0) / float(image["width"]),
                "bbox_center_y_fraction": ((ymin + ymax) / 2.0) / float(image["height"]),
                "person_pixel_height": ymax - ymin,
                "image_width": int(image["width"]),
                "image_height": int(image["height"]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    vcoco_root = args.vcoco_root.resolve()
    annotation_root = args.coco_annotations.resolve()
    image_root = args.image_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_by_split = {}
    image_by_split = {}
    vcoco_hashes = {}
    for split in SPLITS:
        path = vcoco_root / "data" / "vcoco" / f"vcoco_{split}.json"
        target_by_split[split], image_by_split[split] = load_targets(path)
        vcoco_hashes[path.name] = sha256_file(path)

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            annotation_overlap = set(target_by_split[left]) & set(target_by_split[right])
            image_overlap = set(image_by_split[left].values()) & set(image_by_split[right].values())
            if annotation_overlap or image_overlap:
                raise RuntimeError(
                    f"Official V-COCO split overlap {left}/{right}: "
                    f"{len(annotation_overlap)} people, {len(image_overlap)} images"
                )

    requested = set().union(*(set(values) for values in target_by_split.values()))
    annotations, images, coco_hashes = load_coco_records(annotation_root, requested)

    split_frames = {}
    artifact_hashes = {}
    for split in SPLITS:
        rows = build_rows(
            split,
            target_by_split[split],
            image_by_split[split],
            annotations,
            images,
            image_root,
        )
        frame = pd.DataFrame(rows).sort_values(["image_id", "annotation_id"], ignore_index=True)
        split_frames[split] = frame
        path = output_dir / f"vcoco_{split}_persons.csv"
        frame.to_csv(path, index=False)
        artifact_hashes[path.name] = sha256_file(path)

    combined = pd.concat(split_frames.values(), ignore_index=True).sort_values(
        ["external_split", "image_id", "annotation_id"], ignore_index=True
    )
    combined_path = output_dir / "vcoco_all_persons.csv"
    combined.to_csv(combined_path, index=False)
    artifact_hashes[combined_path.name] = sha256_file(combined_path)

    split_summary = {}
    for split, frame in split_frames.items():
        eligible = frame[frame["legacy_eligible"]]
        split_summary[split] = {
            "all_target_positive_people": len(frame),
            "unique_images": int(frame["image_id"].nunique()),
            "legacy_eligible_people": len(eligible),
            "legacy_eligible_images": int(eligible["image_id"].nunique()),
            "legacy_class_counts": {
                str(key): int(value)
                for key, value in eligible["label_3"].value_counts().sort_index().items()
            },
            "source_action_combinations": dict(sorted(Counter(frame["source_actions"]).items())),
            "ambiguous_posture_people": int(frame["posture_label"].eq("ambiguous").sum()),
            "ambiguous_gait_people": int(frame["gait_label"].eq("ambiguous").sum()),
            "images_present": int(
                frame.drop_duplicates("image_id")["image_present"].astype(bool).sum()
            ),
        }

    summary = {
        "status": "VCOCO_V2_SPLITS_AND_SOURCE_ONTOLOGY_LOCK_INPUT_READY",
        "selection_policy": {
            "train": "adaptation_and_inner_model_selection",
            "val": "external_model_selection_and_calibration",
            "test": "single_confirmatory_evaluation_after_lock",
        },
        "test_annotation_counts_inspected": True,
        "test_images_or_predictions_inspected": False,
        "test_class_counts_forbidden_for_model_or_prior_selection": True,
        "ontology_status": "source_tags_factorized_not_human_harmonized",
        "official_split_image_overlap": 0,
        "official_split_person_overlap": 0,
        "vcoco_git_revision": git_revision(vcoco_root),
        "vcoco_annotation_sha256": vcoco_hashes,
        "coco_annotation_sha256": coco_hashes,
        "split_summary": split_summary,
        "artifact_sha256": artifact_hashes,
    }
    summary_path = output_dir / "vcoco_v2_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
