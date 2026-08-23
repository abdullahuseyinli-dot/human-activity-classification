"""Build a person-level, three-class V-COCO train/validation external-evaluation manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file

TARGET_TO_FOUR = {
    "sit": "sitting",
    "stand": "standing",
    "walk": "walking",
    "run": "running",
}
TARGET_TO_THREE = {
    "sit": "sitting",
    "stand": "standing",
    "walk": "walking_running",
    "run": "walking_running",
}


def resolve_target_actions(actions: set[str]) -> tuple[str | None, str | None]:
    """Map V-COCO's nested verbs to the mutually exclusive activity task.

    V-COCO commonly annotates ``stand`` together with ``walk`` or ``run``. The
    more specific locomotion verb takes precedence. Simultaneous ``sit`` and
    ``stand`` remains incompatible and is excluded.
    """

    dynamic = actions & {"walk", "run"}
    if dynamic:
        label_4 = TARGET_TO_FOUR[next(iter(dynamic))] if len(dynamic) == 1 else None
        return "walking_running", label_4
    if actions == {"sit", "stand"}:
        return None, None
    if actions == {"sit"}:
        return "sitting", "sitting"
    if actions == {"stand"}:
        return "standing", "standing"
    raise RuntimeError(f"Unexpected V-COCO target-action set: {sorted(actions)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcoco-root", type=Path, required=True)
    parser.add_argument("--coco-annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--vcoco-split", choices=["train", "val", "trainval"], default="trainval"
    )
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


def load_vcoco_targets(path: Path) -> tuple[dict[int, set[str]], dict[int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = [record for record in payload if record["action_name"] in TARGET_TO_THREE]
    if {record["action_name"] for record in selected} != set(TARGET_TO_THREE):
        raise RuntimeError("V-COCO release is missing a declared target action")
    positive: dict[int, set[str]] = defaultdict(set)
    image_by_annotation = {}
    reference_ids = None
    for record in selected:
        annotation_ids = [int(value) for value in record["ann_id"]]
        image_ids = [int(value) for value in record["image_id"]]
        labels = [int(value) for value in record["label"]]
        if not (len(annotation_ids) == len(image_ids) == len(labels)):
            raise RuntimeError(f"V-COCO arrays differ for action {record['action_name']}")
        if reference_ids is None:
            reference_ids = annotation_ids
        elif annotation_ids != reference_ids:
            raise RuntimeError("V-COCO target actions do not share an annotation order")
        for annotation_id, image_id, label in zip(
            annotation_ids, image_ids, labels, strict=True
        ):
            previous = image_by_annotation.setdefault(annotation_id, image_id)
            if previous != image_id:
                raise RuntimeError(f"V-COCO image id drift for annotation {annotation_id}")
            if label == 1:
                positive[annotation_id].add(str(record["action_name"]))
            elif label != 0:
                raise RuntimeError(f"Unexpected V-COCO label {label}")
    return dict(positive), image_by_annotation


def load_relevant_coco(
    annotations_root: Path, requested_annotation_ids: set[int]
) -> tuple[dict[int, dict], dict[int, tuple[dict, str]], dict[str, str]]:
    annotations = {}
    images = {}
    source_hashes = {}
    for partition in ("train2014", "val2014"):
        path = annotations_root / f"instances_{partition}.json"
        source_hashes[path.name] = sha256_file(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload["annotations"]:
            annotation_id = int(record["id"])
            if annotation_id in requested_annotation_ids:
                if annotation_id in annotations:
                    raise RuntimeError(f"COCO annotation id is not unique: {annotation_id}")
                annotations[annotation_id] = record
        partition_images = {int(record["id"]): record for record in payload["images"]}
        relevant_images = {
            int(record["image_id"])
            for record in annotations.values()
            if int(record["image_id"]) in partition_images
        }
        for image_id in relevant_images:
            record = partition_images[image_id]
            if image_id in images:
                raise RuntimeError(f"COCO image id is not unique across partitions: {image_id}")
            images[image_id] = (record, partition)
        del payload
    missing = requested_annotation_ids - set(annotations)
    if missing:
        raise RuntimeError(f"COCO annotations are missing {len(missing)} V-COCO person ids")
    return annotations, images, source_hashes


def main() -> None:
    args = parse_args()
    vcoco_root = args.vcoco_root.resolve()
    annotations_root = args.coco_annotations.resolve()
    image_root = args.image_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    vcoco_path = vcoco_root / "data" / "vcoco" / f"vcoco_{args.vcoco_split}.json"
    target_actions, vcoco_image_by_annotation = load_vcoco_targets(vcoco_path)
    coco_annotations, coco_images, coco_hashes = load_relevant_coco(
        annotations_root, set(target_actions)
    )

    action_labels_by_image: dict[int, set[str]] = defaultdict(set)
    for annotation_id, actions in target_actions.items():
        image_id = vcoco_image_by_annotation[annotation_id]
        label_3, _ = resolve_target_actions(actions)
        if label_3 is None:
            action_labels_by_image[image_id].update({"sitting", "standing"})
        else:
            action_labels_by_image[image_id].add(label_3)

    rows = []
    exclusions = defaultdict(int)
    for annotation_id, actions in sorted(target_actions.items()):
        annotation = coco_annotations[annotation_id]
        image_id = int(annotation["image_id"])
        if image_id != int(vcoco_image_by_annotation[annotation_id]):
            raise RuntimeError(f"V-COCO/COCO image mismatch for annotation {annotation_id}")
        if int(annotation["category_id"]) != 1:
            raise RuntimeError(f"V-COCO agent annotation is not a COCO person: {annotation_id}")
        label_3, label_4 = resolve_target_actions(actions)
        if label_3 is None:
            exclusions["incompatible_sit_and_stand"] += 1
            continue
        image_record, partition = coco_images[image_id]
        x, y, width, height = (float(value) for value in annotation["bbox"])
        xmin = max(0.0, x)
        ymin = max(0.0, y)
        xmax = min(float(image_record["width"]), x + width)
        ymax = min(float(image_record["height"]), y + height)
        if xmax <= xmin or ymax <= ymin:
            exclusions["invalid_person_box"] += 1
            continue
        file_name = str(image_record["file_name"])
        rows.append(
            {
                "person_id": f"{image_id}_{annotation_id}",
                "image_id": str(image_id),
                "annotation_id": str(annotation_id),
                "image_path": str((image_root / partition / file_name).resolve()),
                "image_url": f"http://images.cocodataset.org/{partition}/{file_name}",
                "file_name": file_name,
                "coco_partition": partition,
                "external_split": args.vcoco_split,
                "source_actions": "|".join(sorted(actions)),
                "label_4": label_4 or "",
                "label_3": label_3,
                "bbox_xmin": xmin,
                "bbox_ymin": ymin,
                "bbox_xmax": xmax,
                "bbox_ymax": ymax,
                "bbox_area_fraction": ((xmax - xmin) * (ymax - ymin))
                / (float(image_record["width"]) * float(image_record["height"])),
                "image_width": int(image_record["width"]),
                "image_height": int(image_record["height"]),
                "image_level_unambiguous": len(action_labels_by_image[image_id]) == 1,
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["image_id", "annotation_id"], ignore_index=True
    )
    frame.to_csv(output_dir / "vcoco_person_manifest_uninspected.csv", index=False)
    image_frame = frame.drop_duplicates("image_id").copy()
    provenance = {
        "status": "VCOCO_EXTERNAL_MANIFEST_UNINSPECTED",
        "selection_role": "none",
        "vcoco_split": args.vcoco_split,
        "vcoco_git_revision": git_revision(vcoco_root),
        "vcoco_annotation_sha256": sha256_file(vcoco_path),
        "coco_annotation_sha256": coco_hashes,
        "target_actions": sorted(TARGET_TO_THREE),
        "label_policy": {
            "walk_or_run_with_stand": "walking_running",
            "sit_with_stand": "exclude_incompatible",
            "walk_with_run": "walking_running_with_undefined_four_class_label",
        },
        "person_rows": len(frame),
        "unique_images": frame["image_id"].nunique(),
        "image_level_unambiguous_images": int(
            image_frame["image_level_unambiguous"].astype(bool).sum()
        ),
        "class_counts_person": frame["label_3"].value_counts().sort_index().to_dict(),
        "excluded_person_records": dict(sorted(exclusions.items())),
        "images_downloaded": 0,
        "model_predictions_read": 0,
        "polar_test_rows_read": 0,
        "test_used_for_selection": False,
    }
    (output_dir / "provenance_uninspected.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
