"""Cache normalized COCO ground-truth pose features for a diagnostic V-COCO oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hac.polar import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--train-keypoints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_annotations(path: Path) -> tuple[dict[int, dict], list[tuple[int, int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    people = {
        int(annotation["id"]): annotation
        for annotation in payload["annotations"]
        if int(annotation.get("category_id", -1)) == 1
    }
    person_categories = [item for item in payload["categories"] if int(item["id"]) == 1]
    if len(person_categories) != 1:
        raise RuntimeError("COCO keypoint metadata does not contain one person category")
    skeleton = [
        (int(left) - 1, int(right) - 1) for left, right in person_categories[0]["skeleton"]
    ]
    return people, skeleton


def pose_features(row: pd.Series, annotation: dict, skeleton: list[tuple[int, int]]) -> np.ndarray:
    values = np.asarray(annotation["keypoints"], dtype=np.float32).reshape(-1, 3)
    if values.shape != (17, 3):
        raise RuntimeError(f"Unexpected keypoint shape for annotation {annotation['id']}")
    width = max(float(row["bbox_xmax"]) - float(row["bbox_xmin"]), 1.0)
    height = max(float(row["bbox_ymax"]) - float(row["bbox_ymin"]), 1.0)
    labelled = values[:, 2] > 0
    visible = values[:, 2] == 2
    x = np.zeros(17, dtype=np.float32)
    y = np.zeros(17, dtype=np.float32)
    x[labelled] = (values[labelled, 0] - float(row["bbox_xmin"])) / width
    y[labelled] = (values[labelled, 1] - float(row["bbox_ymin"])) / height
    base = np.column_stack([x, y, labelled.astype(np.float32), visible.astype(np.float32)])
    bones = []
    for left, right in skeleton:
        valid = bool(labelled[left] and labelled[right])
        delta_x = float(x[right] - x[left]) if valid else 0.0
        delta_y = float(y[right] - y[left]) if valid else 0.0
        bones.extend([delta_x, delta_y, float(np.hypot(delta_x, delta_y)), float(valid)])
    aggregate = np.asarray(
        [
            labelled.mean(),
            visible.mean(),
            float(x[labelled].mean()) if labelled.any() else 0.0,
            float(y[labelled].mean()) if labelled.any() else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([base.ravel(), np.asarray(bones, dtype=np.float32), aggregate])


def main() -> None:
    args = parse_args()
    lock_path = args.protocol_lock.resolve()
    train_path = args.train_manifest.resolve()
    val_path = args.val_manifest.resolve()
    keypoints_path = args.train_keypoints.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("Pose diagnostics require the locked V-COCO v2 protocol")
    if sha256_file(train_path) != lock["artifact_sha256"]["vcoco_train_clean.csv"]:
        raise RuntimeError("Locked train manifest drift")
    if sha256_file(val_path) != lock["artifact_sha256"]["vcoco_val_clean.csv"]:
        raise RuntimeError("Locked validation manifest drift")

    frames = []
    for split, path in (("train", train_path), ("val", val_path)):
        frame = pd.read_csv(path, dtype={"person_id": str, "image_id": str})
        if set(frame["coco_partition"].astype(str)) != {"train2014"}:
            raise RuntimeError(f"Unexpected COCO partition in V-COCO {split}")
        frame["split"] = split
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    if rows["person_id"].duplicated().any():
        raise RuntimeError("Development person identifiers are not unique")
    annotations, skeleton = load_annotations(keypoints_path)
    missing = [value for value in rows["annotation_id"].astype(int) if value not in annotations]
    if missing:
        raise RuntimeError(f"COCO keypoint annotations are missing {len(missing)} V-COCO people")
    features = np.stack(
        [
            pose_features(row, annotations[int(row["annotation_id"])], skeleton)
            for _, row in rows.iterrows()
        ]
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise RuntimeError("Pose feature cache contains non-finite values")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.npy"
    rows_path = output_dir / "rows.csv"
    np.save(feature_path, features)
    rows[
        [
            "person_id",
            "image_id",
            "split",
            "label_3",
            "posture_label",
            "motion_label",
            "gait_label",
            "bbox_area_fraction",
            "bbox_aspect_ratio",
            "bbox_center_x_fraction",
            "bbox_center_y_fraction",
            "person_pixel_height",
        ]
    ].to_csv(rows_path, index=False)
    visible_counts = features[:, 3::4][:, :17].sum(axis=1)
    provenance = {
        "status": "VCOCO_V2_DEVELOPMENT_FEATURE_CACHE_COMPLETE",
        "model_kind": "coco_gt_pose_oracle",
        "representation": "bbox_normalized_keypoints_visibility_and_skeleton",
        "view": "ground_truth_person_keypoints",
        "preprocess": "coco_keypoints_normalized_to_ground_truth_person_box",
        "image_size": 0,
        "rows": len(rows),
        "train_rows": int(rows["split"].eq("train").sum()),
        "val_rows": int(rows["split"].eq("val").sum()),
        "feature_dimensions": int(features.shape[1]),
        "people_with_no_labelled_keypoints": int((visible_counts == 0).sum()),
        "diagnostic_scope": (
            "ground-truth COCO pose oracle; not a deployable or clean external-transfer input"
        ),
        "test_rows_read": 0,
        "test_predictions_run": False,
        "protocol_lock_sha256": sha256_file(lock_path),
        "source_sha256": {
            "train_manifest": sha256_file(train_path),
            "val_manifest": sha256_file(val_path),
            "coco_person_keypoints_train2014": sha256_file(keypoints_path),
        },
        "artifact_sha256": {
            feature_path.name: sha256_file(feature_path),
            rows_path.name: sha256_file(rows_path),
        },
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
