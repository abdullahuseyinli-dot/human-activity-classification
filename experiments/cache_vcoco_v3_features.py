"""Cache gated multiresolution and box-intervention features for V-COCO v3."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from hac.polar import image_view, sha256_file
from hac.vcoco_v3_representations import (
    BOX_PERTURBATIONS,
    build_feature_model,
    build_feature_transform,
    local_checkpoint_evidence,
    model_provenance,
    perturb_person_box,
)


class FeatureDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, *, view: str, perturbation: str, transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.view = view
        self.perturbation = perturbation
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = perturb_person_box(self.frame.iloc[index].to_dict(), self.perturbation)
        with Image.open(str(row["image_path"])) as image:
            crop = image_view(image.convert("RGB"), row, self.view)
            return self.transform(crop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nested-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/nested_stacks/summary.json"),
    )
    parser.add_argument(
        "--spatial-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/spatial/summary.json"),
    )
    parser.add_argument("--stage", choices=["spatial", "representation"], required=True)
    parser.add_argument(
        "--v2-lock",
        type=Path,
        default=Path(".runs/polar_v2/locked_protocol/vcoco_v2_protocol_lock.json"),
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path(".runs/polar_v2/locked_protocol/vcoco_train_clean.csv"),
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path(".runs/polar_v2/locked_protocol/vcoco_val_clean.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-kind", choices=["dinov2_base", "dinov3_base", "siglip2_base"], required=True
    )
    parser.add_argument(
        "--view",
        choices=[
            "full_frame",
            "person_tight",
            "person_context_10",
            "person_context_25",
            "person_context_50",
            "person_context_25_background_blur",
            "person_context_25_background_mask",
        ],
        required=True,
    )
    parser.add_argument(
        "--preprocess",
        choices=["aspect_preserving_pad", "official_square_resize"],
        required=True,
    )
    parser.add_argument("--image-size", type=int, choices=[224, 336, 448], required=True)
    parser.add_argument("--box-perturbation", choices=tuple(BOX_PERTURBATIONS), default="none")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def validate_gate(args: argparse.Namespace) -> tuple[dict, dict, dict | None]:
    nested = json.loads(args.nested_summary.resolve().read_text(encoding="utf-8"))
    if nested.get("status") != "VCOCO_V3_NESTED_CACHED_FUSION_DEVELOPMENT_COMPLETE":
        raise RuntimeError("Spatial and representation caches require the nested-fusion result")
    if nested.get("official_v2_test_rows_read") != 0:
        raise RuntimeError("The nested stage crossed the consumed-test boundary")
    spatial = None
    if args.stage == "spatial" and args.model_kind != "dinov2_base":
        raise RuntimeError("The spatial cache stage is limited to the declared DINOv2 model")
    if args.stage == "representation":
        spatial = json.loads(args.spatial_summary.resolve().read_text(encoding="utf-8"))
        if spatial.get("status") != "VCOCO_V3_SPATIAL_DEVELOPMENT_COMPLETE":
            raise RuntimeError("Representation caching requires the completed spatial stage")
        if spatial.get("official_v2_test_rows_read") != 0:
            raise RuntimeError("The spatial stage crossed the consumed-test boundary")
    lock = json.loads(args.v2_lock.resolve().read_text(encoding="utf-8"))
    for split, path in (("train", args.train_manifest), ("val", args.val_manifest)):
        if sha256_file(path.resolve()) != lock["artifact_sha256"][f"vcoco_{split}_clean.csv"]:
            raise RuntimeError(f"Locked {split} development manifest drift")
    return nested, lock, spatial


@torch.inference_mode()
def extract(model, loader, device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    for pixels in tqdm(loader, desc="extracting frozen features", unit="batch"):
        pixels = pixels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output.append(model(pixels).float().cpu().numpy())
    return np.concatenate(output).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Batch size must be positive and workers cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("V-COCO v3 feature extraction requires CUDA")
    _, _, spatial = validate_gate(args)
    frames = []
    for split, path in (("train", args.train_manifest), ("val", args.val_manifest)):
        frame = pd.read_csv(path.resolve(), dtype={"person_id": str, "image_id": str})
        frame["split"] = split
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if frame["person_id"].duplicated().any():
        raise RuntimeError("Development person identifiers are not unique")
    missing = frame[~frame["image_path"].map(lambda value: Path(str(value)).is_file())]
    if len(missing):
        raise FileNotFoundError(f"Development cohort is missing {len(missing)} images")

    transform = build_feature_transform(args.model_kind, args.preprocess, args.image_size)
    loader = DataLoader(
        FeatureDataset(
            frame,
            view=args.view,
            perturbation=args.box_perturbation,
            transform=transform,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    started = time.perf_counter()
    model = build_feature_model(args.model_kind).to(device)
    features = extract(model, loader, device)
    checkpoint = local_checkpoint_evidence(args.model_kind)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "features.npy"
    rows_path = output_dir / "rows.csv"
    np.save(features_path, features)
    row_columns = [
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
    frame[row_columns].to_csv(rows_path, index=False)
    provenance = {
        "status": "VCOCO_V3_GATED_DEVELOPMENT_FEATURE_CACHE_COMPLETE",
        "stage": args.stage,
        "model_kind": args.model_kind,
        "model": model_provenance(args.model_kind),
        "checkpoint": checkpoint,
        "view": args.view,
        "preprocess": args.preprocess,
        "image_size": args.image_size,
        "box_perturbation": args.box_perturbation,
        "people": len(frame),
        "source_images": int(frame["image_id"].nunique()),
        "feature_dimensions": int(features.shape[1]),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "runtime_seconds": time.perf_counter() - started,
        "official_v2_test_rows_read": 0,
        "official_v2_test_predictions_run": False,
        "source_sha256": {
            "nested_stage": sha256_file(args.nested_summary.resolve()),
            "train_manifest": sha256_file(args.train_manifest.resolve()),
            "val_manifest": sha256_file(args.val_manifest.resolve()),
            "feature_extractor_source": sha256_file(Path(__file__).resolve()),
            "representation_source": sha256_file(
                Path(__file__).resolve().parents[1] / "src/hac/vcoco_v3_representations.py"
            ),
        },
        "artifact_sha256": {
            rows_path.name: sha256_file(rows_path),
            features_path.name: sha256_file(features_path),
        },
    }
    if spatial is not None:
        provenance["source_sha256"]["spatial_stage"] = sha256_file(args.spatial_summary.resolve())
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
