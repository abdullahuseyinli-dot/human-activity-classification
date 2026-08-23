"""Cache frozen person-level features for V-COCO v2 development splits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import ConvNeXt_Small_Weights
from tqdm.auto import tqdm

from hac.augmentations import build_aspect_preserving_eval_transform, build_eval_transform
from hac.polar import image_view, sha256_file
from hac.polar_features import PinnedDinoFeatureModel

SIGLIP2_MODEL_SPEC = {
    "model_id": "google/siglip2-base-patch16-224",
    "revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    "image_mean": (0.5, 0.5, 0.5),
    "image_std": (0.5, 0.5, 0.5),
}


class FeatureDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, *, view: str, transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.view = str(view)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        from PIL import Image

        row = self.frame.iloc[index]
        path = str(row["image_path"])
        with Image.open(path) as image:
            pixels = self.transform(image_view(image.convert("RGB"), row, self.view))
        return {"pixel_values": pixels, "person_id": str(row["person_id"])}


class ConvNeXtFeatureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = models.convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        values = self.model.features(pixel_values)
        values = self.model.avgpool(values)
        values = self.model.classifier[0](values)
        return self.model.classifier[1](values)


class Siglip2FeatureModel(nn.Module):
    """Revision-pinned SigLIP2 vision encoder without the unused text tower."""

    def __init__(self) -> None:
        super().__init__()
        from transformers import SiglipVisionModel

        self.model = SiglipVisionModel.from_pretrained(
            SIGLIP2_MODEL_SPEC["model_id"], revision=SIGLIP2_MODEL_SPEC["revision"]
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=pixel_values)
        if output.pooler_output is not None:
            return output.pooler_output
        return output.last_hidden_state.mean(dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-kind",
        choices=["dinov2_base", "convnext_small", "siglip2_base"],
        required=True,
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
        "--preprocess", choices=["legacy_center_crop", "aspect_preserving_pad"], required=True
    )
    parser.add_argument("--image-size", type=int, choices=[224, 336, 448], default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def validate_inputs(lock_path: Path, paths: dict[str, Path]) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("Feature extraction requires the locked V-COCO v2 protocol")
    for split, path in paths.items():
        if sha256_file(path) != lock["artifact_sha256"][f"vcoco_{split}_clean.csv"]:
            raise RuntimeError(f"Locked {split} manifest drift")
    if lock["test_access"]["model_predictions_run"]:
        raise RuntimeError("Protocol reports prior target-test model access")
    return lock


def build_model(model_kind: str) -> nn.Module:
    if model_kind == "dinov2_base":
        return PinnedDinoFeatureModel("dinov2_base", "final_cls")
    if model_kind == "convnext_small":
        return ConvNeXtFeatureModel()
    if model_kind == "siglip2_base":
        return Siglip2FeatureModel()
    raise ValueError(f"Unknown model kind: {model_kind}")


def build_transform(model_kind: str, preprocess: str, image_size: int):
    if preprocess == "legacy_center_crop":
        if model_kind == "siglip2_base":
            raise ValueError("SigLIP2 is evaluated with its declared aspect-preserving normalization")
        return build_eval_transform(image_size)
    if model_kind == "siglip2_base":
        return build_aspect_preserving_eval_transform(
            image_size,
            mean=SIGLIP2_MODEL_SPEC["image_mean"],
            std=SIGLIP2_MODEL_SPEC["image_std"],
            fill=(128, 128, 128),
        )
    return build_aspect_preserving_eval_transform(image_size)


def model_provenance(model_kind: str) -> dict[str, str]:
    if model_kind == "dinov2_base":
        from hac.polar_models import DINO_MODEL_SPECS

        return dict(DINO_MODEL_SPECS[model_kind])
    if model_kind == "siglip2_base":
        return {
            "model_id": str(SIGLIP2_MODEL_SPEC["model_id"]),
            "revision": str(SIGLIP2_MODEL_SPEC["revision"]),
        }
    if model_kind == "convnext_small":
        return {"model_id": "torchvision/convnext_small", "revision": "IMAGENET1K_V1"}
    raise ValueError(f"Unknown model kind: {model_kind}")


@torch.inference_mode()
def extract(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    values = []
    for batch in tqdm(loader, desc="extracting frozen features", unit="batch"):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            features = model(pixels)
        values.append(features.float().cpu().numpy())
    return np.concatenate(values).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch size must be positive and workers cannot be negative")
    train_path = args.train_manifest.resolve()
    val_path = args.val_manifest.resolve()
    lock_path = args.protocol_lock.resolve()
    validate_inputs(lock_path, {"train": train_path, "val": val_path})
    frames = []
    for split, path in (("train", train_path), ("val", val_path)):
        frame = pd.read_csv(path, dtype={"person_id": str, "image_id": str})
        frame["split"] = split
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if frame["person_id"].duplicated().any():
        raise RuntimeError("Development person identifiers are not unique")
    missing_images = frame.loc[
        ~frame["image_path"].map(lambda value: Path(str(value)).is_file()), "image_path"
    ]
    if len(missing_images):
        raise RuntimeError(f"Development cohort is missing {len(missing_images)} images")

    transform = build_transform(args.model_kind, args.preprocess, args.image_size)
    loader = DataLoader(
        FeatureDataset(frame, view=args.view, transform=transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    model = build_model(args.model_kind).to(device)
    started = time.perf_counter()
    features = extract(model, loader, device)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.npy"
    rows_path = output_dir / "rows.csv"
    np.save(feature_path, features)
    frame[
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
    provenance = {
        "status": "VCOCO_V2_DEVELOPMENT_FEATURE_CACHE_COMPLETE",
        "model_kind": args.model_kind,
        "model": model_provenance(args.model_kind),
        "representation": "final_cls" if args.model_kind == "dinov2_base" else "pooled",
        "view": args.view,
        "preprocess": args.preprocess,
        "image_size": args.image_size,
        "rows": len(frame),
        "train_rows": int(frame["split"].eq("train").sum()),
        "val_rows": int(frame["split"].eq("val").sum()),
        "feature_dimensions": int(features.shape[1]),
        "runtime_seconds": time.perf_counter() - started,
        "device": str(device),
        "test_rows_read": 0,
        "test_predictions_run": False,
        "protocol_lock_sha256": sha256_file(lock_path),
        "source_sha256": {
            "train_manifest": sha256_file(train_path),
            "val_manifest": sha256_file(val_path),
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
