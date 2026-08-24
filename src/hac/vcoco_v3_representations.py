"""Pinned representations and deterministic box interventions for V-COCO v3."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from hac.augmentations import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_aspect_preserving_eval_transform,
)
from hac.polar import sha256_file
from hac.polar_features import PinnedDinoFeatureModel

DINO_V3_MODEL_SPEC = {
    "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "revision": "5931719e67bbdb9737e363e781fb0c67687896bc",
    "representation": "pooler_output",
    "access": "manually_gated",
}

SIGLIP2_MODEL_SPEC = {
    "model_id": "google/siglip2-base-patch16-224",
    "revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    "image_mean": (0.5, 0.5, 0.5),
    "image_std": (0.5, 0.5, 0.5),
}

BOX_PERTURBATIONS = {
    "none": (0.0, 0.0, 1.0),
    "shift_left_05": (-0.05, 0.0, 1.0),
    "shift_right_05": (0.05, 0.0, 1.0),
    "shift_up_05": (0.0, -0.05, 1.0),
    "shift_down_05": (0.0, 0.05, 1.0),
    "scale_090": (0.0, 0.0, 0.90),
    "scale_110": (0.0, 0.0, 1.10),
}


class PinnedAutoFeatureModel(nn.Module):
    def __init__(self, model_id: str, revision: str) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_id, revision=revision)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=pixel_values)
        if getattr(output, "pooler_output", None) is not None:
            return output.pooler_output
        return output.last_hidden_state[:, 0]


class PinnedSiglip2FeatureModel(nn.Module):
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


def build_feature_model(model_kind: str) -> nn.Module:
    if model_kind == "dinov2_base":
        return PinnedDinoFeatureModel("dinov2_base", "final_cls")
    if model_kind == "dinov3_base":
        return PinnedAutoFeatureModel(
            DINO_V3_MODEL_SPEC["model_id"], DINO_V3_MODEL_SPEC["revision"]
        )
    if model_kind == "siglip2_base":
        return PinnedSiglip2FeatureModel()
    raise ValueError(f"Unknown representation: {model_kind}")


def model_provenance(model_kind: str) -> dict:
    if model_kind == "dinov3_base":
        return dict(DINO_V3_MODEL_SPEC)
    if model_kind == "siglip2_base":
        return {
            "model_id": SIGLIP2_MODEL_SPEC["model_id"],
            "revision": SIGLIP2_MODEL_SPEC["revision"],
            "representation": "pooler_output",
        }
    if model_kind == "dinov2_base":
        from hac.polar_models import DINO_MODEL_SPECS

        return {**DINO_MODEL_SPECS[model_kind], "representation": "final_cls"}
    raise ValueError(f"Unknown representation: {model_kind}")


def local_checkpoint_evidence(model_kind: str) -> dict:
    """Hash the pinned local configuration and weight files after model loading."""

    from huggingface_hub import snapshot_download

    specification = model_provenance(model_kind)
    root = Path(
        snapshot_download(
            repo_id=specification["model_id"],
            revision=specification["revision"],
            local_files_only=True,
        )
    )
    files = []
    weight_count = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        is_weight = path.suffix in {".safetensors", ".bin"}
        is_configuration = path.suffix == ".json"
        if not is_weight and not is_configuration:
            continue
        weight_count += int(is_weight)
        files.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
                "role": "weight" if is_weight else "configuration",
            }
        )
    if not weight_count:
        raise RuntimeError(f"The local {model_kind} snapshot does not contain checkpoint weights")
    return {
        "model_id": specification["model_id"],
        "revision": specification["revision"],
        "files": files,
        "total_bytes": int(sum(item["bytes"] for item in files)),
    }


def build_feature_transform(model_kind: str, preprocess: str, image_size: int):
    if preprocess not in {"aspect_preserving_pad", "official_square_resize"}:
        raise ValueError(f"Unknown preprocessing policy: {preprocess}")
    if model_kind == "siglip2_base":
        mean = SIGLIP2_MODEL_SPEC["image_mean"]
        std = SIGLIP2_MODEL_SPEC["image_std"]
        fill = (128, 128, 128)
    else:
        mean, std, fill = IMAGENET_MEAN, IMAGENET_STD, (124, 116, 104)
    if preprocess == "aspect_preserving_pad":
        return build_aspect_preserving_eval_transform(image_size, mean=mean, std=std, fill=fill)
    return transforms.Compose(
        [
            transforms.Resize(
                (int(image_size), int(image_size)), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def perturb_person_box(row: Mapping, perturbation: str) -> dict:
    """Apply a declared translation or scale intervention and clip it to the image."""

    if perturbation not in BOX_PERTURBATIONS:
        raise ValueError(f"Unknown box perturbation: {perturbation}")
    shift_x, shift_y, scale = BOX_PERTURBATIONS[perturbation]
    x1, y1, x2, y2 = (
        float(row["bbox_xmin"]),
        float(row["bbox_ymin"]),
        float(row["bbox_xmax"]),
        float(row["bbox_ymax"]),
    )
    width, height = x2 - x1, y2 - y1
    center_x = 0.5 * (x1 + x2) + shift_x * width
    center_y = 0.5 * (y1 + y2) + shift_y * height
    width *= scale
    height *= scale
    image_width = float(row["image_width"])
    image_height = float(row["image_height"])
    output = dict(row)
    output["bbox_xmin"] = max(0.0, center_x - 0.5 * width)
    output["bbox_ymin"] = max(0.0, center_y - 0.5 * height)
    output["bbox_xmax"] = min(image_width, center_x + 0.5 * width)
    output["bbox_ymax"] = min(image_height, center_y + 0.5 * height)
    if output["bbox_xmax"] - output["bbox_xmin"] < 1.0:
        raise ValueError("Perturbed box is narrower than one pixel")
    if output["bbox_ymax"] - output["bbox_ymin"] < 1.0:
        raise ValueError("Perturbed box is shorter than one pixel")
    output["input_bbox_area_fraction"] = (
        (output["bbox_xmax"] - output["bbox_xmin"])
        * (output["bbox_ymax"] - output["bbox_ymin"])
        / (image_width * image_height)
    )
    output["box_perturbation"] = perturbation
    return output


def aggregate_perturbation_probabilities(
    probabilities: Mapping[str, np.ndarray],
    *,
    method: str,
) -> np.ndarray:
    """Aggregate matched box-intervention predictions without selecting on a holdout."""

    if set(probabilities) != set(BOX_PERTURBATIONS):
        raise ValueError("Robust aggregation requires every declared perturbation")
    values = np.stack([np.asarray(probabilities[name]) for name in BOX_PERTURBATIONS], axis=0)
    if method == "mean_probability":
        output = values.mean(axis=0)
    elif method == "mean_log_probability":
        output = np.exp(np.log(np.clip(values, 1e-8, 1.0)).mean(axis=0))
    else:
        raise ValueError(f"Unknown perturbation aggregation: {method}")
    return output / output.sum(axis=1, keepdims=True)
