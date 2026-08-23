"""Cache frozen pretrained features for POLAR train/validation view screening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import ConvNeXt_Small_Weights
from tqdm.auto import tqdm
from transformers import AutoModel

from hac.augmentations import build_eval_transform
from hac.polar import image_view, sha256_file
from hac.polar_models import DINO_MODEL_SPECS


class PolarFeatureDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, view: str) -> None:
        self.frame = frame.reset_index(drop=True)
        self.view = view
        self.transform = build_eval_transform()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as image:
            source = image.convert("RGB")
            pixels = self.transform(image_view(source, row, self.view))
        return {"pixel_values": pixels, "row": index}


class DinoFeatureModel(nn.Module):
    def __init__(self, model_id: str, revision: str, representation: str) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id, revision=revision)
        self.representation = representation

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=self.representation == "last4_cls_mean_patch",
        )
        if self.representation == "last4_cls_mean_patch":
            return official_multilayer_features(output.hidden_states, self.backbone.layernorm)
        if getattr(output, "pooler_output", None) is not None:
            return output.pooler_output
        return output.last_hidden_state[:, 0]


def official_multilayer_features(
    hidden_states: tuple[torch.Tensor, ...], layernorm: nn.Module
) -> torch.Tensor:
    """Match the official DINOv2 four-block linear-classifier representation."""

    if len(hidden_states) < 4:
        raise ValueError("DINOv2 multi-layer features require at least four hidden states")
    normalized = [layernorm(state) for state in hidden_states[-4:]]
    class_tokens = [state[:, 0] for state in normalized]
    mean_patch_token = normalized[-1][:, 1:].mean(dim=1)
    return torch.cat([*class_tokens, mean_patch_token], dim=1)


class ConvNeXtFeatureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        values = self.backbone.features(pixel_values)
        values = self.backbone.avgpool(values)
        values = self.backbone.classifier[0](values)
        return self.backbone.classifier[1](values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-kind", choices=["convnext_small", "dinov2_small", "dinov2_base"], required=True
    )
    parser.add_argument(
        "--view", choices=["full_frame", "person_context_10", "person_context_25"], required=True
    )
    parser.add_argument(
        "--representation",
        choices=["final_cls", "last4_cls_mean_patch"],
        default="final_cls",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def checkpoint_evidence(model_kind: str) -> dict:
    if model_kind in DINO_MODEL_SPECS:
        model_id = DINO_MODEL_SPECS[model_kind]["model_id"]
        revision = DINO_MODEL_SPECS[model_kind]["revision"]
        root = Path(
            snapshot_download(repo_id=model_id, revision=revision, local_files_only=True)
        ).resolve()
        files = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".safetensors", ".bin"}:
                files.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        return {"model_id": model_id, "revision": revision, "files": files}
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "convnext_small-0c510722.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Expected torchvision checkpoint is unavailable: {checkpoint}")
    return {
        "model_id": "torchvision/convnext_small",
        "revision": "ConvNeXt_Small_Weights.IMAGENET1K_V1",
        "files": [
            {
                "path": checkpoint.name,
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
            }
        ],
    }


def build_model(model_kind: str, representation: str) -> nn.Module:
    if model_kind == "convnext_small":
        if representation != "final_cls":
            raise ValueError("Multi-layer representation is available only for DINOv2")
        return ConvNeXtFeatureModel()
    model_id = DINO_MODEL_SPECS[model_kind]["model_id"]
    revision = DINO_MODEL_SPECS[model_kind]["revision"]
    return DinoFeatureModel(model_id, revision, representation)


@torch.inference_mode()
def extract(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    batches = []
    for batch in tqdm(loader, desc="caching frozen features", unit="batch"):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            features = model(pixels)
        batches.append(features.float().cpu().numpy())
    return np.concatenate(batches, axis=0)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = sha256_file(args.manifest)
    frame = pd.read_csv(args.manifest, dtype={"image_id": str})
    frame = frame.sort_values("image_id").reset_index(drop=True)
    if set(frame["split"].astype(str)) - {"train", "val"}:
        raise ValueError("Feature screening accepts only the development manifest")
    if frame["image_id"].duplicated().any():
        raise ValueError("Development image identifiers must be unique")

    provenance_path = output_dir / "provenance.json"
    features_path = output_dir / "features.npy"
    rows_path = output_dir / "rows.csv"
    expected = {
        "status": "DEVELOPMENT_FEATURE_CACHE",
        "model_kind": args.model_kind,
        "view": args.view,
        "representation": args.representation,
        "manifest_sha256": manifest_hash,
        "rows": len(frame),
        "test_rows": 0,
        "test_labels_read": False,
    }
    if features_path.is_file() and provenance_path.is_file() and rows_path.is_file():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        previous_representation = previous.get("representation", "final_cls")
        expected_without_representation = {
            key: value for key, value in expected.items() if key != "representation"
        }
        if previous_representation != args.representation or any(
            previous.get(key) != value for key, value in expected_without_representation.items()
        ):
            raise RuntimeError("Existing feature cache provenance differs from this request")
        features = np.load(features_path, mmap_mode="r")
        if features.shape[0] != len(frame):
            raise RuntimeError("Existing feature cache row count changed")
        print(json.dumps(previous, indent=2, sort_keys=True), flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        PolarFeatureDataset(frame, args.view),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = build_model(args.model_kind, args.representation).to(device)
    features = extract(model, loader, device)
    np.save(features_path, features)
    pd.DataFrame(
        {"row": np.arange(len(frame)), "image_id": frame["image_id"], "split": frame["split"]}
    ).to_csv(rows_path, index=False)
    provenance = {
        **expected,
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "device": str(device),
        "torch": torch.__version__,
        "checkpoint": checkpoint_evidence(args.model_kind),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
