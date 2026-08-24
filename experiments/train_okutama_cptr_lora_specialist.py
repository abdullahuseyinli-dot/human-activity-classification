"""Train the declared top-block DINOv2 LoRA centre-frame specialist on CUDA."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score, log_loss
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from hac.cptr_features import CPTR_STORE_STATUS, parse_window_boxes, parse_window_frames
from hac.polar import image_view, sha256_file
from hac.polar_training import warmup_cosine_scheduler
from hac.training import seed_everything
from hac.vcoco_v3_neural import (
    FactorizedClassifier,
    PairedPersonSafeTransform,
    configure_parameter_efficient_backbone,
    decode_factorized_logits,
    extract_backbone_features,
    parameter_count_summary,
)
from hac.vcoco_v3_representations import (
    build_feature_model,
    local_checkpoint_evidence,
    model_provenance,
)
from hac.vcoco_v3_temporal_training import hierarchical_class_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiments/okutama_cptr_protocol.json")
    )
    parser.add_argument(
        "--protocol-lock", type=Path, default=Path(".runs/cptr/protocol_lock.json")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/development_manifest.csv"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Path to the provider-supplied Okutama-Action training-frame archive",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".runs/cptr/lora_specialist-v2")
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--maximum-epochs", type=int, default=8)
    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def frame_member(recording_id: str, frame: int) -> str:
    drone, part_of_day, _ = recording_id.split(".")
    period = {"1": "Morning", "2": "Noon"}[part_of_day]
    return (
        f"Drone{drone}/{period}/Extracted-Frames-1280x720/"
        f"{recording_id}/{frame}.jpg"
    )


class CentreFrameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, archive: Path, *, transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.archive_path = archive
        self.transform = transform
        self.class_to_index = {"sitting": 0, "standing": 1, "walking_running": 2}
        self._archive: zipfile.ZipFile | None = None

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_archive"] = None
        return state

    def _zip(self) -> zipfile.ZipFile:
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.archive_path)
        return self._archive

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        frames = parse_window_frames(row["window_frames"])
        boxes = parse_window_boxes(row["window_boxes_1280x720"])
        centre = int(row["center_frame_index"])
        frame = int(frames[centre])
        x1, y1, x2, y2 = map(float, boxes[centre])
        member = frame_member(str(row["video_id"]), frame)
        with self._zip().open(member) as source, Image.open(io.BytesIO(source.read())) as image:
            rgb = image.convert("RGB")
            box = {
                "bbox_xmin": x1,
                "bbox_ymin": y1,
                "bbox_xmax": x2,
                "bbox_ymax": y2,
                "image_width": rgb.width,
                "image_height": rgb.height,
            }
            crop = image_view(rgb, box, "person_tight")
            pixels, _ = self.transform(crop, crop)
        return {
            "pixels": pixels,
            "label": self.class_to_index[str(row["label"])],
            "sample_id": str(row["sample_id"]),
            "feature_index": int(row["feature_index"]),
        }


class LoRASpecialist(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = FactorizedClassifier(hidden_dim, dropout)

    def forward(
        self, pixels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.backbone(pixel_values=pixels)
        features = extract_backbone_features(output)
        posture, motion = self.classifier(features)
        return decode_factorized_logits(posture, motion), posture, motion, features


def primary_loss(
    posture: torch.Tensor,
    motion: torch.Tensor,
    labels: torch.Tensor,
    posture_weight: torch.Tensor,
    motion_weight: torch.Tensor,
    *,
    label_smoothing: float,
) -> torch.Tensor:
    posture_loss = F.cross_entropy(
        posture,
        (labels != 0).long(),
        weight=posture_weight,
        label_smoothing=label_smoothing,
    )
    upright = labels != 0
    if torch.any(upright):
        motion_loss = F.cross_entropy(
            motion[upright],
            (labels[upright] == 2).long(),
            weight=motion_weight,
            label_smoothing=label_smoothing,
        )
    else:
        motion_loss = motion.sum() * 0.0
    return posture_loss + motion_loss


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    values = probabilities / probabilities.sum(axis=1, keepdims=True).clip(min=1e-12)
    predictions = values.argmax(axis=1)
    per_class = f1_score(labels, predictions, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "macro_f1": float(per_class.mean()),
        "accuracy": float(np.mean(predictions == labels)),
        "log_loss": float(log_loss(labels, values, labels=[0, 1, 2])),
        "sitting_f1": float(per_class[0]),
        "standing_f1": float(per_class[1]),
        "walking_running_f1": float(per_class[2]),
        "worst_class_f1": float(per_class.min()),
    }


@torch.inference_mode()
def evaluate(
    model: LoRASpecialist,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    labels = []
    probabilities = []
    sample_ids = []
    for batch in loader:
        pixels = batch["pixels"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda"):
            predicted, _, _, _ = model(pixels)
        labels.append(batch["label"].numpy())
        probabilities.append(predicted.float().cpu().numpy())
        sample_ids.extend(map(str, batch["sample_id"]))
    label_values = np.concatenate(labels)
    probability_values = np.concatenate(probabilities)
    return metrics(label_values, probability_values), {
        "sample_ids": np.asarray(sample_ids),
        "labels": label_values,
        "probabilities": probability_values,
    }


def main() -> None:
    args = parse_args()
    if args.workers < 0 or args.maximum_epochs < 3:
        raise ValueError("LoRA execution parameters are invalid")
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA specialist fitting requires CUDA; CPU fallback is disabled")
    protocol_path = args.protocol.resolve()
    lock_path = args.protocol_lock.resolve()
    manifest_path = args.manifest.resolve()
    archive_path = args.archive.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "OKUTAMA_CPTR_PROTOCOL_LOCKED_BEFORE_FIT":
        raise RuntimeError("The CPTR protocol is not locked")
    if lock["source_sha256"]["protocol"] != sha256_file(protocol_path):
        raise RuntimeError("The CPTR protocol changed after locking")
    if lock["source_sha256"]["development_manifest"] != sha256_file(manifest_path):
        raise RuntimeError("The development manifest changed after locking")
    if archive_path.stat().st_size != int(lock["archive_evidence"]["bytes"]):
        raise RuntimeError("The development archive byte count changed")
    frame = pd.read_csv(
        manifest_path,
        dtype={"sample_id": str, "recording_id": str, "track_id": str},
    )
    train_frame = frame[frame["split"].eq("train")].reset_index(drop=True)
    validation_frame = frame[frame["split"].eq("validation")].reset_index(drop=True)
    train_transform = PairedPersonSafeTransform(
        image_size=224,
        random_erasing_probability=0.10,
        training=True,
    )
    evaluation_transform = PairedPersonSafeTransform(image_size=224, training=False)
    specification = protocol["parameter_efficient_adaptation"]
    training = protocol["training"]
    seed_everything(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        CentreFrameDataset(train_frame, archive_path, transform=train_transform),
        batch_size=int(specification["batch_size"]),
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        generator=generator,
    )
    validation_loader = DataLoader(
        CentreFrameDataset(validation_frame, archive_path, transform=evaluation_transform),
        batch_size=32,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    wrapper = build_feature_model("dinov2_base")
    backbone = wrapper.backbone
    hidden_dim = int(backbone.config.hidden_size)
    replacements = configure_parameter_efficient_backbone(
        backbone,
        strategy="lora_top_blocks",
        top_blocks=int(specification["top_blocks"]),
        lora_rank=int(specification["lora_rank"]),
        lora_alpha=float(specification["lora_alpha"]),
        lora_dropout=float(specification["lora_dropout"]),
    )
    model = LoRASpecialist(backbone, hidden_dim, float(protocol["architecture"]["dropout"]))
    device = torch.device("cuda")
    model.to(device)
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats(device)
    labels = train_frame["label"].map(
        {"sitting": 0, "standing": 1, "walking_running": 2}
    ).to_numpy(dtype=int)
    posture_weight, motion_weight = hierarchical_class_weights(labels, device)
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]
    lora_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("backbone.")
    ]
    if not head_parameters or not lora_parameters:
        raise RuntimeError("The LoRA specialist lacks declared trainable parameter groups")
    optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": 3e-4, "weight_decay": 0.015},
            {"params": lora_parameters, "lr": 5e-5, "weight_decay": 0.015},
        ]
    )
    accumulation = int(specification["gradient_accumulation"])
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=optimizer_steps_per_epoch * args.maximum_epochs,
        warmup_fraction=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=4096.0)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "status": "OKUTAMA_CPTR_LORA_SPECIALIST_REQUEST",
        "seed": int(args.seed),
        "train_samples": int(len(train_frame)),
        "validation_samples": int(len(validation_frame)),
        "adaptation": specification,
        "maximum_epochs": int(args.maximum_epochs),
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "protocol": sha256_file(protocol_path),
            "protocol_lock": sha256_file(lock_path),
            "manifest": sha256_file(manifest_path),
            "archive": lock["archive_evidence"]["sha256"],
            "runner": sha256_file(Path(__file__).resolve()),
            "neural_module": sha256_file(
                Path(__file__).resolve().parents[1] / "src/hac/vcoco_v3_neural.py"
            ),
        },
    }
    request_hash = hashlib.sha256(
        json.dumps(request, sort_keys=True).encode("utf-8")
    ).hexdigest()
    write_json(output_dir / "request.json", {**request, "request_sha256": request_hash})
    history = []
    best = None
    stale = 0
    best_path = output_dir / "best_checkpoint.pt"
    started = time.perf_counter()
    for epoch in range(args.maximum_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(train_loader):
            pixels = batch["pixels"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda"):
                _, posture, motion, _ = model(pixels)
                loss = primary_loss(
                    posture,
                    motion,
                    target,
                    posture_weight,
                    motion_weight,
                    label_smoothing=float(training["label_smoothing"]),
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            losses.append(float(loss.detach()))
            boundary = (step + 1) % accumulation == 0 or step + 1 == len(train_loader)
            if boundary:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(training["gradient_clip_norm"]),
                    error_if_nonfinite=True,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        validation_metrics, _ = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "validation": validation_metrics,
        }
        history.append(row)
        write_json(output_dir / "history.json", history)
        print(json.dumps(row, sort_keys=True), flush=True)
        if best is None or validation_metrics["macro_f1"] > best["macro_f1"]:
            best = validation_metrics
            stale = 0
            torch.save(
                {
                    "request_sha256": request_hash,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_metrics": validation_metrics,
                },
                best_path,
            )
        else:
            stale += 1
        if epoch >= 2 and stale >= 2:
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics, predictions = evaluate(model, validation_loader, device)
    predictions_path = output_dir / "validation_predictions.npz"
    np.savez_compressed(predictions_path, **predictions)

    full_frame = frame.sort_values("feature_index", kind="stable").reset_index(drop=True)
    if not np.array_equal(full_frame["feature_index"].to_numpy(), np.arange(len(full_frame))):
        raise RuntimeError("The LoRA feature cache is not aligned to packed feature indices")
    full_loader = DataLoader(
        CentreFrameDataset(full_frame, archive_path, transform=evaluation_transform),
        batch_size=32,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    feature_path = output_dir / "features.npy"
    feature_store = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(full_frame), hidden_dim),
    )
    model.eval()
    with torch.inference_mode():
        for batch in full_loader:
            pixels = batch["pixels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda"):
                _, _, _, features = model(pixels)
            indices = batch["feature_index"].numpy()
            feature_store[indices] = features.float().cpu().numpy().astype(np.float16)
    feature_store.flush()
    del feature_store
    store_path = output_dir / "store.json"
    store = {
        "status": CPTR_STORE_STATUS,
        "store_kind": "supervised_dinov2_top_block_lora_centre_features",
        "samples": int(len(full_frame)),
        "feature_dim": hidden_dim,
        "model": model_provenance("dinov2_base"),
        "checkpoint": local_checkpoint_evidence("dinov2_base"),
        "lora_replacements": replacements,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "arrays": {
            "features": {"path": feature_path.name, "sha256": sha256_file(feature_path)}
        },
        "source_checkpoint_sha256": sha256_file(best_path),
    }
    write_json(store_path, store)
    summary = {
        "status": "OKUTAMA_CPTR_LORA_SPECIALIST_COMPLETE",
        "seed": int(args.seed),
        "best_epoch": int(checkpoint["epoch"]),
        "validation_metrics": validation_metrics,
        "model_parameters": parameter_count_summary(model),
        "lora_replacements": replacements,
        "training_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "runtime_seconds": time.perf_counter() - started,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "request_sha256": request_hash,
        "artifact_sha256": {
            "best_checkpoint.pt": sha256_file(best_path),
            "history.json": sha256_file(output_dir / "history.json"),
            "validation_predictions.npz": sha256_file(predictions_path),
            "store.json": sha256_file(store_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
