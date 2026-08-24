"""Cache confidence-weighted DINOv2 body-region tokens for CPTR on CUDA."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from hac.cptr_features import BODY_REGION_NAMES, CPTR_STORE_STATUS, parse_window_boxes
from hac.polar import image_view, sha256_file
from hac.vcoco_v3_representations import (
    build_feature_model,
    build_feature_transform,
    local_checkpoint_evidence,
    model_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiments/okutama_cptr_protocol.json")
    )
    parser.add_argument(
        "--protocol-lock", type=Path, default=Path(".runs/cptr/protocol_lock.json")
    )
    parser.add_argument(
        "--centres",
        type=Path,
        default=Path(".runs/vcoco_v3/okutama/development_audit/development_centres.csv"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Path to the provider-supplied Okutama-Action training-frame archive",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(".runs/cptr/part_features"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    return parser.parse_args()


def frame_member(recording_id: str, frame: int) -> str:
    drone, part_of_day, _ = recording_id.split(".")
    period = {"1": "Morning", "2": "Noon"}[part_of_day]
    return (
        f"Drone{drone}/{period}/Extracted-Frames-1280x720/"
        f"{recording_id}/{frame}.jpg"
    )


def parse_frames(value: object) -> list[int]:
    return [int(item) for item in str(value).split(";")]


class PartFrameDataset(Dataset):
    def __init__(
        self,
        archive_path: Path,
        centres: pd.DataFrame,
        pending: np.ndarray,
        *,
        transform,
    ) -> None:
        self.archive_path = archive_path
        self.centres = centres.reset_index(drop=True)
        self.pending = np.asarray(pending, dtype=np.int64)
        self.transform = transform
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
        return len(self.pending)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | float]:
        flat_index = int(self.pending[index])
        sample_index, time_index = divmod(flat_index, 17)
        row = self.centres.iloc[sample_index]
        frames = parse_frames(row["window_frames"])
        boxes = parse_window_boxes(row["window_boxes_1280x720"])
        frame = frames[time_index]
        x1, y1, x2, y2 = map(float, boxes[time_index])
        member = frame_member(str(row["recording_id"]), frame)
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
            pixels = self.transform(image_view(rgb, box, "person_tight"))
        occluded_values = [int(item) for item in str(row["window_occluded"]).split(";")]
        return {
            "pixels": pixels,
            "sample_index": sample_index,
            "time_index": time_index,
            "box_width": x2 - x1,
            "box_height": y2 - y1,
            "occluded": occluded_values[time_index],
        }


def anatomical_region_weights(
    box_width: torch.Tensor,
    box_height: torch.Tensor,
    *,
    grid_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return soft body-region queries over aspect-preserving DINO patch grids."""

    batch = len(box_width)
    axis = (torch.arange(grid_size, device=device, dtype=torch.float32) + 0.5) / grid_size
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    grid_x = grid_x.reshape(1, -1).expand(batch, -1)
    grid_y = grid_y.reshape(1, -1).expand(batch, -1)
    ratio = box_width.to(device=device, dtype=torch.float32) / box_height.to(
        device=device, dtype=torch.float32
    ).clamp_min(1e-6)
    content_width = torch.where(ratio >= 1.0, torch.ones_like(ratio), ratio)
    content_height = torch.where(ratio >= 1.0, 1.0 / ratio, torch.ones_like(ratio))
    content_width = content_width.clamp(max=1.0)
    content_height = content_height.clamp(max=1.0)
    start_x = 0.5 * (1.0 - content_width)
    start_y = 0.5 * (1.0 - content_height)
    body_x = (grid_x - start_x[:, None]) / content_width[:, None].clamp_min(1e-6)
    body_y = (grid_y - start_y[:, None]) / content_height[:, None].clamp_min(1e-6)
    content = (
        (body_x >= 0.0) & (body_x <= 1.0) & (body_y >= 0.0) & (body_y <= 1.0)
    )
    centres = torch.tensor(
        (
            (0.50, 0.13),
            (0.50, 0.37),
            (0.50, 0.58),
            (0.25, 0.40),
            (0.75, 0.40),
            (0.30, 0.80),
            (0.70, 0.80),
        ),
        device=device,
    )
    scales = torch.tensor(
        (
            (0.32, 0.14),
            (0.38, 0.18),
            (0.34, 0.12),
            (0.22, 0.22),
            (0.22, 0.22),
            (0.22, 0.23),
            (0.22, 0.23),
        ),
        device=device,
    )
    distance = (
        (body_x[:, None] - centres[None, :, 0, None]) / scales[None, :, 0, None]
    ).square() + (
        (body_y[:, None] - centres[None, :, 1, None]) / scales[None, :, 1, None]
    ).square()
    weights = torch.exp(-0.5 * distance) * content[:, None]
    empty = weights.sum(dim=2) <= 1e-8
    if torch.any(empty):
        nearest = distance.argmin(dim=2)
        weights = weights.clone()
        rows, regions = torch.where(empty)
        weights[rows, regions, nearest[rows, regions]] = 1.0
    return weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.checkpoint_interval < 1:
        raise ValueError("Part-cache execution parameters are invalid")
    if not torch.cuda.is_available():
        raise RuntimeError("CPTR part-token extraction requires CUDA")
    protocol_path = args.protocol.resolve()
    lock_path = args.protocol_lock.resolve()
    centres_path = args.centres.resolve()
    archive_path = args.archive.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "OKUTAMA_CPTR_PROTOCOL_LOCKED_BEFORE_FIT":
        raise RuntimeError("The CPTR protocol is not locked")
    if lock["source_sha256"]["protocol"] != sha256_file(protocol_path):
        raise RuntimeError("The CPTR protocol changed after locking")
    if lock["source_sha256"]["development_centres"] != sha256_file(centres_path):
        raise RuntimeError("The provider centre manifest changed after locking")
    if archive_path.stat().st_size != int(lock["archive_evidence"]["bytes"]):
        raise RuntimeError("The development archive byte count changed")
    if tuple(protocol["input"]["body_regions"]) != BODY_REGION_NAMES:
        raise RuntimeError("The declared CPTR body regions changed")
    centres = pd.read_csv(centres_path, dtype={"sample_id": str, "recording_id": str})
    samples, frames = len(centres), 17
    if samples != int(lock["development_samples"]):
        raise RuntimeError("The CPTR development centre count changed")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request_core = {
        "status": "OKUTAMA_CPTR_PART_CACHE_REQUEST",
        "samples": samples,
        "frames_per_sample": frames,
        "body_regions": list(BODY_REGION_NAMES),
        "model": model_provenance("dinov2_base"),
        "image_size": 224,
        "preprocess": "aspect_preserving_pad",
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "protocol": sha256_file(protocol_path),
            "protocol_lock": sha256_file(lock_path),
            "centres": sha256_file(centres_path),
            "archive": lock["archive_evidence"]["sha256"],
            "extractor": sha256_file(Path(__file__).resolve()),
        },
    }
    request_hash = hashlib.sha256(
        json.dumps(request_core, sort_keys=True).encode("utf-8")
    ).hexdigest()
    request_path = output_dir / "request.json"
    if request_path.is_file():
        previous = json.loads(request_path.read_text(encoding="utf-8"))
        if previous.get("request_sha256") != request_hash:
            raise RuntimeError("The part-cache directory contains a different request")
    else:
        write_json(request_path, {**request_core, "request_sha256": request_hash})

    completed_path = output_dir / "completed.npy"
    completed = (
        np.load(completed_path)
        if completed_path.is_file()
        else np.zeros((samples, frames), dtype=bool)
    )
    if completed.shape != (samples, frames):
        raise RuntimeError("The resumable part-cache mask changed shape")
    pending = np.flatnonzero(~completed.ravel())
    transform = build_feature_transform("dinov2_base", "aspect_preserving_pad", 224)
    loader = DataLoader(
        PartFrameDataset(archive_path, centres, pending, transform=transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    wrapper = build_feature_model("dinov2_base").to(device).eval()
    backbone = wrapper.backbone
    started = time.perf_counter()
    token_path = output_dir / "part_tokens.npy"
    confidence_path = output_dir / "part_confidence.npy"
    token_store = confidence_store = None
    if token_path.is_file() and confidence_path.is_file():
        token_store = np.load(token_path, mmap_mode="r+")
        confidence_store = np.load(confidence_path, mmap_mode="r+")
        if token_store.shape[:3] != (samples, frames, len(BODY_REGION_NAMES)):
            raise RuntimeError("The resumable body-token store changed shape")
        if confidence_store.shape != token_store.shape[:3]:
            raise RuntimeError("The body-token confidence store does not align")

    for batch_index, batch in enumerate(
        tqdm(loader, desc="DINO body regions", unit="batch")
    ):
        pixels = batch["pixels"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda"):
            output = backbone(pixel_values=pixels)
            patch_tokens = output.last_hidden_state[:, 1:]
        patch_count = patch_tokens.shape[1]
        grid_size = int(round(patch_count**0.5))
        if grid_size * grid_size != patch_count:
            raise RuntimeError("DINO patch tokens do not form a square grid")
        weights = anatomical_region_weights(
            batch["box_width"],
            batch["box_height"],
            grid_size=grid_size,
            device=device,
        )
        pooled = torch.einsum("brp,bpd->brd", weights, patch_tokens.float())
        resolution = torch.sqrt(
            batch["box_width"].float() * batch["box_height"].float()
        )
        resolution = (resolution / 64.0).clamp(0.05, 1.0)
        visibility = torch.where(
            batch["occluded"].bool(),
            torch.full_like(resolution, 0.35),
            torch.ones_like(resolution),
        )
        confidence = (resolution * visibility)[:, None].expand(-1, len(BODY_REGION_NAMES))
        if token_store is None:
            token_store = np.lib.format.open_memmap(
                token_path,
                mode="w+",
                dtype=np.float16,
                shape=(samples, frames, len(BODY_REGION_NAMES), pooled.shape[-1]),
            )
            confidence_store = np.lib.format.open_memmap(
                confidence_path,
                mode="w+",
                dtype=np.float16,
                shape=(samples, frames, len(BODY_REGION_NAMES)),
            )
        sample_index = batch["sample_index"].numpy()
        time_index = batch["time_index"].numpy()
        token_store[sample_index, time_index] = pooled.cpu().numpy().astype(np.float16)
        confidence_store[sample_index, time_index] = confidence.numpy().astype(np.float16)
        completed[sample_index, time_index] = True
        if (batch_index + 1) % args.checkpoint_interval == 0:
            token_store.flush()
            confidence_store.flush()
            np.save(completed_path, completed)
    if token_store is None or confidence_store is None or not completed.all():
        raise RuntimeError("CPTR body-region extraction ended incomplete")
    token_store.flush()
    confidence_store.flush()
    np.save(completed_path, completed)
    part_token_dim = int(token_store.shape[-1])
    mean_confidence = float(np.asarray(confidence_store, dtype=np.float32).mean())
    del wrapper, backbone, token_store, confidence_store
    torch.cuda.empty_cache()
    gc.collect()
    checkpoint = local_checkpoint_evidence("dinov2_base")
    arrays = {
        "part_tokens": token_path,
        "part_confidence": confidence_path,
    }
    summary = {
        "status": CPTR_STORE_STATUS,
        "store_kind": "confidence_weighted_dinov2_body_region_tokens",
        "samples": samples,
        "frames_per_sample": frames,
        "body_regions": list(BODY_REGION_NAMES),
        "part_token_dim": part_token_dim,
        "mean_part_confidence": mean_confidence,
        "model": model_provenance("dinov2_base"),
        "checkpoint": checkpoint,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "runtime_seconds": time.perf_counter() - started,
        "request_sha256": request_hash,
        "arrays": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in arrays.items()
        },
        "artifact_sha256": {
            completed_path.name: sha256_file(completed_path),
        },
    }
    write_json(output_dir / "store.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
