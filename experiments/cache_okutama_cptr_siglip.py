"""Cache pinned frozen SigLIP centre-frame person features for CPTR on CUDA."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, SiglipVisionModel

from hac.cptr_features import CPTR_STORE_STATUS, parse_window_boxes, parse_window_frames
from hac.polar import image_view, sha256_file
from hac.vcoco_v3_neural import extract_backbone_features

MODEL_ID = "google/siglip-base-patch16-224"
MODEL_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"


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
    parser.add_argument("--output-dir", type=Path, default=Path(".runs/cptr/siglip_features"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def frame_member(video_id: str, frame: int) -> str:
    drone, part_of_day, _ = video_id.split(".")
    period = {"1": "Morning", "2": "Noon"}[part_of_day]
    return (
        f"Drone{drone}/{period}/Extracted-Frames-1280x720/"
        f"{video_id}/{frame}.jpg"
    )


class SiglipCentreDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, archive: Path, processor) -> None:
        self.frame = frame.reset_index(drop=True)
        self.archive_path = archive
        self.processor = processor
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
        x1, y1, x2, y2 = map(float, boxes[centre])
        member = frame_member(str(row["video_id"]), int(frames[centre]))
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
            pixels = self.processor(images=crop, return_tensors="pt")["pixel_values"][0]
        return {
            "pixels": pixels,
            "feature_index": int(row["feature_index"]),
        }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("SigLIP cache execution parameters are invalid")
    if not torch.cuda.is_available():
        raise RuntimeError("SigLIP extraction requires CUDA; CPU fallback is disabled")
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
    ).sort_values("feature_index", kind="stable")
    if not np.array_equal(frame["feature_index"].to_numpy(), np.arange(len(frame))):
        raise RuntimeError("SigLIP rows are not aligned to packed feature indices")

    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=("config.json", "preprocessor_config.json", "model.safetensors"),
        )
    ).resolve()
    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = SiglipVisionModel.from_pretrained(snapshot, local_files_only=True)
    device = torch.device("cuda")
    model.to(device).eval().requires_grad_(False)
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats(device)
    loader = DataLoader(
        SiglipCentreDataset(frame, archive_path, processor),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    local_files = []
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        path = snapshot / name
        if path.is_file():
            local_files.append(
                {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    request = {
        "status": "OKUTAMA_CPTR_SIGLIP_CACHE_REQUEST",
        "samples": int(len(frame)),
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "representation": "vision_pooler",
        "calibration_labels_read": 0,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "protocol": sha256_file(protocol_path),
            "protocol_lock": sha256_file(lock_path),
            "manifest": sha256_file(manifest_path),
            "archive": lock["archive_evidence"]["sha256"],
            "extractor": sha256_file(Path(__file__).resolve()),
        },
    }
    request_hash = hashlib.sha256(
        json.dumps(request, sort_keys=True).encode("utf-8")
    ).hexdigest()
    write_json(output_dir / "request.json", {**request, "request_sha256": request_hash})
    feature_path = output_dir / "features.npy"
    feature_store = None
    started = time.perf_counter()
    for batch in tqdm(loader, desc="SigLIP centre features", unit="batch"):
        pixels = batch["pixels"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda"):
            output = model(pixel_values=pixels)
            features = extract_backbone_features(output).float()
        if feature_store is None:
            feature_store = np.lib.format.open_memmap(
                feature_path,
                mode="w+",
                dtype=np.float16,
                shape=(len(frame), features.shape[1]),
            )
        indices = batch["feature_index"].numpy()
        feature_store[indices] = features.cpu().numpy().astype(np.float16)
    if feature_store is None:
        raise RuntimeError("SigLIP extraction returned no features")
    feature_store.flush()
    feature_dim = int(feature_store.shape[1])
    del feature_store
    summary = {
        "status": CPTR_STORE_STATUS,
        "store_kind": "frozen_siglip_centre_person_features",
        "samples": int(len(frame)),
        "feature_dim": feature_dim,
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "representation": "vision_pooler",
        },
        "checkpoint_files": local_files,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "runtime_seconds": time.perf_counter() - started,
        "calibration_labels_read": 0,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "request_sha256": request_hash,
        "arrays": {
            "features": {"path": feature_path.name, "sha256": sha256_file(feature_path)}
        },
        "artifact_sha256": {
            "request.json": sha256_file(output_dir / "request.json")
        },
        "protocol_siglip_stage": protocol["execution_order"].index(
            "top_block_lora_and_centre_siglip_specialists"
        ),
    }
    write_json(output_dir / "store.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
