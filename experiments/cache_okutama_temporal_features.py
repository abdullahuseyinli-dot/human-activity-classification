"""Cache packed Okutama tight/context frame features on CUDA."""

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
        "--partition", choices=["development", "confirmation"], default="development"
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--centres",
        type=Path,
        default=Path(".runs/vcoco_v3/okutama/development_audit/development_centres.csv"),
    )
    parser.add_argument(
        "--audit-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/okutama/development_audit/summary.json"),
    )
    parser.add_argument(
        "--representation-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/representations/evaluation/summary.json"),
    )
    parser.add_argument(
        "--protocol-amendment",
        type=Path,
        default=Path(".runs/vcoco_v3/protocol/external_cuda_amendment_lock.json"),
    )
    parser.add_argument(
        "--pipeline-lock",
        type=Path,
        help="Required only for the sealed confirmation partition",
    )
    parser.add_argument(
        "--representation-metrics",
        type=Path,
        default=Path(
            ".runs/vcoco_v3/representations/evaluation/nested_source_tag_metrics.csv"
        ),
    )
    parser.add_argument("--model-kind", choices=["dinov2_base", "dinov3_base"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
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


def parse_window(value: str, *, converter) -> list:
    return [converter(item) for item in str(value).split(";")]


class OkutamaFrameDataset(Dataset):
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

    def __getitem__(self, index: int) -> dict:
        flat_index = int(self.pending[index])
        sample_index, time_index = divmod(flat_index, 17)
        row = self.centres.iloc[sample_index]
        frames = parse_window(row["window_frames"], converter=int)
        boxes = parse_window(
            row["window_boxes_1280x720"],
            converter=lambda item: tuple(map(float, item.split(","))),
        )
        frame = frames[time_index]
        x1, y1, x2, y2 = boxes[time_index]
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
            tight = self.transform(image_view(rgb, box, "person_tight"))
            context = self.transform(image_view(rgb, box, "person_context_25"))
        return {
            "pixels": torch.stack((tight, context)),
            "sample_index": sample_index,
            "time_index": time_index,
        }


def geometry_from_centres(centres: pd.DataFrame) -> np.ndarray:
    all_boxes = []
    for value in centres["window_boxes_1280x720"]:
        boxes = parse_window(
            value,
            converter=lambda item: tuple(map(float, item.split(","))),
        )
        if len(boxes) != 17:
            raise RuntimeError("An Okutama temporal window does not contain 17 boxes")
        all_boxes.append(boxes)
    boxes = np.asarray(all_boxes, dtype=np.float32)
    width = np.clip(boxes[..., 2] - boxes[..., 0], 1e-6, None)
    height = np.clip(boxes[..., 3] - boxes[..., 1], 1e-6, None)
    center_x = (boxes[..., 0] + boxes[..., 2]) / (2.0 * 1280.0)
    center_y = (boxes[..., 1] + boxes[..., 3]) / (2.0 * 720.0)
    edge = np.minimum.reduce([center_x, 1.0 - center_x, center_y, 1.0 - center_y])
    return np.stack(
        (
            np.log(np.clip(width * height / (1280.0 * 720.0), 1e-8, 1.0)),
            np.log(width / height),
            center_x,
            center_y,
            np.log(height),
            edge,
        ),
        axis=-1,
    ).astype(np.float32)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, dict, dict, dict]:
    if args.batch_size < 1 or args.workers < 0 or args.checkpoint_interval < 1:
        raise ValueError("Batch size and checkpoint interval must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Okutama feature extraction requires CUDA")
    archive_path = args.archive.resolve()
    centres_path = args.centres.resolve()
    audit_path = args.audit_summary.resolve()
    representation_path = args.representation_summary.resolve()
    metrics_path = args.representation_metrics.resolve()
    amendment_path = args.protocol_amendment.resolve()
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if (
        amendment.get("status")
        != "VCOCO_V3_EXTERNAL_CUDA_AMENDMENT_LOCKED_BEFORE_TARGET_FITTING"
    ):
        raise RuntimeError("The external CUDA protocol amendment is not locked")
    if amendment.get("cuda", {}).get("cpu_fallback_permitted") is not False:
        raise RuntimeError("The external protocol amendment permits CPU fallback")
    if amendment["source_sha256"].get("okutama_cache_source") != sha256_file(
        Path(__file__).resolve()
    ):
        raise RuntimeError("The amended Okutama cache implementation changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if amendment["source_sha256"].get("okutama_protocol") != audit["source_sha256"].get(
        "okutama_protocol"
    ):
        raise RuntimeError("The Okutama audit belongs to a different protocol amendment")
    if args.partition == "development":
        if audit.get("status") != "OKUTAMA_DEVELOPMENT_ARCHIVE_AND_CENTRES_AUDITED":
            raise RuntimeError("The Okutama development audit is incomplete")
        if audit.get("confirmation_archive_opened"):
            raise RuntimeError("The development cache cannot follow an early confirmation open")
        archive_evidence = audit["development_archive"]
    else:
        if audit.get("status") != "OKUTAMA_CONFIRMATION_ARCHIVE_AND_CENTRES_AUDITED":
            raise RuntimeError("The Okutama confirmation audit is incomplete")
        if audit.get("confirmation_archive_opened") is not True:
            raise RuntimeError("The confirmation open was not recorded")
        if args.pipeline_lock is None:
            raise ValueError("Confirmation caching requires --pipeline-lock")
        pipeline_path = args.pipeline_lock.resolve()
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        if pipeline.get("status") != "VCOCO_V3_TEMPORAL_PIPELINE_LOCKED_BEFORE_CONFIRMATION":
            raise RuntimeError("The temporal pipeline is not locked for confirmation")
        if sha256_file(pipeline_path) != audit["source_sha256"]["pipeline_lock"]:
            raise RuntimeError("The confirmation audit belongs to a different pipeline lock")
        archive_evidence = audit["confirmation_archive"]
    if sha256_file(archive_path) != archive_evidence["sha256"]:
        raise RuntimeError("The audited Okutama archive changed")
    if sha256_file(centres_path) != audit["artifact_sha256"][centres_path.name]:
        raise RuntimeError("The audited Okutama centre manifest changed")
    representation = json.loads(representation_path.read_text(encoding="utf-8"))
    if representation.get("status") != "VCOCO_V3_MATCHED_REPRESENTATION_DEVELOPMENT_COMPLETE":
        raise RuntimeError("Okutama caching requires the completed representation screen")
    if sha256_file(metrics_path) != representation["artifact_sha256"][metrics_path.name]:
        raise RuntimeError("The representation metrics changed")
    metrics = pd.read_csv(metrics_path)
    eligible = metrics[metrics["family"].isin({"dinov2_base", "dinov3_base"})].sort_values(
        ["macro_f1", "locomotion_f1", "log_loss", "family"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    if len(eligible) != 2 or str(eligible.iloc[0]["family"]) != args.model_kind:
        raise RuntimeError("The requested Okutama backbone is not the locked best DINO family")
    centres = pd.read_csv(
        centres_path,
        dtype={"sample_id": str, "recording_id": str, "scenario_id": str, "track_id": str},
    )
    if len(centres) != int(audit["selected_centres"]):
        raise RuntimeError("The Okutama centre count changed")
    expected_partition = "train" if args.partition == "development" else "test"
    if set(centres["provider_partition"].astype(str)) != {expected_partition}:
        raise RuntimeError("The Okutama centre manifest belongs to a different partition")
    return centres, audit, representation, archive_evidence


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    centres, audit, representation, archive_evidence = validate_inputs(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request_core = {
        "status": "VCOCO_V3_OKUTAMA_FEATURE_CACHE_REQUEST",
        "partition": f"provider_{'train' if args.partition == 'development' else 'test'}",
        "model_kind": args.model_kind,
        "image_size": 224,
        "preprocess": "aspect_preserving_pad",
        "views": ["person_tight", "person_context_25"],
        "samples": len(centres),
        "frames_per_sample": 17,
        "center_frame_index": 8,
        "confirmation_archive_opened": args.partition == "confirmation",
        "source_sha256": {
            "archive": archive_evidence["sha256"],
            "centres": audit["artifact_sha256"][args.centres.resolve().name],
            "audit_summary": sha256_file(args.audit_summary.resolve()),
            "representation_summary": sha256_file(args.representation_summary.resolve()),
            "representation_metrics": sha256_file(args.representation_metrics.resolve()),
            "external_cuda_amendment": sha256_file(args.protocol_amendment.resolve()),
            "extractor": sha256_file(Path(__file__).resolve()),
            "representation_module": sha256_file(
                Path(__file__).resolve().parents[1] / "src/hac/vcoco_v3_representations.py"
            ),
        },
    }
    if args.partition == "confirmation":
        request_core["source_sha256"]["pipeline_lock"] = sha256_file(
            args.pipeline_lock.resolve()
        )
    request_hash = hashlib.sha256(
        json.dumps(request_core, sort_keys=True).encode("utf-8")
    ).hexdigest()
    request_path = output_dir / "request.json"
    if request_path.is_file():
        previous = json.loads(request_path.read_text(encoding="utf-8"))
        if previous.get("request_sha256") != request_hash:
            raise RuntimeError("The output directory contains a different cache request")
    else:
        write_json(request_path, {**request_core, "request_sha256": request_hash})

    completed_path = output_dir / "completed.npy"
    if completed_path.is_file():
        completed = np.load(completed_path)
        if completed.shape != (len(centres), 17):
            raise RuntimeError("The Okutama cache checkpoint shape changed")
    else:
        completed = np.zeros((len(centres), 17), dtype=bool)
    pending = np.flatnonzero(~completed.ravel())
    transform = build_feature_transform(args.model_kind, "aspect_preserving_pad", 224)
    loader = DataLoader(
        OkutamaFrameDataset(args.archive.resolve(), centres, pending, transform=transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    model = build_feature_model(args.model_kind).to(device).eval()
    started = time.perf_counter()
    tight_path = output_dir / "tight.npy"
    context_path = output_dir / "context.npy"
    tight = context = None
    if tight_path.is_file() and context_path.is_file():
        tight = np.load(tight_path, mmap_mode="r+")
        context = np.load(context_path, mmap_mode="r+")
        if tight.shape[:2] != completed.shape or context.shape != tight.shape:
            raise RuntimeError("The resumable Okutama embedding arrays changed shape")

    for batch_index, batch in enumerate(
        tqdm(loader, desc="extracting Okutama frame features", unit="batch")
    ):
        pixels = batch["pixels"].to(device, non_blocking=True)
        batch_size, views = pixels.shape[:2]
        with torch.autocast(device_type="cuda"):
            embeddings = model(pixels.flatten(0, 1)).float()
        embeddings = embeddings.reshape(batch_size, views, -1).cpu().numpy()
        if tight is None:
            shape = (len(centres), 17, embeddings.shape[-1])
            tight = np.lib.format.open_memmap(tight_path, mode="w+", dtype=np.float32, shape=shape)
            context = np.lib.format.open_memmap(
                context_path, mode="w+", dtype=np.float32, shape=shape
            )
        sample_index = batch["sample_index"].numpy()
        time_index = batch["time_index"].numpy()
        tight[sample_index, time_index] = embeddings[:, 0]
        context[sample_index, time_index] = embeddings[:, 1]
        completed[sample_index, time_index] = True
        if (batch_index + 1) % args.checkpoint_interval == 0:
            tight.flush()
            context.flush()
            np.save(completed_path, completed)

    if tight is None or context is None or not completed.all():
        raise RuntimeError("Okutama feature extraction ended with incomplete rows")
    tight.flush()
    context.flush()
    np.save(completed_path, completed)
    del tight, context
    gc.collect()
    geometry_path = output_dir / "geometry.npy"
    np.save(geometry_path, geometry_from_centres(centres))
    checkpoint = local_checkpoint_evidence(args.model_kind)

    store_path = output_dir / "store.json"
    store = {
        "status": "VCOCO_V3_PACKED_TEMPORAL_FEATURE_STORE_COMPLETE",
        "partition": f"provider_{'train' if args.partition == 'development' else 'test'}",
        "model_kind": args.model_kind,
        "model": model_provenance(args.model_kind),
        "checkpoint": checkpoint,
        "samples": len(centres),
        "frames_per_sample": 17,
        "center_frame_index": 8,
        "feature_dimensions": int(np.load(tight_path, mmap_mode="r").shape[-1]),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "confirmation_archive_opened": args.partition == "confirmation",
        "arrays": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in (
                ("tight", tight_path),
                ("context", context_path),
                ("geometry", geometry_path),
            )
        },
        "runtime_seconds": time.perf_counter() - started,
        "request_sha256": request_hash,
    }
    write_json(store_path, store)

    metadata = centres.copy()
    metadata["video_id"] = metadata["recording_id"]
    metadata["provider_track_id"] = metadata["track_id"]
    metadata["recording_id"] = metadata["scenario_id"]
    metadata["track_id"] = metadata["video_id"] + "::" + metadata["provider_track_id"]
    metadata["frame_count"] = 17
    metadata["center_frame_index"] = 8
    metadata["frames_per_second"] = 16.0
    metadata["feature_path"] = str(store_path)
    metadata["feature_index"] = np.arange(len(metadata), dtype=int)
    if args.partition == "confirmation":
        metadata["split"] = "confirmation"
    metadata_path = output_dir / f"{args.partition}_metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    summary = {
        "status": (
            "VCOCO_V3_OKUTAMA_DEVELOPMENT_FEATURE_STORE_COMPLETE"
            if args.partition == "development"
            else "VCOCO_V3_OKUTAMA_CONFIRMATION_FEATURE_STORE_COMPLETE"
        ),
        "samples": len(metadata),
        "scenarios": int(metadata["recording_id"].nunique()),
        "videos": int(metadata["video_id"].nunique()),
        "model_kind": args.model_kind,
        "feature_dimensions": store["feature_dimensions"],
        "extraction_device": torch.cuda.get_device_name(0),
        "confirmation_archive_opened": args.partition == "confirmation",
        "confirmation_open_number": 1 if args.partition == "confirmation" else 0,
        "representation_stage_sha256": sha256_file(args.representation_summary.resolve()),
        "source_sha256": request_core["source_sha256"],
        "artifact_sha256": {
            store_path.name: sha256_file(store_path),
            metadata_path.name: sha256_file(metadata_path),
            **{path.name: sha256_file(path) for path in (tight_path, context_path, geometry_path)},
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
