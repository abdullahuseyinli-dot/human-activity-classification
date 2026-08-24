"""Cache raw and camera-compensated Okutama person trajectories for CPTR."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from hac.cptr_features import (
    CPTR_STORE_STATUS,
    TRAJECTORY_SEQUENCE_DIM,
    TRAJECTORY_SUMMARY_DIM,
    affine_diagnostics,
    build_trajectory_features,
    parse_occlusion_mask,
    parse_window_boxes,
    parse_window_frames,
)
from hac.polar import sha256_file


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
    parser.add_argument("--output-dir", type=Path, default=Path(".runs/cptr/motion_features"))
    parser.add_argument("--working-width", type=int, default=320)
    parser.add_argument("--working-height", type=int, default=180)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def frame_prefix(recording_id: str) -> str:
    drone, part_of_day, _ = recording_id.split(".")
    period = {"1": "Morning", "2": "Noon"}[part_of_day]
    return f"Drone{drone}/{period}/Extracted-Frames-1280x720/{recording_id}/"


def decode_gray(
    archive: zipfile.ZipFile,
    member: str,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    encoded = np.frombuffer(archive.read(member), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode {member}")
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def phase_translation(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, float]:
    shift, response = cv2.phaseCorrelate(
        np.asarray(previous, dtype=np.float32),
        np.asarray(current, dtype=np.float32),
    )
    transform = np.asarray(
        ((1.0, 0.0, shift[0]), (0.0, 1.0, shift[1]), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return transform, float(np.clip(response, 0.0, 1.0) * 0.35)


def estimate_pair_transform(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    maximum_corners: int,
    minimum_inliers: int,
) -> tuple[np.ndarray, float, str, int, int]:
    points = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=maximum_corners,
        qualityLevel=0.01,
        minDistance=6,
        blockSize=7,
    )
    if points is None or len(points) < minimum_inliers:
        transform, quality = phase_translation(previous, current)
        return transform, quality, "phase_correlation", 0, 0
    tracked, status, error = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    if tracked is None or status is None or error is None:
        transform, quality = phase_translation(previous, current)
        return transform, quality, "phase_correlation", 0, 0
    usable = status[:, 0].astype(bool) & np.isfinite(error[:, 0]) & (error[:, 0] < 30.0)
    usable_count = int(usable.sum())
    if usable_count < minimum_inliers:
        transform, quality = phase_translation(previous, current)
        return transform, quality, "phase_correlation", usable_count, 0
    affine, inliers = cv2.estimateAffinePartial2D(
        points[usable],
        tracked[usable],
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=750,
        confidence=0.995,
        refineIters=10,
    )
    if affine is None or inliers is None:
        transform, quality = phase_translation(previous, current)
        return transform, quality, "phase_correlation", usable_count, 0
    transform = np.eye(3, dtype=np.float64)
    transform[:2] = affine
    determinant = float(np.linalg.det(transform[:2, :2]))
    scale = math.sqrt(max(abs(determinant), 1e-12))
    rotation = math.atan2(transform[1, 0], transform[0, 0])
    translation = float(np.linalg.norm(transform[:2, 2]))
    inlier_count = int(inliers.sum())
    plausible = (
        inlier_count >= minimum_inliers
        and 0.94 <= scale <= 1.06
        and abs(rotation) <= math.radians(8.0)
        and translation <= 40.0
        and np.all(np.isfinite(transform))
    )
    if not plausible:
        fallback, quality = phase_translation(previous, current)
        return fallback, quality, "phase_correlation", usable_count, inlier_count
    inlier_fraction = inlier_count / max(usable_count, 1)
    support = min(1.0, inlier_count / 80.0)
    quality = float(np.clip(inlier_fraction * support, 0.0, 1.0))
    return transform, quality, "sparse_lk_ransac", usable_count, inlier_count


def full_resolution_transform(
    small_transform: np.ndarray,
    *,
    working_width: int,
    working_height: int,
) -> np.ndarray:
    scale = np.asarray(
        (
            (working_width / 1280.0, 0.0, 0.0),
            (0.0, working_height / 720.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return np.linalg.inv(scale) @ small_transform @ scale


def estimate_recording(
    archive: zipfile.ZipFile,
    recording_id: str,
    required_frames: np.ndarray,
    *,
    working_width: int,
    working_height: int,
    maximum_corners: int,
    minimum_inliers: int,
) -> dict[str, np.ndarray | int | float]:
    prefix = frame_prefix(recording_id)
    available = []
    for name in archive.namelist():
        if name.startswith(prefix) and name.lower().endswith(".jpg"):
            available.append(int(Path(name).stem))
    frame_numbers = np.asarray(sorted(available), dtype=np.int64)
    if not set(map(int, required_frames)).issubset(set(map(int, frame_numbers))):
        missing = sorted(set(map(int, required_frames)) - set(map(int, frame_numbers)))
        raise RuntimeError(f"Recording {recording_id} lacks required frames: {missing[:5]}")
    lower, upper = int(required_frames.min()), int(required_frames.max())
    keep = (frame_numbers >= lower) & (frame_numbers <= upper)
    frame_numbers = frame_numbers[keep]
    if not len(frame_numbers):
        raise RuntimeError(f"Recording {recording_id} has no selected frames")
    cumulative = np.repeat(np.eye(3, dtype=np.float64)[None], len(frame_numbers), axis=0)
    pair_quality = np.ones(len(frame_numbers), dtype=np.float32)
    methods = np.zeros(len(frame_numbers), dtype=np.int8)
    usable_counts = np.zeros(len(frame_numbers), dtype=np.int16)
    inlier_counts = np.zeros(len(frame_numbers), dtype=np.int16)
    previous = decode_gray(
        archive,
        f"{prefix}{int(frame_numbers[0])}.jpg",
        width=working_width,
        height=working_height,
    )
    for index in range(1, len(frame_numbers)):
        current = decode_gray(
            archive,
            f"{prefix}{int(frame_numbers[index])}.jpg",
            width=working_width,
            height=working_height,
        )
        transform, quality, method, usable, inliers = estimate_pair_transform(
            previous,
            current,
            maximum_corners=maximum_corners,
            minimum_inliers=minimum_inliers,
        )
        transform = full_resolution_transform(
            transform,
            working_width=working_width,
            working_height=working_height,
        )
        cumulative[index] = transform @ cumulative[index - 1]
        pair_quality[index] = quality
        methods[index] = 1 if method == "sparse_lk_ransac" else 2
        usable_counts[index] = min(usable, np.iinfo(np.int16).max)
        inlier_counts[index] = min(inliers, np.iinfo(np.int16).max)
        previous = current
    return {
        "frame_numbers": frame_numbers,
        "cumulative": cumulative,
        "pair_quality": pair_quality,
        "methods": methods,
        "usable_counts": usable_counts,
        "inlier_counts": inlier_counts,
    }


def relative_quality(pair_quality: np.ndarray, source: int, centre: int) -> float:
    if source == centre:
        return 1.0
    lower, upper = sorted((source, centre))
    path = np.asarray(pair_quality[lower + 1 : upper + 1], dtype=float)
    if not len(path):
        return 0.0
    return float(np.clip(np.exp(np.mean(np.log(np.clip(path, 1e-4, 1.0)))), 0.0, 1.0))


def open_array(path: Path, shape: tuple[int, ...], dtype) -> np.ndarray:
    if path.is_file():
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise RuntimeError(f"Resumable motion array changed shape or dtype: {path}")
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def main() -> None:
    args = parse_args()
    if args.working_width < 64 or args.working_height < 64:
        raise ValueError("Camera-motion working resolution is too small")
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
        raise RuntimeError("The development centres changed after locking")
    if archive_path.stat().st_size != int(lock["archive_evidence"]["bytes"]):
        raise RuntimeError("The development archive byte count changed")
    camera_spec = protocol["camera_compensation"]
    if [args.working_width, args.working_height] != camera_spec["working_resolution"]:
        raise RuntimeError("The camera-motion working resolution differs from the protocol")
    centres = pd.read_csv(centres_path, dtype={"sample_id": str, "recording_id": str})
    if len(centres) != int(lock["development_samples"]):
        raise RuntimeError("The CPTR centre count changed")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    recording_dir = output_dir / "recordings"
    recording_dir.mkdir(parents=True, exist_ok=True)
    request_core = {
        "status": "OKUTAMA_CPTR_MOTION_CACHE_REQUEST",
        "samples": int(len(centres)),
        "working_resolution": [args.working_width, args.working_height],
        "camera_compensation": camera_spec,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "protocol": sha256_file(protocol_path),
            "protocol_lock": sha256_file(lock_path),
            "centres": sha256_file(centres_path),
            "archive": lock["archive_evidence"]["sha256"],
            "runner": sha256_file(Path(__file__).resolve()),
        },
    }
    request_hash = hashlib.sha256(
        json.dumps(request_core, sort_keys=True).encode("utf-8")
    ).hexdigest()
    request_path = output_dir / "request.json"
    if request_path.is_file():
        previous = json.loads(request_path.read_text(encoding="utf-8"))
        if previous.get("request_sha256") != request_hash:
            raise RuntimeError("The motion-cache output directory contains a different request")
    else:
        write_json(request_path, {**request_core, "request_sha256": request_hash})

    started = time.perf_counter()
    recording_summaries: list[dict] = []
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("The development archive failed its CRC audit")
        for recording_id, rows in tqdm(
            centres.groupby("recording_id", sort=True),
            desc="camera compensation",
            unit="recording",
        ):
            required = np.unique(
                np.concatenate([parse_window_frames(value) for value in rows["window_frames"]])
            )
            cache_path = recording_dir / f"{recording_id}.npz"
            if cache_path.is_file():
                with np.load(cache_path, allow_pickle=False) as payload:
                    result = {name: np.asarray(payload[name]) for name in payload.files}
                if not set(map(int, required)).issubset(
                    set(map(int, result["frame_numbers"]))
                ):
                    raise RuntimeError(f"Cached recording {recording_id} lost required frames")
            else:
                result = estimate_recording(
                    archive,
                    str(recording_id),
                    required,
                    working_width=args.working_width,
                    working_height=args.working_height,
                    maximum_corners=int(camera_spec["maximum_corners"]),
                    minimum_inliers=int(camera_spec["minimum_inliers"]),
                )
                np.savez_compressed(cache_path, **result)
            methods = np.asarray(result["methods"])
            qualities = np.asarray(result["pair_quality"])
            recording_summaries.append(
                {
                    "recording_id": str(recording_id),
                    "frames": int(len(result["frame_numbers"])),
                    "ransac_pairs": int(np.sum(methods == 1)),
                    "phase_fallback_pairs": int(np.sum(methods == 2)),
                    "mean_pair_quality": float(np.mean(qualities[1:])),
                    "minimum_pair_quality": float(np.min(qualities[1:])),
                    "artifact_sha256": sha256_file(cache_path),
                }
            )

    samples, frames = len(centres), 17
    raw_sequence = open_array(
        output_dir / "raw_sequence.npy",
        (samples, frames, TRAJECTORY_SEQUENCE_DIM),
        np.float32,
    )
    raw_summary = open_array(
        output_dir / "raw_summary.npy", (samples, TRAJECTORY_SUMMARY_DIM), np.float32
    )
    compensated_sequence = open_array(
        output_dir / "compensated_sequence.npy",
        (samples, frames, TRAJECTORY_SEQUENCE_DIM),
        np.float32,
    )
    compensated_summary = open_array(
        output_dir / "compensated_summary.npy",
        (samples, TRAJECTORY_SUMMARY_DIM),
        np.float32,
    )
    camera_quality_store = open_array(
        output_dir / "camera_quality.npy", (samples, frames), np.float32
    )
    homography_store = open_array(
        output_dir / "to_centre_homography.npy", (samples, frames, 3, 3), np.float32
    )
    completed_path = output_dir / "completed.npy"
    completed = (
        np.load(completed_path)
        if completed_path.is_file()
        else np.zeros(samples, dtype=bool)
    )
    if completed.shape != (samples,):
        raise RuntimeError("The resumable motion completion mask changed")
    recording_cache: dict[str, dict[str, np.ndarray]] = {}
    for index in tqdm(np.flatnonzero(~completed), desc="person trajectories", unit="sample"):
        row = centres.iloc[int(index)]
        recording_id = str(row["recording_id"])
        if recording_id not in recording_cache:
            with np.load(recording_dir / f"{recording_id}.npz", allow_pickle=False) as payload:
                recording_cache[recording_id] = {
                    name: np.asarray(payload[name]) for name in payload.files
                }
        recording = recording_cache[recording_id]
        available = np.asarray(recording["frame_numbers"], dtype=np.int64)
        position = {int(value): item for item, value in enumerate(available)}
        window_frames = parse_window_frames(row["window_frames"])
        boxes = parse_window_boxes(row["window_boxes_1280x720"])
        occluded = parse_occlusion_mask(row["window_occluded"], expected=frames)
        indices = np.asarray([position[int(value)] for value in window_frames], dtype=np.int64)
        centre_position = position[int(row["center_frame"])]
        cumulative = np.asarray(recording["cumulative"], dtype=np.float64)
        pair_quality = np.asarray(recording["pair_quality"], dtype=np.float64)
        centre_transform = cumulative[centre_position]
        transforms = np.empty((frames, 3, 3), dtype=np.float64)
        qualities = np.empty(frames, dtype=np.float32)
        for time_index, source_position in enumerate(indices):
            try:
                transforms[time_index] = centre_transform @ np.linalg.inv(
                    cumulative[source_position]
                )
            except np.linalg.LinAlgError:
                transforms[time_index] = np.eye(3)
                qualities[time_index] = 0.0
                continue
            qualities[time_index] = relative_quality(
                pair_quality,
                int(source_position),
                int(centre_position),
            )
        raw_values, raw_statistic, _ = build_trajectory_features(
            boxes,
            window_frames,
            ~occluded,
            centre_index=8,
        )
        compensated_values, compensated_statistic, qualities = build_trajectory_features(
            boxes,
            window_frames,
            ~occluded,
            centre_index=8,
            to_centre_homographies=transforms,
            camera_quality=qualities,
        )
        raw_sequence[index] = raw_values
        raw_summary[index] = raw_statistic
        compensated_sequence[index] = compensated_values
        compensated_summary[index] = compensated_statistic
        camera_quality_store[index] = qualities
        homography_store[index] = transforms.astype(np.float32)
        completed[index] = True
        if (int(index) + 1) % 500 == 0:
            for array in (
                raw_sequence,
                raw_summary,
                compensated_sequence,
                compensated_summary,
                camera_quality_store,
                homography_store,
            ):
                array.flush()
            np.save(completed_path, completed)
    if not completed.all():
        raise RuntimeError("The CPTR motion cache is incomplete")
    for array in (
        raw_sequence,
        raw_summary,
        compensated_sequence,
        compensated_summary,
        camera_quality_store,
        homography_store,
    ):
        array.flush()
    np.save(completed_path, completed)
    array_paths = {
        name: output_dir / f"{name}.npy"
        for name in (
            "raw_sequence",
            "raw_summary",
            "compensated_sequence",
            "compensated_summary",
            "camera_quality",
            "to_centre_homography",
        )
    }
    recording_path = output_dir / "camera_estimation_by_recording.csv"
    pd.DataFrame(recording_summaries).to_csv(recording_path, index=False)
    camera_diagnostics = affine_diagnostics(np.asarray(homography_store).reshape(-1, 3, 3))
    summary = {
        "status": CPTR_STORE_STATUS,
        "store_kind": "raw_and_camera_compensated_trajectory",
        "samples": samples,
        "frames_per_sample": frames,
        "trajectory_sequence_dim": TRAJECTORY_SEQUENCE_DIM,
        "trajectory_summary_dim": TRAJECTORY_SUMMARY_DIM,
        "recordings": int(centres["recording_id"].nunique()),
        "mean_camera_quality": float(np.asarray(camera_quality_store).mean()),
        "low_quality_fraction": float(np.mean(np.asarray(camera_quality_store) < 0.25)),
        "mean_absolute_camera_translation": float(
            np.linalg.norm(camera_diagnostics[:, :2], axis=1).mean()
        ),
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "runtime_seconds": time.perf_counter() - started,
        "request_sha256": request_hash,
        "arrays": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in array_paths.items()
        },
        "artifact_sha256": {
            recording_path.name: sha256_file(recording_path),
            completed_path.name: sha256_file(completed_path),
        },
    }
    write_json(output_dir / "store.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
