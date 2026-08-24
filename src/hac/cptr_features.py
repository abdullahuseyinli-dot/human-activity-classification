"""Feature contracts and datasets for the CPTR development study."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hac.vcoco_v3_temporal import TEMPORAL_CLASS_NAMES, validate_temporal_manifest

CPTR_STORE_STATUS = "OKUTAMA_CPTR_FEATURE_STORE_COMPLETE"
BASE_GEOMETRY_DIM = 6
TRAJECTORY_SEQUENCE_DIM = 21
TRAJECTORY_SUMMARY_DIM = 58
QUALITY_DIM = 8
BODY_REGION_NAMES = (
    "head_shoulders",
    "torso",
    "pelvis",
    "left_upper_side",
    "right_upper_side",
    "left_leg",
    "right_leg",
)


def parse_semicolon_values(value: object, *, converter) -> list:
    return [converter(item) for item in str(value).split(";")]


def parse_window_frames(value: object) -> np.ndarray:
    output = np.asarray(parse_semicolon_values(value, converter=int), dtype=np.int64)
    if output.ndim != 1 or len(output) < 2 or np.any(np.diff(output) < 0):
        raise ValueError("Window frame numbers must be a nondecreasing vector")
    return output


def parse_window_boxes(value: object) -> np.ndarray:
    boxes = parse_semicolon_values(
        value,
        converter=lambda item: tuple(map(float, item.split(","))),
    )
    output = np.asarray(boxes, dtype=np.float32)
    if output.ndim != 2 or output.shape[1] != 4:
        raise ValueError("Window boxes must have shape [time, 4]")
    if np.any(output[:, 2:] <= output[:, :2]):
        raise ValueError("Window boxes must have positive width and height")
    return output


def parse_occlusion_mask(value: object, *, expected: int) -> np.ndarray:
    values = np.asarray(parse_semicolon_values(value, converter=int), dtype=np.int64)
    if values.shape != (expected,) or not set(values.tolist()).issubset({0, 1}):
        raise ValueError("Window occlusion must be a binary vector aligned with frames")
    return values.astype(bool)


def sample_indices_with_centre(
    frame_count: int,
    *,
    centre_index: int,
    samples: int,
    span_frames: int,
) -> tuple[np.ndarray, int]:
    """Uniformly sample a window while guaranteeing that its labelled centre is present."""

    if min(frame_count, samples, span_frames) < 1 or not 0 <= centre_index < frame_count:
        raise ValueError("Temporal sampling parameters are invalid")
    half = (span_frames - 1) / 2.0
    positions = np.linspace(centre_index - half, centre_index + half, samples)
    indices = np.rint(np.clip(positions, 0, frame_count - 1)).astype(np.int64)
    closest = int(np.argmin(np.abs(indices - centre_index)))
    indices[closest] = centre_index
    indices.sort(kind="stable")
    centre_positions = np.flatnonzero(indices == centre_index)
    if not len(centre_positions):
        raise RuntimeError("Centre-preserving temporal sampling failed")
    return indices, int(centre_positions[len(centre_positions) // 2])


def box_state(boxes: np.ndarray, *, image_width: float = 1280.0, image_height: float = 720.0):
    values = np.asarray(boxes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("boxes must have shape [time, 4]")
    width = np.clip(values[:, 2] - values[:, 0], 1e-6, None)
    height = np.clip(values[:, 3] - values[:, 1], 1e-6, None)
    centre_x = (values[:, 0] + values[:, 2]) / (2.0 * image_width)
    centre_y = (values[:, 1] + values[:, 3]) / (2.0 * image_height)
    edge = np.minimum.reduce((centre_x, 1.0 - centre_x, centre_y, 1.0 - centre_y))
    return np.stack(
        (
            centre_x,
            centre_y,
            np.log(np.clip(width * height / (image_width * image_height), 1e-8, 1.0)),
            np.log(width / height),
            np.log(np.clip(height / image_height, 1e-8, 1.0)),
            edge,
        ),
        axis=1,
    )


def _interpolate_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    valid = np.asarray(valid, dtype=bool)
    if output.ndim != 2 or valid.shape != (len(output),):
        raise ValueError("Values and validity mask do not align")
    positions = np.arange(len(output), dtype=float)
    if not valid.any():
        return np.zeros_like(output)
    for column in range(output.shape[1]):
        finite = valid & np.isfinite(output[:, column])
        if finite.any():
            output[:, column] = np.interp(positions, positions[finite], output[finite, column])
        else:
            output[:, column] = 0.0
    return output


def _gradient(values: np.ndarray, timing: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    safe_timing = np.asarray(timing, dtype=np.float64).copy()
    for index in range(1, len(safe_timing)):
        if safe_timing[index] <= safe_timing[index - 1]:
            safe_timing[index] = safe_timing[index - 1] + 1e-3
    return np.gradient(values, safe_timing, axis=0, edge_order=1)


def trajectory_summary(state: np.ndarray, timing: np.ndarray, *, centre_index: int) -> np.ndarray:
    values = np.asarray(state, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not 0 <= centre_index < len(values):
        raise ValueError("Trajectory state must have shape [time, 6] with a valid centre")
    summaries: list[float] = []
    for column in range(6):
        series = values[:, column]
        summaries.extend(
            (
                float(series[0]),
                float(series[centre_index]),
                float(series[-1]),
                float(series[-1] - series[0]),
                float(series.mean()),
                float(series.std()),
            )
        )
    derivatives = _gradient(values[:, :5], timing)
    for column in range(5):
        series = derivatives[:, column]
        summaries.extend(
            (
                float(series.mean()),
                float(series.std()),
                float(np.max(np.abs(series))),
            )
        )
    speed = np.linalg.norm(derivatives[:, :2], axis=1)
    path = float(np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1).sum())
    net = float(np.linalg.norm(values[-1, :2] - values[0, :2]))
    summaries.extend(
        (
            float(speed.mean()),
            float(speed.std()),
            float(speed.max()),
            float(np.median(speed)),
            path,
            net,
            net / max(path, 1e-8),
        )
    )
    output = np.asarray(summaries, dtype=np.float32)
    if output.shape != (TRAJECTORY_SUMMARY_DIM,):
        raise RuntimeError(f"Trajectory summary width changed: {output.shape}")
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def apply_homography(points: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transforms = np.asarray(transforms, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or transforms.shape != (len(points), 3, 3):
        raise ValueError("Points and homographies must align over time")
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    warped = np.einsum("tij,tj->ti", transforms, homogeneous)
    return warped[:, :2] / np.clip(warped[:, 2:], 1e-8, None)


def affine_diagnostics(transforms: np.ndarray) -> np.ndarray:
    values = np.asarray(transforms, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError("transforms must have shape [time, 3, 3]")
    linear = values[:, :2, :2]
    determinant = np.linalg.det(linear)
    scale = np.sqrt(np.clip(np.abs(determinant), 1e-12, None))
    rotation = np.arctan2(linear[:, 1, 0], linear[:, 0, 0])
    return np.stack(
        (
            values[:, 0, 2] / 1280.0,
            values[:, 1, 2] / 720.0,
            rotation,
            np.log(scale),
        ),
        axis=1,
    )


def build_trajectory_features(
    boxes: np.ndarray,
    frame_numbers: np.ndarray,
    valid_mask: np.ndarray,
    *,
    centre_index: int,
    frame_rate: float = 30.0,
    to_centre_homographies: np.ndarray | None = None,
    camera_quality: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 21-D temporal signal, 58-D summary, and per-frame camera quality."""

    boxes = np.asarray(boxes, dtype=np.float64)
    frames = np.asarray(frame_numbers, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if boxes.shape != (len(frames), 4) or valid.shape != (len(frames),):
        raise ValueError("Trajectory boxes, frames, and validity do not align")
    if frame_rate <= 0.0 or not 0 <= centre_index < len(frames):
        raise ValueError("Trajectory timing is invalid")
    state = box_state(boxes)
    if to_centre_homographies is None:
        transforms = np.repeat(np.eye(3, dtype=np.float64)[None], len(frames), axis=0)
        quality = np.ones(len(frames), dtype=np.float64)
    else:
        transforms = np.asarray(to_centre_homographies, dtype=np.float64)
        if transforms.shape != (len(frames), 3, 3):
            raise ValueError("Camera homographies do not align with the trajectory")
        quality = np.asarray(camera_quality, dtype=np.float64)
        if quality.shape != (len(frames),):
            raise ValueError("Camera quality does not align with the trajectory")
        corners = np.stack(
            (
                boxes[:, [0, 1]],
                boxes[:, [2, 1]],
                boxes[:, [2, 3]],
                boxes[:, [0, 3]],
            ),
            axis=1,
        )
        corrected_corners = np.stack(
            [apply_homography(corners[:, item], transforms) for item in range(4)],
            axis=1,
        )
        corrected_boxes = np.concatenate(
            (
                corrected_corners.min(axis=1),
                corrected_corners.max(axis=1),
            ),
            axis=1,
        )
        state = box_state(corrected_boxes)
    state = _interpolate_invalid(state, valid)
    timing = (frames - frames[0]) / frame_rate
    derivative = _gradient(state[:, :5], timing)
    speed = np.linalg.norm(derivative[:, :2], axis=1, keepdims=True)
    acceleration_xy = _gradient(derivative[:, :2], timing)
    acceleration = np.linalg.norm(acceleration_xy, axis=1, keepdims=True)
    camera = affine_diagnostics(transforms)
    sequence = np.concatenate(
        (
            state,
            derivative,
            speed,
            acceleration_xy,
            acceleration,
            camera,
            quality[:, None],
            valid.astype(np.float64)[:, None],
        ),
        axis=1,
    ).astype(np.float32)
    if sequence.shape != (len(frames), TRAJECTORY_SEQUENCE_DIM):
        raise RuntimeError(f"Trajectory sequence width changed: {sequence.shape}")
    summary = trajectory_summary(state, timing, centre_index=centre_index)
    return (
        np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0),
        summary,
        np.clip(quality, 0.0, 1.0).astype(np.float32),
    )


class _ArrayStore:
    def __init__(self, declaration_path: Path, *, expected_status: str) -> None:
        self.path = declaration_path.resolve()
        self.declaration = json.loads(self.path.read_text(encoding="utf-8"))
        if self.declaration.get("status") != expected_status:
            raise RuntimeError(f"Feature store is incomplete: {self.path}")
        self.arrays: dict[str, np.ndarray] = {}
        for name, item in self.declaration.get("arrays", {}).items():
            self.arrays[name] = np.load((self.path.parent / item["path"]).resolve(), mmap_mode="r")

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]


class CPTRFeatureDataset(Dataset):
    """Join the locked v3 frame store to CPTR motion and optional specialist stores."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        manifest_directory: str | Path,
        motion_store: str | Path | None = None,
        part_store: str | Path | None = None,
        siglip_store: str | Path | None = None,
        pose_store: str | Path | None = None,
        use_compensated_trajectory: bool = True,
        short_samples: int = 8,
        short_seconds: float = 0.5,
        long_samples: int = 8,
        long_seconds: float = 1.0,
    ) -> None:
        self.frame = validate_temporal_manifest(frame).reset_index(drop=True)
        self.manifest_directory = Path(manifest_directory).resolve()
        self.class_to_index = {name: index for index, name in enumerate(TEMPORAL_CLASS_NAMES)}
        self.short_samples = int(short_samples)
        self.short_seconds = float(short_seconds)
        self.long_samples = int(long_samples)
        self.long_seconds = float(long_seconds)
        self.use_compensated_trajectory = bool(use_compensated_trajectory)
        self._base_stores: dict[Path, _ArrayStore] = {}
        self.motion = (
            _ArrayStore(Path(motion_store), expected_status=CPTR_STORE_STATUS)
            if motion_store is not None
            else None
        )
        self.parts = (
            _ArrayStore(Path(part_store), expected_status=CPTR_STORE_STATUS)
            if part_store is not None
            else None
        )
        self.siglip = (
            _ArrayStore(Path(siglip_store), expected_status=CPTR_STORE_STATUS)
            if siglip_store is not None
            else None
        )
        self.pose = (
            _ArrayStore(Path(pose_store), expected_status=CPTR_STORE_STATUS)
            if pose_store is not None
            else None
        )
        feature_indices = pd.to_numeric(self.frame.get("feature_index"), errors="coerce")
        if feature_indices.isna().any():
            raise ValueError("CPTR requires aligned packed feature indices")
        self.feature_indices = feature_indices.to_numpy(dtype=np.int64)
        for store in (self.motion, self.parts, self.siglip, self.pose):
            if store is None:
                continue
            expected = int(store.declaration.get("samples", -1))
            maximum = int(self.feature_indices.max(initial=-1))
            if expected <= maximum:
                raise RuntimeError("An auxiliary store is shorter than the requested feature indices")

    def __len__(self) -> int:
        return len(self.frame)

    def _path(self, value: object) -> Path:
        path = Path(str(value))
        return path.resolve() if path.is_absolute() else (self.manifest_directory / path).resolve()

    def _base_store(self, path: Path) -> _ArrayStore:
        store = self._base_stores.get(path)
        if store is None:
            store = _ArrayStore(
                path,
                expected_status="VCOCO_V3_PACKED_TEMPORAL_FEATURE_STORE_COMPLETE",
            )
            self._base_stores[path] = store
        return store

    @staticmethod
    def _bool(value: object) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes"}

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        feature_index = int(self.feature_indices[index])
        store = self._base_store(self._path(row["feature_path"]))
        tight = np.asarray(store["tight"][feature_index], dtype=np.float32)
        context = np.asarray(store["context"][feature_index], dtype=np.float32)
        geometry = np.asarray(store["geometry"][feature_index], dtype=np.float32)
        if tight.shape != context.shape or geometry.shape != (len(tight), BASE_GEOMETRY_DIM):
            raise RuntimeError("Base CPTR temporal features are not aligned")
        combined = np.concatenate((tight, context, geometry), axis=1)
        frames = len(combined)
        centre = int(row["center_frame_index"])
        frames_per_second = float(row["frames_per_second"])
        short_indices, short_centre = sample_indices_with_centre(
            frames,
            centre_index=centre,
            samples=self.short_samples,
            span_frames=max(1, int(round(frames_per_second * self.short_seconds))),
        )
        long_indices, long_centre = sample_indices_with_centre(
            frames,
            centre_index=centre,
            samples=self.long_samples,
            span_frames=max(1, int(round(frames_per_second * self.long_seconds))),
        )
        if "window_occluded" in row and not pd.isna(row["window_occluded"]):
            occluded = parse_occlusion_mask(row["window_occluded"], expected=frames)
        else:
            occluded = np.zeros(frames, dtype=bool)
        valid = ~occluded

        if self.motion is not None:
            prefix = "compensated" if self.use_compensated_trajectory else "raw"
            trajectory_sequence = np.asarray(
                self.motion[f"{prefix}_sequence"][feature_index], dtype=np.float32
            )
            trajectory_summary_values = np.asarray(
                self.motion[f"{prefix}_summary"][feature_index], dtype=np.float32
            )
            camera_quality = (
                np.asarray(self.motion["camera_quality"][feature_index], dtype=np.float32)
                if self.use_compensated_trajectory
                else np.ones(frames, dtype=np.float32)
            )
        else:
            trajectory_sequence = np.zeros((frames, TRAJECTORY_SEQUENCE_DIM), dtype=np.float32)
            trajectory_summary_values = np.zeros(TRAJECTORY_SUMMARY_DIM, dtype=np.float32)
            camera_quality = np.ones(frames, dtype=np.float32)

        if self.parts is not None:
            part_tokens = np.asarray(self.parts["part_tokens"][feature_index], dtype=np.float32)
            part_confidence = np.asarray(
                self.parts["part_confidence"][feature_index], dtype=np.float32
            )
        else:
            part_tokens = np.zeros((frames, len(BODY_REGION_NAMES), 1), dtype=np.float32)
            part_confidence = np.zeros((frames, len(BODY_REGION_NAMES)), dtype=np.float32)

        if self.siglip is not None:
            siglip = np.asarray(self.siglip["features"][feature_index], dtype=np.float32)
        else:
            siglip = np.zeros(1, dtype=np.float32)
        if self.pose is not None:
            pose = np.asarray(self.pose["pose"][feature_index], dtype=np.float32)
        else:
            pose = np.zeros((frames, 1, 3), dtype=np.float32)

        centre_geometry = geometry[centre]
        part_quality = float(part_confidence.mean()) if part_confidence.size else 0.0
        straightness = float(trajectory_summary_values[-1])
        quality = np.asarray(
            (
                float(occluded[centre]),
                float(occluded.mean()),
                float(centre_geometry[0]),
                float(centre_geometry[1]),
                float(centre_geometry[5]),
                float(camera_quality[valid].mean()) if valid.any() else 0.0,
                part_quality,
                straightness,
            ),
            dtype=np.float32,
        )
        gait_mapping = {"walking": 0, "running": 1}
        gait_target = gait_mapping.get(str(row.get("walking_running_subtype", "")), -1)
        return {
            "static_features": torch.from_numpy(combined[centre].copy()),
            "short_features": torch.from_numpy(combined[short_indices].copy()),
            "short_valid_mask": torch.from_numpy(valid[short_indices].copy()),
            "short_centre_index": short_centre,
            "long_features": torch.from_numpy(combined[long_indices].copy()),
            "long_valid_mask": torch.from_numpy(valid[long_indices].copy()),
            "long_centre_index": long_centre,
            "trajectory_sequence": torch.from_numpy(trajectory_sequence.copy()),
            "trajectory_summary": torch.from_numpy(trajectory_summary_values.copy()),
            "trajectory_valid_mask": torch.from_numpy(valid.copy()),
            "camera_quality": torch.from_numpy(camera_quality.copy()),
            "part_tokens": torch.from_numpy(part_tokens[long_indices].copy()),
            "part_confidence": torch.from_numpy(part_confidence[long_indices].copy()),
            "part_valid_mask": torch.from_numpy(valid[long_indices].copy()),
            "part_centre_index": long_centre,
            "pose": torch.from_numpy(pose[long_indices].copy()),
            "pose_valid_mask": torch.from_numpy(valid[long_indices].copy()),
            "pose_centre_index": long_centre,
            "siglip_features": torch.from_numpy(siglip.copy()),
            "quality_features": torch.from_numpy(quality),
            "label": self.class_to_index[str(row["label"])],
            "transition_target": int(self._bool(row.get("transition_window", False))),
            "gait_target": gait_target,
            "occlusion_target": float(occluded.mean() > 0.0),
            "sample_id": str(row["sample_id"]),
            "recording_id": str(row["recording_id"]),
            "scenario_id": str(row["scenario_id"]),
            "track_id": str(row["track_id"]),
            "feature_index": feature_index,
        }


def model_kwargs_from_batch(
    batch: Mapping[str, object],
    device: torch.device,
    *,
    use_long: bool,
    use_trajectory: bool,
    use_parts: bool,
    use_pose: bool,
    use_siglip: bool,
) -> dict[str, object]:
    required = (
        "static_features",
        "short_features",
        "short_valid_mask",
        "quality_features",
    )
    output: dict[str, object] = {
        name: batch[name].to(device, non_blocking=True) for name in required  # type: ignore[union-attr]
    }
    output["short_centre_index"] = int(batch["short_centre_index"][0])  # type: ignore[index]
    if use_long:
        for name in ("long_features", "long_valid_mask"):
            output[name] = batch[name].to(device, non_blocking=True)  # type: ignore[union-attr]
        output["long_centre_index"] = int(batch["long_centre_index"][0])  # type: ignore[index]
    if use_trajectory:
        for name in (
            "trajectory_sequence",
            "trajectory_summary",
            "trajectory_valid_mask",
            "camera_quality",
        ):
            output[name] = batch[name].to(device, non_blocking=True)  # type: ignore[union-attr]
    if use_parts:
        for name in ("part_tokens", "part_confidence", "part_valid_mask"):
            output[name] = batch[name].to(device, non_blocking=True)  # type: ignore[union-attr]
        output["part_centre_index"] = int(batch["part_centre_index"][0])  # type: ignore[index]
    if use_pose:
        for name in ("pose", "pose_valid_mask"):
            output[name] = batch[name].to(device, non_blocking=True)  # type: ignore[union-attr]
        output["pose_centre_index"] = int(batch["pose_centre_index"][0])  # type: ignore[index]
    if use_siglip:
        output["siglip_features"] = batch["siglip_features"].to(  # type: ignore[union-attr]
            device, non_blocking=True
        )
    return output


def _repeat_centre(values: torch.Tensor, centre: int) -> torch.Tensor:
    return values[:, centre : centre + 1].expand_as(values).clone()


def motion_null_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    output = dict(kwargs)
    short_centre = int(output["short_centre_index"])
    output["short_features"] = _repeat_centre(output["short_features"], short_centre)  # type: ignore[arg-type]
    if "long_features" in output:
        long_centre = int(output["long_centre_index"])
        output["long_features"] = _repeat_centre(output["long_features"], long_centre)  # type: ignore[arg-type]
    if "trajectory_sequence" in output:
        sequence = output["trajectory_sequence"]  # type: ignore[assignment]
        centre = sequence.shape[1] // 2
        null_sequence = _repeat_centre(sequence, centre)
        null_sequence[:, :, 6:19] = 0.0
        output["trajectory_sequence"] = null_sequence
        output["trajectory_summary"] = torch.zeros_like(output["trajectory_summary"])  # type: ignore[arg-type]
    if "part_tokens" in output:
        centre = int(output["part_centre_index"])
        output["part_tokens"] = _repeat_centre(output["part_tokens"], centre)  # type: ignore[arg-type]
        output["part_confidence"] = _repeat_centre(  # type: ignore[arg-type]
            output["part_confidence"], centre
        )
    if "pose" in output:
        centre = int(output["pose_centre_index"])
        output["pose"] = _repeat_centre(output["pose"], centre)  # type: ignore[arg-type]
    return output


def reversed_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    output = dict(kwargs)
    temporal_names = (
        "short_features",
        "short_valid_mask",
        "long_features",
        "long_valid_mask",
        "trajectory_sequence",
        "trajectory_valid_mask",
        "camera_quality",
        "part_tokens",
        "part_confidence",
        "part_valid_mask",
        "pose",
        "pose_valid_mask",
    )
    for name in temporal_names:
        if name in output:
            output[name] = torch.flip(output[name], dims=(1,))  # type: ignore[arg-type]
    for prefix in ("short", "long", "part", "pose"):
        key = f"{prefix}_centre_index"
        values_key = f"{prefix}_features" if prefix in {"short", "long"} else prefix
        if key in output and values_key in output:
            output[key] = output[values_key].shape[1] - 1 - int(output[key])  # type: ignore[index,union-attr]
    return output


def jittered_camera_kwargs(
    kwargs: Mapping[str, object],
    *,
    maximum_shift: float = 0.01,
) -> dict[str, object]:
    output = dict(kwargs)
    static = output["static_features"].clone()  # type: ignore[union-attr]
    batch = static.shape[0]
    shift = (torch.rand(batch, 2, device=static.device, dtype=static.dtype) * 2.0 - 1.0)
    shift = shift * maximum_shift
    static[:, -4:-2] += shift
    output["static_features"] = static
    for name in ("short_features", "long_features"):
        if name in output:
            values = output[name].clone()  # type: ignore[union-attr]
            values[:, :, -4:-2] += shift[:, None]
            output[name] = values
    return output


def coherent_feature_augmentation(
    kwargs: Mapping[str, object],
    *,
    feature_noise: float,
    geometry_jitter: float,
) -> dict[str, object]:
    output = dict(kwargs)
    static = output["static_features"]  # type: ignore[assignment]
    batch, width = static.shape
    visual_width = width - BASE_GEOMETRY_DIM
    noise = torch.randn(batch, visual_width, device=static.device, dtype=static.dtype)
    noise = noise * float(feature_noise)
    augmented_static = static.clone()
    augmented_static[:, :visual_width] += noise
    output["static_features"] = augmented_static
    for name in ("short_features", "long_features"):
        if name not in output:
            continue
        values = output[name].clone()  # type: ignore[union-attr]
        values[:, :, :visual_width] += noise[:, None]
        output[name] = values
    if geometry_jitter > 0.0:
        output = jittered_camera_kwargs(output, maximum_shift=geometry_jitter)
    return output
