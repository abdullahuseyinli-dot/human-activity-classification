"""Leakage-safe pose/velocity mechanism control for the temporal study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hac.vcoco_v3_cuda_heads import CudaStandardizedLogisticRegression
from hac.vcoco_v3_models import CudaPrimalLinearSVM
from hac.vcoco_v3_temporal import (
    TEMPORAL_CLASS_NAMES,
    pose_velocity_summary,
    uniform_clip_indices,
    validate_temporal_manifest,
)


class PoseControlUnavailableError(RuntimeError):
    """Raised when the declared normalized pose evidence is not available."""


@dataclass(frozen=True)
class PoseControlFeatures:
    values: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    recording_ids: np.ndarray
    track_ids: np.ndarray


def _feature_path(manifest_directory: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest_directory / path).resolve()


def extract_pose_control_features(
    frame: pd.DataFrame,
    *,
    candidate: dict,
    manifest_directory: str | Path,
    confidence_threshold: float = 0.2,
) -> PoseControlFeatures:
    """Extract fixed pose summaries without opening rows outside ``frame``."""

    rows = validate_temporal_manifest(frame).reset_index(drop=True)
    root = Path(manifest_directory).resolve()
    samples = int(candidate["uniform_samples"])
    window_seconds = float(candidate["window_seconds"])
    if samples < 2 or window_seconds <= 0.0:
        raise ValueError("The pose control requires at least two samples and a positive window")
    summaries = []
    labels = []
    class_to_index = {name: index for index, name in enumerate(TEMPORAL_CLASS_NAMES)}
    for row in rows.itertuples(index=False):
        path = _feature_path(root, str(row.feature_path))
        with np.load(path, allow_pickle=False) as payload:
            if "pose" not in payload.files:
                raise PoseControlUnavailableError(
                    f"Normalized pose is absent for development sample {row.sample_id}"
                )
            pose = np.asarray(payload["pose"], dtype=float)
        expected_frames = int(row.frame_count)
        if pose.ndim != 3 or pose.shape[0] != expected_frames or pose.shape[2] not in {2, 3}:
            raise ValueError(f"Pose feature shape drift: {path}")
        span_frames = max(1, int(round(float(row.frames_per_second) * window_seconds)))
        indices = uniform_clip_indices(
            expected_frames,
            center_index=int(row.center_frame_index),
            samples=samples,
            span_frames=span_frames,
        )
        summaries.append(
            pose_velocity_summary(
                pose[indices],
                frames_per_second=float(row.frames_per_second),
                frame_indices=indices,
                confidence_threshold=confidence_threshold,
            )
        )
        labels.append(class_to_index[str(row.label)])
    if not summaries:
        raise ValueError("The pose-control partition is empty")
    values = np.stack(summaries).astype(np.float32, copy=False)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Pose-control summaries contain non-finite values")
    return PoseControlFeatures(
        values=values,
        labels=np.asarray(labels, dtype=np.int64),
        sample_ids=rows["sample_id"].astype(str).to_numpy(dtype=str),
        recording_ids=rows["recording_id"].astype(str).to_numpy(dtype=str),
        track_ids=rows["track_id"].astype(str).to_numpy(dtype=str),
    )


class CudaStandardizedPoseSVM:
    """Standardize pose summaries and fit the declared linear SVM entirely on CUDA."""

    def __init__(
        self,
        *,
        c_value: float,
        class_weight: str,
        maximum_iterations: int,
        tolerance: float,
        seed: int,
    ) -> None:
        self.c_value = float(c_value)
        self.class_weight = str(class_weight)
        self.maximum_iterations = int(maximum_iterations)
        self.tolerance = float(tolerance)
        self.seed = int(seed)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.svm_: CudaPrimalLinearSVM | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> CudaStandardizedPoseSVM:
        if not torch.cuda.is_available():
            raise RuntimeError("The declared pose-control SVM requires CUDA")
        values = np.asarray(features, dtype=np.float32)
        targets = np.asarray(labels, dtype=np.int64)
        if values.ndim != 2 or targets.ndim != 1 or len(values) != len(targets):
            raise ValueError("Pose-control features and labels do not align")
        device = torch.device("cuda")
        tensor = torch.as_tensor(values, device=device)
        mean = tensor.mean(dim=0)
        scale = tensor.var(dim=0, correction=0).sqrt()
        scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
        standardized = ((tensor - mean) / scale).cpu().numpy()
        model = CudaPrimalLinearSVM(
            c_value=self.c_value,
            class_weight=self.class_weight,
            maximum_iterations=self.maximum_iterations,
            tolerance=self.tolerance,
            seed=self.seed,
        ).fit(standardized, targets)
        self.mean_ = mean.cpu().numpy()
        self.scale_ = scale.cpu().numpy()
        self.svm_ = model
        self.classes_ = np.asarray(model.classes_)
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.svm_ is None:
            raise RuntimeError("The CUDA pose-control SVM has not been fitted")
        if self.svm_.coef_ is None or self.svm_.intercept_ is None:
            raise RuntimeError("The CUDA pose-control SVM parameters are absent")
        device = torch.device("cuda")
        values = torch.as_tensor(np.asarray(features, dtype=np.float32), device=device)
        mean = torch.as_tensor(self.mean_, device=device)
        scale = torch.as_tensor(self.scale_, device=device)
        coefficients = torch.as_tensor(self.svm_.coef_, device=device)
        intercept = torch.as_tensor(self.svm_.intercept_, device=device)
        with torch.inference_mode():
            scores = ((values - mean) / scale) @ coefficients + intercept
        return scores.cpu().numpy()


def build_pose_svm(
    *,
    c_value: float,
    class_weight: str,
    seed: int,
    maximum_iterations: int = 800,
    tolerance: float = 1e-4,
) -> CudaStandardizedPoseSVM:
    if c_value <= 0.0:
        raise ValueError("SVM C must be positive")
    if class_weight not in {"none", "balanced"}:
        raise ValueError("class_weight must be 'none' or 'balanced'")
    return CudaStandardizedPoseSVM(
        c_value=float(c_value),
        class_weight=class_weight,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
        seed=int(seed),
    )


def pose_decision_scores(model: CudaStandardizedPoseSVM, values: np.ndarray) -> np.ndarray:
    scores = np.asarray(model.decision_function(values), dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(TEMPORAL_CLASS_NAMES):
        raise RuntimeError("The pose SVM did not produce one score per declared class")
    return scores


def fit_pose_score_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    maximum_iterations: int = 400,
    tolerance: float = 1e-5,
) -> CudaStandardizedLogisticRegression:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.shape != (len(labels), len(TEMPORAL_CLASS_NAMES)):
        raise ValueError("Pose score and label rows do not align")
    if set(np.unique(labels)) != set(range(len(TEMPORAL_CLASS_NAMES))):
        raise RuntimeError("Pose score calibration requires all declared classes")
    calibrator = CudaStandardizedLogisticRegression(
        c_value=1.0,
        class_weight="none",
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
        seed=int(seed),
    )
    calibrator.fit(scores, labels)
    return calibrator


def predict_pose_probabilities(bundle: dict, values: np.ndarray) -> np.ndarray:
    model = bundle.get("svm")
    calibrator = bundle.get("score_calibrator")
    if not isinstance(model, CudaStandardizedPoseSVM) or not isinstance(
        calibrator, CudaStandardizedLogisticRegression
    ):
        raise ValueError("Pose-control bundle is malformed")
    probabilities = np.asarray(
        calibrator.predict_proba(pose_decision_scores(model, values)), dtype=float
    )
    if not np.array_equal(calibrator.classes_, np.arange(len(TEMPORAL_CLASS_NAMES))):
        raise RuntimeError("Pose calibrator class order drifted")
    if probabilities.shape != (len(values), len(TEMPORAL_CLASS_NAMES)):
        raise RuntimeError("Pose calibrator returned an invalid probability matrix")
    return probabilities
