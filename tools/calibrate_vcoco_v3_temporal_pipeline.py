"""Calibrate locked temporal models and seal the pipeline before confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.vcoco_v3_cuda_heads import (
    CudaStandardizedLogisticRegression,
    cuda_logistic_fit_audit,
    reset_cuda_logistic_fit_audit,
)
from hac.vcoco_v3_temporal import (
    aps_nonconformity_scores,
    aps_prediction_sets,
    evaluate_routing_curve,
    fit_aps_threshold,
    teacher_advantage_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", type=Path, default=Path("experiments/vcoco_v3_temporal_grid.json")
    )
    parser.add_argument(
        "--temporal-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/temporal_grid_lock.json"),
    )
    parser.add_argument(
        "--development-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/development_final/summary.json"),
    )
    parser.add_argument(
        "--manifest-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/temporal_manifest_lock.json"),
    )
    parser.add_argument(
        "--model-root", type=Path, default=Path(".runs/vcoco_v3/temporal/pipeline_models")
    )
    parser.add_argument(
        "--pose-control-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/pose_control/summary.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".runs/vcoco_v3/temporal/pipeline_lock")
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def model_path(root: Path, role: str, candidate: str | None, seed: int) -> Path:
    if role == "student":
        return root / "student" / str(candidate) / f"seed-{seed}"
    return root / role / f"seed-{seed}"


def load_model_run(path: Path, role: str, candidate: str | None, seed: int) -> dict:
    summary_path = path / "summary.json"
    predictions_path = path / "calibration_predictions.npz"
    checkpoint_path = path / "checkpoint.pt"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "status": "VCOCO_V3_TEMPORAL_FINAL_MODEL_COMPLETE",
        "model_role": role,
        "student_candidate": candidate,
        "seed": seed,
        "confirmation_samples_read": 0,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise RuntimeError(f"Final temporal model drift at {path}: {field}")
    for artifact in (predictions_path, checkpoint_path):
        if sha256_file(artifact) != summary["artifact_sha256"][artifact.name]:
            raise RuntimeError(f"Final temporal artifact drift at {path}: {artifact.name}")
    with np.load(predictions_path, allow_pickle=False) as payload:
        sample_ids = payload["sample_ids"].astype(str)
        recording_ids = payload["recording_ids"].astype(str)
        labels = payload["labels"].astype(int)
        probabilities = payload["probabilities"].astype(float)
        identifiability_scores = payload["identifiability_scores"].astype(float)
    return {
        "summary_path": summary_path,
        "predictions_path": predictions_path,
        "checkpoint_path": checkpoint_path,
        "sample_ids": sample_ids,
        "recording_ids": recording_ids,
        "labels": labels,
        "probabilities": probabilities,
        "identifiability_scores": identifiability_scores,
    }


def load_pose_control(summary_path: Path) -> tuple[dict, dict | None, dict[str, str]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    status = summary.get("status")
    if status not in {
        "VCOCO_V3_POSE_CONTROL_COMPLETE",
        "VCOCO_V3_POSE_CONTROL_UNAVAILABLE",
    }:
        raise RuntimeError(
            "The pose/velocity control has not reached a terminal pre-confirmation state"
        )
    if summary.get("confirmation_samples_read") != 0:
        raise RuntimeError("The pose/velocity control opened confirmation data before locking")
    if status == "VCOCO_V3_POSE_CONTROL_UNAVAILABLE":
        return summary, None, {}
    root = summary_path.parent
    predictions_path = root / "calibration_predictions.npz"
    bundle_path = root / "model.joblib"
    for artifact in (predictions_path, bundle_path):
        if sha256_file(artifact) != summary["artifact_sha256"][artifact.name]:
            raise RuntimeError(f"Pose-control artifact drift: {artifact.name}")
    with np.load(predictions_path, allow_pickle=False) as payload:
        values = {
            "sample_ids": payload["sample_ids"].astype(str),
            "recording_ids": payload["recording_ids"].astype(str),
            "labels": payload["labels"].astype(int),
            "probabilities": payload["probabilities"].astype(float),
            "identifiability_scores": np.asarray([], dtype=float),
        }
    return (
        summary,
        values,
        {
            "summary": sha256_file(summary_path),
            "model": sha256_file(bundle_path),
            "calibration_predictions": sha256_file(predictions_path),
        },
    )


def aggregate_runs(runs: list[dict]) -> dict:
    reference = runs[0]
    for run in runs[1:]:
        for field in ("sample_ids", "recording_ids", "labels"):
            if not np.array_equal(run[field], reference[field]):
                raise RuntimeError("Calibration rows differ across final model seeds")
    score_arrays = [run["identifiability_scores"] for run in runs]
    scores = (
        np.stack(score_arrays).mean(axis=0) if score_arrays[0].size else np.asarray([], dtype=float)
    )
    return {
        "sample_ids": reference["sample_ids"],
        "recording_ids": reference["recording_ids"],
        "labels": reference["labels"],
        "probabilities": np.stack([run["probabilities"] for run in runs]).mean(axis=0),
        "identifiability_scores": scores,
    }


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0))
    scaled = logits / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    values = np.exp(scaled)
    return values / values.sum(axis=1, keepdims=True)


def fit_temperature(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    if not torch.cuda.is_available():
        raise RuntimeError("Temperature calibration requires CUDA")
    device = torch.device("cuda")
    logits = torch.as_tensor(
        np.log(np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)),
        dtype=torch.float64,
        device=device,
    )
    targets = torch.as_tensor(labels, dtype=torch.long, device=device)
    raw_temperature = torch.tensor(
        -1.3862943611198906,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.LBFGS(
        [raw_temperature],
        lr=1.0,
        max_iter=100,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = 0.25 + 3.75 * torch.sigmoid(raw_temperature)
        loss = torch.nn.functional.cross_entropy(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float((0.25 + 3.75 * torch.sigmoid(raw_temperature)).detach().cpu())
    if not np.isfinite(temperature) or not 0.25 <= temperature <= 4.0:
        raise RuntimeError("Temperature calibration failed")
    return temperature


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Temporal routing calibration requires CUDA")
    grid_path = args.grid.resolve()
    lock_path = args.temporal_lock.resolve()
    development_path = args.development_summary.resolve()
    manifest_lock_path = args.manifest_lock.resolve()
    model_root = args.model_root.resolve()
    pose_summary_path = args.pose_control_summary.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    manifest_lock = json.loads(manifest_lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_TEMPORAL_GRID_LOCKED_BEFORE_FIT":
        raise RuntimeError("The temporal grid is not locked")
    if development.get("status") != "VCOCO_V3_TEMPORAL_DEVELOPMENT_COMPLETE":
        raise RuntimeError("Temporal development is incomplete")
    if manifest_lock.get("confirmation_feature_arrays_opened") != 0:
        raise RuntimeError("Confirmation features were opened before pipeline calibration")
    seeds = list(map(int, lock["seeds"]))
    classification_student = str(development["classification_student"])
    routing_student = str(development["routing_student"])
    requested = {
        "static": ("static", None),
        "teacher": ("teacher", None),
        "classification_student": ("student", classification_student),
        "routing_student": ("student", routing_student),
    }
    aggregates = {}
    model_evidence = {}
    for name, (role, candidate) in requested.items():
        runs = []
        for seed in seeds:
            path = model_path(model_root, role, candidate, seed)
            run = load_model_run(path, role, candidate, seed)
            runs.append(run)
            model_evidence[path.relative_to(model_root).as_posix()] = {
                "summary": sha256_file(run["summary_path"]),
                "checkpoint": sha256_file(run["checkpoint_path"]),
                "calibration_predictions": sha256_file(run["predictions_path"]),
            }
        aggregates[name] = aggregate_runs(runs)
    reference = aggregates["static"]
    for name, values in aggregates.items():
        for field in ("sample_ids", "recording_ids", "labels"):
            if not np.array_equal(values[field], reference[field]):
                raise RuntimeError(f"Calibration rows differ for {name}")
    pose_summary, pose_values, pose_evidence = load_pose_control(pose_summary_path)
    if pose_values is not None:
        for field in ("sample_ids", "recording_ids", "labels"):
            if not np.array_equal(pose_values[field], reference[field]):
                raise RuntimeError(f"Pose-control calibration rows differ: {field}")
        aggregates["pose_velocity_svm"] = pose_values

    temperatures = {}
    calibrated = {}
    metric_rows = []
    for name, values in aggregates.items():
        temperature = fit_temperature(reference["labels"], values["probabilities"])
        temperatures[name] = temperature
        calibrated[name] = temperature_scale(values["probabilities"], temperature)
        before = classification_metrics(reference["labels"], values["probabilities"])
        after = classification_metrics(reference["labels"], calibrated[name])
        metric_rows.extend(
            [
                {"family": name, "calibrated": False, **before},
                {"family": name, "calibrated": True, **after},
            ]
        )

    margin = float(grid["distillation"]["teacher_advantage_minimum_log_likelihood_gain"])
    advantage = teacher_advantage_targets(
        reference["labels"],
        calibrated["static"],
        calibrated["teacher"],
        minimum_log_likelihood_gain=margin,
    )
    raw_scores = aggregates["routing_student"]["identifiability_scores"]
    if raw_scores.shape != advantage.shape:
        raise RuntimeError("Routing scores are absent or do not align with calibration targets")
    if len(np.unique(advantage)) < 2:
        routing_calibrator = None
        calibrated_scores = raw_scores
        routing_optimizer = None
    else:
        logits = np.log(
            np.clip(raw_scores, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - raw_scores, 1e-6, 1.0)
        )
        reset_cuda_logistic_fit_audit()
        routing = grid["routing"]
        model = CudaStandardizedLogisticRegression(
            c_value=float(routing["calibrator_C"]),
            class_weight="none",
            maximum_iterations=int(routing["calibrator_maximum_iterations"]),
            tolerance=float(routing["calibrator_gradient_tolerance"]),
            seed=20260905,
        )
        model.fit(logits[:, None], advantage)
        calibrated_scores = model.predict_proba(logits[:, None])[:, 1]
        raw_coefficient = float(model.coef_[0] / model.scale_[0])
        raw_intercept = float(
            model.intercept_[0] - model.mean_[0] * model.coef_[0] / model.scale_[0]
        )
        routing_calibrator = {
            "coefficient": raw_coefficient,
            "intercept": raw_intercept,
            "input": "logit_of_seed_mean_identifiability_probability",
            "solver": "pytorch_cuda_lbfgs_logistic",
        }
        routing_optimizer = cuda_logistic_fit_audit()[0]
    routing_curve = evaluate_routing_curve(
        reference["labels"],
        calibrated["classification_student"],
        calibrated["teacher"],
        calibrated_scores,
        clip_fractions=grid["routing"]["clip_budget_fractions"],
        advantage_targets=advantage,
    )

    aps_thresholds = {}
    aps_rows = []
    scores = aps_nonconformity_scores(reference["labels"], calibrated["classification_student"])
    for miscoverage in grid["prediction_sets"]["miscoverage_levels"]:
        threshold = fit_aps_threshold(scores, miscoverage=float(miscoverage))
        membership = aps_prediction_sets(calibrated["classification_student"], threshold)
        coverage = membership[np.arange(len(reference["labels"])), reference["labels"]].mean()
        key = f"alpha_{float(miscoverage):g}"
        aps_thresholds[key] = threshold
        aps_rows.append(
            {
                "miscoverage": float(miscoverage),
                "threshold": threshold,
                "empirical_coverage": float(coverage),
                "mean_set_size": float(membership.sum(axis=1).mean()),
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "calibration_metrics.csv"
    routing_path = output_dir / "calibration_routing_curve.csv"
    aps_path = output_dir / "calibration_prediction_sets.csv"
    probabilities_path = output_dir / "calibration_predictions.npz"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(routing_curve).to_csv(routing_path, index=False)
    pd.DataFrame(aps_rows).to_csv(aps_path, index=False)
    np.savez_compressed(
        probabilities_path,
        sample_ids=reference["sample_ids"],
        recording_ids=reference["recording_ids"],
        labels=reference["labels"],
        advantage_targets=advantage,
        calibrated_routing_scores=calibrated_scores,
        **{f"{name}_probabilities": values for name, values in calibrated.items()},
    )
    calibration = {
        "temperature": temperatures,
        "pose_control_status": pose_summary["status"],
        "routing_calibrator": routing_calibrator,
        "routing_enabled": bool(
            development["routing_eligible_for_calibration"] and routing_calibrator is not None
        ),
        "routing_clip_budget_fractions": grid["routing"]["clip_budget_fractions"],
        "aps_thresholds": aps_thresholds,
        "aps_miscoverage_levels": grid["prediction_sets"]["miscoverage_levels"],
        "teacher_advantage_minimum_log_likelihood_gain": margin,
        "routing_calibrator_optimizer": routing_optimizer,
        "calibration_device": torch.cuda.get_device_name(0),
    }
    calibration_path = output_dir / "calibration.json"
    write_json(calibration_path, calibration)
    summary = {
        "status": "VCOCO_V3_TEMPORAL_PIPELINE_LOCKED_BEFORE_CONFIRMATION",
        "classification_student": classification_student,
        "routing_student": routing_student,
        "selected_teacher": development["selected_teacher"],
        "seeds": seeds,
        "calibration_samples": len(reference["labels"]),
        "calibration_recordings": int(pd.Series(reference["recording_ids"]).nunique()),
        "confirmation_feature_arrays_opened": 0,
        "confirmation_evaluations_run": 0,
        "calibration_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "source_sha256": {
            "temporal_grid": sha256_file(grid_path),
            "temporal_grid_lock": sha256_file(lock_path),
            "development_summary": sha256_file(development_path),
            "temporal_manifest_lock": sha256_file(manifest_lock_path),
            "pose_control_summary": sha256_file(pose_summary_path),
            "calibration_runner": sha256_file(Path(__file__).resolve()),
            "cuda_heads_source": sha256_file(
                Path(__file__).resolve().parents[1] / "src/hac/vcoco_v3_cuda_heads.py"
            ),
        },
        "model_artifact_sha256": model_evidence,
        "pose_control_artifact_sha256": pose_evidence,
        "artifact_sha256": {
            metrics_path.name: sha256_file(metrics_path),
            routing_path.name: sha256_file(routing_path),
            aps_path.name: sha256_file(aps_path),
            probabilities_path.name: sha256_file(probabilities_path),
            calibration_path.name: sha256_file(calibration_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
