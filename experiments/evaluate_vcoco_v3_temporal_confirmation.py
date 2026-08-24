"""Open the sealed temporal confirmation split once and evaluate the locked pipeline."""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_analysis import per_class_metrics
from hac.vcoco_v3_models import (
    CLASS_NAMES,
    holm_adjust,
    locomotion_f1,
    paired_cluster_bootstrap,
)
from hac.vcoco_v3_pose_control import (
    extract_pose_control_features,
    predict_pose_probabilities,
)
from hac.vcoco_v3_temporal import (
    StaticIdentifiabilityStudent,
    aps_prediction_sets,
    evaluate_routing_curve,
    routed_probabilities,
)
from hac.vcoco_v3_temporal_training import (
    build_temporal_development_model,
    evaluate_temporal_development,
    make_temporal_loader,
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
        "--pipeline-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/pipeline_lock/summary.json"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/pipeline_lock/calibration.json"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--provider-confirmation-summary",
        type=Path,
        help="Required for an external provider-test confirmation partition",
    )
    parser.add_argument(
        "--confirmation-cache-summary",
        type=Path,
        help="Required for an external provider-test confirmation partition",
    )
    parser.add_argument(
        "--source-only-summary",
        type=Path,
        help="Label-blind source-only prediction summary for provider-test confirmation",
    )
    parser.add_argument(
        "--source-only-predictions",
        type=Path,
        help="Label-blind source-only probabilities for provider-test confirmation",
    )
    parser.add_argument(
        "--model-root", type=Path, default=Path(".runs/vcoco_v3/temporal/pipeline_models")
    )
    parser.add_argument(
        "--pose-control-root",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/pose_control"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".runs/vcoco_v3/temporal/confirmation")
    )
    parser.add_argument("--workers", type=int, default=2)
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


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def okutama_subgroup_metrics(
    frame: pd.DataFrame,
    labels: np.ndarray,
    family_probabilities: dict[str, np.ndarray],
    grid: dict,
) -> list[dict]:
    definitions = grid.get("subgroup_definitions")
    if not definitions:
        return []
    required = {
        "walking_running_subtype",
        "transition_window",
        "center_occluded",
        "window_any_occluded",
        "drone_view",
        "part_of_day",
        "recording_id",
        "bbox_area_fraction",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Okutama confirmation metadata lacks subgroup columns: {missing}")
    values: dict[str, pd.Series] = {
        "walking_running_subtype": frame["walking_running_subtype"].fillna("not_applicable"),
        "transition_window": frame["transition_window"].astype(str).str.lower(),
        "center_occluded": frame["center_occluded"].astype(str).str.lower(),
        "window_any_occluded": frame["window_any_occluded"].astype(str).str.lower(),
        "drone_view": frame["drone_view"].astype(str),
        "part_of_day": frame["part_of_day"].astype(str),
        "scenario": frame["recording_id"].astype(str),
        "person_scale": pd.cut(
            frame["bbox_area_fraction"].astype(float),
            bins=definitions["person_scale_area_fraction_edges"],
            labels=definitions["person_scale_names"],
            include_lowest=True,
            right=False,
        ).astype(str),
    }
    minimum = int(definitions["minimum_reported_rows"])
    output = []
    for family, probabilities in family_probabilities.items():
        predictions = probabilities.argmax(axis=1)
        for axis, axis_values in values.items():
            for value in sorted(axis_values.unique()):
                mask = axis_values.eq(value).to_numpy()
                if int(mask.sum()) < minimum:
                    continue
                output.append(
                    {
                        "family": family,
                        "axis": axis,
                        "value": str(value),
                        "samples": int(mask.sum()),
                        "accuracy": float((predictions[mask] == labels[mask]).mean()),
                        "macro_f1_all_classes": float(
                            f1_score(
                                labels[mask],
                                predictions[mask],
                                labels=np.arange(len(CLASS_NAMES)),
                                average="macro",
                                zero_division=0,
                            )
                        ),
                        "locomotion_f1": float(
                            f1_score(
                                labels[mask] == 2,
                                predictions[mask] == 2,
                                average="binary",
                                zero_division=0,
                            )
                        ),
                    }
                )
    return output


def load_model(
    path: Path,
    *,
    role: str,
    candidate: str | None,
    seed: int,
    lock: dict,
    grid: dict,
    teacher_candidate: dict,
    expected_evidence: dict,
    device: torch.device,
) -> torch.nn.Module:
    summary_path = path / "summary.json"
    checkpoint_path = path / "checkpoint.pt"
    if sha256_file(summary_path) != expected_evidence["summary"]:
        raise RuntimeError(f"Locked final model summary drift: {path}")
    if sha256_file(checkpoint_path) != expected_evidence["checkpoint"]:
        raise RuntimeError(f"Locked final model checkpoint drift: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "VCOCO_V3_TEMPORAL_FINAL_MODEL_COMPLETE"
        or summary.get("model_role") != role
        or summary.get("student_candidate") != candidate
        or summary.get("seed") != seed
        or summary.get("confirmation_samples_read") != 0
    ):
        raise RuntimeError(f"Locked final model metadata drift: {path}")
    if role == "student":
        model = StaticIdentifiabilityStudent(
            int(lock["input_dimensions"]),
            hidden_dim=int(grid["architecture"]["static_hidden_dim"]),
            dropout=float(grid["architecture"]["dropout"]),
        )
    else:
        model = build_temporal_development_model(
            role,
            input_dimensions=int(lock["input_dimensions"]),
            architecture=grid["architecture"],
            maximum_length=max(int(item["uniform_samples"]) for item in lock["teacher_candidates"]),
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("teacher_candidate") != teacher_candidate:
        raise RuntimeError(f"Final model used a different teacher window: {path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def load_locked_pose_control(
    root: Path,
    *,
    pipeline: dict,
    calibration: dict,
) -> dict | None:
    summary_path = root / "summary.json"
    if sha256_file(summary_path) != pipeline["source_sha256"]["pose_control_summary"]:
        raise RuntimeError("The locked pose-control summary changed")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != calibration.get("pose_control_status"):
        raise RuntimeError("Pose-control status differs between calibration artifacts")
    if summary.get("confirmation_samples_read") != 0:
        raise RuntimeError("The pose control opened confirmation data before pipeline locking")
    if summary["status"] == "VCOCO_V3_POSE_CONTROL_UNAVAILABLE":
        if pipeline.get("pose_control_artifact_sha256"):
            raise RuntimeError(
                "An unavailable pose control unexpectedly has locked model artifacts"
            )
        return None
    if summary["status"] != "VCOCO_V3_POSE_CONTROL_COMPLETE":
        raise RuntimeError("The pose control is not in a recognized locked state")
    bundle_path = root / "model.joblib"
    evidence = pipeline["pose_control_artifact_sha256"]
    if sha256_file(bundle_path) != evidence["model"]:
        raise RuntimeError("The locked pose-control model changed")
    bundle = joblib.load(bundle_path)
    if bundle.get("version") != "vcoco-v3-pose-control-2-cuda":
        raise RuntimeError("The locked pose-control solver version changed")
    if bundle.get("teacher_candidate") != pipeline["selected_teacher"]:
        raise RuntimeError("The pose control used a different temporal window")
    return bundle


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    started = time.perf_counter()
    grid_path = args.grid.resolve()
    lock_path = args.temporal_lock.resolve()
    pipeline_path = args.pipeline_lock.resolve()
    calibration_path = args.calibration.resolve()
    manifest_path = args.manifest.resolve()
    model_root = args.model_root.resolve()
    pose_root = args.pose_control_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    open_path = output_dir / "confirmation_open.json"
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "VCOCO_V3_TEMPORAL_CONFIRMATION_COMPLETE":
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
            return
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_TEMPORAL_GRID_LOCKED_BEFORE_FIT":
        raise RuntimeError("The temporal grid is not locked")
    if pipeline.get("status") != "VCOCO_V3_TEMPORAL_PIPELINE_LOCKED_BEFORE_CONFIRMATION":
        raise RuntimeError("The temporal pipeline is not locked for confirmation")
    if sha256_file(calibration_path) != pipeline["artifact_sha256"][calibration_path.name]:
        raise RuntimeError("The locked calibration parameters changed")
    pipeline_hash = sha256_file(pipeline_path)
    confirmation_provenance = {}
    source_only_predictions_path = None
    source_only_summary = None
    if grid.get("dataset", {}).get("confirmation_partition") == "provider_test":
        if (
            args.provider_confirmation_summary is None
            or args.confirmation_cache_summary is None
            or args.source_only_summary is None
            or args.source_only_predictions is None
        ):
            raise ValueError(
                "External provider-test evaluation requires audit, cache, and source-only evidence"
            )
        provider_path = args.provider_confirmation_summary.resolve()
        cache_path = args.confirmation_cache_summary.resolve()
        source_only_summary_path = args.source_only_summary.resolve()
        source_only_predictions_path = args.source_only_predictions.resolve()
        provider = json.loads(provider_path.read_text(encoding="utf-8"))
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        source_only_summary = json.loads(
            source_only_summary_path.read_text(encoding="utf-8")
        )
        if provider.get("status") != "OKUTAMA_CONFIRMATION_ARCHIVE_AND_CENTRES_AUDITED":
            raise RuntimeError("The provider confirmation audit is incomplete")
        if provider.get("confirmation_open_number") != 1:
            raise RuntimeError("The provider confirmation open number changed")
        if cache.get("status") != "VCOCO_V3_OKUTAMA_CONFIRMATION_FEATURE_STORE_COMPLETE":
            raise RuntimeError("The provider confirmation feature cache is incomplete")
        if cache.get("confirmation_open_number") != 1:
            raise RuntimeError("The confirmation cache belongs to a different open")
        if cache["source_sha256"].get("audit_summary") != sha256_file(provider_path):
            raise RuntimeError("The confirmation cache belongs to a different provider audit")
        if cache["source_sha256"].get("pipeline_lock") != pipeline_hash:
            raise RuntimeError("The confirmation cache belongs to a different pipeline lock")
        if cache["artifact_sha256"].get(manifest_path.name) != sha256_file(manifest_path):
            raise RuntimeError("The confirmation manifest changed after CUDA caching")
        if (
            source_only_summary.get("status")
            != "OKUTAMA_SOURCE_ONLY_TRANSFER_PREDICTIONS_COMPLETE"
        ):
            raise RuntimeError("The source-only confirmation predictions are incomplete")
        if source_only_summary.get("target_partition") != "confirmation":
            raise RuntimeError("The source-only predictions are not for confirmation")
        if source_only_summary.get("target_labels_read") != 0:
            raise RuntimeError("Source-only confirmation fitting read target labels")
        if source_only_summary["source_sha256"].get("pipeline_lock") != pipeline_hash:
            raise RuntimeError("The source-only predictions belong to a different pipeline")
        if (
            source_only_summary["source_sha256"].get("target_cache_summary")
            != sha256_file(cache_path)
        ):
            raise RuntimeError("The source-only predictions belong to a different feature cache")
        if (
            source_only_summary["artifact_sha256"].get(source_only_predictions_path.name)
            != sha256_file(source_only_predictions_path)
        ):
            raise RuntimeError("The source-only confirmation predictions changed after fitting")
        confirmation_provenance = {
            "provider_confirmation_summary": sha256_file(provider_path),
            "confirmation_cache_summary": sha256_file(cache_path),
            "confirmation_manifest": sha256_file(manifest_path),
            "source_only_summary": sha256_file(source_only_summary_path),
            "source_only_predictions": sha256_file(source_only_predictions_path),
        }
    pose_bundle = load_locked_pose_control(
        pose_root,
        pipeline=pipeline,
        calibration=calibration,
    )
    if open_path.is_file():
        opened = json.loads(open_path.read_text(encoding="utf-8"))
        if opened.get("pipeline_lock_sha256") != pipeline_hash:
            raise RuntimeError("Confirmation was already opened for a different pipeline")
        if opened.get("status") == "COMPLETE":
            raise RuntimeError("Confirmation open ledger is complete but its summary is missing")
    else:
        write_json(
            open_path,
            {
                "status": "STARTED",
                "opened_at_utc": datetime.now(UTC).isoformat(),
                "pipeline_lock_sha256": pipeline_hash,
                "manifest_sha256": sha256_file(manifest_path),
                "confirmation_open_number": 1,
            },
        )

    try:
        frame = pd.read_csv(
            manifest_path,
            dtype={"sample_id": str, "recording_id": str, "track_id": str},
        )
        confirmation = frame[frame["split"].eq("confirmation")].reset_index(drop=True)
        if not len(confirmation):
            raise RuntimeError("The sealed temporal confirmation split is empty")
        if not torch.cuda.is_available():
            raise RuntimeError("Temporal confirmation inference requires CUDA")
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
        teacher_candidate = pipeline["selected_teacher"]
        loader = make_temporal_loader(
            confirmation,
            candidate=teacher_candidate,
            manifest_directory=manifest_path.parent,
            batch_size=int(grid["training"]["batch_size"]),
            shuffle=False,
            seed=0,
            workers=args.workers,
        )
        requested = {
            "static": ("static", None),
            "teacher": ("teacher", None),
            "classification_student": ("student", pipeline["classification_student"]),
            "routing_student": ("student", pipeline["routing_student"]),
        }
        aggregates = {}
        for name, (role, candidate) in requested.items():
            seed_predictions = []
            seed_scores = []
            reference = None
            for seed in pipeline["seeds"]:
                path = model_path(model_root, role, candidate, int(seed))
                evidence = pipeline["model_artifact_sha256"][
                    path.relative_to(model_root).as_posix()
                ]
                model = load_model(
                    path,
                    role=role,
                    candidate=candidate,
                    seed=int(seed),
                    lock=lock,
                    grid=grid,
                    teacher_candidate=teacher_candidate,
                    expected_evidence=evidence,
                    device=device,
                )
                result = evaluate_temporal_development(
                    model, "static" if role == "student" else role, loader, device
                )
                if reference is None:
                    reference = result
                else:
                    for field in ("sample_ids", "recording_ids", "labels"):
                        if not np.array_equal(result[field], reference[field]):
                            raise RuntimeError(f"Confirmation seed rows differ for {name}")
                seed_predictions.append(result["probabilities"])
                if result["identifiability_scores"] is not None:
                    seed_scores.append(result["identifiability_scores"])
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            aggregates[name] = {
                "sample_ids": reference["sample_ids"],
                "recording_ids": reference["recording_ids"],
                "labels": reference["labels"],
                "probabilities": np.stack(seed_predictions).mean(axis=0),
                "identifiability_scores": (
                    np.stack(seed_scores).mean(axis=0) if seed_scores else np.asarray([])
                ),
            }
        reference = aggregates["static"]
        if not np.array_equal(
            confirmation["sample_id"].astype(str).to_numpy(), reference["sample_ids"]
        ):
            raise RuntimeError("Confirmation metadata order differs from model predictions")
        source_only_probabilities = None
        if source_only_predictions_path is not None:
            with np.load(source_only_predictions_path, allow_pickle=False) as payload:
                source_only_indices = payload["target_feature_indices"].astype(int)
                source_only_probabilities = payload["probabilities"].astype(float)
                source_only_classes = tuple(map(str, payload["class_names"].tolist()))
            if source_only_classes != CLASS_NAMES:
                raise RuntimeError("Source-only confirmation classes changed")
            expected_indices = confirmation["feature_index"].to_numpy(dtype=int)
            if not np.array_equal(source_only_indices, expected_indices):
                raise RuntimeError("Source-only confirmation feature indices do not align")
            if source_only_probabilities.shape != (len(confirmation), len(CLASS_NAMES)):
                raise RuntimeError("Source-only confirmation probabilities have the wrong shape")
            if not np.isfinite(source_only_probabilities).all():
                raise RuntimeError("Source-only confirmation probabilities are non-finite")
            if not np.allclose(source_only_probabilities.sum(axis=1), 1.0, atol=1e-5):
                raise RuntimeError("Source-only confirmation probabilities are not normalized")
            if source_only_summary.get("target_feature_rows") != len(confirmation):
                raise RuntimeError("Source-only confirmation row count changed")
        if pose_bundle is not None:
            pose_features = extract_pose_control_features(
                confirmation,
                candidate=teacher_candidate,
                manifest_directory=manifest_path.parent,
                confidence_threshold=float(pose_bundle["confidence_threshold"]),
            )
            for field, values in (
                ("sample_ids", pose_features.sample_ids),
                ("recording_ids", pose_features.recording_ids),
                ("labels", pose_features.labels),
            ):
                if not np.array_equal(values, reference[field]):
                    raise RuntimeError(f"Pose-control confirmation rows differ: {field}")
            aggregates["pose_velocity_svm"] = {
                "sample_ids": pose_features.sample_ids,
                "recording_ids": pose_features.recording_ids,
                "labels": pose_features.labels,
                "probabilities": predict_pose_probabilities(pose_bundle, pose_features.values),
                "identifiability_scores": np.asarray([]),
            }
        calibrated = {
            name: temperature_scale(values["probabilities"], calibration["temperature"][name])
            for name, values in aggregates.items()
        }
        raw_scores = aggregates["routing_student"]["identifiability_scores"]
        calibrator = calibration["routing_calibrator"]
        if calibrator is None:
            routing_scores = raw_scores
        else:
            raw_logits = np.log(
                np.clip(raw_scores, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - raw_scores, 1e-6, 1.0)
            )
            routed_logits = float(calibrator["coefficient"]) * raw_logits + float(
                calibrator["intercept"]
            )
            routing_scores = 1.0 / (1.0 + np.exp(-routed_logits))
        routing_curve = evaluate_routing_curve(
            reference["labels"],
            calibrated["classification_student"],
            calibrated["teacher"],
            routing_scores,
            clip_fractions=calibration["routing_clip_budget_fractions"],
        )
        family_probabilities = dict(calibrated)
        if source_only_probabilities is not None:
            family_probabilities["source_only_static"] = source_only_probabilities
        for fraction in calibration["routing_clip_budget_fractions"]:
            count = int(math.ceil(len(routing_scores) * float(fraction)))
            order = np.argsort(-routing_scores, kind="stable")
            route = np.zeros(len(routing_scores), dtype=bool)
            route[order[:count]] = True
            family_probabilities[f"hybrid_budget_{float(fraction):g}"] = routed_probabilities(
                calibrated["classification_student"], calibrated["teacher"], route
            )
        metric_rows = []
        class_rows = []
        for family, probabilities in family_probabilities.items():
            metrics = classification_metrics(reference["labels"], probabilities)
            metrics["locomotion_f1"] = locomotion_f1(reference["labels"], probabilities)
            metric_rows.append({"family": family, **metrics})
            class_rows.extend(
                {"family": family, **row}
                for row in per_class_metrics(reference["labels"], probabilities, CLASS_NAMES)
            )
        comparison_families = [
            "teacher",
            "classification_student",
            *(["source_only_static"] if source_only_probabilities is not None else []),
            *(["pose_velocity_svm"] if pose_bundle is not None else []),
            *[
                f"hybrid_budget_{float(value):g}"
                for value in calibration["routing_clip_budget_fractions"]
                if float(value) not in {0.0, 1.0}
            ],
        ]
        confirmation_seed = int(
            grid["split_policy"].get(
                "confirmation_seed",
                grid["split_policy"].get("confirmation_bootstrap_seed"),
            )
        )
        uncertainty = {
            family: paired_cluster_bootstrap(
                reference["labels"],
                family_probabilities[family],
                family_probabilities["static"],
                reference["recording_ids"],
                resamples=int(grid["selection"]["bootstrap_resamples"]),
                seed=confirmation_seed + 100_000 + index,
            )
            for index, family in enumerate(comparison_families)
        }
        macro_adjusted = holm_adjust(
            {
                family: comparison["macro_f1"]["two_sided_p"]
                for family, comparison in uncertainty.items()
            }
        )
        locomotion_adjusted = holm_adjust(
            {
                family: comparison["per_class_f1"]["walking_running"]["two_sided_p"]
                for family, comparison in uncertainty.items()
            }
        )
        for family, comparison in uncertainty.items():
            comparison["holm_adjusted_macro_p"] = macro_adjusted[family]
            comparison["holm_adjusted_locomotion_p"] = locomotion_adjusted[family]
        aps_rows = []
        for key, threshold in calibration["aps_thresholds"].items():
            membership = aps_prediction_sets(calibrated["classification_student"], float(threshold))
            coverage = membership[np.arange(len(reference["labels"])), reference["labels"]].mean()
            aps_rows.append(
                {
                    "threshold_id": key,
                    "threshold": float(threshold),
                    "empirical_coverage": float(coverage),
                    "mean_set_size": float(membership.sum(axis=1).mean()),
                }
            )

        metrics_path = output_dir / "confirmation_metrics.csv"
        class_path = output_dir / "confirmation_per_class.csv"
        routing_path = output_dir / "confirmation_routing_curve.csv"
        aps_path = output_dir / "confirmation_prediction_sets.csv"
        uncertainty_path = output_dir / "confirmation_uncertainty.json"
        predictions_path = output_dir / "confirmation_predictions.npz"
        subgroups_path = output_dir / "confirmation_subgroups.csv"
        pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
        pd.DataFrame(class_rows).to_csv(class_path, index=False)
        pd.DataFrame(routing_curve).to_csv(routing_path, index=False)
        pd.DataFrame(aps_rows).to_csv(aps_path, index=False)
        write_json(uncertainty_path, uncertainty)
        pd.DataFrame(
            okutama_subgroup_metrics(
                confirmation,
                reference["labels"],
                family_probabilities,
                grid,
            )
        ).to_csv(subgroups_path, index=False)
        np.savez_compressed(
            predictions_path,
            sample_ids=reference["sample_ids"],
            recording_ids=reference["recording_ids"],
            labels=reference["labels"],
            routing_scores=routing_scores,
            **{f"{name}_probabilities": values for name, values in family_probabilities.items()},
        )
        summary = {
            "status": "VCOCO_V3_TEMPORAL_CONFIRMATION_COMPLETE",
            "endpoint": "provider_labels_mapped_to_locked_three_class_ontology",
            "samples": len(reference["labels"]),
            "recordings": int(pd.Series(reference["recording_ids"]).nunique()),
            "confirmation_open_number": 1,
            "inference_device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "routing_enabled_by_validation_gate": bool(calibration["routing_enabled"]),
            "pose_control_status": calibration["pose_control_status"],
            "pipeline_lock_sha256": pipeline_hash,
            "confirmation_provenance_sha256": confirmation_provenance,
            "runtime_seconds": time.perf_counter() - started,
            "artifact_sha256": {
                metrics_path.name: sha256_file(metrics_path),
                class_path.name: sha256_file(class_path),
                routing_path.name: sha256_file(routing_path),
                aps_path.name: sha256_file(aps_path),
                uncertainty_path.name: sha256_file(uncertainty_path),
                predictions_path.name: sha256_file(predictions_path),
                subgroups_path.name: sha256_file(subgroups_path),
            },
        }
        write_json(summary_path, summary)
        opened = json.loads(open_path.read_text(encoding="utf-8"))
        opened.update(
            {
                "status": "COMPLETE",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "summary_sha256": sha256_file(summary_path),
            }
        )
        write_json(open_path, opened)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except Exception as error:
        opened = json.loads(open_path.read_text(encoding="utf-8"))
        opened.update(
            {
                "status": "FAILED_RESUMABLE_SAME_OPEN",
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(open_path, opened)
        raise


if __name__ == "__main__":
    main()
