"""Export the compact V-COCO v3 and Okutama motion-study evidence package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file

EXPECTED_STATUS = {
    "annotation": "VCOCO_V3_HUMAN_PILOT_AUDIT_COMPLETE",
    "nested": "VCOCO_V3_NESTED_CACHED_FUSION_DEVELOPMENT_COMPLETE",
    "spatial": "VCOCO_V3_SPATIAL_DEVELOPMENT_COMPLETE",
    "representations": "VCOCO_V3_MATCHED_REPRESENTATION_DEVELOPMENT_COMPLETE",
    "neural": "VCOCO_V3_NEURAL_STAGE_NOT_ELIGIBLE",
    "amendment": "VCOCO_V3_EXTERNAL_CUDA_AMENDMENT_LOCKED_BEFORE_TARGET_FITTING",
    "development_audit": "OKUTAMA_DEVELOPMENT_ARCHIVE_AND_CENTRES_AUDITED",
    "fewshot": "OKUTAMA_FEWSHOT_TRANSFER_DEVELOPMENT_COMPLETE",
    "teacher": "VCOCO_V3_TEMPORAL_TEACHER_SELECTED",
    "crossfit": "VCOCO_V3_TEMPORAL_CROSSFIT_TARGETS_LOCKED",
    "development": "VCOCO_V3_TEMPORAL_DEVELOPMENT_COMPLETE",
    "pose": "VCOCO_V3_POSE_CONTROL_UNAVAILABLE",
    "pipeline": "VCOCO_V3_TEMPORAL_PIPELINE_LOCKED_BEFORE_CONFIRMATION",
    "confirmation_audit": "OKUTAMA_CONFIRMATION_ARCHIVE_AND_CENTRES_AUDITED",
    "confirmation_features": "VCOCO_V3_OKUTAMA_CONFIRMATION_FEATURE_STORE_COMPLETE",
    "confirmation_source": "OKUTAMA_SOURCE_ONLY_TRANSFER_PREDICTIONS_COMPLETE",
    "confirmation": "VCOCO_V3_TEMPORAL_CONFIRMATION_COMPLETE",
}

METRIC_COLUMNS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "log_loss",
    "brier_score",
    "ece",
    "adaptive_ece",
    "classwise_ece",
    "locomotion_f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path(".runs/vcoco_v3"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/vcoco_v3"))
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def require_status(name: str, payload: dict) -> None:
    expected = EXPECTED_STATUS[name]
    if payload.get("status") != expected:
        raise RuntimeError(f"{name} is incomplete: expected {expected}")


def require_declared_artifact(summary_path: Path, artifact_name: str) -> Path:
    summary = read_json(summary_path)
    expected = summary.get("artifact_sha256", {}).get(artifact_name)
    artifact = summary_path.parent / artifact_name
    if not isinstance(expected, str) or not artifact.is_file():
        raise RuntimeError(f"Missing declared artifact: {artifact}")
    if sha256_file(artifact) != expected:
        raise RuntimeError(f"Artifact hash drift: {artifact}")
    return artifact


def require_cuda_summary(path: Path, expected_hash: str, device: str) -> None:
    if sha256_file(path) != expected_hash:
        raise RuntimeError(f"CUDA run summary hash drift: {path}")
    summary = read_json(path)
    if summary.get("training_device") != device:
        raise RuntimeError(f"Non-CUDA temporal run found: {path}")
    if "+cu" not in str(summary.get("torch_version", "")):
        raise RuntimeError(f"CUDA PyTorch provenance missing: {path}")


def validate_cuda_runs(root: Path, summaries: dict, device: str) -> int:
    checked = 0
    temporal = root / "temporal"

    teacher = summaries["teacher"]
    for key, artifacts in teacher["run_artifact_sha256"].items():
        require_cuda_summary(
            temporal / "development" / key / "summary.json", artifacts["summary"], device
        )
        checked += 1

    crossfit = summaries["crossfit"]
    for key, artifacts in crossfit["run_artifact_sha256"].items():
        require_cuda_summary(
            temporal / "crossfit" / key / "summary.json", artifacts["summary"], device
        )
        checked += 1

    development = summaries["development"]
    for key, artifacts in development["run_artifact_sha256"].items():
        if not key.startswith(("distilled_static/", "identifiability_conditioned_static/")):
            continue
        require_cuda_summary(
            temporal / "students" / key / "summary.json", artifacts["summary"], device
        )
        checked += 1

    pipeline = summaries["pipeline"]
    for key, artifacts in pipeline["model_artifact_sha256"].items():
        require_cuda_summary(
            temporal / "pipeline_models" / key / "summary.json", artifacts["summary"], device
        )
        checked += 1

    return checked


def copy_declared_csv(summary_path: Path, name: str, output: Path) -> pd.DataFrame:
    source = require_declared_artifact(summary_path, name)
    frame = pd.read_csv(source)
    write_csv(output / name, frame)
    return frame


def build_fewshot_summary(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, budget), group in source.groupby(["method", "budget_per_class"], sort=True):
        row: dict[str, int | float | str] = {
            "method": str(method),
            "budget_per_class": int(budget),
            "fit_samples": int(group["fit_samples"].iloc[0]),
            "seeds": int(len(group)),
        }
        for metric in ("accuracy", "macro_f1", "locomotion_f1", "log_loss"):
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["budget_per_class", "method"], ignore_index=True)


def build_temporal_development(teacher: dict, development_path: Path) -> pd.DataFrame:
    rows = []
    for family, metrics in (
        ("static", teacher["static_validation_metrics"]),
        ("teacher", teacher["selected_teacher_validation_metrics"]),
    ):
        rows.append({"family": family, "split": "validation", **metrics})

    students = pd.read_csv(
        require_declared_artifact(development_path, "student_validation_metrics.csv")
    )
    students.insert(1, "split", "validation")
    rows.extend(students.to_dict(orient="records"))
    return pd.DataFrame(rows)[["family", "split", *METRIC_COLUMNS]]


def build_source_development(root: Path, paths: dict[str, Path]) -> pd.DataFrame:
    sources = {
        "nested_stacks": (paths["nested"], "nested_source_tag_metrics.csv"),
        "spatial": (paths["spatial"], "spatial_metrics.csv"),
        "representations": (paths["representations"], "nested_source_tag_metrics.csv"),
    }
    frames = []
    for stage, (summary_path, filename) in sources.items():
        frame = pd.read_csv(require_declared_artifact(summary_path, filename))
        frame.insert(0, "stage", stage)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    return result[["stage", "family", *METRIC_COLUMNS]].sort_values(
        ["stage", "macro_f1", "family"], ascending=[True, False, True], ignore_index=True
    )


def build_subgroup_deltas(subgroups: pd.DataFrame) -> pd.DataFrame:
    index = ["axis", "value"]
    baseline = subgroups.loc[subgroups["family"].eq("static")].set_index(index)
    rows = []
    families = (
        "teacher",
        "classification_student",
        "routing_student",
        "hybrid_budget_0.5",
        "source_only_static",
    )
    for family in families:
        current = subgroups.loc[subgroups["family"].eq(family)].set_index(index)
        if set(current.index) != set(baseline.index):
            raise RuntimeError(f"Subgroup definitions differ for {family}")
        for key, record in current.iterrows():
            reference = baseline.loc[key]
            if int(record["samples"]) != int(reference["samples"]):
                raise RuntimeError(f"Subgroup support differs for {family}: {key}")
            rows.append(
                {
                    "family": family,
                    "axis": key[0],
                    "value": key[1],
                    "samples": int(record["samples"]),
                    "accuracy_delta_vs_static": float(record["accuracy"] - reference["accuracy"]),
                    "macro_f1_delta_vs_static": float(
                        record["macro_f1_all_classes"] - reference["macro_f1_all_classes"]
                    ),
                    "locomotion_f1_delta_vs_static": float(
                        record["locomotion_f1"] - reference["locomotion_f1"]
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["family", "axis", "value"], ignore_index=True)


def main() -> None:
    args = parse_args()
    root = args.runs_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "annotation": root / "annotation/final/summary.json",
        "nested": root / "nested_stacks/summary.json",
        "spatial": root / "spatial/summary.json",
        "representations": root / "representations/evaluation/summary.json",
        "neural": root / "neural/neural_grid_lock.json",
        "amendment": root / "protocol/external_cuda_amendment_lock.json",
        "development_audit": root / "okutama/development_audit/summary.json",
        "fewshot": root / "okutama/fewshot/summary.json",
        "teacher": root / "temporal/teacher_selection_lock.json",
        "crossfit": root / "temporal/crossfit_aggregate/summary.json",
        "development": root / "temporal/development_final/summary.json",
        "pose": root / "temporal/pose_control/summary.json",
        "pipeline": root / "temporal/pipeline_lock/summary.json",
        "confirmation_audit": root / "okutama/confirmation_audit/summary.json",
        "confirmation_features": root / "okutama/confirmation_features/dinov2_base/summary.json",
        "confirmation_source": root / "okutama/confirmation_source_only/summary.json",
        "confirmation": root / "temporal/confirmation/summary.json",
    }
    summaries = {name: read_json(path) for name, path in paths.items()}
    for name, summary in summaries.items():
        require_status(name, summary)

    amendment = summaries["amendment"]
    device = amendment["cuda"]["device"]
    if (
        amendment["cuda"].get("available") is not True
        or amendment["cuda"].get("cpu_fallback_permitted") is not False
        or "+cu" not in amendment["cuda"].get("torch_version", "")
    ):
        raise RuntimeError("The external experiment is not bound to CUDA-only fitting")
    if summaries["fewshot"].get("training_device") != device:
        raise RuntimeError("Few-shot fitting did not use the locked CUDA device")
    if summaries["pipeline"].get("calibration_device") != device:
        raise RuntimeError("Pipeline calibration did not use the locked CUDA device")
    if summaries["confirmation"].get("inference_device") != device:
        raise RuntimeError("Confirmation inference did not use the locked CUDA device")
    if summaries["confirmation_features"].get("model_kind") != "dinov2_base":
        raise RuntimeError("Confirmation features do not use the selected representation")
    if summaries["confirmation_source"].get("target_labels_read") != 0:
        raise RuntimeError("Source-only confirmation fitting read target labels")
    if summaries["pipeline"].get("confirmation_evaluations_run") != 0:
        raise RuntimeError("Pipeline lock was not written before confirmation")
    if {
        summaries["confirmation_audit"].get("confirmation_open_number"),
        summaries["confirmation_features"].get("confirmation_open_number"),
        summaries["confirmation"].get("confirmation_open_number"),
    } != {1}:
        raise RuntimeError("Confirmation was not opened exactly once")

    cuda_temporal_runs = validate_cuda_runs(root, summaries, device)
    if cuda_temporal_runs != 100:
        raise RuntimeError(f"Expected 100 CUDA temporal runs, found {cuda_temporal_runs}")

    annotation = summaries["annotation"]
    annotation_public = {
        "status": annotation["status"],
        "evidence_scope": annotation["evidence_scope"],
        "primary_selection_rule": annotation["primary_selection_rule"],
        "primary_task_presentations": annotation["primary_task_presentations"],
        "unique_content_rows": annotation["unique_content_rows"],
        "unique_images": annotation["unique_images"],
        "resolved_3class_rows": annotation["resolved_3class_rows"],
        "not_resolved_3class_rows": annotation["not_resolved_3class_rows"],
        "clear_walking_or_running_rows": annotation["clear_walking_or_running_rows"],
        "non_gait_locomotion_rows": annotation["non_gait_locomotion_rows"],
        "complete_repeat_pairs": annotation["complete_repeat_pairs"],
        "intrarater_repeat_agreement": annotation["intrarater_repeat_agreement"],
        "interrater_agreement_available": annotation["interrater_agreement_available"],
        "human_harmonized_performance_claim_permitted": annotation[
            "human_harmonized_performance_claim_permitted"
        ],
        "human_pilot_labels_used_for_candidate_selection": annotation[
            "human_pilot_labels_used_for_candidate_selection"
        ],
        "surplus_responses_preserved_and_excluded": annotation[
            "surplus_responses_preserved_and_excluded"
        ],
        "source_summary_sha256": sha256_file(paths["annotation"]),
    }
    write_json(output / "annotation_summary.json", annotation_public)
    for filename in (
        "axis_counts.csv",
        "cohort_summary.csv",
        "error_resolution.csv",
        "joint_translation_gait.csv",
        "repeat_summary.csv",
    ):
        copy_declared_csv(paths["annotation"], filename, output)

    source_development = build_source_development(root, paths)
    write_csv(output / "source_tag_development_metrics.csv", source_development)
    promotions = {
        "status": "VCOCO_V3_SOURCE_TAG_DEVELOPMENT_EXPORTED",
        "nested_stacks": {
            "best_family": summaries["nested"]["best_family"],
            "best_macro_f1": summaries["nested"]["best_macro_f1"],
            "decisions": read_json(
                require_declared_artifact(paths["nested"], "promotion_decisions.json")
            ),
        },
        "spatial": {
            "best_family": summaries["spatial"]["best_family"],
            "best_macro_f1": summaries["spatial"]["best_macro_f1"],
            "decisions": read_json(
                require_declared_artifact(paths["spatial"], "spatial_promotions.json")
            ),
        },
        "representations": {
            "best_family": summaries["representations"]["best_family"],
            "best_macro_f1": summaries["representations"]["best_macro_f1"],
            "decisions": read_json(
                require_declared_artifact(paths["representations"], "promotion_decisions.json")
            ),
        },
        "neural_adaptation": {
            "status": summaries["neural"]["status"],
            "reason": summaries["neural"]["reason"],
        },
    }
    write_json(output / "source_tag_promotion_decisions.json", promotions)

    fewshot_source = pd.read_csv(require_declared_artifact(paths["fewshot"], "metrics.csv"))
    write_csv(output / "okutama_fewshot_summary.csv", build_fewshot_summary(fewshot_source))

    temporal_development = build_temporal_development(summaries["teacher"], paths["development"])
    write_csv(output / "temporal_development_metrics.csv", temporal_development)
    crossfit = summaries["crossfit"]
    write_json(
        output / "temporal_crossfit_summary.json",
        {
            "status": crossfit["status"],
            "samples": crossfit["samples"],
            "recordings": crossfit["recordings"],
            "folds": crossfit["folds"],
            "seeds": crossfit["seeds"],
            "static_oof_metrics": crossfit["static_oof_metrics"],
            "teacher_oof_metrics": crossfit["teacher_oof_metrics"],
            "teacher_advantage_definition": {
                "minimum_log_likelihood_gain": crossfit[
                    "teacher_advantage_minimum_log_likelihood_gain"
                ],
                "positive_count": crossfit["teacher_advantage_positive_count"],
                "fraction": crossfit["teacher_advantage_fraction"],
            },
            "source_summary_sha256": sha256_file(paths["crossfit"]),
        },
    )

    copy_declared_csv(paths["pipeline"], "calibration_metrics.csv", output)
    copy_declared_csv(paths["pipeline"], "calibration_routing_curve.csv", output)

    confirmation_metrics = copy_declared_csv(
        paths["confirmation"], "confirmation_metrics.csv", output
    )
    copy_declared_csv(paths["confirmation"], "confirmation_per_class.csv", output)
    copy_declared_csv(paths["confirmation"], "confirmation_routing_curve.csv", output)
    copy_declared_csv(paths["confirmation"], "confirmation_prediction_sets.csv", output)
    subgroups = copy_declared_csv(paths["confirmation"], "confirmation_subgroups.csv", output)
    write_csv(output / "confirmation_subgroup_deltas.csv", build_subgroup_deltas(subgroups))
    uncertainty_path = require_declared_artifact(
        paths["confirmation"], "confirmation_uncertainty.json"
    )
    write_json(output / "confirmation_uncertainty.json", read_json(uncertainty_path))

    confirmation = summaries["confirmation"]
    write_json(
        output / "confirmation_summary.json",
        {
            "status": confirmation["status"],
            "endpoint": confirmation["endpoint"],
            "samples": confirmation["samples"],
            "videos": summaries["confirmation_audit"]["recordings"],
            "scenario_clusters": summaries["confirmation_audit"]["scenarios"],
            "confirmation_open_number": confirmation["confirmation_open_number"],
            "routing_enabled_by_validation_gate": confirmation[
                "routing_enabled_by_validation_gate"
            ],
            "pose_control_status": confirmation["pose_control_status"],
            "frame_backbone": summaries["pipeline"]["selected_teacher"]["frame_backbone"],
            "selected_teacher": summaries["pipeline"]["selected_teacher"],
            "inference_device": confirmation["inference_device"],
            "torch_version": confirmation["torch_version"],
            "pipeline_lock_sha256": confirmation["pipeline_lock_sha256"],
            "source_summary_sha256": sha256_file(paths["confirmation"]),
            "source_artifact_sha256": confirmation["artifact_sha256"],
        },
    )

    development_audit = summaries["development_audit"]
    confirmation_audit = summaries["confirmation_audit"]
    write_json(
        output / "okutama_dataset_summary.json",
        {
            "status": "OKUTAMA_DEVELOPMENT_AND_CONFIRMATION_AUDIT_EXPORTED",
            "development": {
                "samples": development_audit["selected_centres"],
                "recordings": development_audit["recordings"],
                "scenarios": development_audit["scenarios"],
                "tracks": development_audit["stable_tracks"],
                "transition_windows": development_audit["transition_centres"],
                "center_occluded": development_audit["occluded_centres"],
                "window_any_occluded": development_audit["windows_with_occlusion"],
                "class_counts": development_audit["counts"],
                "archive_bytes": development_audit["development_archive"]["bytes"],
                "archive_sha256": development_audit["development_archive"]["sha256"],
            },
            "confirmation": {
                "samples": confirmation_audit["selected_centres"],
                "recordings": confirmation_audit["recordings"],
                "scenarios": confirmation_audit["scenarios"],
                "tracks": confirmation_audit["stable_tracks"],
                "transition_windows": confirmation_audit["transition_centres"],
                "center_occluded": confirmation_audit["occluded_centres"],
                "window_any_occluded": confirmation_audit["windows_with_occlusion"],
                "class_counts": confirmation_audit["counts"],
                "archive_bytes": confirmation_audit["confirmation_archive"]["bytes"],
                "archive_sha256": confirmation_audit["confirmation_archive"]["sha256"],
                "open_number": confirmation_audit["confirmation_open_number"],
            },
        },
    )

    pipeline_hash = sha256_file(paths["pipeline"])
    if pipeline_hash != confirmation["pipeline_lock_sha256"]:
        raise RuntimeError("Confirmation does not bind to the current pipeline lock")
    write_json(
        output / "protocol_lineage.json",
        {
            "status": "VCOCO_V3_PORTABLE_PROTOCOL_LINEAGE_COMPLETE",
            "protocol_version": annotation["protocol_version"],
            "base_protocol_lock_sha256": amendment["base_protocol_lock_sha256"],
            "external_cuda_amendment_sha256": sha256_file(paths["amendment"]),
            "external_cuda_amendment_version": amendment["amendment_version"],
            "repository_revision_at_cuda_lock": amendment["repository_revision_at_lock"],
            "cuda": amendment["cuda"],
            "verified_cuda_temporal_runs": cuda_temporal_runs,
            "spatial_cuda_logistic_fits": summaries["spatial"]["cuda_logistic_fits"],
            "representation_cuda_logistic_fits": summaries["representations"]["cuda_logistic_fits"],
            "pipeline_lock_sha256": pipeline_hash,
            "confirmation_open_number": confirmation["confirmation_open_number"],
            "confirmation_used_for_selection": False,
        },
    )

    metrics = confirmation_metrics.set_index("family")
    uncertainty = read_json(uncertainty_path)
    teacher_gain = uncertainty["teacher"]["macro_f1"]
    routed_gain = uncertainty["hybrid_budget_0.5"]["macro_f1"]
    manifest = {
        "status": "VCOCO_V3_PORTABLE_EVIDENCE_COMPLETE",
        "endpoint": confirmation["endpoint"],
        "confirmation_samples": confirmation["samples"],
        "confirmation_scenarios": confirmation_audit["scenarios"],
        "confirmation_open_number": confirmation["confirmation_open_number"],
        "confirmation_used_for_selection": False,
        "selected_frame_backbone": "dinov2_base",
        "teacher_macro_f1": float(metrics.loc["teacher", "macro_f1"]),
        "static_macro_f1": float(metrics.loc["static", "macro_f1"]),
        "teacher_macro_f1_gain": teacher_gain,
        "half_budget_macro_f1": float(metrics.loc["hybrid_budget_0.5", "macro_f1"]),
        "half_budget_macro_f1_gain": routed_gain,
        "pipeline_lock_sha256": pipeline_hash,
        "external_cuda_amendment_sha256": sha256_file(paths["amendment"]),
        "exporter_sha256": sha256_file(Path(__file__).resolve()),
        "artifacts": {},
        "local_artifacts_retained": [
            "private annotation rows and image identifiers",
            "dataset archives and person crops",
            "feature tensors and dense probabilities",
            "model checkpoints and optimizer traces",
        ],
    }
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "evidence_manifest.json":
            manifest["artifacts"][path.name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    write_json(output / "evidence_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
