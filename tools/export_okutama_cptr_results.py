"""Export the compact Okutama CPTR development evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file

COMPONENTS = (
    (
        "raw_trajectory",
        "compact raw bounding-box kinematics",
        ".runs/cptr/candidates/trajectory_raw/seed-42-v3/summary.json",
        "legacy temporal, seed 42",
        None,
        "control retained",
    ),
    (
        "camera_compensated_trajectory",
        "LK/RANSAC camera-compensated kinematics",
        ".runs/cptr/candidates/trajectory_compensated/seed-42-v2/summary.json",
        "raw trajectory",
        "raw_trajectory",
        "control retained",
    ),
    (
        "centre_short",
        "center-conditioned 0.5 s residual",
        ".runs/cptr/candidates/centre_short/seed-42-v1/summary.json",
        "legacy temporal, seed 42",
        None,
        "retained",
    ),
    (
        "dual_clock",
        "shared short and long clocks",
        ".runs/cptr/candidates/dual_clock/seed-42-v1/summary.json",
        "center-conditioned short residual",
        "centre_short",
        "not retained",
    ),
    (
        "dual_clock_specialized",
        "posture/motion-specialized clock roles",
        ".runs/cptr/adaptive/dual_clock_specialized/seed-42/summary.json",
        "center-conditioned short residual",
        "centre_short",
        "not retained",
    ),
    (
        "centre_short_trajectory",
        "short residual plus compensated trajectory",
        ".runs/cptr/adaptive/centre_short_trajectory/seed-42/summary.json",
        "center-conditioned short residual",
        "centre_short",
        "not retained",
    ),
    (
        "centre_short_parts",
        "short residual plus confidence-masked body regions",
        ".runs/cptr/adaptive/centre_short_parts/seed-42/summary.json",
        "center-conditioned short residual",
        "centre_short",
        "retained for five-seed evaluation",
    ),
    (
        "integrated_cptr",
        "short residual, parts, and compensated trajectory",
        ".runs/cptr/adaptive/cptr_integrated/seed-42/summary.json",
        "short residual plus body regions",
        "centre_short_parts",
        "not retained",
    ),
    (
        "counterfactual_original",
        "motion-null and invariance objective",
        ".runs/cptr/stage2/centre_short_parts_counterfactual/seed-42/summary.json",
        "short residual plus body regions",
        "centre_short_parts",
        "not retained",
    ),
    (
        "counterfactual_refined",
        "legacy-preserving motion-null objective",
        ".runs/cptr/stage3/centre_short_parts_counterfactual_refined/seed-42/summary.json",
        "short residual plus body regions",
        "centre_short_parts",
        "transition subgroup retained as diagnostic",
    ),
    (
        "masked_initialisation",
        "target-video masked feature pretraining",
        ".runs/cptr/stage3/centre_short_parts_masked_only/seed-42/summary.json",
        "short residual plus body regions",
        "centre_short_parts",
        "not retained",
    ),
    (
        "siglip_posture_specialist",
        "frozen center-frame SigLIP posture expert",
        ".runs/cptr/stage4/centre_short_parts_siglip/seed-42/summary.json",
        "short residual plus body regions",
        "centre_short_parts",
        "not retained",
    ),
    (
        "group_dro",
        "scenario-weighted GroupDRO",
        ".runs/cptr/stage4/centre_short_parts_group_dro/seed-42/summary.json",
        "short residual plus body regions",
        "centre_short_parts",
        "not retained",
    ),
    (
        "top_block_lora",
        "DINOv2 top-block Q/V LoRA specialist",
        ".runs/cptr/lora_specialist-v4/summary.json",
        "legacy temporal, seed 42",
        None,
        "not retained",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("results/okutama_cptr"))
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_utf8_lf(path: Path, text: str) -> None:
    """Write deterministic portable text without platform newline translation."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    path.write_bytes(normalized.encode("utf-8"))


def write_json(path: Path, payload: object) -> None:
    write_utf8_lf(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    write_utf8_lf(path, frame.to_csv(index=False, lineterminator="\n"))


def copy_portable_text(source: Path, destination: Path) -> None:
    """Copy a locked text artifact into the portable LF-normalized package."""
    write_utf8_lf(destination, source.read_text(encoding="utf-8"))


def require_artifact(summary_path: Path, name: str) -> Path:
    summary = read_json(summary_path)
    source = summary_path.parent / name
    expected = summary.get("artifact_sha256", {}).get(name)
    if not source.is_file() or not isinstance(expected, str):
        raise RuntimeError(f"Missing declared artifact: {source}")
    if sha256_file(source) != expected:
        raise RuntimeError(f"Artifact hash drift: {source}")
    return source


def component_rows(repository: Path) -> pd.DataFrame:
    summaries = {}
    rows = []
    reference_summary = read_json(
        repository / ".runs/cptr/candidates/centre_short/seed-42-v1/summary.json"
    )
    seed42_legacy = float(reference_summary["same_seed_legacy_temporal_metrics"]["macro_f1"])
    for component_id, description, relative, reference, parent, decision in COMPONENTS:
        path = repository / relative
        summary = read_json(path)
        request_path = path.parent / "request.json"
        request = read_json(request_path)
        request_payload = {key: value for key, value in request.items() if key != "request_sha256"}
        request_payload_sha256 = hashlib.sha256(
            json.dumps(request_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if summary.get("request_sha256") != request.get("request_sha256"):
            raise RuntimeError(f"Summary/request mismatch: {path.parent}")
        if request_payload_sha256 != request.get("request_sha256"):
            raise RuntimeError(f"Request payload hash mismatch: {request_path}")
        source = request.get("source_sha256", {})
        metrics = summary.get("validation_metrics", {})
        macro_f1 = float(metrics["macro_f1"])
        summaries[component_id] = macro_f1
        if parent is None:
            incremental = macro_f1 - seed42_legacy
        else:
            incremental = macro_f1 - summaries[parent]
        rows.append(
            {
                "component_id": component_id,
                "description": description,
                "seed": int(summary.get("seed", 42)),
                "validation_macro_f1": macro_f1,
                "validation_accuracy": float(metrics["accuracy"]),
                "validation_log_loss": float(metrics["log_loss"]),
                "delta_vs_seed42_legacy": macro_f1 - seed42_legacy,
                "incremental_reference": reference,
                "incremental_macro_f1_delta": incremental,
                "decision": decision,
                "run_status": summary["status"],
                "summary_sha256": sha256_file(path),
                "request_payload_sha256": request_payload_sha256,
                "request_file_sha256": sha256_file(request_path),
                "model_module_sha256": source.get("model_module"),
                "neural_module_sha256": source.get("neural_module"),
                "feature_module_sha256": source.get("feature_module"),
                "training_module_sha256": source.get("training_module"),
                "runner_sha256": source.get("runner"),
                "grid_sha256": source.get("grid"),
                "grid_lock_sha256": source.get("grid_lock"),
            }
        )
    return pd.DataFrame(rows)


def crossfit_contract_audit(repository: Path) -> dict:
    requests = []
    for fold in range(5):
        for seed in range(42, 47):
            path = (
                repository
                / ".runs/cptr/crossfit/centre_short_parts"
                / f"fold-{fold}"
                / f"seed-{seed}"
                / "request.json"
            )
            request = read_json(path)
            payload = {key: value for key, value in request.items() if key != "request_sha256"}
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
            if digest != request.get("request_sha256"):
                raise RuntimeError(f"Cross-fit request payload hash mismatch: {path}")
            requests.append(request)

    source_keys = ("model_module", "feature_module", "runner", "candidate_grid", "plan")
    shared = {
        key: sorted({request["source_sha256"][key] for request in requests}) for key in source_keys
    }
    if any(len(values) != 1 for values in shared.values()):
        raise RuntimeError("Cross-fit requests do not share one implementation snapshot")

    adaptive_lock = read_json(repository / ".runs/cptr/adaptive_grid_lock.json")
    locked_grid = adaptive_lock["source_sha256"]["adaptive_grid"]
    if shared["candidate_grid"] != [locked_grid]:
        raise RuntimeError("Cross-fit requests do not match the adaptive-grid lock")

    return {
        "status": "VERIFIED_WITH_HISTORICAL_SCHEMA_LIMITATIONS",
        "requests_verified": len(requests),
        "shared_source_sha256": {key: values[0] for key, values in shared.items()},
        "candidate_grid_matches_adaptive_lock": True,
        "historical_request_schema_omissions": ["src/hac/cptr_training.py"],
        "historical_plan_lock_schema_omissions": [
            "candidate_grid",
            "candidate_grid_lock",
            "src/hac/cptr_training.py",
        ],
        "scope": "Contract-completeness audit; reported metrics are unchanged.",
    }


def headline_rows(development: dict) -> pd.DataFrame:
    rows = []
    for scope_key, scope in (
        ("development_validation", development["development_validation"]),
        ("grouped_crossfit_oof", development["grouped_crossfit_oof"]),
    ):
        for model_key, metrics in (
            ("v3_temporal_baseline", scope["baseline_metrics"]),
            ("centre_short_parts", scope["candidate_metrics"]),
        ):
            rows.append(
                {
                    "scope": scope_key,
                    "model": model_key,
                    "samples": int(scope["samples"]),
                    "recordings": int(scope["recordings"]),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def portable_decision(lock: dict, development: dict) -> dict:
    return {
        "status": lock["status"],
        "candidate_id": lock["candidate_id"],
        "decision": lock["decision"],
        "development_validation": development["development_validation"],
        "grouped_crossfit_oof": development["grouped_crossfit_oof"],
        "promotion_checks": development["promotion_checks"],
        "promotion_passed": development["promotion_passed"],
        "worst_recording_macro_f1_delta": development["worst_recording_macro_f1_delta"],
        "fixed_epochs": development["fixed_epochs"],
        "seeds": development["seeds"],
        "folds": development["folds"],
        "validation_samples_read": development["validation_samples_read"],
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": lock["source_sha256"],
    }


def render_readme(decision: dict, faithfulness: dict) -> str:
    validation = decision["development_validation"]
    oof = decision["grouped_crossfit_oof"]
    real = faithfulness["diagnostics"]["real"]
    motion_null = faithfulness["diagnostics"]["motion_null"]
    no_parts = faithfulness["diagnostics"]["parts_missing"]
    reverse = faithfulness["diagnostics"]["reverse_temporal_order"]
    jitter = faithfulness["diagnostics"]["background_camera_jitter"]
    return f"""# Okutama CPTR development evidence

This package records the camera-compensated part-trajectory residual (CPTR) development
study on the provider-train portion of Okutama-Action. The evaluated family combines
frozen v3 static and temporal experts with center-conditioned temporal residuals,
camera-aware box trajectories, confidence-masked body-region tokens, quality gates,
counterfactual objectives, target-video masked pretraining, and parameter-efficient
specialists.

The strongest component was the center-conditioned short-window plus body-region
model. Its five-seed ensemble reached **{validation["candidate_metrics"]["macro_f1"]:.4f}
macro-F1** on the fixed three-recording validation split, compared with
**{validation["baseline_metrics"]["macro_f1"]:.4f}** for the frozen temporal baseline.
The gain was concentrated in standing F1
(**{validation["baseline_metrics"]["standing_f1"]:.4f} to
{validation["candidate_metrics"]["standing_f1"]:.4f}**).

The recording-grouped OOF result did not reproduce that gain: the center-plus-parts
residual reached
**{oof["candidate_metrics"]["macro_f1"]:.4f}**, while the matched temporal baseline
reached **{oof["baseline_metrics"]["macro_f1"]:.4f}** across 4,977 samples from 11
recordings. The existing temporal ensemble therefore remains the default model. The
center-and-parts branch is retained as a documented research component.

## Main comparison

| Evaluation | Baseline macro-F1 | Center + parts residual | Change | Recordings |
| --- | ---: | ---: | ---: | ---: |
| Fixed validation | {validation["baseline_metrics"]["macro_f1"]:.4f} | {validation["candidate_metrics"]["macro_f1"]:.4f} | {validation["macro_f1_delta"]:+.4f} | {validation["recordings"]} |
| Five-fold grouped OOF | {oof["baseline_metrics"]["macro_f1"]:.4f} | {oof["candidate_metrics"]["macro_f1"]:.4f} | {oof["macro_f1_delta"]:+.4f} | {oof["recordings"]} |

The exact recording-level prediction-swap test gives the OOF comparison directly;
`uncertainty.json` also contains 10,000-resample paired recording-cluster intervals.

## What the interventions show

- Repeating the center frame and removing relative temporal evidence reduced
  validation macro-F1 by **{motion_null["macro_f1_delta_real_minus_intervention"]:.4f}**
  and reduced mean true-class log probability by
  **{motion_null["mean_true_class_log_probability_gain_real_minus_intervention"]:.4f}**.
- Removing the body-region stream reduced macro-F1 by
  **{no_parts["macro_f1_delta_real_minus_intervention"]:.4f}** on validation.
- Reversing temporal order reduced macro-F1 by
  **{reverse["macro_f1_delta_real_minus_intervention"]:.4f}**; deterministic camera
  jitter changed it by **{jitter["macro_f1_delta_real_minus_intervention"]:.4f}**.
- Clear-window OOF performance was slightly above baseline, but occluded-window
  macro-F1 was lower. This quality-dependent instability is the largest failure mode
  observed in this evaluation.

The intervention reference result is {real["metrics"]["macro_f1"]:.4f} macro-F1.
Classifier-forward latency is measured from cached DINOv2 and part features; feature
extraction is intentionally reported separately.

## Files

- `development_decision.json` - aggregate metrics, gates, fixed epochs, and hashes;
- `headline_metrics.csv` - validation and grouped-OOF model metrics;
- `component_ablation.csv` - sequential development trace with request and source hashes;
- `fold_seed_metrics.csv` - all 25 fold/seed runs plus ensembles;
- `subgroup_metrics.csv` and `recording_metrics.csv` - transition, visibility, and
  recording-level results;
- `faithfulness_metrics.csv` and `faithfulness_summary.json` - intervention and gate
  evidence;
- `uncertainty.json` - paired cluster intervals and exact group-swap tests;
- `provenance.json` and `evidence_manifest.json` - model revisions, CUDA runtime, and
  file hashes.

Dataset frames, pretrained weights, cached embeddings, checkpoints, and dense
prediction arrays remain outside Git under their original terms.
"""


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    output = (repository / args.output_dir).resolve()
    development_summary_path = repository / ".runs/cptr/development_final/summary.json"
    development_lock_path = repository / ".runs/cptr/development_lock.json"
    faithfulness_summary_path = (
        repository / ".runs/cptr/development_final/faithfulness/summary.json"
    )
    development = read_json(development_summary_path)
    lock = read_json(development_lock_path)
    faithfulness = read_json(faithfulness_summary_path)
    if lock.get("status") != "OKUTAMA_CPTR_DEVELOPMENT_LOCKED_NO_PROMOTION":
        raise RuntimeError("The CPTR development decision is not locked")
    if development.get("status") != "OKUTAMA_CPTR_DEVELOPMENT_COMPLETE_NO_PROMOTION":
        raise RuntimeError("The CPTR development evidence is incomplete")
    if faithfulness.get("status") != "OKUTAMA_CPTR_FAITHFULNESS_COMPLETE":
        raise RuntimeError("The CPTR faithfulness evidence is incomplete")
    if sha256_file(development_summary_path) != lock["source_sha256"]["development_summary"]:
        raise RuntimeError("The development summary changed after locking")
    if sha256_file(faithfulness_summary_path) != lock["source_sha256"]["faithfulness_summary"]:
        raise RuntimeError("The faithfulness summary changed after locking")

    output.mkdir(parents=True, exist_ok=True)
    decision = portable_decision(lock, development)
    write_json(output / "development_decision.json", decision)
    write_csv(output / "headline_metrics.csv", headline_rows(development))
    write_csv(output / "component_ablation.csv", component_rows(repository))
    source_artifacts = {
        "fold_seed_metrics.csv": require_artifact(
            development_summary_path, "fold_seed_metrics.csv"
        ),
        "subgroup_metrics.csv": require_artifact(development_summary_path, "subgroup_metrics.csv"),
        "recording_metrics.csv": require_artifact(
            development_summary_path, "recording_metrics.csv"
        ),
        "uncertainty.json": require_artifact(development_summary_path, "uncertainty.json"),
        "faithfulness_metrics.csv": require_artifact(
            faithfulness_summary_path, "intervention_metrics.csv"
        ),
    }
    for name, source in source_artifacts.items():
        copy_portable_text(source, output / name)
    portable_faithfulness = {
        key: faithfulness[key]
        for key in (
            "status",
            "candidate_id",
            "seeds",
            "samples",
            "diagnostics",
            "gate_summary",
            "latency",
            "latency_feature_extraction_excluded",
            "validation_samples_read",
            "calibration_samples_read",
            "confirmation_samples_read",
            "checkpoint_sha256",
            "source_sha256",
        )
    }
    write_json(output / "faithfulness_summary.json", portable_faithfulness)

    protocol = read_json(repository / "experiments/okutama_cptr_protocol.json")
    part_store = read_json(repository / ".runs/cptr/part_features/store.json")
    motion_store = read_json(repository / ".runs/cptr/motion_features/store.json")
    siglip_store = read_json(repository / ".runs/cptr/siglip_features/store.json")
    promotion_summary = read_json(
        repository / ".runs/cptr/promotion/centre_short_parts/seed-42/summary.json"
    )
    provenance = {
        "status": "OKUTAMA_CPTR_PORTABLE_EVIDENCE_COMPLETE",
        "dataset": "Okutama-Action provider train",
        "candidate_id": "centre_short_parts",
        "frozen_frame_backbone": {
            "model_id": part_store["model"]["model_id"],
            "revision": part_store["model"]["revision"],
        },
        "siglip_control": {
            "model_id": siglip_store["model"]["model_id"],
            "revision": siglip_store["model"]["revision"],
        },
        "camera_compensation": protocol["camera_compensation"],
        "motion_feature_runtime_seconds": motion_store["runtime_seconds"],
        "part_feature_runtime_seconds": part_store["runtime_seconds"],
        "siglip_feature_runtime_seconds": siglip_store["runtime_seconds"],
        "training_device": promotion_summary["training_device"],
        "torch_version": promotion_summary["torch_version"],
        "seeds": development["seeds"],
        "crossfit_folds": development["folds"],
        "development_samples": {
            "train": development["grouped_crossfit_oof"]["samples"],
            "validation": development["development_validation"]["samples"],
            "calibration_read": 0,
            "confirmation_read": 0,
        },
        "source_sha256": {
            "execution_protocol": lock["source_sha256"]["protocol"],
            "portable_protocol": sha256_file(repository / "experiments/okutama_cptr_protocol.json"),
            "protocol_lock": lock["source_sha256"]["protocol_lock"],
            "crossfit_plan_lock": lock["source_sha256"]["crossfit_plan_lock"],
            "development_lock": sha256_file(development_lock_path),
            "tested_top_level_dependency_snapshot": sha256_file(
                repository / "requirements-v3-lock.txt"
            ),
            "exporter": sha256_file(Path(__file__).resolve()),
        },
        "portable_protocol_normalization": {
            "field": "input.raw_development_archive",
            "reason": "replace the machine-local archive location with a runtime path",
            "method_configuration_changed": False,
        },
        "crossfit_contract_audit": crossfit_contract_audit(repository),
    }
    write_json(output / "provenance.json", provenance)
    write_utf8_lf(output / "README.md", render_readme(decision, faithfulness))

    manifest_path = output / "evidence_manifest.json"
    artifacts = {}
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path == manifest_path:
            continue
        artifacts[path.name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "schema_version": 2,
        "study": "okutama_cptr_development",
        "hash_algorithm": "sha256",
        "text_encoding": "utf-8",
        "text_line_endings": "LF",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    write_json(manifest_path, manifest)
    print(f"CPTR portable evidence written: {output}")


if __name__ == "__main__":
    main()
