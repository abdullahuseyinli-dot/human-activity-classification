"""Build the immutable POLAR final-selection specification from development evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file

CONFIRMED_NEURAL_RUNS = {
    "convnext_small_full": [
        "adaptation/convnext_small_full_backbone_ld070_seed42",
        "ensemble_confirmation/convnext_small_full_backbone_ld070_seed52",
        "ensemble_confirmation/convnext_small_full_backbone_ld070_seed62",
    ],
    "dinov2_base_top4": [
        "capacity/dinov2_base_context25_top4_seed42",
        "ensemble_confirmation/dinov2_base_context25_top4_seed52",
        "ensemble_confirmation/dinov2_base_context25_top4_seed62",
    ],
    "dinov2_small_moderate": [
        "regularization/dinov2_small_context25_full_moderate_seed42",
        "confirmation/dinov2_small_context25_full_moderate_seed52",
        "confirmation/dinov2_small_context25_full_moderate_seed62",
    ],
}

FINAL_CONFIGURATION_FIELDS = (
    "model_kind",
    "task",
    "view",
    "unfreeze_strategy",
    "top_n_blocks",
    "augmentation",
    "batch_size",
    "grad_accum_steps",
    "effective_batch_size",
    "workers",
    "head_lr",
    "backbone_lr",
    "weight_decay",
    "layer_decay",
    "dropout",
    "mixup_alpha",
    "label_smoothing",
    "class_balance",
    "warmup_fraction",
    "gradient_clip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--probe-results", type=Path, required=True)
    parser.add_argument("--rbf-summary", type=Path, required=True)
    parser.add_argument("--scale-provenance", type=Path, required=True)
    parser.add_argument("--scale-summary", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def checked_json(path: Path, *, status: str | None = None) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if status is not None and payload.get("status") != status:
        raise RuntimeError(f"Unexpected evidence status at {path}: {payload.get('status')}")
    if payload.get("test_rows_read") != 0 or payload.get("test_used_for_selection"):
        raise RuntimeError(f"Development evidence violates the test gate: {path}")
    return payload


def confirmed_neural_fits(training_root: Path) -> tuple[dict, list[str], dict]:
    final_fits = {}
    all_run_ids = []
    component_evidence = {}
    for component, run_ids in CONFIRMED_NEURAL_RUNS.items():
        summaries = []
        for run_id in run_ids:
            summary_path = training_root / run_id / "summary.json"
            summary = checked_json(summary_path, status="COMPLETE")
            summaries.append((run_id, summary_path, summary))
        seeds = [int(summary["configuration"]["seed"]) for _, _, summary in summaries]
        if seeds != [42, 52, 62]:
            raise RuntimeError(f"Unexpected confirmation seeds for {component}: {seeds}")
        reference = summaries[0][2]["configuration"]
        for _, _, summary in summaries[1:]:
            observed = summary["configuration"]
            for field in FINAL_CONFIGURATION_FIELDS:
                if observed[field] != reference[field]:
                    raise RuntimeError(f"Confirmation configuration drift: {component}.{field}")
        best_epochs = [int(summary["best_epoch"]) for _, _, summary in summaries]
        fixed_epochs = int(statistics.median(best_epochs))
        configuration = {field: reference[field] for field in FINAL_CONFIGURATION_FIELDS}
        configuration["fixed_epochs"] = fixed_epochs
        final_fits[component] = {
            "seeds": seeds,
            "output_dir_pattern": f"neural/{component}/seed_{{seed}}",
            "configuration": configuration,
            "epoch_rule": "median_best_epoch_across_three_confirmation_seeds",
            "confirmation_best_epochs": best_epochs,
        }
        all_run_ids.extend(run_ids)
        component_evidence[component] = {
            "kind": "three_seed_neural_probability_mean",
            "confirmation_run_ids": run_ids,
            "confirmation_macro_f1": [
                float(summary["best_validation_metrics"]["macro_f1"]) for _, _, summary in summaries
            ],
            "summary_sha256": {
                str(seed): sha256_file(summary_path)
                for seed, (_, summary_path, _) in zip(seeds, summaries, strict=True)
            },
        }
    return final_fits, all_run_ids, component_evidence


def selected_logistic_row(frame: pd.DataFrame, task: str) -> pd.Series:
    candidates = frame[
        frame["screen"].astype(str).eq("official_multilayer")
        & frame["task"].astype(str).eq(task)
        & frame["candidate"]
        .astype(str)
        .eq("fusion__dinov2_base__full_frame__dinov2_base__person_context_10")
    ].copy()
    if candidates.empty:
        raise RuntimeError(f"Official multilayer probe evidence is missing for {task}")
    candidates["macro_f1"] = pd.to_numeric(candidates["macro_f1"])
    candidates["log_loss"] = pd.to_numeric(candidates["log_loss"])
    return candidates.sort_values(
        ["macro_f1", "log_loss", "class_weight"],
        ascending=[False, True, True],
        ignore_index=True,
    ).iloc[0]


def logistic_configuration(row: pd.Series) -> dict:
    return {
        "model_kind": "dinov2_base",
        "representation": "last4_cls_mean_patch",
        "views": ["full_frame", "person_context_10"],
        "task": str(row["task"]),
        "classifier": "standardized_multinomial_logistic",
        "C": float(row["C"]),
        "class_weight": str(row["class_weight"]),
        "seed": 42,
        "solver": "lbfgs",
        "max_iter": 2_000,
        "tol": 1e-5,
    }


def final_probe_fits(probe_results: Path, rbf_summary_path: Path) -> tuple[dict, dict]:
    probe_frame = pd.read_csv(probe_results)
    four_row = selected_logistic_row(probe_frame, "label_4")
    three_row = selected_logistic_row(probe_frame, "label_3")
    rbf = checked_json(rbf_summary_path)
    rbf_config = rbf["configuration"]
    if (
        rbf.get("representation") != "last4_cls_mean_patch"
        or float(rbf_config["C"]) != 10.0
        or float(rbf_config["gamma_multiplier"]) != 1.0
        or rbf_config["class_weight"] != "none"
        or rbf_config["calibration"] != "five_fold_sigmoid"
    ):
        raise RuntimeError("Transferred RBF evidence differs from the declared protocol")
    fits = {
        "dinov2_base_multilayer_logistic": {
            "output_dir": "probes/dinov2_base_multilayer_logistic",
            "configuration": logistic_configuration(four_row),
        },
        "dinov2_base_multilayer_rbf": {
            "output_dir": "probes/dinov2_base_multilayer_rbf",
            "configuration": {
                "model_kind": "dinov2_base",
                "representation": "last4_cls_mean_patch",
                "views": ["full_frame", "person_context_10"],
                "task": "label_4",
                "classifier": "calibrated_rbf_svm",
                "C": 10.0,
                "class_weight": "none",
                "seed": int(rbf_config["seed"]),
                "kernel": "rbf",
                "gamma_multiplier": 1.0,
                "calibration": "sigmoid",
                "calibration_folds": int(rbf_config["calibration_folds"]),
                "calibration_ensemble": True,
                "cache_size_mb": 4_096,
            },
        },
        "dinov2_base_multilayer_logistic_3class": {
            "output_dir": "probes/dinov2_base_multilayer_logistic_3class",
            "configuration": logistic_configuration(three_row),
        },
    }
    evidence = {
        "dinov2_base_multilayer_logistic": {
            "kind": "frozen_feature_classifier",
            "development_macro_f1": float(four_row["macro_f1"]),
            "probe_results_sha256": sha256_file(probe_results),
        },
        "dinov2_base_multilayer_rbf": {
            "kind": "frozen_feature_classifier",
            "development_macro_f1": float(rbf["metrics"]["macro_f1"]),
            "rbf_summary_sha256": sha256_file(rbf_summary_path),
        },
    }
    return fits, evidence


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol_json.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "1.4.0":
        raise RuntimeError("Final selection requires POLAR protocol 1.4.0")

    development_path = args.development_manifest.resolve()
    development = pd.read_csv(development_path, dtype={"image_id": str})
    if set(development["split"].astype(str)) != {"train", "val"}:
        raise RuntimeError("Final selection input must contain development rows only")
    test_path = args.test_manifest.resolve()
    test_bytes = test_path.read_bytes()
    test_rows = sum(1 for _ in test_bytes.splitlines()) - 1

    training_root = args.training_root.resolve()
    neural_fits, confirmation_run_ids, components = confirmed_neural_fits(training_root)
    probe_fits, probe_components = final_probe_fits(
        args.probe_results.resolve(), args.rbf_summary.resolve()
    )
    components.update(probe_components)

    validation_dir = args.validation_dir.resolve()
    blend = checked_json(
        validation_dir / "validation_blend.json",
        status="DEVELOPMENT_ONLY_VALIDATION_BLEND",
    )
    expected_components = {
        "convnext_small_full",
        "dinov2_base_top4",
        "dinov2_small_moderate",
        "dinov2_base_multilayer_logistic",
        "dinov2_base_multilayer_rbf",
    }
    if set(blend["weights"]) != expected_components or set(components) != expected_components:
        raise RuntimeError("Final development blend differs from the confirmed component pool")

    external_path = args.external_manifest.resolve()
    external = pd.read_csv(external_path, dtype={"person_id": str, "image_id": str})
    external_unambiguous = external["image_level_unambiguous"].astype(str).str.lower().eq("true")

    scale_provenance_path = args.scale_provenance.resolve()
    scale_summary_path = args.scale_summary.resolve()
    checked_json(scale_provenance_path)

    implementation_files = [
        "experiments/fit_polar_final_model.py",
        "experiments/fit_polar_final_probe.py",
        "experiments/evaluate_polar_final.py",
        "experiments/evaluate_vcoco_external.py",
        "experiments/screen_polar_embedding_classifiers.py",
        "experiments/train_polar_candidate.py",
        "src/hac/augmentations.py",
        "src/hac/config.py",
        "src/hac/data.py",
        "src/hac/metrics.py",
        "src/hac/models.py",
        "src/hac/polar.py",
        "src/hac/polar_analysis.py",
        "src/hac/polar_features.py",
        "src/hac/polar_models.py",
        "src/hac/polar_training.py",
        "src/hac/training.py",
    ]
    selection = {
        "status": "DEVELOPMENT_SELECTION_SPEC",
        "protocol_version": protocol["protocol_version"],
        "selection_basis": "development_train_validation_only",
        "data": {
            "development_manifest_sha256": sha256_file(development_path),
            "development_rows": len(development),
            "quarantine_sha256": sha256_file(args.quarantine.resolve()),
        },
        "confirmation_run_ids": confirmation_run_ids,
        "final_neural_fits": neural_fits,
        "final_probe_fits": probe_fits,
        "ensemble": {
            "aggregation": "weighted_arithmetic_mean_of_probability_rows",
            "neural_seed_aggregation": "arithmetic_mean_before_component_weighting",
            "frozen_probe_component": "dinov2_base_multilayer_logistic",
            "components": components,
            "weights": blend["weights"],
            "development_metrics": blend["metrics"],
        },
        "calibration": {
            "neural_components": "native_softmax",
            "logistic_components": "native_multinomial_probabilities",
            "rbf_component": "five_fold_sigmoid_calibration",
            "ensemble_posthoc_calibration": "none",
            "selection_metric_includes": ["log_loss", "ece_15"],
        },
        "evaluation": {
            "primary_task": "label_4",
            "primary_metric": "macro_f1",
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 20_260_822,
            "secondary_task": "label_3",
            "report_component_results": True,
        },
        "faithfulness": {
            "selection_role": "none",
            "cohort_rows": 256,
            "cohort_sampling": "deterministic_class_and_bbox_area_stratified",
            "cohort_seed": 20_260_822,
            "families": ["convnext_small_full", "dinov2_base_top4"],
            "target": "locked_predicted_class_probability",
            "attribution_methods": {
                "convnext_small_full": "gradcam",
                "dinov2_base_top4": "integrated_gradients",
            },
            "integrated_gradient_steps": 16,
            "perturbation_grid": [8, 8],
            "deletion_insertion_fractions": [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            "metrics": [
                "deletion_auc",
                "insertion_auc",
                "random_deletion_auc",
                "positive_attribution_mass_in_person_bbox",
                "pointing_game",
                "person_vs_matched_context_occlusion",
                "full_crop_probability_consistency",
                "target_sensitivity",
                "parameter_randomization_sanity",
            ],
            "strata": ["class", "bbox_area_quartile"],
            "test_used_for_attribution_selection": False,
        },
        "fault_robustness": {
            "selection_role": "none",
            "reported_separately_from_faithfulness": True,
            "input_pixel_bit_flip_rates": [0.0, 1e-5, 1e-4, 1e-3],
            "head_parameter_bit_flips": [0, 1, 4, 16],
            "seeds": [20_260_822, 20_260_823, 20_260_824],
        },
        "external_validation": {
            "dataset": "V-COCO trainval",
            "manifest_sha256": sha256_file(external_path),
            "person_rows": len(external),
            "unique_images": int(external["image_id"].nunique()),
            "image_level_unambiguous_rows": int(external_unambiguous.sum()),
            "mixed_image_policy": "exclude_from_image_level_evaluation",
            "selection_role": "none",
            "polar_test_rows_read": 0,
        },
        "scale_evidence": {
            "provenance_sha256": sha256_file(scale_provenance_path),
            "summary_sha256": sha256_file(scale_summary_path),
            "test_rows_read": 0,
        },
        "test_gate": {
            "test_manifest_sha256": sha256_file(test_path),
            "expected_rows": test_rows,
            "probe_batch_size": 32,
            "workers": 4,
        },
        "implementation_files": implementation_files,
        "test_rows_read": 0,
        "test_used_for_selection": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
