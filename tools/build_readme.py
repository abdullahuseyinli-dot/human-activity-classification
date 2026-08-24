"""Generate the repository README from locked POLAR, V-COCO, and temporal evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file

ROOT = Path(__file__).resolve().parents[1]

POLAR_NAMES = {
    "locked_ensemble": "Locked probability ensemble",
    "dinov2_base_multilayer_rbf": "DINOv2-B multilayer + calibrated RBF SVM",
    "dinov2_base_multilayer_logistic": "DINOv2-B multilayer + logistic regression",
    "dinov2_base_top4": "DINOv2-B, top four blocks adapted",
    "dinov2_small_moderate": "DINOv2-S, full adaptation",
    "convnext_small_full": "ConvNeXt-S, full adaptation",
}
VCOCO_NAMES = {
    "scale_conditioned_stacking": "Scale-conditioned DINO stack",
    "historical_v1_dino": "Historical source-only DINO",
}
VCOCO_V3_NAMES = {
    "source_only_static": ("Source-only static transfer", "0%"),
    "static": ("Target-supervised static", "0%"),
    "classification_student": ("Distilled static student", "0%"),
    "hybrid_budget_0.5": ("Routed student + teacher", "50%"),
    "teacher": ("Temporal teacher", "100%"),
}
CLASS_NAMES = {
    "sitting": "Sitting",
    "standing": "Standing",
    "walking": "Walking",
    "running": "Running",
    "walking_running": "Walking/running",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_readme(repository: Path = ROOT) -> str:
    results = repository / "results"
    vcoco_results = results / "vcoco_v2"
    vcoco_v3_results = results / "vcoco_v3"
    cptr_results = results / "okutama_cptr"

    polar_test = read_json(results / "polar_test_summary.json")
    polar_external = read_json(results / "polar_external_summary.json")
    faithfulness = read_json(results / "polar_faithfulness_summary.json")
    fault = read_json(results / "polar_fault_summary.json")
    polar_evidence = read_json(results / "polar_final_evidence_manifest.json")
    polar_selection = read_json(results / "polar_final_selection_lock.json")
    polar_audit = read_json(results / "polar_data_audit.json")
    polar_uncertainty = read_json(results / "polar_test_uncertainty.json")
    fit_manifest = read_json(results / "polar_final_fit_manifest.json")
    extension = read_json(results / "polar_extension_summary.json")
    exploratory = read_json(results / "polar_exploratory_summary.json")

    polar_statuses = {
        polar_test.get("status"): "LOCKED_FINAL_TEST_COMPLETE",
        polar_external.get("status"): "LOCKED_EXTERNAL_EVALUATION_COMPLETE",
        faithfulness.get("status"): "LOCKED_POLAR_FAITHFULNESS_COMPLETE",
        fault.get("status"): "LOCKED_POLAR_FAULT_ROBUSTNESS_COMPLETE",
        polar_evidence.get("status"): "LOCKED_POLAR_PORTFOLIO_EVIDENCE",
        exploratory.get("status"): "POSTHOC_EXPLORATORY_ANALYSIS_COMPLETE",
    }
    if any(actual != expected for actual, expected in polar_statuses.items()):
        raise RuntimeError("README generation requires complete locked POLAR evidence")
    polar_lock_hashes = {
        item["selection_lock_sha256"]
        for item in (polar_test, polar_external, faithfulness, fault, polar_evidence)
    }
    polar_lock_hash = polar_evidence["selection_lock_sha256"]
    if polar_lock_hashes != {polar_lock_hash}:
        raise RuntimeError("POLAR evidence does not share one final selection lock")
    if polar_test.get("test_used_for_selection") or polar_external.get("test_used_for_selection"):
        raise RuntimeError("Test-selected POLAR evidence cannot be promoted")

    vcoco_protocol = read_json(vcoco_results / "protocol_lock.json")
    vcoco_selection = read_json(vcoco_results / "final_selection_lock.json")
    vcoco_gate = read_json(vcoco_results / "test_access_gate.json")
    vcoco_summary = read_json(vcoco_results / "official_test_summary.json")
    vcoco_evidence = read_json(vcoco_results / "evidence_manifest.json")
    vcoco_uncertainty = read_json(vcoco_results / "official_test_uncertainty.json")
    vcoco_statuses = {
        vcoco_protocol.get("status"): "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING",
        vcoco_selection.get("status"): "VCOCO_V2_FINAL_SELECTION_LOCKED_PRE_TEST",
        vcoco_gate.get("status"): "VCOCO_V2_OFFICIAL_TEST_GATE_OPEN",
        vcoco_summary.get("status"): "VCOCO_V2_OFFICIAL_TEST_EVALUATION_COMPLETE",
        vcoco_evidence.get("status"): "VCOCO_V2_PORTABLE_EVIDENCE_COMPLETE",
    }
    if any(actual != expected for actual, expected in vcoco_statuses.items()):
        raise RuntimeError("README generation requires complete locked V-COCO v2 evidence")
    vcoco_protocol_hash = vcoco_protocol["source_lock_sha256"]
    vcoco_selection_hash = vcoco_evidence["selection_lock_sha256"]
    if {
        vcoco_evidence["protocol_lock_sha256"],
        vcoco_selection["protocol_lock_sha256"],
        vcoco_summary["protocol_lock_sha256"],
    } != {vcoco_protocol_hash}:
        raise RuntimeError("V-COCO protocol lineage does not align")
    if {
        vcoco_selection["source_lock_sha256"],
        vcoco_gate["selection_lock_sha256"],
        vcoco_summary["selection_lock_sha256"],
    } != {vcoco_selection_hash}:
        raise RuntimeError("V-COCO selection lineage does not align")
    if (
        vcoco_summary.get("test_used_for_selection") is not False
        or vcoco_summary.get("official_test_label_open_count") != 1
        or vcoco_gate.get("official_test_label_open_count") != 1
    ):
        raise RuntimeError("V-COCO official-test access contract changed")

    vcoco_v3_evidence = read_json(vcoco_v3_results / "evidence_manifest.json")
    vcoco_v3_summary = read_json(vcoco_v3_results / "confirmation_summary.json")
    vcoco_v3_uncertainty = read_json(vcoco_v3_results / "confirmation_uncertainty.json")
    if (
        vcoco_v3_evidence.get("status") != "VCOCO_V3_PORTABLE_EVIDENCE_COMPLETE"
        or vcoco_v3_summary.get("status") != "VCOCO_V3_TEMPORAL_CONFIRMATION_COMPLETE"
        or vcoco_v3_evidence.get("confirmation_open_number") != 1
        or vcoco_v3_summary.get("confirmation_open_number") != 1
        or vcoco_v3_evidence.get("confirmation_used_for_selection") is not False
    ):
        raise RuntimeError("README generation requires complete motion-study evidence")
    if vcoco_v3_evidence.get("pipeline_lock_sha256") != vcoco_v3_summary.get(
        "pipeline_lock_sha256"
    ):
        raise RuntimeError("Motion-study pipeline lineage does not align")

    cptr_decision = read_json(cptr_results / "development_decision.json")
    cptr_provenance = read_json(cptr_results / "provenance.json")
    cptr_manifest = read_json(cptr_results / "evidence_manifest.json")
    if (
        cptr_decision.get("status") != "OKUTAMA_CPTR_DEVELOPMENT_LOCKED_NO_PROMOTION"
        or cptr_decision.get("promotion_passed") is not False
        or cptr_provenance.get("status") != "OKUTAMA_CPTR_PORTABLE_EVIDENCE_COMPLETE"
        or cptr_manifest.get("study") != "okutama_cptr_development"
    ):
        raise RuntimeError("README generation requires complete CPTR development evidence")
    for name, evidence in cptr_manifest["artifacts"].items():
        path = cptr_results / name
        if not path.is_file() or sha256_file(path) != evidence["sha256"]:
            raise RuntimeError(f"CPTR portable evidence drift: {name}")

    polar_metrics = pd.read_csv(results / "polar_test_metrics.csv")
    polar_rows = []
    for row in polar_metrics.itertuples(index=False):
        interval = polar_uncertainty[row.candidate]
        polar_rows.append(
            [
                POLAR_NAMES.get(row.candidate, row.candidate),
                f"{row.macro_f1:.3f}",
                f"[{interval['ci_95_low']:.3f}, {interval['ci_95_high']:.3f}]",
                f"{row.accuracy:.3f}",
                f"{row.log_loss:.3f}",
                f"{row.ece:.3f}",
            ]
        )
    polar_table = markdown_table(
        ["Predeclared candidate", "Macro-F1", "95% CI", "Accuracy", "Log loss", "ECE"],
        polar_rows,
    )

    polar_per_class = pd.read_csv(results / "polar_test_per_class.csv")
    polar_per_class = polar_per_class[polar_per_class["candidate"].eq("locked_ensemble")]
    polar_per_class_table = markdown_table(
        ["Class", "Precision", "Recall", "F1", "Support"],
        [
            [
                CLASS_NAMES.get(str(row["class"]), str(row["class"]).title()),
                f"{row['precision']:.3f}",
                f"{row['recall']:.3f}",
                f"{row['f1']:.3f}",
                f"{int(row['support']):,}",
            ]
            for _, row in polar_per_class.iterrows()
        ],
    )

    vcoco_metrics = pd.read_csv(vcoco_results / "official_test_metrics.csv")
    vcoco_table = markdown_table(
        [
            "Official-test method",
            "Macro-F1",
            "Accuracy",
            "Balanced accuracy",
            "Log loss",
            "ECE",
        ],
        [
            [
                VCOCO_NAMES[row.method],
                f"{row.macro_f1:.4f}",
                f"{row.accuracy:.4f}",
                f"{row.balanced_accuracy:.4f}",
                f"{row.log_loss:.4f}",
                f"{row.ece:.4f}",
            ]
            for row in vcoco_metrics.itertuples(index=False)
        ],
    )

    vcoco_per_class = pd.read_csv(vcoco_results / "official_test_per_class.csv").set_index(
        ["method", "class"]
    )
    vcoco_per_class_table = markdown_table(
        ["Class", "Stack F1", "Baseline F1", "Change", "Support"],
        [
            [
                CLASS_NAMES[class_name],
                f"{vcoco_per_class.loc[('scale_conditioned_stacking', class_name), 'f1']:.4f}",
                f"{vcoco_per_class.loc[('historical_v1_dino', class_name), 'f1']:.4f}",
                (
                    f"{vcoco_per_class.loc[('scale_conditioned_stacking', class_name), 'f1'] - vcoco_per_class.loc[('historical_v1_dino', class_name), 'f1']:+.4f}"
                ),
                (
                    f"{int(vcoco_per_class.loc[('scale_conditioned_stacking', class_name), 'support']):,}"
                ),
            ]
            for class_name in ("sitting", "standing", "walking_running")
        ],
    )

    vcoco_v3_metrics = pd.read_csv(vcoco_v3_results / "confirmation_metrics.csv").set_index(
        "family"
    )
    vcoco_v3_table = markdown_table(
        ["Confirmation method", "Clip fraction", "Macro-F1", "Accuracy", "Locomotion F1"],
        [
            [
                VCOCO_V3_NAMES[family][0],
                VCOCO_V3_NAMES[family][1],
                f"{vcoco_v3_metrics.loc[family, 'macro_f1']:.4f}",
                f"{vcoco_v3_metrics.loc[family, 'accuracy']:.4f}",
                f"{vcoco_v3_metrics.loc[family, 'locomotion_f1']:.4f}",
            ]
            for family in VCOCO_V3_NAMES
        ],
    )

    polar_primary = polar_test["primary_metrics"]
    polar_primary_ci = polar_uncertainty["locked_ensemble"]
    polar_paired = polar_uncertainty["locked_ensemble_paired_deltas"]
    smallest_polar_delta = min(polar_paired.items(), key=lambda item: item[1]["point_estimate"])
    secondary = pd.read_csv(results / "polar_test_secondary_metrics.csv").iloc[0]
    scale = sorted(extension["scale_curve"], key=lambda row: row["actual_train_size"])

    conv_faith = faithfulness["aggregate"]["convnext_small_full"]
    dino_faith = faithfulness["aggregate"]["dinov2_base_top4"]
    aggregate_fault = {
        (row["family"], row["condition"], float(row["level"])): row
        for row in fault["aggregate_results"]
        if str(row["fault_seed"]) == "aggregate"
    }
    conv_input = aggregate_fault[("convnext_small_full", "uint8_input_bit_flip_rate", 0.001)]
    dino_input = aggregate_fault[("dinov2_base_top4", "uint8_input_bit_flip_rate", 0.001)]
    conv_head = aggregate_fault[
        ("convnext_small_full", "symmetric_int8_head_weight_bit_flips", 16.0)
    ]
    dino_head = aggregate_fault[("dinov2_base_top4", "symmetric_int8_head_weight_bit_flips", 16.0)]

    polar_weights = polar_selection["ensemble"]["weights"]
    clean_counts = polar_audit["clean_target_counts"]
    train_rows = sum(clean_counts["train"].values())
    validation_rows = sum(clean_counts["val"].values())
    test_rows = sum(clean_counts["test"].values())
    logistic = fit_manifest["probes"]["dinov2_base_multilayer_logistic"]
    rbf = fit_manifest["probes"]["dinov2_base_multilayer_rbf"]

    vcoco_primary = vcoco_summary["primary_metrics"]
    vcoco_final = vcoco_selection["final_test"]
    vcoco_development = vcoco_selection["selection"]["validation_metrics"]
    comparisons = vcoco_selection["development_comparisons"]
    factorized_gain = comparisons["factorized_minus_flat_same_features"]
    single_view_gain = comparisons["champion_minus_single_view_dino"]
    augmix_difference = comparisons["champion_minus_augmix_lpft"]

    vcoco_strata = pd.read_csv(vcoco_results / "official_test_strata.csv").set_index(
        ["stratum", "value", "method"]
    )
    small_stack = vcoco_strata.loc[
        ("area_quartile", "Q1_small", "scale_conditioned_stacking"),
        "macro_f1_fixed_classes",
    ]
    small_baseline = vcoco_strata.loc[
        ("area_quartile", "Q1_small", "historical_v1_dino"),
        "macro_f1_fixed_classes",
    ]
    large_stack = vcoco_strata.loc[
        ("area_quartile", "Q4_large", "scale_conditioned_stacking"),
        "macro_f1_fixed_classes",
    ]
    large_baseline = vcoco_strata.loc[
        ("area_quartile", "Q4_large", "historical_v1_dino"),
        "macro_f1_fixed_classes",
    ]
    selective = read_json(vcoco_results / "official_test_selective_metrics.json")[
        "scale_conditioned_stacking"
    ]
    coverage_90 = next(
        row for row in selective["coverage_points"] if row["requested_coverage"] == 0.9
    )
    coverage_70 = next(
        row for row in selective["coverage_points"] if row["requested_coverage"] == 0.7
    )

    v3_teacher_gain = vcoco_v3_uncertainty["teacher"]["macro_f1"]
    v3_half_gain = vcoco_v3_uncertainty["hybrid_budget_0.5"]["macro_f1"]
    cptr_validation = cptr_decision["development_validation"]
    cptr_oof = cptr_decision["grouped_crossfit_oof"]
    cptr_subgroups = pd.read_csv(cptr_results / "subgroup_metrics.csv").set_index(
        ["scope", "subgroup"]
    )

    return f"""# Human Activity Classification Under Domain and Temporal Shift

[![Quality gates](https://github.com/abdullahuseyinli-dot/human-activity-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/abdullahuseyinli-dot/human-activity-classification/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/Code-MIT-0F766E.svg)](LICENSE)
[![Study report: v3.0.0](https://img.shields.io/badge/study_report-v3.0.0-0F766E.svg)](output/pdf/vcoco_v3_motion_identifiability_v3.0.0.pdf)

A leakage-audited study of still-image and short-clip human activity classification
across POLAR, V-COCO, and Okutama-Action. The repository covers source-overlap control,
transfer learning, frozen representations, person-centric multiview features,
factorized targets, motion identifiability, budgeted temporal inference, calibrated
classifiers, selection locks, one-time test gates, attribution sanity checks, and
bounded fault injection.

## Motion identifiability and budgeted temporal inference

The [technical report](output/pdf/vcoco_v3_motion_identifiability_v3.0.0.pdf),
[Markdown source](docs/VCOCO_V3_MOTION_IDENTIFIABILITY.md),
[portable evidence](results/vcoco_v3/README.md), and
[execution runbook](docs/VCOCO_V3_EXECUTION_RUNBOOK.md) document the tracked-video
extension. All target-domain model fitting ran on CUDA, and the complete pipeline was
fixed before the Okutama provider test archive was opened once.

![Okutama confirmation comparison](assets/vcoco_v3_confirmation_comparison.png)

{vcoco_v3_table}

The temporal teacher improves over the matched static model by **{v3_teacher_gain["point_estimate"]:+.4f} macro-F1**, with a 95% paired
scenario-cluster interval of **[{v3_teacher_gain["ci_95_low"]:+.4f}, {v3_teacher_gain["ci_95_high"]:+.4f}]**. The fixed 50% clip policy retains **{v3_half_gain["point_estimate"] / v3_teacher_gain["point_estimate"]:.1%}**
of that gain and reaches **{vcoco_v3_metrics.loc["hybrid_budget_0.5", "macro_f1"]:.4f} macro-F1**; its interval is **[{v3_half_gain["ci_95_low"]:+.4f}, {v3_half_gain["ci_95_high"]:+.4f}]**.

The matched static distillation result is neutral on confirmation. DINOv3-B also does
not displace DINOv2-B under the fixed frozen-representation screen. In contrast, short
temporal context improves every class F1 and four of five confirmation scenarios.

![Fixed-budget temporal routing](assets/vcoco_v3_routing_curve.png)

## Okutama camera-compensated part-trajectory residual architecture study

The [development report](output/pdf/okutama_cptr_development_v3.0.0.pdf),
[Markdown source](docs/OKUTAMA_CPTR_DEVELOPMENT.md), and
[portable evidence](results/okutama_cptr/README.md) document the follow-up
camera-compensated part-trajectory residual (CPTR) architecture study. Frozen static
and temporal anchors were extended with center-conditioned
residuals, camera-compensated kinematics, confidence-masked body-region tokens,
quality-aware gates, counterfactual objectives, masked target-video adaptation,
GroupDRO, frozen SigLIP, and top-block LoRA controls.

| Development evaluation | Temporal baseline | Center + parts | Change |
| --- | ---: | ---: | ---: |
| Fixed validation, {cptr_validation["recordings"]} recordings | {cptr_validation["baseline_metrics"]["macro_f1"]:.4f} | {cptr_validation["candidate_metrics"]["macro_f1"]:.4f} | {cptr_validation["macro_f1_delta"]:+.4f} |
| Five-fold OOF, {cptr_oof["recordings"]} recordings | {cptr_oof["baseline_metrics"]["macro_f1"]:.4f} | {cptr_oof["candidate_metrics"]["macro_f1"]:.4f} | {cptr_oof["macro_f1_delta"]:+.4f} |

The validation improvement is concentrated in standing F1 ({cptr_validation["baseline_metrics"]["standing_f1"]:.4f} to {cptr_validation["candidate_metrics"]["standing_f1"]:.4f}), but it
does not persist across recording-grouped OOF evaluation. Occluded-window OOF macro-F1
changes by {cptr_subgroups.loc[("crossfit_oof", "window_occluded"), "macro_f1_delta"]:+.4f} relative to the baseline, while clear-window performance is slightly positive.
The existing temporal ensemble remains the default; the new branch is retained with
its complete component, uncertainty, and faithfulness evidence.

## Person-level V-COCO transfer - study v2

The versioned [technical report](output/pdf/vcoco_v2_external_transfer_v2.0.0.pdf),
[Markdown source](docs/VCOCO_V2_EXTERNAL_TRANSFER.md),
[release notes](docs/releases/POLAR_STUDY_V2.0.0.md), and
[SHA-256 manifest](results/polar_study_v2.0.0_manifest.json) document the complete
follow-up study.

![Official V-COCO test comparison](assets/vcoco_v2_official_test_comparison.png)

The selected system uses two aspect-preserving DINOv2-B person views, cross-fitted
base probabilities, and five box-geometry features. It was trained on the official
V-COCO training split, selected on validation, refitted on
{vcoco_selection["final_fit"]["training_people"]:,} development people, and evaluated
after the test labels were opened once.

{vcoco_table}

The official-test gain is **{vcoco_uncertainty["point_estimate"]:+.4f} macro-F1**, with
a 95% image-cluster bootstrap interval of
**[{vcoco_uncertainty["ci_95_low"]:+.4f}, {vcoco_uncertainty["ci_95_high"]:+.4f}]**.
Validation macro-F1 was **{vcoco_development["macro_f1"]:.4f}** and test macro-F1 was
**{vcoco_primary["macro_f1"]:.4f}** on {vcoco_final["expected_people"]:,} people from
{vcoco_final["expected_images"]:,} images.

{vcoco_per_class_table}

### Measured patterns and controlled gains

- **Person scale and crop construction:** the stack gains
  **{small_stack - small_baseline:+.4f} macro-F1** in the smallest box-area quartile,
  compared with **{large_stack - large_baseline:+.4f}** in the largest quartile.
  Aspect-preserving person views greatly reduce the observed correlation between
  correctness and apparent person size.
- **Multiview complementarity:** the stack improves over the best single-view DINO
  control by **{single_view_gain["point_estimate"]:+.4f}**, with a 95% interval of
  **[{single_view_gain["ci_95_low"]:+.4f}, {single_view_gain["ci_95_high"]:+.4f}]**.
- **Target structure:** factorizing seated/upright posture and
  stationary/locomoting motion improves over the matched flat classifier by
  **{factorized_gain["point_estimate"]:+.4f}**, interval
  **[{factorized_gain["ci_95_low"]:+.4f}, {factorized_gain["ci_95_high"]:+.4f}]**.
- **Neural adaptation:** LP-FT with AugMix is close to the selected stack; their
  development difference is **{augmix_difference["point_estimate"]:+.4f}**, interval
  **[{augmix_difference["ci_95_low"]:+.4f}, {augmix_difference["ci_95_high"]:+.4f}]**.
  The frozen stack therefore keeps the simpler final fit.
- **Calibrated abstention:** at 90% coverage, the stack reaches
  **{coverage_90["accuracy"]:.4f} accuracy** and **{coverage_90["macro_f1"]:.4f}
  macro-F1**. At 70% coverage, it reaches **{coverage_70["accuracy"]:.4f} accuracy**
  and **{coverage_70["macro_f1"]:.4f} macro-F1**.

![V-COCO person-scale gains](assets/vcoco_v2_scale_gain.png)

![V-COCO selective prediction](assets/vcoco_v2_selective_prediction.png)

## POLAR v1: source benchmark

The original four-class benchmark compares ConvNeXt and DINOv2 adaptation, linear and
nonlinear classifiers on frozen representations, and a development-locked probability
ensemble.

![Held-out POLAR comparison](assets/polar_test_comparison.png)

The ensemble reached **{polar_primary["macro_f1"]:.3f} macro-F1** (95% stratified
bootstrap interval **[{polar_primary_ci["ci_95_low"]:.3f},
{polar_primary_ci["ci_95_high"]:.3f}]**) and **{polar_primary["accuracy"]:.3f}
accuracy** on {test_rows:,} held-out images. Its smallest paired gain over a component
was **{smallest_polar_delta[1]["point_estimate"]:+.3f} macro-F1**, interval
**[{smallest_polar_delta[1]["ci_95_low"]:+.3f},
{smallest_polar_delta[1]["ci_95_high"]:+.3f}]**.

{polar_table}

The secondary three-class mapping, with walking and running combined, reached
**{secondary.macro_f1:.3f} macro-F1** and **{secondary.accuracy:.3f} accuracy**.

{polar_per_class_table}

![Locked POLAR confusion matrix](assets/polar_confusion_matrix.png)

### What changed the POLAR result

- The frozen DINOv2-B validation curve rose from
  **{scale[0]["macro_f1_mean"]:.3f}** at {scale[0]["actual_train_size"]:,} training
  images to **{scale[-1]["macro_f1_mean"]:.3f}** at
  {scale[-1]["actual_train_size"]:,}.
- The locked neural configurations use dropout 0.10, no MixUp, no label smoothing,
  and either mild or moderate augmentation. Interventions that reduced validation
  performance remain visible in the evidence tables.
- The final blend fixed its development-selected weights at ConvNeXt-S
  {polar_weights["convnext_small_full"]:.0%}, DINOv2-S
  {polar_weights["dinov2_small_moderate"]:.0%}, adapted DINOv2-B
  {polar_weights["dinov2_base_top4"]:.0%}, logistic regression
  {polar_weights["dinov2_base_multilayer_logistic"]:.0%}, and calibrated RBF SVM
  {polar_weights["dinov2_base_multilayer_rbf"]:.0%}.

![POLAR data-scale curve](assets/polar_scale_curve.png)

### Linear versus nonlinear final-stage classifiers

The calibrated RBF SVM is the strongest standalone POLAR component at 0.927
macro-F1. The multinomial logistic model reaches 0.926 with better log loss
(0.176 versus 0.228), fits in {logistic["fit_seconds"]:.1f} seconds, and serializes to
{logistic["pipeline_bytes"] / 1_000_000:.1f} MB. The RBF pipeline takes
{rbf["fit_seconds"] / 60:.1f} minutes and serializes to
{rbf["pipeline_bytes"] / 1_000_000:.1f} MB. The SVM contributes a small nonlinear
margin; logistic regression is the lighter calibrated endpoint.

## Attribution and bounded fault response

The attribution audit uses a deterministic 256-image cohort balanced by class and
person-box-area quartile. ConvNeXt Grad-CAM has a targeted-versus-random deletion gap
of **{conv_faith["deletion_selectivity_gap"]["mean"]:.3f}**, concentrates
**{conv_faith["person_attribution_mass_lift"]["mean"]:.2f}x** more attribution in the
person box than uniform area, and produces a person-minus-context probability drop of
**{conv_faith["person_minus_context_occlusion_drop"]["mean"]:.3f}**. DINOv2-B
integrated gradients show **{dino_faith["person_attribution_mass_lift"]["mean"]:.2f}x**
area-normalized lift but retain high correlations after target and parameter
randomization, so the maps are used as coarse localization diagnostics.

![Attribution faithfulness](assets/polar_faithfulness.png)

![Attribution sanity checks](assets/polar_attribution_sanity.png)

Bit-flip experiments are reported separately. At a 0.1% input bit-flip rate,
prediction agreement with the clean models is
**{conv_input["prediction_agreement_with_clean"]:.3f}** for ConvNeXt-S and
**{dino_input["prediction_agreement_with_clean"]:.3f}** for DINOv2-B. Sixteen flips
per quantized classifier-weight matrix retain
**{conv_head["prediction_agreement_with_clean"]:.3f}** and
**{dino_head["prediction_agreement_with_clean"]:.3f}** agreement, respectively, on
the same cohort.

![Fault robustness](assets/polar_fault_robustness.png)

## Evaluation controls and evidence lineage

### POLAR

- Clean split: {train_rows:,} train, {validation_rows:,} validation, and
  {test_rows:,} test images.
- {polar_audit["quarantine_images"]} images in
  {polar_audit["quarantine_components"]} confirmed cross-split source-related
  components were quarantined before supervised fitting.
- Candidate selection, blend weights, epochs, and classifier settings were fixed on
  development evidence before the test cache opened.
- The portable summaries share selection-lock SHA-256 {polar_lock_hash}.

### V-COCO v2

- Official memberships remain intact after quarantining 60 test images used in the
  earlier external audit.
- Source-image groups stay together during cross-validation and image-cluster
  bootstrapping.
- The selected stack, evaluator, dependencies, and historical baseline were bound
  before the single official test-label open.
- Protocol lock: {vcoco_protocol_hash}.
- Selection lock: {vcoco_selection_hash}.

### Motion-identifiability extension

- The 130-presentation human pilot is descriptive and is excluded from candidate
  selection.
- Okutama scenarios and synchronized drone views stay together across train,
  validation, calibration, and confirmation boundaries.
- The external protocol disables CPU fitting and records 100 CUDA temporal runs.
- Model artifacts, temperatures, routing budgets, and prediction-set thresholds were
  locked before the single confirmation open.
- Pipeline lock: {vcoco_v3_evidence["pipeline_lock_sha256"]}.

### Camera-compensated part-trajectory residual architecture development

- Camera, trajectory, part, counterfactual, masked-adaptation, specialist, and robust
  training components were evaluated in a fixed sequence.
- The strongest component was repeated with five seeds and 25 grouped cross-fit runs;
  every fit used CUDA and preserved recording boundaries.
- The fixed validation split contains three recordings, so its cluster bootstrap is
  accompanied by the eight-permutation exact recording-swap test.
- The architecture did not pass the aggregate and grouped-OOF gates. Calibration and
  confirmation data remain unopened for this branch.

Checkpoints, image paths, dense probabilities, and large feature tensors remain
outside Git. The tracked evidence is path-sanitized and hash-indexed.

The [scientific validation plan](docs/SCIENTIFIC_VALIDATION_PLAN.md) maps each known
limitation to a measurement, acceptance gate, and permitted claim. It covers
independent annotation, external-domain replication, grouped inference, matched
baselines, operational measurements, and a clean-environment replay.

## Reports and release files

| Artifact | Purpose |
| --- | --- |
| [Motion-identifiability report](output/pdf/vcoco_v3_motion_identifiability_v3.0.0.pdf) | Sealed static, temporal, distillation, and routing study |
| [CPTR development report](output/pdf/okutama_cptr_development_v3.0.0.pdf) | Camera compensation, part tokens, residual fusion, cross-fit, and failure analysis |
| [Scientific validation plan](docs/SCIENTIFIC_VALIDATION_PLAN.md) | Remaining limitations, required measurements, and claim gates |
| [Study v3.0.0 version notes](docs/releases/HUMAN_ACTIVITY_STUDY_V3.0.0.md) | Candidate scope, headline results, and validation commands |
| [Study v3.0.0 manifest](results/human_activity_study_v3.0.0_manifest.json) | SHA-256 inventory of the release candidate |
| [CPTR development evidence](results/okutama_cptr/README.md) | Component screens, grouped OOF results, uncertainty, and faithfulness |
| [Motion-identifiability evidence](results/vcoco_v3/README.md) | CUDA lineage, metrics, uncertainty, and subgroup tables |
| [V-COCO v2 report](output/pdf/vcoco_v2_external_transfer_v2.0.0.pdf) | Person-level transfer study |
| [V-COCO v2 evidence](results/vcoco_v2/README.md) | Locks, metrics, uncertainty, and mechanism tables |
| [POLAR v1 report](output/pdf/polar_public_report_v1.0.0.pdf) | Original four-class source benchmark |
| [Executed notebook](human_activity_classification.ipynb) | Compact, code-backed result walkthrough |
| [Result lineage](docs/RESULT_LINEAGE.md) | Complete study artifact map |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | Dataset and model terms |

## Repository layout

~~~text
.
|-- human_activity_classification.ipynb  # executed evidence narrative
|-- src/hac/                             # reusable data, model, metric, and audit code
|-- experiments/                         # staged fitting and evaluation runners
|-- tools/                               # dataset, export, validation, and figure utilities
|-- results/                             # portable locks, metrics, uncertainty, and hashes
|-- assets/                              # publication figures
|-- docs/                                # protocols, reports, release notes, and lineage
|-- tests/                               # fast invariants and evidence-contract tests
+-- .github/workflows/ci.yml             # Linux quality gates
~~~

## Reproduce the environment

Python 3.11 is the recorded runtime. Install the PyTorch build appropriate for the
machine, then install the project:

~~~bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,dev,report]"
~~~

The staged commands and acceptance boundaries are documented in
[experiments/README.md](experiments/README.md). Validate a checkout with:

~~~bash
python -m ruff check .
python -m compileall -q src experiments tools
python -m pytest
python tools/validate_repository.py
python tools/build_study_release_manifest.py --check
python tools/build_v3_release_manifest.py --check
python tools/verify_v3_release_archive.py
~~~

The historical v1/v2 environment remains frozen in `requirements-lock.txt`. The tested
top-level dependency versions for the v3 CUDA work are recorded in
`requirements-v3-lock.txt`; the execution runbook shows the matching PyTorch CUDA
installation order.

## Citation, references, and license

Citation metadata is provided in [CITATION.cff](CITATION.cff).

> Huseyinli, A. (2026). *Human Activity Classification Under Domain and Temporal
> Shift* (Version 3.0.0) [Computer software].
> https://github.com/abdullahuseyinli-dot/human-activity-classification

- [POLAR dataset](https://doi.org/10.17632/hvnsh7rwz7.1)
- [V-COCO](https://arxiv.org/abs/1505.04474)
- [Okutama-Action](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w34/html/Barekatain_Okutama-Action_An_Aerial_CVPR_2017_paper.html)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [DINOv3](https://arxiv.org/abs/2508.10104)
- [ConvNeXt](https://arxiv.org/abs/2201.03545)
- [SigLIP2](https://arxiv.org/abs/2502.14786)
- [SigLIP](https://arxiv.org/abs/2303.15343)
- [AugMix](https://arxiv.org/abs/1912.02781)
- [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292)

Original code and documentation are MIT licensed. Dataset media, qualitative
source-image composites, and pretrained checkpoints are excluded from the current
distributable tree; upstream data and model terms are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
"""


def main() -> None:
    readme = build_readme(ROOT)
    (ROOT / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    print("Wrote README.md from locked POLAR, V-COCO, temporal, and CPTR evidence")


if __name__ == "__main__":
    main()
