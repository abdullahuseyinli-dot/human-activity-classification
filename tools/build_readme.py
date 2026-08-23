"""Generate the repository README from locked POLAR and V-COCO evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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

    polar_primary = polar_test["primary_metrics"]
    polar_primary_ci = polar_uncertainty["locked_ensemble"]
    polar_paired = polar_uncertainty["locked_ensemble_paired_deltas"]
    smallest_polar_delta = min(
        polar_paired.items(), key=lambda item: item[1]["point_estimate"]
    )
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
    dino_head = aggregate_fault[
        ("dinov2_base_top4", "symmetric_int8_head_weight_bit_flips", 16.0)
    ]

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

    return f"""# Source-Overlap-Controlled Human Activity Classification

[![Quality gates](https://github.com/abdullahuseyinli-dot/human-activity-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/abdullahuseyinli-dot/human-activity-classification/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/Code-MIT-0F766E.svg)](LICENSE)
[![Study report: v2.0.0](https://img.shields.io/badge/study_report-v2.0.0-0F766E.svg)](output/pdf/vcoco_v2_external_transfer_v2.0.0.pdf)

A leakage-audited study of still-image human activity classification across POLAR and
V-COCO. The repository covers source-overlap control, transfer learning, frozen
representations, person-centric multiview features, factorized targets, calibrated
classifiers, selection locks, one-time test gates, attribution sanity checks, and
bounded fault injection.

## V-COCO v2: person-level external transfer

The current [technical report](output/pdf/vcoco_v2_external_transfer_v2.0.0.pdf),
[Markdown source](docs/VCOCO_V2_EXTERNAL_TRANSFER.md),
[release notes](docs/releases/POLAR_STUDY_V2.0.0.md), and
[SHA-256 manifest](results/polar_study_v2.0.0_manifest.json) document the complete
follow-up study.

![Official V-COCO test comparison](assets/vcoco_v2_official_test_comparison.png)

The selected system uses two aspect-preserving DINOv2-B person views, cross-fitted
base probabilities, and five box-geometry features. It was trained on the official
V-COCO training split, selected on validation, refitted on
{vcoco_selection['final_fit']['training_people']:,} development people, and evaluated
after the test labels were opened once.

{vcoco_table}

The official-test gain is **{vcoco_uncertainty['point_estimate']:+.4f} macro-F1**, with
a 95% image-cluster bootstrap interval of
**[{vcoco_uncertainty['ci_95_low']:+.4f}, {vcoco_uncertainty['ci_95_high']:+.4f}]**.
Validation macro-F1 was **{vcoco_development['macro_f1']:.4f}** and test macro-F1 was
**{vcoco_primary['macro_f1']:.4f}** on {vcoco_final['expected_people']:,} people from
{vcoco_final['expected_images']:,} images.

{vcoco_per_class_table}

### Measured sources of improvement

- **Person scale and crop construction:** the stack gains
  **{small_stack - small_baseline:+.4f} macro-F1** in the smallest box-area quartile,
  compared with **{large_stack - large_baseline:+.4f}** in the largest quartile.
  Aspect-preserving person views remove most of the historical dependence on person
  height and box area.
- **Multiview complementarity:** the stack improves over the best single-view DINO
  control by **{single_view_gain['point_estimate']:+.4f}**, with a 95% interval of
  **[{single_view_gain['ci_95_low']:+.4f}, {single_view_gain['ci_95_high']:+.4f}]**.
- **Target structure:** factorizing seated/upright posture and
  stationary/locomoting motion improves over the matched flat classifier by
  **{factorized_gain['point_estimate']:+.4f}**, interval
  **[{factorized_gain['ci_95_low']:+.4f}, {factorized_gain['ci_95_high']:+.4f}]**.
- **Neural adaptation:** LP-FT with AugMix is close to the selected stack; their
  development difference is **{augmix_difference['point_estimate']:+.4f}**, interval
  **[{augmix_difference['ci_95_low']:+.4f}, {augmix_difference['ci_95_high']:+.4f}]**.
  The frozen stack therefore keeps the simpler final fit.
- **Calibrated abstention:** at 90% coverage, the stack reaches
  **{coverage_90['accuracy']:.4f} accuracy** and **{coverage_90['macro_f1']:.4f}
  macro-F1**. At 70% coverage, it reaches **{coverage_70['accuracy']:.4f} accuracy**
  and **{coverage_70['macro_f1']:.4f} macro-F1**.

![V-COCO person-scale gains](assets/vcoco_v2_scale_gain.png)

![V-COCO selective prediction](assets/vcoco_v2_selective_prediction.png)

## POLAR v1: source benchmark

The original four-class benchmark compares ConvNeXt and DINOv2 adaptation, linear and
nonlinear classifiers on frozen representations, and a development-locked probability
ensemble.

![Held-out POLAR comparison](assets/polar_test_comparison.png)

The ensemble reached **{polar_primary['macro_f1']:.3f} macro-F1** (95% stratified
bootstrap interval **[{polar_primary_ci['ci_95_low']:.3f},
{polar_primary_ci['ci_95_high']:.3f}]**) and **{polar_primary['accuracy']:.3f}
accuracy** on {test_rows:,} held-out images. Its smallest paired gain over a component
was **{smallest_polar_delta[1]['point_estimate']:+.3f} macro-F1**, interval
**[{smallest_polar_delta[1]['ci_95_low']:+.3f},
{smallest_polar_delta[1]['ci_95_high']:+.3f}]**.

{polar_table}

The secondary three-class mapping, with walking and running combined, reached
**{secondary.macro_f1:.3f} macro-F1** and **{secondary.accuracy:.3f} accuracy**.

{polar_per_class_table}

![Locked POLAR confusion matrix](assets/polar_confusion_matrix.png)

### What changed the POLAR result

- The frozen DINOv2-B validation curve rose from
  **{scale[0]['macro_f1_mean']:.3f}** at {scale[0]['actual_train_size']:,} training
  images to **{scale[-1]['macro_f1_mean']:.3f}** at
  {scale[-1]['actual_train_size']:,}.
- The locked neural configurations use dropout 0.10, no MixUp, no label smoothing,
  and either mild or moderate augmentation. Interventions that reduced validation
  performance remain visible in the evidence tables.
- The final blend fixed its development-selected weights at ConvNeXt-S
  {polar_weights['convnext_small_full']:.0%}, DINOv2-S
  {polar_weights['dinov2_small_moderate']:.0%}, adapted DINOv2-B
  {polar_weights['dinov2_base_top4']:.0%}, logistic regression
  {polar_weights['dinov2_base_multilayer_logistic']:.0%}, and calibrated RBF SVM
  {polar_weights['dinov2_base_multilayer_rbf']:.0%}.

![POLAR data-scale curve](assets/polar_scale_curve.png)

### Linear versus nonlinear final-stage classifiers

The calibrated RBF SVM is the strongest standalone POLAR component at 0.927
macro-F1. The multinomial logistic model reaches 0.926 with better log loss
(0.176 versus 0.228), fits in {logistic['fit_seconds']:.1f} seconds, and serializes to
{logistic['pipeline_bytes'] / 1_000_000:.1f} MB. The RBF pipeline takes
{rbf['fit_seconds'] / 60:.1f} minutes and serializes to
{rbf['pipeline_bytes'] / 1_000_000:.1f} MB. The SVM contributes a small nonlinear
margin; logistic regression is the lighter calibrated endpoint.

## Attribution and bounded fault response

The attribution audit uses a deterministic 256-image cohort balanced by class and
person-box-area quartile. ConvNeXt Grad-CAM has a targeted-versus-random deletion gap
of **{conv_faith['deletion_selectivity_gap']['mean']:.3f}**, concentrates
**{conv_faith['person_attribution_mass_lift']['mean']:.2f}x** more attribution in the
person box than uniform area, and produces a person-minus-context probability drop of
**{conv_faith['person_minus_context_occlusion_drop']['mean']:.3f}**. DINOv2-B
integrated gradients show **{dino_faith['person_attribution_mass_lift']['mean']:.2f}x**
area-normalized lift but retain high correlations after target and parameter
randomization, so the maps are used as coarse localization diagnostics.

![Attribution faithfulness](assets/polar_faithfulness.png)

![Attribution sanity checks](assets/polar_attribution_sanity.png)

Bit-flip experiments are reported separately. At a 0.1% input bit-flip rate,
prediction agreement with the clean models is
**{conv_input['prediction_agreement_with_clean']:.3f}** for ConvNeXt-S and
**{dino_input['prediction_agreement_with_clean']:.3f}** for DINOv2-B. Sixteen flips
per quantized classifier-weight matrix retain
**{conv_head['prediction_agreement_with_clean']:.3f}** and
**{dino_head['prediction_agreement_with_clean']:.3f}** agreement, respectively, on
the same cohort.

![Fault robustness](assets/polar_fault_robustness.png)

## Evaluation controls and evidence lineage

### POLAR

- Clean split: {train_rows:,} train, {validation_rows:,} validation, and
  {test_rows:,} test images.
- {polar_audit['quarantine_images']} images in
  {polar_audit['quarantine_components']} confirmed cross-split source-related
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

Checkpoints, image paths, dense probabilities, and large feature tensors remain
outside Git. The tracked evidence is path-sanitized and hash-indexed.

## Reports and release files

| Artifact | Purpose |
| --- | --- |
| [V-COCO v2 report](output/pdf/vcoco_v2_external_transfer_v2.0.0.pdf) | Current person-level transfer study |
| [V-COCO v2 evidence](results/vcoco_v2/README.md) | Locks, metrics, uncertainty, and mechanism tables |
| [POLAR v1 report](output/pdf/polar_public_report_v1.0.0.pdf) | Original four-class source benchmark |
| [Executed notebook](human_activity_classification.ipynb) | Compact, code-backed result walkthrough |
| [Result lineage](docs/RESULT_LINEAGE.md) | Source benchmark artifact map |
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
python -m pip install -e ".[notebook,dev]"
~~~

The staged commands and acceptance boundaries are documented in
[experiments/README.md](experiments/README.md). Validate a checkout with:

~~~bash
python -m ruff check .
python -m compileall -q src experiments tools
python -m pytest
python tools/validate_repository.py
python tools/build_study_release_manifest.py --check
~~~

## Citation, references, and license

GitHub's citation panel reads [CITATION.cff](CITATION.cff). The Zenodo DOI will be
added to the citation metadata after the v2 deposit.

> Huseyinli, A. (2026). *Improving Person-Level External Transfer on V-COCO*
> (Version 2.0.0) [Technical report].
> https://github.com/abdullahuseyinli-dot/human-activity-classification

- [POLAR dataset](https://doi.org/10.17632/hvnsh7rwz7.1)
- [V-COCO](https://arxiv.org/abs/1505.04474)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [ConvNeXt](https://arxiv.org/abs/2201.03545)
- [SigLIP2](https://arxiv.org/abs/2502.14786)
- [AugMix](https://arxiv.org/abs/1912.02781)
- [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292)

Original code and documentation are MIT licensed. Dataset images, annotations, and
pretrained weights retain their upstream terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
"""


def main() -> None:
    readme = build_readme(ROOT)
    (ROOT / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    print("Wrote README.md from locked POLAR and V-COCO evidence")


if __name__ == "__main__":
    main()
