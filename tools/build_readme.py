"""Generate the repository README from locked POLAR evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DISPLAY_NAMES = {
    "locked_ensemble": "Locked probability ensemble",
    "dinov2_base_multilayer_rbf": "DINOv2-B multilayer + calibrated RBF SVM",
    "dinov2_base_multilayer_logistic": "DINOv2-B multilayer + logistic regression",
    "dinov2_base_top4": "DINOv2-B, top four blocks adapted",
    "dinov2_small_moderate": "DINOv2-S, full adaptation",
    "convnext_small_full": "ConvNeXt-S, full adaptation",
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
    test = read_json(results / "polar_test_summary.json")
    external = read_json(results / "polar_external_summary.json")
    faithfulness = read_json(results / "polar_faithfulness_summary.json")
    fault = read_json(results / "polar_fault_summary.json")
    evidence = read_json(results / "polar_final_evidence_manifest.json")
    selection = read_json(results / "polar_final_selection_lock.json")
    audit = read_json(results / "polar_data_audit.json")
    uncertainty = read_json(results / "polar_test_uncertainty.json")
    fit_manifest = read_json(results / "polar_final_fit_manifest.json")
    extension = read_json(results / "polar_extension_summary.json")

    required_statuses = {
        test.get("status"): "LOCKED_FINAL_TEST_COMPLETE",
        external.get("status"): "LOCKED_EXTERNAL_EVALUATION_COMPLETE",
        faithfulness.get("status"): "LOCKED_POLAR_FAITHFULNESS_COMPLETE",
        fault.get("status"): "LOCKED_POLAR_FAULT_ROBUSTNESS_COMPLETE",
        evidence.get("status"): "LOCKED_POLAR_PORTFOLIO_EVIDENCE",
    }
    if any(actual != expected for actual, expected in required_statuses.items()):
        raise RuntimeError("README generation requires complete locked evidence")
    lock_hashes = {
        item["selection_lock_sha256"] for item in (test, external, faithfulness, fault, evidence)
    }
    lock_hash = evidence["selection_lock_sha256"]
    if lock_hashes != {lock_hash}:
        raise RuntimeError("README evidence does not share one final selection lock")
    if test.get("test_used_for_selection") or external.get("test_used_for_selection"):
        raise RuntimeError("Test-selected evidence cannot be promoted")

    metrics = pd.read_csv(results / "polar_test_metrics.csv")
    result_rows = []
    for row in metrics.itertuples(index=False):
        interval = uncertainty[row.candidate]
        result_rows.append(
            [
                DISPLAY_NAMES.get(row.candidate, row.candidate),
                f"{row.macro_f1:.3f}",
                f"[{interval['ci_95_low']:.3f}, {interval['ci_95_high']:.3f}]",
                f"{row.accuracy:.3f}",
                f"{row.log_loss:.3f}",
                f"{row.ece:.3f}",
            ]
        )
    results_table = markdown_table(
        ["Predeclared candidate", "Macro-F1", "95% CI", "Accuracy", "Log loss", "ECE"],
        result_rows,
    )

    per_class = pd.read_csv(results / "polar_test_per_class.csv")
    per_class = per_class[per_class["candidate"].eq("locked_ensemble")]
    per_class_table = markdown_table(
        ["Class", "Precision", "Recall", "F1", "Support"],
        [
            [
                str(row["class"]).replace("_", " ").title(),
                f"{row['precision']:.3f}",
                f"{row['recall']:.3f}",
                f"{row['f1']:.3f}",
                f"{int(row['support']):,}",
            ]
            for _, row in per_class.iterrows()
        ],
    )

    external_metrics = pd.read_csv(results / "polar_external_image_metrics.csv")
    external_metrics = external_metrics[
        external_metrics["candidate"].isin(
            ["locked_ensemble_collapsed", "dinov2_base_top4", "dinov2_base_multilayer_rbf"]
        )
    ]
    external_names = {
        "locked_ensemble_collapsed": "Locked ensemble (collapsed to three classes)",
        "dinov2_base_top4": "DINOv2-B, top four blocks adapted",
        "dinov2_base_multilayer_rbf": "DINOv2-B multilayer + calibrated RBF SVM",
    }
    external_table = markdown_table(
        ["External candidate", "Macro-F1", "Accuracy", "Log loss", "ECE"],
        [
            [
                external_names[row.candidate],
                f"{row.macro_f1:.3f}",
                f"{row.accuracy:.3f}",
                f"{row.log_loss:.3f}",
                f"{row.ece:.3f}",
            ]
            for row in external_metrics.itertuples(index=False)
        ],
    )

    primary = test["primary_metrics"]
    primary_ci = uncertainty["locked_ensemble"]
    paired = uncertainty["locked_ensemble_paired_deltas"]
    smallest_delta = min(paired.items(), key=lambda item: item[1]["point_estimate"])
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

    weights = selection["ensemble"]["weights"]
    weight_text = ", ".join(
        f"{DISPLAY_NAMES.get(name, name)} {value:.0%}" for name, value in weights.items()
    )
    clean_counts = audit["clean_target_counts"]
    train_rows = sum(clean_counts["train"].values())
    validation_rows = sum(clean_counts["val"].values())
    test_rows = sum(clean_counts["test"].values())
    logistic = fit_manifest["probes"]["dinov2_base_multilayer_logistic"]
    rbf = fit_manifest["probes"]["dinov2_base_multilayer_rbf"]

    return f"""# Leakage-Safe Human Activity Classification

[![Quality gates](https://github.com/abdullahuseyinli-dot/human-activity-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/abdullahuseyinli-dot/human-activity-classification/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/Code-MIT-0F766E.svg)](LICENSE)

A locked, leakage-audited transfer-learning study for recognizing **sitting, standing,
walking, and running** in still images. The primary benchmark uses the POLAR dataset,
compares ConvNeXt and DINOv2 adaptation strategies, tests linear and nonlinear
classifiers on frozen representations, and evaluates external transfer and attribution
faithfulness without using test results for selection.

![Held-out POLAR comparison](assets/polar_test_comparison.png)

The predeclared ensemble achieved **{primary['macro_f1']:.3f} macro-F1**
(95% stratified-bootstrap CI **[{primary_ci['ci_95_low']:.3f},
{primary_ci['ci_95_high']:.3f}]**) and **{primary['accuracy']:.3f} accuracy** on
{test_rows:,} held-out POLAR images. Its smallest paired gain over any component was
**+{smallest_delta[1]['point_estimate']:.3f} macro-F1**, with a positive 95% interval
**[{smallest_delta[1]['ci_95_low']:.3f}, {smallest_delta[1]['ci_95_high']:.3f}]**.

> This is a reproducible benchmark result, not a state-of-the-art claim. No published
> result was found with the same cleaned four-class subset, quarantine policy, fixed
> split, and metric.

## Held-out results

{results_table}

The secondary three-class mapping (walking and running combined) reached
**{secondary.macro_f1:.3f} macro-F1** and **{secondary.accuracy:.3f} accuracy**.
Walking remains the hardest primary class.

{per_class_table}

![Locked confusion matrix](assets/polar_confusion_matrix.png)

## What changed the result

The study separates data scale, representation, adaptation depth, regularization, and
model diversity instead of treating training as a single opaque run.

- **Data scale mattered:** the frozen DINOv2-B validation curve rose from
  **{scale[0]['macro_f1_mean']:.3f}** at {scale[0]['actual_train_size']:,} training images
  to **{scale[-1]['macro_f1_mean']:.3f}** at {scale[-1]['actual_train_size']:,}.
- **Person-aware views mattered:** the strongest DINO branches use deterministic person
  crops with declared context, while ConvNeXt retained the full frame.
- **Moderate regularization won selectively:** the locked neural configurations use
  dropout 0.10, no MixUp, no label smoothing, and either mild or moderate augmentation;
  interventions that reduced validation performance remain in the evidence tables.
- **Complementarity mattered:** the final weights were fixed on development data as
  {weight_text}. Every paired held-out interval favors the locked blend.

![Data-scale curve](assets/polar_scale_curve.png)

## Was an SVM useful at the final stage?

Yes—as a representation probe, not as the default deployment choice. A calibrated RBF
SVM on 7,680-dimensional DINOv2-B multilayer features reached **0.927 macro-F1**, the
strongest standalone held-out component. The standardized multinomial logistic model
reached **0.926**, but had better log loss (**0.176** versus **0.228**), fitted in
{logistic['fit_seconds']:.1f} seconds, and occupied {logistic['pipeline_bytes'] / 1_000_000:.1f}
MB. The RBF pipeline took {rbf['fit_seconds'] / 60:.1f} minutes and occupied
{rbf['pipeline_bytes'] / 1_000_000:.1f} MB. The nonlinear margin adds a small accuracy
gain, while logistic regression is the more practical calibrated endpoint.

## External transfer: the result does not travel unchanged

The locked models were evaluated without retuning on a clean V-COCO train/validation
subset. An exact/perceptual overlap audit compared 16,614 clean POLAR records with
4,123 V-COCO images and found **zero confirmed source-related pairs**. Image-level
evaluation uses {external['image_level_rows']:,} unambiguous images; person-level
evaluation uses {external['person_rows']:,} annotations.

{external_table}

![External V-COCO transfer](assets/polar_external_validation.png)

The locked ensemble falls from **{secondary.macro_f1:.3f}** in-domain three-class
macro-F1 to **{external['primary_image_metrics']['macro_f1']:.3f}** externally. The
adapted DINOv2-B component transfers best descriptively at
**{external['best_observed_image_metrics']['macro_f1']:.3f}**. This is evidence of a
substantial domain and annotation-policy gap, not evidence that the external set should
be used to retune the locked result.

## Faithfulness and fault robustness

The attribution audit uses a deterministic 256-image cohort balanced by class and
person-box-area quartile. ConvNeXt Grad-CAM has a targeted-versus-random deletion gap
of **{conv_faith['deletion_selectivity_gap']['mean']:.3f}**, concentrates
**{conv_faith['person_attribution_mass_lift']['mean']:.2f}x** more attribution in the
person box than uniform area, and produces a person-minus-context probability drop of
**{conv_faith['person_minus_context_occlusion_drop']['mean']:.3f}**. DINOv2-B integrated
gradients localizes on people but shows only **{dino_faith['person_attribution_mass_lift']['mean']:.2f}x**
area-normalized lift and retains high correlations after target and parameter
randomization. It is therefore presented as a limited localization diagnostic, not a
fully validated causal explanation.

![BBox-aware faithfulness](assets/polar_faithfulness.png)

![Attribution sanity checks](assets/polar_attribution_sanity.png)

Bit-flip experiments are reported separately from faithfulness. At a 0.1% exact input
bit-flip rate, prediction agreement with the clean models was
**{conv_input['prediction_agreement_with_clean']:.3f}** for ConvNeXt-S and
**{dino_input['prediction_agreement_with_clean']:.3f}** for DINOv2-B. Sixteen flips per
quantized classifier weight matrix retained **{conv_head['prediction_agreement_with_clean']:.3f}**
and **{dino_head['prediction_agreement_with_clean']:.3f}** agreement, respectively, on
this cohort. These are bounded software fault-injection results, not hardware safety
certification.

![Fault robustness](assets/polar_fault_robustness.png)

## Leakage controls and evidence lineage

- POLAR clean split: {train_rows:,} train, {validation_rows:,} validation, and
  {test_rows:,} test images.
- {audit['quarantine_images']} images in {audit['quarantine_components']} confirmed
  cross-split source-related components were quarantined before supervised fitting.
- All candidate selection, blend weights, epochs, and classifier hyperparameters were
  locked on development evidence before the test cache opened.
- Nine neural fits and three frozen-feature probes completed and were hash-verified
  before the single test evaluation.
- The test access gate records one official open, and every exported summary shares
  selection-lock SHA-256 `{lock_hash}`.
- Checkpoints, local image paths, dense probabilities, and full-resolution attribution
  maps remain outside Git; the tracked evidence is path-sanitized and hash-indexed.

Start with the [rendered technical report](output/pdf/polar_technical_report.pdf),
[source report](docs/POLAR_TECHNICAL_REPORT.md),
[portfolio article](docs/PORTFOLIO_ARTICLE.md), and
[result lineage](docs/RESULT_LINEAGE.md). The older 285-image COCO study is retained as
a [historical benchmark](docs/LEGACY_COCO_STUDY.md), not the portfolio headline.

## Repository layout

```text
.
├── human_activity_classification.ipynb  # executed evidence narrative
├── src/hac/                             # reusable data, model, metric, and audit code
├── experiments/                         # staged selection, fitting, and evaluation runners
├── tools/                               # dataset, export, validation, and figure utilities
├── results/                             # portable locks, metrics, uncertainty, and hashes
├── assets/                              # publication figures
├── docs/                                # protocols, report, article, and lineage
├── tests/                               # fast invariants and evidence-contract tests
└── .github/workflows/ci.yml             # Linux quality gates
```

## Reproduce the environment

Python 3.11 is the recorded runtime. Install the PyTorch build appropriate for the
machine, then install the project:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,dev]"
```

Obtain POLAR v1 from its publisher, verify the archive hashes in
[`experiments/polar_study_protocol.json`](experiments/polar_study_protocol.json), and
build the audited local manifest:

```bash
python tools/prepare_polar.py \\
  --annotations-dir /path/to/Annotations \\
  --images-dir /path/to/JPEGImages \\
  --image-sets-dir /path/to/ImageSets \\
  --output-dir .runs/polar_data \\
  --legacy-manifest data/manifest.csv
```

The long-running stages are documented in [experiments/README.md](experiments/README.md).
Validate a checkout with:

```bash
python -m ruff check .
python -m compileall -q src experiments tools
python -m pytest
python tools/validate_repository.py
```

## Scope and limitations

- Subject and capture-session identifiers are unavailable, so subject-independent
  generalization cannot be claimed.
- The four-class task is a cleaned subset of POLAR, not the full nine-label benchmark.
- V-COCO has a different source and annotation policy; its results diagnose transfer
  rather than rank models for the POLAR task.
- The RBF result is a large frozen-representation probe, not an end-to-end serving
  design.
- Attribution localization, perturbation, and randomization tests do not prove human-like
  or causal reasoning.

## References and license

- [POLAR dataset](https://doi.org/10.17632/hvnsh7rwz7.1)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [ConvNeXt](https://arxiv.org/abs/2201.03545)
- [V-COCO](https://arxiv.org/abs/1505.04474)
- [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292)

Original code and documentation are MIT licensed. Dataset images, annotations, and
pretrained weights retain their upstream terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
"""


def main() -> None:
    readme = build_readme(ROOT)
    (ROOT / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    print("Wrote README.md from locked POLAR evidence")


if __name__ == "__main__":
    main()
