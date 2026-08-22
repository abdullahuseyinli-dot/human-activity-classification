"""Generate the repository README from locked, tracked experiment evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    return parser.parse_args()


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    results = repository / "results"
    with (results / "selection_lock.json").open(encoding="utf-8") as handle:
        model_lock = json.load(handle)
    with (results / "downstream_selection_lock.json").open(encoding="utf-8") as handle:
        downstream_lock = json.load(handle)
    test_metrics = pd.read_csv(results / "locked_test_metrics.csv")
    champion = test_metrics.loc[test_metrics["selected_champion"].astype(bool)].iloc[0]
    intervals = pd.read_csv(results / "test_bootstrap_intervals.csv")
    champion_interval = intervals[
        (intervals["method"] == champion["method"]) & (intervals["metric"] == "macro_f1")
    ].iloc[0]
    parameters = pd.read_csv(results / "model_parameter_summary.csv").set_index("model_kind")
    faithfulness_selection = pd.read_csv(
        results / "faithfulness_method_selection.csv"
    )
    faithfulness_test = pd.read_csv(results / "faithfulness_test_summary.csv")

    result_rows = []
    for row in test_metrics.itertuples(index=False):
        result_rows.append(
            [
                str(row.display_name),
                f"{row.accuracy:.3f}",
                f"{row.macro_f1:.3f}",
                f"{row.log_loss:.3f}",
                "**OOF champion**" if bool(row.selected_champion) else "Comparator",
            ]
        )

    config_rows = []
    for family, display_name in (
        ("convnext_small", "ConvNeXt-Small"),
        ("dinov2_small", "DINOv2-Small"),
    ):
        record = model_lock["selected"][family]
        config = record["config"]
        parameter_row = parameters.loc[family]
        strategy = str(config["unfreeze_strategy"])
        if config.get("top_n_blocks"):
            strategy += f" ({int(config['top_n_blocks'])} blocks)"
        config_rows.append(
            [
                display_name,
                str(record["candidate_id"]),
                strategy,
                f"{float(config['dropout']):.2f}",
                str(config["augmentation_strength"]),
                f"{float(parameter_row.trainable_percent):.1f}%",
            ]
        )

    results_table = markdown_table(
        ["Locked method", "Accuracy", "Macro-F1", "Log-loss", "Role"], result_rows
    )
    config_table = markdown_table(
        ["Family", "Candidate", "Adaptation", "Dropout", "Augmentation", "Trainable"],
        config_rows,
    )

    explanation_names = {
        "hirescam": "HiResCAM",
        "gradient_attention_rollout": "Gradient-attention rollout",
        "weighted_hirescam+gradient_attention_rollout": (
            "0.1 HiResCAM + 0.9 gradient-attention rollout"
        ),
    }
    faithfulness_rows = []
    for row in faithfulness_test.itertuples(index=False):
        faithfulness_rows.append(
            [
                str(row.display_name),
                explanation_names.get(str(row.method), str(row.method)),
                (
                    f"{row.road_combined_mean:.3f} "
                    f"[{row.road_combined_ci_2_5:.3f}, {row.road_combined_ci_97_5:.3f}]"
                ),
                f"{row.deletion_auc_mean:.3f}",
                f"{row.insertion_auc_mean:.3f}",
                f"{row.selectivity_gap_mean:.3f}",
                f"{row.faithfulness_spearman_mean:.3f}",
            ]
        )
    faithfulness_table = markdown_table(
        [
            "Locked model",
            "Explanation",
            "ROADCombined ↑ (95% CI)",
            "Deletion AUC ↓",
            "Insertion AUC ↑",
            "Random gap ↑",
            "Subset ρ ↑",
        ],
        faithfulness_rows,
    )
    selected_explanations = faithfulness_selection[
        faithfulness_selection["selected"].astype(bool)
    ].set_index("family")
    conv_oof_road = float(selected_explanations.loc["convnext_small", "road_combined_mean"])
    dino_oof_road = float(selected_explanations.loc["dinov2_small", "road_combined_mean"])

    readme = f"""# Human Activity Classification with ConvNeXt and DINOv2

A leakage-safe small-data transfer-learning benchmark for classifying **sitting**,
**standing**, and **walking/running** from still images. The project compares an
ImageNet-pretrained ConvNeXt-Small with DINOv2-Small features, then evaluates
OOF-locked seed ensembles, probability blending, and SVM representation probes.

![Final method comparison](assets/final_method_comparison.png)

The selected method—**{champion["display_name"]}**—reached
**{champion["macro_f1"]:.3f} macro-F1** and **{champion["accuracy"]:.3f} accuracy**
on the fixed 43-image test split. Its stratified-bootstrap macro-F1 interval is
**[{champion_interval["ci_2_5"]:.3f}, {champion_interval["ci_97_5"]:.3f}]**. The
interval matters: this is a carefully controlled small benchmark, not a claim of
deployment-level certainty.

## Reproducibility improvements

The released pipeline keeps model selection, final training, and promoted results
within one evidence lineage. Freeze depth and dropout propagate through every
training branch, while full-pool retraining replays the median cross-validation
learning-rate schedule instead of silently changing optimization behaviour.

Exact source fingerprints, selection locks, and evidence-handling rules are
documented in [the result lineage](docs/RESULT_LINEAGE.md).

## Results

{results_table}

The champion is selected by pooled OOF macro-F1 before downstream test arrays are
read. Test scores for the remaining locked comparators are reported for context,
not used for tuning.

## Locked model configurations

{config_table}

![ConvNeXt architecture](assets/convnext_architecture.png)

![DINOv2 architecture](assets/dinov2_architecture.png)

The experiment compares dropout, MixUp, light RandAugment, random-erasing removal,
label smoothing, weight decay, and multiple freeze depths as controlled OOF
interventions. A regularizer is retained only when the dataset supports it.

## Attribution faithfulness

![Faithfulness perturbation curves](assets/faithfulness_perturbation_curves.png)

{faithfulness_table}

Attribution methods are selected without reading test explanations. A fixed
36-image, class-balanced OOF audit selected HiResCAM for ConvNeXt
(ROADCombined **{conv_oof_road:.3f}**) and class-specific gradient-attention
rollout for DINOv2 (**{dino_oof_road:.3f}**). ConvNeXt's Grad-CAM and HiResCAM
scores were effectively tied; the machine-readable ordering is retained rather
than presenting the difference as a substantive gain.

The audit perturbs a common 16 by 16 patch grid using ROAD imputation,
blur-baseline deletion/insertion, and matched random removal. It also checks
parameter randomization, target-class sensitivity, horizontal-flip
equivariance, and agreement across the three final seeds. Raw DINOv2 attention
rollout remains an ineligible class-agnostic negative control.

## Evaluation design

- **Data:** 285 checksum-verified COCO images; 242 development and 43 fixed test.
- **Primary metric:** pooled OOF macro-F1; lower fold variability and log-loss are
  deterministic tie-breakers.
- **Selection:** three-fold coarse screen followed by five-fold confirmation.
- **Final training:** seeds 42, 52, and 62 with fixed folds and fold-derived epoch/LR
  schedules; all full-pool models finish before the test gate opens.
- **Calibration:** OOF temperature scaling transferred unchanged to final models.
- **Inference:** center crop versus horizontal-flip TTA selected from OOF evidence.
- **Downstream:** seed averaging, blend weight, and SVM parameters selected OOF-only.
- **Uncertainty:** 2,000 stratified bootstrap resamples and paired champion deltas.
- **Explanations:** OOF method selection followed by locked-test perturbation,
  parameter-randomization, specificity, and stability checks.

The complete contract is in [the experiment protocol](docs/EXPERIMENT_PROTOCOL.md).

## Repository layout

```text
.
├── human_activity_classification.ipynb  # compact, executed portfolio narrative
├── src/hac/                             # reusable data, model, metric, and training code
├── experiments/                         # staged selection/final-analysis runners
├── data/manifest.csv                    # URLs, fixed splits, labels, and checksums
├── results/                             # compact locked evidence; no checkpoints
├── assets/                              # tracked architecture and result figures
├── docs/                                # protocol and result lineage
├── tests/                               # fast protocol/config/metric tests
└── tools/                               # download, export, and notebook build utilities
```

## Quick start

Python 3.11 is the recorded environment.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,dev]"
python tools/download_dataset.py --manifest data/manifest.csv
jupyter lab human_activity_classification.ipynb
```

Install the PyTorch wheel appropriate for the local CUDA or CPU platform when it
differs from the default package index. The exact recorded versions are listed in
`requirements-lock.txt`.

## Reproduce the staged experiment

The code-only pipeline notebook is tracked separately from the concise portfolio
notebook so the latter remains readable on GitHub.

```bash
python experiments/recover_experiment.py \\
  --source-notebook experiments/pipeline_source.ipynb \\
  --manifest data/manifest.csv \\
  --artifact-root .runs/selection \\
  --stage coarse
```

Promote only the desired candidates to `--stage confirm`, then create the
configuration lock:

```bash
python experiments/select_candidates.py \\
  --selection-root .runs/selection \\
  --output .runs/selection_lock.json
```

Final multi-seed retraining and downstream analysis are deliberately separate:

```bash
python experiments/finalize_experiment.py \\
  --source-notebook experiments/pipeline_source.ipynb \\
  --manifest data/manifest.csv \\
  --artifact-root .runs/final \\
  --selection-lock .runs/selection_lock.json

python experiments/analyze_final.py \\
  --artifact-root .runs/final \\
  --output-dir .runs/final_analysis

python experiments/evaluate_faithfulness.py \\
  --manifest data/manifest.csv \\
  --final-root .runs/final \\
  --analysis-dir .runs/final_analysis \\
  --output-dir .runs/faithfulness
```

Bulky checkpoints, logits, local paths, and interrupted runs remain under
`.runs/` and are ignored by Git. The tracked `results/` directory contains the
path-sanitized evidence needed to audit the notebook.

## Quality gates

```bash
python -m ruff check .
python -m compileall -q src experiments tools
python -m pytest
python tools/validate_repository.py
```

The same checks run in GitHub Actions. The portable manifest is validated for
row counts, label vocabulary, unique IDs and hashes, the fixed test contract,
and cross-boundary perceptual near-duplicates.

## Limitations

- The test split contains 43 images, so point estimates have substantial
  uncertainty even with careful locking.
- Subject and capture-session identifiers are unavailable; subject-independent
  generalization cannot be claimed.
- Images come from COCO and may reward scene context as well as body pose.
- The SVM result is a representation probe, not an end-to-end deployment stack.
- Attribution metrics measure sensitivity under declared patch perturbations;
  they do not prove causal or human-like reasoning.
- Parameter-randomization, target-specificity, and flip-stability checks use a
  deterministic nine-image class-balanced audit cohort.

## Technical references

- [ConvNeXt: A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [Dropout](https://www.jmlr.org/papers/v15/srivastava14a.html)
- [MixUp](https://arxiv.org/abs/1710.09412)
- [RandAugment](https://arxiv.org/abs/1909.13719)
- [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629)
- [ROAD: Remove and Debias](https://proceedings.mlr.press/v162/rong22a.html)
- [Pixel-flipping Evaluation of Neural Explanations](https://arxiv.org/abs/1509.06321)
- [Transformer Interpretability Beyond Attention Visualization](https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html)
- [Sanity Checks for Saliency Maps](https://papers.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html)

## License

Original source code and documentation are licensed under the [MIT License](LICENSE).
COCO source images and pretrained model components retain their upstream terms;
see [the third-party notices](THIRD_PARTY_NOTICES.md).
"""

    (repository / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    print(f"Wrote README.md for {downstream_lock['champion_method']}")


if __name__ == "__main__":
    main()
