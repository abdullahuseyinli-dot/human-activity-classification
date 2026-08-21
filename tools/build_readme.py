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

## Engineering corrections

The evidence pipeline resolves a mismatch between selected freeze depth and the
later DINOv2 training branch, removes stale result narratives, applies configured
dropout consistently, and rebuilds every promoted result within one evidence
lineage. It also prevents a subtler training mismatch by replaying the median
cross-validation learning-rate schedule during full-pool retraining.

The executed source pipeline is fingerprinted. Exact source fingerprints and the
lineage controls are documented in [the result-lineage record](docs/RESULT_LINEAGE.md).

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

## Technical references

- [ConvNeXt: A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [Dropout](https://www.jmlr.org/papers/v15/srivastava14a.html)
- [MixUp](https://arxiv.org/abs/1710.09412)
- [RandAugment](https://arxiv.org/abs/1909.13719)
- [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629)
"""

    (repository / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    print(f"Wrote README.md for {downstream_lock['champion_method']}")


if __name__ == "__main__":
    main()
