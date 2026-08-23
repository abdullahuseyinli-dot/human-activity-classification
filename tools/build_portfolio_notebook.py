"""Build and execute the compact POLAR study notebook from tracked evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "results/polar_data_audit.json",
    "results/polar_final_selection_lock.json",
    "results/polar_final_fit_manifest.json",
    "results/polar_test_access_gate.json",
    "results/polar_test_metrics.csv",
    "results/polar_test_uncertainty.json",
    "results/polar_test_per_class.csv",
    "results/polar_test_secondary_metrics.csv",
    "results/polar_external_image_metrics.csv",
    "results/polar_external_summary.json",
    "results/polar_faithfulness_summary.json",
    "results/polar_fault_summary.json",
    "results/polar_extension_summary.json",
    "assets/polar_test_comparison.png",
    "assets/polar_confusion_matrix.png",
    "assets/polar_scale_curve.png",
    "assets/polar_external_validation.png",
    "assets/polar_faithfulness.png",
    "assets/polar_attribution_sanity.png",
    "assets/polar_fault_robustness.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "human_activity_classification.ipynb",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown(value: str):
    return nbformat.v4.new_markdown_cell(value.strip())


def code(value: str):
    return nbformat.v4.new_code_cell(value.strip())


def build_notebook(repository: Path) -> dict:
    missing = [name for name in REQUIRED if not (repository / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing notebook evidence: {missing}")

    test = read_json(repository / "results" / "polar_test_summary.json")
    uncertainty = read_json(repository / "results" / "polar_test_uncertainty.json")
    external = read_json(repository / "results" / "polar_external_summary.json")
    primary = test["primary_metrics"]
    interval = uncertainty["locked_ensemble"]

    notebook = nbformat.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    notebook["cells"] = [
        markdown(
            f"""
# Source-Overlap-Controlled Human Activity Classification

## Locked POLAR benchmark

This notebook is the compact, executable evidence narrative for a four-class still-image
posture study. Model selection used the clean POLAR development split; nine neural fits
and three frozen-feature probes completed before the official test cache opened once.

**Locked result:** {primary['macro_f1']:.3f} macro-F1 (95% CI
[{interval['ci_95_low']:.3f}, {interval['ci_95_high']:.3f}]) and
{primary['accuracy']:.3f} accuracy on {test['test_rows_read']:,} held-out images.

The notebook reads only tracked, path-sanitized evidence. It performs no training and
makes no post-test selection decisions.
"""
        ),
        code(
            """
import json
from pathlib import Path

import pandas as pd
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / "results" / "polar_test_summary.json").is_file():
    raise RuntimeError("Run this notebook from the repository root.")

pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda value: f"{value:.4f}")

def load_json(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
"""
        ),
        markdown(
            """
## 1. Evidence boundary

The audit removes confirmed cross-split source relatives before supervised fitting. The
selection lock then fixes every model, seed, epoch count, classifier setting, blend
weight, metric, and bootstrap rule before test access.
"""
        ),
        code(
            """
audit = load_json("polar_data_audit.json")
gate = load_json("polar_test_access_gate.json")
fits = load_json("polar_final_fit_manifest.json")

controls = pd.DataFrame(
    [
        ("Clean development images", fits["development_rows"]),
        ("Clean held-out test images", sum(audit["clean_target_counts"]["test"].values())),
        ("Quarantined source-related images", audit["quarantine_images"]),
        ("Verified neural final fits", sum(len(item["seeds"]) for item in fits["neural"].values())),
        ("Verified final probes", len(fits["probes"])),
        ("Official test-manifest opens", gate["official_test_manifest_open_count"]),
        ("Test rows used for selection", 0),
    ],
    columns=["Control", "Recorded value"],
)
display(controls)
"""
        ),
        markdown(
            """
## 2. Held-out POLAR result

The primary metric is macro-F1. Confidence intervals use 10,000 class-stratified
bootstrap resamples. The ensemble and all component rows were predeclared; the table is
not a post-test leaderboard.

![Predeclared held-out candidates](assets/polar_test_comparison.png)
"""
        ),
        code(
            """
names = {
    "locked_ensemble": "Locked ensemble",
    "dinov2_base_multilayer_rbf": "DINOv2-B + RBF SVM",
    "dinov2_base_multilayer_logistic": "DINOv2-B + logistic",
    "dinov2_base_top4": "DINOv2-B top 4",
    "dinov2_small_moderate": "DINOv2-S full",
    "convnext_small_full": "ConvNeXt-S full",
}
metrics = pd.read_csv(ROOT / "results" / "polar_test_metrics.csv")
intervals = load_json("polar_test_uncertainty.json")
metrics["macro_f1_ci"] = [
    f"[{intervals[key]['ci_95_low']:.3f}, {intervals[key]['ci_95_high']:.3f}]"
    for key in metrics["candidate"]
]
metrics["candidate"] = metrics["candidate"].map(names)
display(metrics[["candidate", "macro_f1", "macro_f1_ci", "accuracy", "log_loss", "ece"]])
"""
        ),
        markdown("![Locked ensemble confusion matrix](assets/polar_confusion_matrix.png)"),
        code(
            """
per_class = pd.read_csv(ROOT / "results" / "polar_test_per_class.csv")
per_class = per_class[per_class["candidate"].eq("locked_ensemble")]
display(per_class[["class", "precision", "recall", "f1", "support"]].reset_index(drop=True))

secondary = pd.read_csv(ROOT / "results" / "polar_test_secondary_metrics.csv")
display(secondary[["candidate", "macro_f1", "accuracy", "log_loss", "ece"]])
"""
        ),
        markdown(
            """
## 3. What improved performance

The frozen DINOv2-B learning curve isolates the effect of training-set size. Adaptation
and regularization screens then test whether additional complexity earns its place.
Dropout and image augmentation were retained; MixUp, label smoothing, inverse-frequency
weights, and removing random erasing did not improve the relevant seed-42 baseline.

![Frozen DINOv2-B learning curve](assets/polar_scale_curve.png)
"""
        ),
        code(
            """
scale = pd.DataFrame(load_json("polar_extension_summary.json")["scale_curve"])
display(scale[["actual_train_size", "macro_f1_mean", "accuracy_mean", "log_loss_mean"]])
"""
        ),
        markdown(
            """
## 4. Linear versus nonlinear final-stage classifiers

The calibrated RBF SVM is the strongest standalone held-out component, but its gain over
logistic regression is small. The RBF artifact is 870.9 MB and took 60.4 minutes to fit;
the logistic artifact is 0.4 MB, fitted in 13.9 seconds, and has better log loss and ECE.
The SVM is useful as a research probe and ensemble component; logistic regression is the
more practical calibrated endpoint.
"""
        ),
        code(
            """
probe_rows = metrics[metrics["candidate"].isin(["DINOv2-B + RBF SVM", "DINOv2-B + logistic"])].copy()
probe_rows["fit_seconds"] = [3625.2294, 13.9214]
probe_rows["artifact_mb"] = [870.8566, 0.4136]
display(probe_rows[["candidate", "macro_f1", "accuracy", "log_loss", "ece", "fit_seconds", "artifact_mb"]])
"""
        ),
        markdown(
            f"""
## 5. External transfer

The locked three-class ensemble reaches 0.961 in-domain macro-F1 but
{external['primary_image_metrics']['macro_f1']:.3f} on
{external['image_level_rows']:,} unambiguous V-COCO images. DINOv2-B top-four adaptation
transfers best descriptively at
{external['best_observed_image_metrics']['macro_f1']:.3f}. The models were not retuned
after this evaluation.

![V-COCO external validation](assets/polar_external_validation.png)
"""
        ),
        code(
            """
external_metrics = pd.read_csv(ROOT / "results" / "polar_external_image_metrics.csv")
display(external_metrics[["candidate", "macro_f1", "accuracy", "log_loss", "ece"]])
"""
        ),
        markdown(
            """
## 6. Attribution: localization is not enough

The fixed 256-image audit combines deletion/insertion, person-box localization,
equal-area person/context occlusion, target sensitivity, and parameter randomization.
ConvNeXt Grad-CAM has stronger causal and sanity evidence. DINOv2-B integrated gradients
localize on people but remain highly correlated after changing the target and resetting
learned layers, so they are not promoted as faithful causal explanations.

![BBox-aware attribution audit](assets/polar_faithfulness.png)

![Target and parameter randomization](assets/polar_attribution_sanity.png)
"""
        ),
        code(
            """
faith = load_json("polar_faithfulness_summary.json")["aggregate"]
rows = []
for family in ("convnext_small_full", "dinov2_base_top4"):
    rows.append(
        {
            "family": names[family],
            "deletion_selectivity_gap": faith[family]["deletion_selectivity_gap"]["mean"],
            "person_area_lift": faith[family]["person_attribution_mass_lift"]["mean"],
            "person_minus_context_drop": faith[family]["person_minus_context_occlusion_drop"]["mean"],
            "alternative_target_rho": faith[family]["target_vs_alternative_attribution_spearman"]["mean"],
            "randomized_cascade_rho": faith[family]["randomized_adapted_cascade_spearman"]["mean"],
        }
    )
display(pd.DataFrame(rows))
"""
        ),
        markdown(
            """
## 7. Bounded bit-flip robustness

Fault injection is reported separately from attribution faithfulness. Exact bit flips
are applied either to the uint8 input tensor or to an int8-quantized classifier weight
matrix. The result measures local prediction stability on the declared cohort; it is not
hardware certification.

![Fault robustness](assets/polar_fault_robustness.png)
"""
        ),
        code(
            """
fault = pd.DataFrame(load_json("polar_fault_summary.json")["aggregate_results"])
fault = fault[fault["fault_seed"].astype(str).isin(["none", "aggregate"])]
display(
    fault[[
        "family", "condition", "level", "macro_f1",
        "prediction_agreement_with_clean", "mean_absolute_probability_drift"
    ]].reset_index(drop=True)
)
"""
        ),
        markdown(
            """
## 8. Conclusions

- Data scale is the largest isolated performance amplifier in this study.
- DINOv2-B representations support strong linear and nonlinear final-stage classifiers.
- Development-locked model diversity produces a statistically supported ensemble gain.
- The external-domain gap is large; the POLAR score is not a deployment guarantee.
- ConvNeXt Grad-CAM passes the declared sanity checks more convincingly than DINOv2-B
  integrated gradients.

No exact state-of-the-art claim is made. The full method, discussion, limitations, and
references are documented in `docs/POLAR_PUBLIC_REPORT.md`.
"""
        ),
    ]
    return notebook


def main() -> None:
    args = parse_args()
    notebook = build_notebook(ROOT)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(nbformat.writes(notebook), encoding="utf-8", newline="\n")
    print(f"Wrote executed notebook: {args.output.resolve()}")


if __name__ == "__main__":
    main()
