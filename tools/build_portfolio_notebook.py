"""Build and execute the repository's compact evidence notebook."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    "results/vcoco_v2/official_test_metrics.csv",
    "results/vcoco_v2/official_test_per_class.csv",
    "results/vcoco_v2/official_test_summary.json",
    "results/vcoco_v2/official_test_uncertainty.json",
    "results/vcoco_v3/confirmation_metrics.csv",
    "results/vcoco_v3/confirmation_per_class.csv",
    "results/vcoco_v3/confirmation_summary.json",
    "results/vcoco_v3/confirmation_uncertainty.json",
    "results/vcoco_v3/protocol_lineage.json",
    "results/okutama_cptr/development_decision.json",
    "results/okutama_cptr/headline_metrics.csv",
    "results/okutama_cptr/subgroup_metrics.csv",
    "results/okutama_cptr/faithfulness_summary.json",
    "assets/polar_test_comparison.png",
    "assets/polar_confusion_matrix.png",
    "assets/polar_scale_curve.png",
    "assets/polar_external_validation.png",
    "assets/polar_faithfulness.png",
    "assets/polar_attribution_sanity.png",
    "assets/polar_fault_robustness.png",
    "assets/vcoco_v2_official_test_comparison.png",
    "assets/vcoco_v2_scale_gain.png",
    "assets/vcoco_v2_selective_prediction.png",
    "assets/vcoco_v3_confirmation_comparison.png",
    "assets/vcoco_v3_routing_curve.png",
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    vcoco = read_json(repository / "results" / "vcoco_v2" / "official_test_summary.json")
    vcoco_interval = read_json(
        repository / "results" / "vcoco_v2" / "official_test_uncertainty.json"
    )
    v3_metrics = {
        row["family"]: row
        for row in read_csv_rows(repository / "results/vcoco_v3/confirmation_metrics.csv")
    }
    v3_interval = read_json(repository / "results/vcoco_v3/confirmation_uncertainty.json")
    v3_summary = read_json(repository / "results/vcoco_v3/confirmation_summary.json")
    cptr = read_json(repository / "results/okutama_cptr/development_decision.json")
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
# Human Activity Classification Under Domain and Temporal Shift

## POLAR, V-COCO, and Okutama-Action evidence

This notebook is the compact, executable evidence narrative for the repository's
still-image, person-level transfer, and short-video studies. It reads only tracked,
path-sanitized evidence and performs no training or post-evaluation selection.

**Locked result:** {primary["macro_f1"]:.3f} macro-F1 (95% CI
[{interval["ci_95_low"]:.3f}, {interval["ci_95_high"]:.3f}]) and
{primary["accuracy"]:.3f} accuracy on {test["test_rows_read"]:,} held-out images.

The person-level V-COCO follow-up reaches
**{vcoco["primary_metrics"]["macro_f1"]:.4f} official-test macro-F1**, improving over
the historical source-only DINO baseline by
**{vcoco_interval["point_estimate"]:+.4f}**, with a 95% image-cluster interval of
[{vcoco_interval["ci_95_low"]:+.4f}, {vcoco_interval["ci_95_high"]:+.4f}].

On sealed Okutama confirmation data, the static target-trained model reaches
**{float(v3_metrics["static"]["macro_f1"]):.4f} macro-F1** and the temporal teacher
reaches **{float(v3_metrics["teacher"]["macro_f1"]):.4f}**. Routing half of the samples
to clips retains **{float(v3_metrics["hybrid_budget_0.5"]["macro_f1"]):.4f}**.
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
## 5. Person-level V-COCO study

The original no-retuning audit identified the external gap: the locked three-class
POLAR ensemble reached 0.961 in-domain macro-F1 and
{external["primary_image_metrics"]["macro_f1"]:.3f} on
{external["image_level_rows"]:,} unambiguous V-COCO images.

The follow-up keeps the official V-COCO memberships, trains on the target training
split, selects on validation, and opens the official test labels once after locking the
final stack. Two aspect-preserving DINOv2-B person views and five geometry features
raise person-level macro-F1 from
{vcoco["baseline_metrics"]["macro_f1"]:.4f} to
{vcoco["primary_metrics"]["macro_f1"]:.4f}.

![Official V-COCO test comparison](assets/vcoco_v2_official_test_comparison.png)

![Person-scale gain](assets/vcoco_v2_scale_gain.png)

![Selective prediction](assets/vcoco_v2_selective_prediction.png)
"""
        ),
        code(
            """
vcoco_metrics = pd.read_csv(ROOT / "results" / "vcoco_v2" / "official_test_metrics.csv")
vcoco_per_class = pd.read_csv(
    ROOT / "results" / "vcoco_v2" / "official_test_per_class.csv"
)
display(
    vcoco_metrics[
        ["method", "macro_f1", "accuracy", "balanced_accuracy", "log_loss", "ece"]
    ]
)
display(vcoco_per_class[["method", "class", "precision", "recall", "f1", "support"]])
"""
        ),
        markdown(
            f"""
## 6. Motion identifiability on Okutama-Action

The sealed confirmation compares the locked static model with an 8-frame, 0.5-second
temporal teacher over {v3_summary["samples"]:,} person instances.
The temporal model changes macro-F1 by
**{v3_interval["teacher"]["macro_f1"]["point_estimate"]:+.4f}** with a 95% paired
scenario-cluster interval of
[{v3_interval["teacher"]["macro_f1"]["ci_95_low"]:+.4f},
{v3_interval["teacher"]["macro_f1"]["ci_95_high"]:+.4f}]. A fixed 50% routing budget
reaches **{float(v3_metrics["hybrid_budget_0.5"]["macro_f1"]):.4f} macro-F1**.

![Static, temporal, and routed confirmation](assets/vcoco_v3_confirmation_comparison.png)

![Fixed-budget temporal routing](assets/vcoco_v3_routing_curve.png)
"""
        ),
        code(
            """
v3_metrics = pd.read_csv(ROOT / "results/vcoco_v3/confirmation_metrics.csv")
selected_v3 = v3_metrics[
    v3_metrics["family"].isin(
        ["source_only_static", "static", "teacher", "hybrid_budget_0.5"]
    )
]
display(
    selected_v3[
        ["family", "macro_f1", "accuracy", "log_loss", "ece"]
    ].reset_index(drop=True)
)

v3_per_class = pd.read_csv(ROOT / "results/vcoco_v3/confirmation_per_class.csv")
display(
    v3_per_class[v3_per_class["family"].isin(["static", "teacher", "hybrid_budget_0.5"])]
    [["family", "class", "precision", "recall", "f1", "support"]]
    .reset_index(drop=True)
)
"""
        ),
        markdown(
            f"""
## 7. CPTR architecture development

The follow-up architecture keeps the static and temporal anchors frozen and evaluates
center-conditioned residuals, camera-compensated trajectories, and confidence-masked
body-region tokens. The center-plus-parts branch changes fixed-validation macro-F1 from
**{cptr["development_validation"]["baseline_metrics"]["macro_f1"]:.4f}** to
**{cptr["development_validation"]["candidate_metrics"]["macro_f1"]:.4f}**. The matched
recording-grouped OOF comparison moves in the other direction:
**{cptr["grouped_crossfit_oof"]["baseline_metrics"]["macro_f1"]:.4f}** for the temporal
baseline and **{cptr["grouped_crossfit_oof"]["candidate_metrics"]["macro_f1"]:.4f}** for
the candidate. The promotion gate therefore keeps the temporal ensemble as the default.

Motion nulling reduces the candidate's validation macro-F1 by
**{read_json(repository / "results/okutama_cptr/faithfulness_summary.json")["diagnostics"]["motion_null"]["macro_f1_delta_real_minus_intervention"]:.4f}**,
while the grouped analysis identifies occluded windows as the largest observed failure
mode. The fixed-split gain is retained as development evidence, not promoted as a
generalized improvement.
"""
        ),
        code(
            """
cptr_headline = pd.read_csv(ROOT / "results/okutama_cptr/headline_metrics.csv")
display(
    cptr_headline[
        ["scope", "model", "samples", "recordings", "macro_f1", "accuracy", "log_loss"]
    ]
)

cptr_subgroups = pd.read_csv(ROOT / "results/okutama_cptr/subgroup_metrics.csv")
display(
    cptr_subgroups[
        cptr_subgroups["scope"].eq("crossfit_oof")
        & cptr_subgroups["subgroup"].isin(["window_clear", "window_occluded", "transition"])
    ][["subgroup", "samples", "baseline_macro_f1", "candidate_macro_f1", "macro_f1_delta"]]
)
"""
        ),
        markdown(
            """
## 8. Attribution: localization is not enough

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
## 9. Bounded bit-flip robustness

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
## 10. Conclusions

- Data scale is the largest isolated performance amplifier in this study.
- DINOv2-B representations support strong linear and nonlinear final-stage classifiers.
- Development-locked model diversity produces a statistically supported ensemble gain.
- Person-centric scale conditioning raises official-test V-COCO macro-F1 from 0.7071
  to 0.8663 and greatly reduces the observed association with apparent person size.
- Factorized posture-motion targets add a smaller, independently supported gain under
  matched feature inputs.
- Short temporal context improves sealed Okutama macro-F1 from 0.7458 to 0.7854; fixed
  50% routing retains 0.7817.
- The center-plus-parts residual improves one fixed split but not recording-grouped OOF,
  showing why the grouped promotion gate is necessary.
- ConvNeXt Grad-CAM passes the declared sanity checks more convincingly than DINOv2-B
  integrated gradients.

The complete method, evidence lineage, discussion, and references are documented in
`docs/POLAR_PUBLIC_REPORT.md`, `docs/VCOCO_V2_EXTERNAL_TRANSFER.md`,
`docs/VCOCO_V3_MOTION_IDENTIFIABILITY.md`, and `docs/OKUTAMA_CPTR_DEVELOPMENT.md`.
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
    for index, cell in enumerate(notebook.cells):
        cell.metadata.pop("execution", None)
        identity = f"{index}\0{cell.cell_type}\0{cell.source}".encode()
        cell["id"] = hashlib.sha256(identity).hexdigest()[:8]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(nbformat.writes(notebook), encoding="utf-8", newline="\n")
    print(f"Wrote executed notebook: {args.output.resolve()}")


if __name__ == "__main__":
    main()
