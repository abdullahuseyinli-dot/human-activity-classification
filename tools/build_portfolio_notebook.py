"""Build and execute the compact portfolio notebook from tracked evidence."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel", default="python3")
    return parser.parse_args()


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def require_files(repository: Path) -> None:
    required = [
        "data/manifest.csv",
        "results/candidate_selection_summary.csv",
        "results/confirmation_ranking.csv",
        "results/selection_lock.json",
        "results/final_seed_metrics.csv",
        "results/downstream_oof_ranking.csv",
        "results/evaluation_policy_oof_selection.csv",
        "results/locked_test_metrics.csv",
        "results/test_bootstrap_intervals.csv",
        "results/champion_paired_bootstrap_difference.csv",
        "assets/final_method_comparison.png",
        "assets/champion_confusion_matrix.png",
        "assets/convnext_architecture.png",
        "assets/dinov2_architecture.png",
        "assets/champion_error_gallery.png",
        "results/champion_error_analysis.csv",
    ]
    missing = [name for name in required if not (repository / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Portfolio evidence is incomplete: {missing}")


def headline_text(
    repository: Path,
) -> tuple[tuple[str, str, float, float], tuple[float, float]]:
    metrics = pd.read_csv(repository / "results" / "locked_test_metrics.csv")
    champion = metrics.loc[metrics["selected_champion"].astype(bool)].iloc[0]
    intervals = pd.read_csv(repository / "results" / "test_bootstrap_intervals.csv")
    interval = intervals[
        (intervals["method"] == champion["method"]) & (intervals["metric"] == "macro_f1")
    ].iloc[0]
    return (
        str(champion["display_name"]),
        str(champion["method"]),
        float(champion["macro_f1"]),
        float(champion["accuracy"]),
    ), (float(interval["ci_2_5"]), float(interval["ci_97_5"]))


def build_notebook(repository: Path) -> dict:
    (headline_name, headline_method, headline_f1, headline_accuracy), interval = headline_text(
        repository
    )
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    notebook["cells"] = [
        markdown(
            f"""
# Human Activity Classification with ConvNeXt and DINOv2

This study compares supervised transfer learning and self-supervised visual
features on 285 still images across **sitting**, **standing**, and
**walking/running**. Model selection is isolated from the fixed 43-image test
split through internal stratified cross-validation.

**Locked result:** {headline_name} reached **{headline_f1:.3f} macro-F1** and
**{headline_accuracy:.3f} accuracy**. The stratified bootstrap interval for
macro-F1 is **[{interval[0]:.3f}, {interval[1]:.3f}]**, which is reported because
the final test set is deliberately small.
"""
        ),
        markdown(
            """
## Evaluation contract

The original train and validation partitions form a 242-image development
pool. All freeze-depth, dropout, augmentation, learning-rate, calibration,
epoch, SVM, and ensemble decisions use out-of-fold predictions from that pool.
The original test partition stays fixed and is evaluated only after the
configuration and downstream-method locks are written.
"""
        ),
        code(
            """
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Image, display

ROOT = Path.cwd().resolve()
if not (ROOT / "pyproject.toml").is_file():
    raise RuntimeError("Run this notebook from the repository root.")
sys.path.insert(0, str(ROOT / "src"))

from hac.protocol import load_and_validate_manifest  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
manifest, protocol = load_and_validate_manifest(
    ROOT / "data" / "manifest.csv", require_images=False
)
pd.DataFrame([{
    "manifest_rows": len(manifest),
    "development_rows": protocol.development_rows,
    "locked_test_rows": protocol.test_rows,
    "manifest_sha256": protocol.manifest_sha256[:12] + "…",
    "test_id_sha256": protocol.test_image_ids_sha256[:12] + "…",
}])
"""
        ),
        markdown("## Dataset profile"),
        code(
            """
split_counts = (
    manifest.groupby(["original_split", "label"])
    .size()
    .rename("images")
    .reset_index()
)
display(split_counts.pivot(index="original_split", columns="label", values="images"))

fig, ax = plt.subplots(figsize=(8.5, 4.2))
sns.countplot(
    data=manifest,
    x="label",
    hue="original_split",
    order=["sitting", "standing", "walking_running"],
    hue_order=["train", "val", "test"],
    palette="Set2",
    ax=ax,
)
ax.set(xlabel="Activity", ylabel="Images", title="Class balance by fixed partition")
ax.legend(title="Partition", frameon=False)
fig.tight_layout()
plt.show()
"""
        ),
        markdown(
            """
The manifest validator checks cardinality, class vocabulary, unique identifiers,
content hashes, and the fixed split contract. It also rejects exact hashes or
perceptual hashes within Hamming distance six when they cross the development
and test boundary. Subject-level independence cannot be claimed because subject
identifiers are not available.
"""
        ),
        markdown("## Controlled regularization screen"),
        code(
            """
candidates = pd.read_csv(ROOT / "results" / "candidate_selection_summary.csv")
coarse = candidates[candidates["stage"] == "coarse"].copy()
coarse["display_model"] = coarse["model_kind"].map({
    "convnext_small": "ConvNeXt-Small",
    "dinov2_small": "DINOv2-Small",
})
display(
    coarse.sort_values(["model_kind", "oof_macro_f1"], ascending=[True, False])[
        ["candidate_id", "display_model", "oof_macro_f1", "oof_log_loss", "derived_final_epochs"]
    ].round(4)
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=False)
for ax, (family, rows) in zip(axes, coarse.groupby("display_model"), strict=True):
    rows = rows.sort_values("oof_macro_f1")
    ax.barh(rows["candidate_id"], rows["oof_macro_f1"], color="#64748B")
    ax.set_title(family)
    ax.set_xlabel("Three-fold OOF macro-F1")
    ax.set_xlim(max(0.0, rows["oof_macro_f1"].min() - 0.05), min(1.0, rows["oof_macro_f1"].max() + 0.03))
fig.suptitle("Controlled dropout, augmentation, and freeze-depth comparisons")
fig.tight_layout()
plt.show()
"""
        ),
        markdown(
            """
Interventions are compared independently where possible: head dropout,
full/partial/head-only adaptation, light RandAugment, random-erasing removal,
MixUp, label smoothing, and weight decay. The purpose is not to maximize the
number of regularizers; it is to retain only changes supported by out-of-fold
performance and stability.
"""
        ),
        markdown("## Five-fold confirmation and configuration lock"),
        code(
            """
confirmation = pd.read_csv(ROOT / "results" / "confirmation_ranking.csv")
display(confirmation.round(4))

with (ROOT / "results" / "selection_lock.json").open(encoding="utf-8") as handle:
    selection_lock = json.load(handle)

locked_rows = []
for family, record in selection_lock["selected"].items():
    locked_rows.append({
        "model": family,
        "candidate": record["candidate_id"],
        "confirmation_macro_f1": record["confirmation_oof_macro_f1"],
        "fold_macro_f1_std": record["confirmation_fold_macro_f1_std"],
        **record["config"],
    })
pd.DataFrame(locked_rows).round(5)
"""
        ),
        markdown("## Locked model architectures"),
        code(
            """
display(Image(filename=str(ROOT / "assets" / "convnext_architecture.png")))
display(Image(filename=str(ROOT / "assets" / "dinov2_architecture.png")))
"""
        ),
        markdown(
            """
Each locked configuration is retrained across seeds 42, 52, and 62. Five-fold
training derives a median best-epoch count per seed. Full-pool training replays
the median fold learning-rate state by epoch, avoiding the inherited mismatch
where validation-driven LR reductions disappeared during final retraining.
"""
        ),
        markdown("## Seed stability and calibration"),
        code(
            """
seed_metrics = pd.read_csv(ROOT / "results" / "final_seed_metrics.csv")
primary = seed_metrics[
    seed_metrics["calibration"].isin(["temperature_scaled", "flip_tta_temperature_scaled"])
].copy()
display(
    primary[
        ["family", "seed", "calibration", "derived_final_epochs", "accuracy", "macro_f1", "log_loss", "ece"]
    ].round(4)
)

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.pointplot(
    data=primary[primary["calibration"] == "temperature_scaled"],
    x="family",
    y="macro_f1",
    hue="seed",
    palette="viridis",
    markers="o",
    linestyles="none",
    ax=ax,
)
ax.set(xlabel="Model family", ylabel="Locked-test macro-F1", title="Final seed variability")
ax.legend(title="Seed", frameon=False)
fig.tight_layout()
plt.show()
"""
        ),
        markdown("## OOF-locked downstream selection"),
        code(
            """
policy = pd.read_csv(ROOT / "results" / "evaluation_policy_oof_selection.csv")
ranking = pd.read_csv(ROOT / "results" / "downstream_oof_ranking.csv")
display(policy.round(4))
display(ranking.round(4))
"""
        ),
        markdown(
            """
The downstream comparison includes calibrated seed averages, an OOF-weighted
ConvNeXt/DINOv2 probability blend, and SVM probes over OOF embeddings. A
center-crop versus center-plus-horizontal-flip policy is also selected per
family using OOF predictions. The test set does not determine any of these
choices.
"""
        ),
        markdown("## Locked test results"),
        code(
            """
test_metrics = pd.read_csv(ROOT / "results" / "locked_test_metrics.csv")
display(
    test_metrics[
        ["display_name", "selected_champion", "accuracy", "macro_f1", "balanced_accuracy", "log_loss", "ece"]
    ].round(4)
)
display(Image(filename=str(ROOT / "assets" / "final_method_comparison.png")))
display(Image(filename=str(ROOT / "assets" / "champion_confusion_matrix.png")))
"""
        ),
        markdown("## Error inspection"),
        code(
            """
errors = pd.read_csv(ROOT / "results" / "champion_error_analysis.csv")
display(
    errors[
        ["image_id", "true_class", "predicted_class", "confidence", "image_url"]
    ].round(3)
)
display(Image(filename=str(ROOT / "assets" / "champion_error_gallery.png")))
"""
        ),
        markdown(
            """
This gallery is diagnostic only and is generated after the downstream lock.
No test image is relabeled, removed, or used to revise the selected method.
"""
        ),
        markdown("## Uncertainty and paired comparison"),
        code(
            """
intervals = pd.read_csv(ROOT / "results" / "test_bootstrap_intervals.csv")
differences = pd.read_csv(ROOT / "results" / "champion_paired_bootstrap_difference.csv")
display(intervals[intervals["metric"] == "macro_f1"].round(4))
display(differences.round(4))
"""
        ),
        markdown(
            f"""
## Conclusion

The OOF selection process chose **{headline_name}** (`{headline_method}`) as the
portfolio headline. Its fixed-test result is **{headline_f1:.3f} macro-F1** and
**{headline_accuracy:.3f} accuracy**. This replaces the notebook's conflicting
historical outputs with one traceable lineage: a fixed test contract, explicit
configuration locks, three final seeds, OOF calibration, and paired uncertainty
reporting.

The small test set limits precision, and the lack of subject identifiers limits
the generalization claim. Results should be read as a careful benchmark on this
manifest, not as a deployment guarantee.
"""
        ),
        markdown(
            """
## Reproduction

1. Install the project and notebook dependencies from `pyproject.toml`.
2. Download the checksum-verified images with
   `python tools/download_dataset.py --manifest data/manifest.csv`.
3. Review `docs/EXPERIMENT_PROTOCOL.md` and the tracked configuration locks.
4. Run the training scripts from the repository root. Bulky checkpoints and
   local experiment artifacts stay under `.runs/` and are intentionally not
   committed.

The tracked results are sufficient to rerun this analysis notebook without
downloading model checkpoints or the source images.
"""
        ),
    ]
    return notebook


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    require_files(repository)
    notebook = build_notebook(repository)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    previous_directory = Path.cwd()
    try:
        os.chdir(repository)
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name=args.kernel,
            resources={"metadata": {"path": str(repository)}},
        )
        client.execute()
    finally:
        os.chdir(previous_directory)

    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nbformat.write(notebook, args.output)
    print(f"Wrote executed notebook: {args.output.resolve()}")


if __name__ == "__main__":
    main()
