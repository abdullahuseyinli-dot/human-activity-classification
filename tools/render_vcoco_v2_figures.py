"""Render the tracked V-COCO v2 result figures from portable evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#17212B"
BLUE = "#176B87"
TEAL = "#0E8A7A"
GOLD = "#D99126"
GRAY = "#A8B1BA"
GRID = "#DDE3E8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/vcoco_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    return parser.parse_args()


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "svg.hashsalt": "vcoco-v2",
        }
    )


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(
        output_dir / f"{name}.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "human-activity-classification"},
    )
    svg_path = output_dir / f"{name}.svg"
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": None, "Creator": "human-activity-classification"},
    )
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    plt.close(fig)


def development_figure(results: Path, output: Path) -> None:
    frame = pd.read_csv(results / "development_candidates.csv").sort_values("macro_f1")
    labels = {
        "historical_source_dino": "Historical source DINO",
        "source_dino_tight_pad": "Source DINO, tight padded",
        "pose_geometry_rbf_oracle": "Pose + geometry oracle",
        "convnext_frozen_probe": "Frozen ConvNeXt-S",
        "siglip2_frozen_probe": "Frozen SigLIP2-B",
        "dino_frozen_single_view": "Frozen DINOv2-B",
        "dino_flat_same_features": "Flat DINO, same inputs",
        "dino_factorized_head": "Factorized DINO head",
        "dino_lpft_mild": "DINO LP-FT, mild",
        "dino_lpft_augmix": "DINO LP-FT, AugMix",
        "dino_scale_conditioned_stack": "Scale-conditioned DINO stack",
    }
    colors = [
        GOLD
        if method == "dino_scale_conditioned_stack"
        else GRAY
        if method == "historical_source_dino"
        else BLUE
        for method in frame["method"]
    ]
    fig, axis = plt.subplots(figsize=(9.2, 5.8))
    bars = axis.barh(
        [labels[value] for value in frame["method"]], frame["macro_f1"], color=colors
    )
    axis.set_xlim(0.65, 0.88)
    axis.set_xlabel("Validation macro-F1")
    axis.set_title("Controlled V-COCO development comparison", loc="left")
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in frame["macro_f1"]], padding=4)
    save(fig, output, "vcoco_v2_development_comparison")


def test_figure(results: Path, output: Path) -> None:
    overall = pd.read_csv(results / "official_test_metrics.csv").set_index("method")
    classes = pd.read_csv(results / "official_test_per_class.csv")
    methods = ["historical_v1_dino", "scale_conditioned_stacking"]
    labels = ["Historical source DINO", "Scale-conditioned stack"]
    categories = ["Macro-F1", "Sitting", "Standing", "Walking/running"]
    values = []
    for method in methods:
        per_class = classes[classes["method"].eq(method)].set_index("class")
        values.append(
            [
                overall.loc[method, "macro_f1"],
                per_class.loc["sitting", "f1"],
                per_class.loc["standing", "f1"],
                per_class.loc["walking_running", "f1"],
            ]
        )
    positions = np.arange(len(categories))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    for index, (label, row, color) in enumerate(zip(labels, values, [GRAY, TEAL], strict=True)):
        bars = axis.bar(positions + (index - 0.5) * width, row, width, label=label, color=color)
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in row], padding=3, fontsize=9)
    axis.set_xticks(positions, categories)
    axis.set_ylim(0.5, 1.0)
    axis.set_ylabel("F1")
    axis.set_title("Locked official-test result", loc="left")
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper right")
    save(fig, output, "vcoco_v2_official_test_comparison")


def fewshot_figure(results: Path, output: Path) -> None:
    frame = pd.read_csv(results / "fewshot_curve.csv")
    x = frame["budget_people_per_class_minimum"].to_numpy()
    mean = frame["macro_f1_mean"].to_numpy()
    low = frame["macro_f1_repeat_interval_low"].to_numpy()
    high = frame["macro_f1_repeat_interval_high"].to_numpy()
    fig, axis = plt.subplots(figsize=(8.4, 4.6))
    axis.fill_between(x, low, high, color=BLUE, alpha=0.16, linewidth=0)
    axis.plot(x, mean, color=BLUE, marker="o", linewidth=2.2)
    for budget, score in zip(x, mean, strict=True):
        axis.annotate(f"{score:.3f}", (budget, score), xytext=(0, 8), textcoords="offset points", ha="center")
    axis.set_xscale("log")
    axis.set_xticks(x, [str(value) for value in x])
    axis.set_ylim(0.2, 0.85)
    axis.set_xlabel("Minimum labeled people per class")
    axis.set_ylabel("Validation macro-F1")
    axis.set_title("Image-grouped DINOv2-B few-shot curve", loc="left")
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    save(fig, output, "vcoco_v2_fewshot_curve")


def scale_figure(results: Path, output: Path) -> None:
    frame = pd.read_csv(results / "official_test_strata.csv")
    specifications = [
        ("area_quartile", ["Q1_small", "Q2", "Q3", "Q4_large"], "Box-area quartile"),
        ("height_quartile", ["Q1_short", "Q2", "Q3", "Q4_tall"], "Person-height quartile"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=True)
    for axis, (stratum, order, title) in zip(axes, specifications, strict=True):
        subset = frame[frame["stratum"].eq(stratum)]
        pivot = subset.pivot(index="value", columns="method", values="macro_f1_fixed_classes")
        gain = (
            pivot["scale_conditioned_stacking"] - pivot["historical_v1_dino"]
        ).reindex(order)
        labels = ["Q1", "Q2", "Q3", "Q4"]
        bars = axis.bar(labels, gain, color=[TEAL, TEAL, BLUE, BLUE])
        axis.bar_label(bars, labels=[f"+{value:.3f}" for value in gain], padding=3)
        axis.set_title(title, loc="left")
        axis.set_xlabel("Small/short → large/tall")
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Paired macro-F1 gain over historical baseline")
    axes[0].set_ylim(0.0, 0.26)
    fig.suptitle("The largest transfer gains occur for small people", x=0.08, ha="left", fontweight="bold")
    save(fig, output, "vcoco_v2_scale_gain")


def selective_figure(results: Path, output: Path) -> None:
    payload = json.loads((results / "official_test_selective_metrics.json").read_text())
    fig, axis = plt.subplots(figsize=(8.4, 4.6))
    specifications = [
        ("historical_v1_dino", "Historical source DINO", GRAY),
        ("scale_conditioned_stacking", "Scale-conditioned stack", TEAL),
    ]
    for key, label, color in specifications:
        rows = sorted(payload[key]["coverage_points"], key=lambda row: row["realized_coverage"])
        axis.plot(
            [row["realized_coverage"] for row in rows],
            [row["accuracy"] for row in rows],
            marker="o",
            linewidth=2.2,
            color=color,
            label=label,
        )
    axis.set_xlim(0.68, 1.01)
    axis.set_ylim(0.68, 0.98)
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Accuracy on retained people")
    axis.set_title("Confidence supports useful selective prediction", loc="left")
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    save(fig, output, "vcoco_v2_selective_prediction")


def main() -> None:
    args = parse_args()
    results = args.results_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure()
    development_figure(results, output)
    test_figure(results, output)
    fewshot_figure(results, output)
    scale_figure(results, output)
    selective_figure(results, output)
    print("Rendered five V-COCO v2 figures in PNG and SVG formats", flush=True)


if __name__ == "__main__":
    main()
