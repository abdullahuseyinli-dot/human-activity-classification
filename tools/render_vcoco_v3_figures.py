"""Render the tracked motion-identifiability figures from portable evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "source_only_static": "#94A3B8",
    "static": "#2563EB",
    "classification_student": "#7C3AED",
    "hybrid_budget_0.5": "#F59E0B",
    "teacher": "#059669",
}

LABELS = {
    "source_only_static": "Source-only transfer",
    "static": "Target static",
    "classification_student": "Distilled static",
    "hybrid_budget_0.5": "50% routed clip",
    "teacher": "Temporal clip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/vcoco_v3"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    return parser.parse_args()


def save_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("png", "svg"):
        output_path = output / f"{stem}.{suffix}"
        figure.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Creator": "human-activity-classification"},
        )
        if suffix == "svg":
            lines = output_path.read_text(encoding="utf-8").splitlines()
            output_path.write_text(
                "\n".join(line.rstrip() for line in lines) + "\n",
                encoding="utf-8",
                newline="\n",
            )
    plt.close(figure)


def comparison_figure(results: Path, output: Path) -> None:
    metrics = pd.read_csv(results / "confirmation_metrics.csv").set_index("family")
    order = [
        "source_only_static",
        "static",
        "classification_student",
        "hybrid_budget_0.5",
        "teacher",
    ]
    selected = metrics.loc[order]

    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    positions = range(len(order))
    bars = axis.barh(
        positions,
        selected["macro_f1"],
        color=[COLORS[name] for name in order],
        height=0.62,
    )
    axis.set_yticks(list(positions), [LABELS[name] for name in order])
    axis.invert_yaxis()
    axis.set_xlim(0.0, 0.82)
    axis.set_xlabel("Macro-F1")
    axis.set_title("Okutama sealed confirmation")
    axis.grid(axis="x", alpha=0.22, linewidth=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    for bar, value in zip(bars, selected["macro_f1"], strict=True):
        axis.text(
            float(value) + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{float(value):.4f}",
            va="center",
            fontsize=10,
        )
    figure.tight_layout()
    save_figure(figure, output, "vcoco_v3_confirmation_comparison")


def routing_figure(results: Path, output: Path) -> None:
    routing = pd.read_csv(results / "confirmation_routing_curve.csv")
    x = routing["observed_clip_fraction"] * 100.0

    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    axis.plot(
        x,
        routing["macro_f1"],
        marker="o",
        linewidth=2.4,
        markersize=6,
        color="#059669",
        label="Macro-F1",
    )
    axis.plot(
        x,
        routing["accuracy"],
        marker="s",
        linewidth=2.0,
        markersize=5.5,
        color="#2563EB",
        label="Accuracy",
    )
    axis.set_xticks([0, 10, 25, 50, 100])
    axis.set_xlim(-2, 102)
    axis.set_ylim(0.72, 0.795)
    axis.set_xlabel("Examples sent to the temporal model (%)")
    axis.set_ylabel("Score")
    axis.set_title("Fixed-budget temporal routing")
    axis.grid(alpha=0.22, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    for x_value, value in zip(x, routing["macro_f1"], strict=True):
        axis.annotate(
            f"{float(value):.4f}",
            (float(x_value), float(value)),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#065F46",
        )
    figure.tight_layout()
    save_figure(figure, output, "vcoco_v3_routing_curve")


def main() -> None:
    args = parse_args()
    results = args.results.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    comparison_figure(results, output)
    routing_figure(results, output)
    print("Rendered V-COCO v3 confirmation figures")


if __name__ == "__main__":
    main()
