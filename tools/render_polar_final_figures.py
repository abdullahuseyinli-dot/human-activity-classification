"""Render publication figures from exported, locked POLAR evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DISPLAY_NAMES = {
    "convnext_small_full": "ConvNeXt-S full",
    "dinov2_small_moderate": "DINOv2-S full",
    "dinov2_base_top4": "DINOv2-B top 4",
    "dinov2_base_multilayer_logistic": "DINOv2-B logistic",
    "dinov2_base_multilayer_rbf": "DINOv2-B RBF SVM",
    "locked_ensemble": "Locked ensemble",
    "locked_ensemble_collapsed": "Locked ensemble",
    "direct_three_class_probe": "Direct 3-class probe",
}
FAMILY_COLORS = {
    "convnext_small_full": "#2563EB",
    "dinov2_small_moderate": "#D97706",
    "dinov2_base_top4": "#7C3AED",
    "dinov2_base_multilayer_logistic": "#059669",
    "dinov2_base_multilayer_rbf": "#DC2626",
    "locked_ensemble": "#111827",
    "locked_ensemble_collapsed": "#111827",
    "direct_three_class_probe": "#64748B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "svg.hashsalt": "polar-locked-study",
        }
    )


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    figure.savefig(
        output_dir / f"{stem}.png",
        dpi=200,
        facecolor="white",
        metadata={"Software": "Matplotlib"},
    )
    figure.savefig(
        svg_path,
        facecolor="white",
        metadata={"Date": None, "Creator": "Matplotlib"},
    )
    # Matplotlib emits trailing spaces in SVG path data. Normalize generated text
    # so repository whitespace checks stay useful and the artifact is deterministic.
    svg = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    plt.close(figure)


def candidate_color(name: str) -> str:
    return FAMILY_COLORS.get(name, "#94A3B8")


def render_test_comparison(results: Path, output: Path) -> None:
    frame = pd.read_csv(results / "polar_test_metrics.csv").sort_values(
        ["macro_f1", "candidate"], ascending=[True, True], ignore_index=True
    )
    colors = [candidate_color(name) for name in frame["candidate"]]
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    bars = axis.barh(
        [DISPLAY_NAMES.get(name, name) for name in frame["candidate"]],
        frame["macro_f1"],
        color=colors,
    )
    lower = max(0.0, float(frame["macro_f1"].min()) - 0.035)
    axis.set_xlim(lower, 1.0)
    axis.set_xlabel("Macro-F1")
    axis.set_title("Held-out POLAR test: predeclared candidates")
    for bar, value in zip(bars, frame["macro_f1"], strict=True):
        axis.text(
            min(float(value) + 0.003, 0.992),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )
    save_figure(figure, output, "polar_test_comparison")


def render_scale_curve(results: Path, output: Path) -> None:
    summary = read_json(results / "polar_extension_summary.json")
    curve = pd.DataFrame(summary["scale_curve"]).sort_values("actual_train_size")
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    axis.plot(
        curve["actual_train_size"],
        curve["macro_f1_mean"],
        marker="o",
        linewidth=2.2,
        color="#7C3AED",
    )
    axis.set_xscale("log")
    axis.set_xticks(curve["actual_train_size"])
    axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axis.set_ylim(max(0.0, float(curve["macro_f1_mean"].min()) - 0.025), 0.95)
    axis.set_xlabel("Development training images (log scale)")
    axis.set_ylabel("Validation macro-F1")
    axis.set_title("Frozen DINOv2-B learning curve")
    for row in curve.itertuples(index=False):
        axis.annotate(
            f"{row.macro_f1_mean:.3f}",
            (row.actual_train_size, row.macro_f1_mean),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )
    save_figure(figure, output, "polar_scale_curve")


def render_confusion(results: Path, output: Path) -> None:
    payload = read_json(results / "polar_test_confusions.json")["locked_ensemble"]
    matrix = np.asarray(payload["row_normalized"], dtype=float)
    names = [str(name).replace("_", " ") for name in payload["class_names"]]
    figure, axis = plt.subplots(figsize=(5.8, 5.0))
    image = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.1%}",
                ha="center",
                va="center",
                color="white" if value >= 0.55 else "#111827",
                fontsize=9,
            )
    axis.set_xticks(range(len(names)), names, rotation=25, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Locked ensemble confusion matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row-normalized rate")
    save_figure(figure, output, "polar_confusion_matrix")


def render_external(results: Path, output: Path) -> None:
    frame = pd.read_csv(results / "polar_external_image_metrics.csv").sort_values(
        ["macro_f1", "candidate"], ascending=[True, True], ignore_index=True
    )
    figure, axis = plt.subplots(figsize=(8.2, 5.1))
    bars = axis.barh(
        [DISPLAY_NAMES.get(name, name) for name in frame["candidate"]],
        frame["macro_f1"],
        color=[candidate_color(name) for name in frame["candidate"]],
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Macro-F1")
    axis.set_title("V-COCO external transfer (image-level, 3 classes)")
    for bar, value in zip(bars, frame["macro_f1"], strict=True):
        axis.text(
            min(float(value) + 0.012, 0.97),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )
    save_figure(figure, output, "polar_external_validation")


def confidence_interval(record: dict) -> tuple[float, float, float]:
    mean = float(record["mean"])
    return mean, mean - float(record["ci95_low"]), float(record["ci95_high"]) - mean


def render_faithfulness(results: Path, output: Path) -> None:
    curves = pd.read_csv(results / "polar_faithfulness_curves.csv")
    summary = read_json(results / "polar_faithfulness_summary.json")
    families = list(summary["protocol"]["families"])
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    curve_styles = {
        "deletion": ("Targeted deletion", "#DC2626", "-"),
        "insertion": ("Insertion", "#059669", "-"),
        "random_deletion": ("Random deletion", "#64748B", "--"),
    }
    for axis, family in zip(axes[0], families, strict=True):
        family_frame = curves[curves["family"].eq(family)]
        means = (
            family_frame.groupby(["curve", "fraction"], as_index=False)[
                "target_probability"
            ]
            .mean()
            .sort_values(["curve", "fraction"])
        )
        for curve_name, (label, color, style) in curve_styles.items():
            values = means[means["curve"].eq(curve_name)]
            axis.plot(
                values["fraction"],
                values["target_probability"],
                marker="o",
                label=label,
                color=color,
                linestyle=style,
            )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Fraction of ranked patches perturbed")
        axis.set_ylabel("Locked target probability")
        axis.set_title(DISPLAY_NAMES.get(family, family))
        axis.legend(fontsize=8)

    metric_panels = (
        (
            axes[1, 0],
            "person_attribution_mass_lift",
            "Attribution mass / bbox area",
            1.0,
        ),
        (
            axes[1, 1],
            "person_minus_context_occlusion_drop",
            "Person minus matched-context probability drop",
            0.0,
        ),
    )
    for axis, metric, title, reference in metric_panels:
        values = [confidence_interval(summary["aggregate"][family][metric]) for family in families]
        means = [item[0] for item in values]
        errors = np.asarray([[item[1] for item in values], [item[2] for item in values]])
        positions = np.arange(len(families))
        axis.bar(
            positions,
            means,
            yerr=errors,
            capsize=4,
            color=[candidate_color(family) for family in families],
        )
        axis.axhline(reference, color="#475569", linewidth=1.0, linestyle="--")
        axis.set_xticks(
            positions,
            [DISPLAY_NAMES.get(family, family) for family in families],
            rotation=12,
            ha="right",
        )
        axis.set_title(title)
        axis.set_ylabel("Mean with 95% bootstrap CI")
    figure.suptitle("BBox-aware attribution faithfulness", fontsize=14)
    figure.tight_layout()
    save_figure(figure, output, "polar_faithfulness")


def render_attribution_sanity(results: Path, output: Path) -> None:
    summary = read_json(results / "polar_faithfulness_summary.json")
    families = list(summary["protocol"]["families"])
    metrics = (
        ("target_vs_alternative_attribution_spearman", "Alternative target"),
        ("randomized_head_spearman", "Randomized head"),
        ("randomized_adapted_cascade_spearman", "Randomized adapted layers"),
    )
    positions = np.arange(len(metrics), dtype=float)
    width = 0.34
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for family_index, family in enumerate(families):
        offsets = positions + (family_index - (len(families) - 1) / 2) * width
        values = [float(summary["aggregate"][family][metric]["mean"]) for metric, _ in metrics]
        bars = axis.bar(
            offsets,
            values,
            width=width,
            color=candidate_color(family),
            label=DISPLAY_NAMES.get(family, family),
        )
        for bar, value in zip(bars, values, strict=True):
            vertical = 0.035 if value >= 0.0 else -0.075
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + vertical,
                f"{value:.2f}",
                ha="center",
                va="bottom" if value >= 0.0 else "top",
                fontsize=8,
            )
    axis.axhline(0.0, color="#475569", linewidth=1.0)
    axis.set_xticks(positions, [label for _, label in metrics])
    axis.set_ylim(-0.65, 1.02)
    axis.set_ylabel("Mean attribution Spearman correlation")
    axis.set_title(
        "Attribution specificity and parameter-randomization sanity\n"
        "(lower correlation indicates greater sensitivity)"
    )
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    save_figure(figure, output, "polar_attribution_sanity")


def render_fault_robustness(results: Path, output: Path) -> None:
    frame = pd.read_csv(results / "polar_fault_robustness_metrics.csv")
    aggregate = frame[frame["fault_seed"].astype(str).eq("aggregate")].copy()
    families = sorted(set(aggregate["family"]))
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    input_rows = aggregate[aggregate["condition"].eq("uint8_input_bit_flip_rate")]
    for family in families:
        values = input_rows[input_rows["family"].eq(family)].sort_values("level")
        axes[0].plot(
            values["level"],
            values["prediction_agreement_with_clean"],
            marker="o",
            color=candidate_color(family),
            label=DISPLAY_NAMES.get(family, family),
        )
    axes[0].set_xscale("symlog", linthresh=1e-5)
    axes[0].set_xlabel("Input bit-flip rate")
    axes[0].set_ylabel("Prediction agreement with clean")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_title("Input-pixel corruption")
    axes[0].legend(fontsize=8)

    head_rows = aggregate[
        aggregate["condition"].eq("symmetric_int8_head_weight_bit_flips")
    ]
    for family in families:
        values = head_rows[head_rows["family"].eq(family)].sort_values("level")
        axes[1].plot(
            values["level"],
            values["prediction_agreement_with_clean"],
            marker="o",
            color=candidate_color(family),
            label=DISPLAY_NAMES.get(family, family),
        )
    axes[1].set_xlabel("Classifier-weight bit flips per seed model")
    axes[1].set_ylabel("Prediction agreement with clean float model")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Int8-quantized head corruption")
    figure.suptitle("Locked neural-component fault robustness", fontsize=14)
    figure.tight_layout()
    save_figure(figure, output, "polar_fault_robustness")


def main() -> None:
    args = parse_args()
    results = args.results_dir.resolve()
    output = args.output_dir.resolve()
    required_statuses = {
        "polar_test_summary.json": "LOCKED_FINAL_TEST_COMPLETE",
        "polar_external_summary.json": "LOCKED_EXTERNAL_EVALUATION_COMPLETE",
        "polar_faithfulness_summary.json": "LOCKED_POLAR_FAITHFULNESS_COMPLETE",
        "polar_fault_summary.json": "LOCKED_POLAR_FAULT_ROBUSTNESS_COMPLETE",
    }
    hashes = set()
    for name, status in required_statuses.items():
        summary = read_json(results / name)
        if summary.get("status") != status:
            raise RuntimeError(f"Incomplete evidence: {name}")
        hashes.add(summary["selection_lock_sha256"])
    if len(hashes) != 1:
        raise RuntimeError("Exported summaries do not share one final selection lock")

    configure_style()
    render_test_comparison(results, output)
    render_scale_curve(results, output)
    render_confusion(results, output)
    render_external(results, output)
    render_faithfulness(results, output)
    render_attribution_sanity(results, output)
    render_fault_robustness(results, output)
    print(f"Rendered locked POLAR figures in {output}", flush=True)


if __name__ == "__main__":
    main()
