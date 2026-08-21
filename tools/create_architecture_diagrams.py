"""Render architecture diagrams from the locked portfolio configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

COLORS = {
    "input": "#DBEAFE",
    "frozen": "#E2E8F0",
    "trainable": "#FEF3C7",
    "feature": "#D1FAE5",
    "head": "#FECACA",
    "output": "#EDE9FE",
    "border": "#334155",
    "text": "#0F172A",
}


def normalize_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    return parser.parse_args()


def box(ax, x: float, title: str, detail: str, status: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, 1.1),
        1.45,
        2.0,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor=color,
        edgecolor=COLORS["border"],
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x + 0.725, 2.65, title, ha="center", va="center", weight="bold", fontsize=9)
    ax.text(x + 0.725, 2.05, detail, ha="center", va="center", fontsize=7.8)
    ax.text(x + 0.725, 1.45, status, ha="center", va="center", fontsize=7.6)


def arrow(ax, left: float, right: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (left, 2.1),
            (right, 2.1),
            arrowstyle="->",
            mutation_scale=13,
            color=COLORS["border"],
            linewidth=1.2,
        )
    )


def render(
    output: Path,
    title: str,
    subtitle: str,
    nodes: list[tuple[str, str, str, str]],
) -> None:
    width = max(12.0, 1.8 * len(nodes))
    fig, ax = plt.subplots(figsize=(width, 4.4))
    positions = [0.25 + 1.8 * index for index in range(len(nodes))]
    for index, (node_title, detail, status, color_key) in enumerate(nodes):
        box(ax, positions[index], node_title, detail, status, COLORS[color_key])
        if index:
            arrow(ax, positions[index - 1] + 1.45, positions[index])
    ax.text(0.25, 3.85, title, ha="left", va="center", fontsize=16, weight="bold")
    ax.text(0.25, 3.48, subtitle, ha="left", va="center", fontsize=9.5, color="#475569")
    ax.set_xlim(0.0, positions[-1] + 1.8)
    ax.set_ylim(0.55, 4.15)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    svg_path = output.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    normalize_svg(svg_path)
    plt.close(fig)


def trainability(strategy: str, node: str) -> tuple[str, str]:
    if strategy == "full_backbone":
        return "Trainable", "trainable"
    if strategy == "last_stage" and node == "Stage 4":
        return "Trainable", "trainable"
    return "Frozen", "frozen"


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    with (repository / "results" / "selection_lock.json").open(encoding="utf-8") as handle:
        lock = json.load(handle)
    parameters = pd.read_csv(repository / "results" / "model_parameter_summary.csv").set_index(
        "model_kind"
    )
    assets = repository / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    conv = lock["selected"]["convnext_small"]
    conv_cfg = conv["config"]
    conv_strategy = conv_cfg["unfreeze_strategy"]
    conv_nodes = [
        ("Input", "224 × 224 RGB", "Deterministic manifest", "input"),
        ("Augmentation", conv_cfg["augmentation_strength"], "Training only", "input"),
    ]
    for stage, shape in (
        ("Stem", "56 × 56 × 96"),
        ("Stage 1", "56 × 56 × 96"),
        ("Stage 2", "28 × 28 × 192"),
        ("Stage 3", "14 × 14 × 384"),
        ("Stage 4", "7 × 7 × 768"),
    ):
        status, color = trainability(conv_strategy, stage)
        conv_nodes.append((stage, shape, status, color))
    conv_nodes.extend(
        [
            ("Pooled feature", "768 dimensions", "Embedding output", "feature"),
            ("Dropout + head", f"p={conv_cfg['dropout']:.2f} → 3 logits", "Trainable", "head"),
            ("Softmax", "3 probabilities", "Inference", "output"),
        ]
    )
    conv_params = parameters.loc["convnext_small"]
    render(
        assets / "convnext_architecture",
        "ConvNeXt-Small transfer-learning path",
        f"Locked strategy: {conv_strategy} | trainable parameters: {conv_params.trainable_percent:.1f}%",
        conv_nodes,
    )

    dino = lock["selected"]["dinov2_small"]
    dino_cfg = dino["config"]
    dino_strategy = dino_cfg["unfreeze_strategy"]
    if dino_strategy == "full_backbone":
        lower_status = upper_status = ("Trainable", "trainable")
        upper_detail = "Blocks 11–12"
    elif dino_strategy == "top_blocks":
        top_n = int(dino_cfg["top_n_blocks"])
        lower_status = ("Frozen; norms trainable", "frozen")
        upper_status = ("Trainable", "trainable")
        upper_detail = f"Top {top_n} blocks"
    else:
        lower_status = upper_status = ("Frozen", "frozen")
        upper_detail = "Blocks 11–12"
    dino_nodes = [
        ("Input", "224 × 224 RGB", "Deterministic manifest", "input"),
        ("Augmentation", dino_cfg["augmentation_strength"], "Training only", "input"),
        ("Patch tokens", "14 × 14 patches", lower_status[0], lower_status[1]),
        ("Transformer", "Lower encoder blocks", lower_status[0], lower_status[1]),
        ("Transformer", upper_detail, upper_status[0], upper_status[1]),
        ("CLS feature", "384 dimensions", "Embedding output", "feature"),
        ("Dropout + head", f"p={dino_cfg['dropout']:.2f} → 3 logits", "Trainable", "head"),
        ("Softmax", "3 probabilities", "Inference", "output"),
    ]
    dino_params = parameters.loc["dinov2_small"]
    render(
        assets / "dinov2_architecture",
        "DINOv2-Small transfer-learning path",
        f"Locked strategy: {dino_strategy} | trainable parameters: {dino_params.trainable_percent:.1f}%",
        dino_nodes,
    )
    print(f"Wrote architecture diagrams to {assets}")


if __name__ == "__main__":
    main()
