"""Render portfolio figures from completed, immutable faithfulness evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hac.augmentations import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform  # noqa: E402
from hac.protocol import load_and_validate_manifest  # noqa: E402

MODEL_DISPLAY = {
    "convnext_small": "ConvNeXt-Small",
    "dinov2_small": "DINOv2-Small",
    "probability_blend": "0.1 ConvNeXt + 0.9 DINOv2",
}
METHOD_DISPLAY = {
    "gradcam": "Grad-CAM",
    "hirescam": "HiResCAM",
    "integrated_gradients": "Integrated gradients",
    "attention_rollout": "Raw attention rollout (diagnostic)",
    "gradient_attention_rollout": "Gradient-attention rollout",
    "weighted_hirescam+gradient_attention_rollout": "Weighted family attribution",
}
CLASS_DISPLAY = {
    0: "Sitting",
    1: "Standing",
    2: "Walking/running",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def display_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        tensor = build_eval_transform()(image.convert("RGB"))
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[None, None]
    values = tensor.permute(1, 2, 0).numpy()
    return np.clip(values * std + mean, 0.0, 1.0)


def render_selection(evidence_dir: Path) -> list[Path]:
    frame = pd.read_csv(evidence_dir / "faithfulness_method_selection.csv")
    frame["label"] = frame.apply(
        lambda row: (
            f"{MODEL_DISPLAY[str(row['family'])]} — "
            f"{METHOD_DISPLAY.get(str(row['method']), str(row['method']))}"
        ),
        axis=1,
    )
    frame = frame.sort_values(["family", "road_combined_mean"], ascending=[True, True])
    colors = ["#0F766E" if bool(value) else "#94A3B8" for value in frame["selected"]]
    figure, axis = plt.subplots(figsize=(11.2, 6.1))
    axis.barh(frame["label"], frame["road_combined_mean"], color=colors)
    axis.set(
        xlabel="OOF ROADCombined (higher is better)",
        ylabel="",
        title="Attribution method selection before test evaluation",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", alpha=0.18)
    figure.text(
        0.99,
        0.015,
        "Teal: selected · raw attention rollout is diagnostic only",
        ha="right",
        fontsize=9,
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    outputs = []
    for suffix in ("png", "svg"):
        path = evidence_dir / f"faithfulness_method_selection.{suffix}"
        figure.savefig(path, dpi=190 if suffix == "png" else None, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    return outputs


def render_curves(evidence_dir: Path) -> list[Path]:
    curves = pd.read_csv(evidence_dir / "faithfulness_test_curves.csv")
    styles = {
        "road_most": ("#DC2626", "ROAD: most relevant"),
        "road_least": ("#2563EB", "ROAD: least relevant"),
        "road_random": ("#64748B", "ROAD: random"),
        "deletion": ("#EA580C", "Blur deletion"),
        "insertion": ("#16A34A", "Blur insertion"),
    }
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
    for axis, model in zip(
        axes, ("convnext_small", "dinov2_small", "probability_blend"), strict=True
    ):
        subset = curves[curves["model"].eq(model)]
        for metric, (color, label) in styles.items():
            rows = subset[subset["metric"].eq(metric)].sort_values("fraction")
            axis.plot(
                rows["fraction"],
                rows["mean"],
                marker="o",
                linewidth=2,
                markersize=4,
                color=color,
                label=label,
            )
        axis.set(
            title=MODEL_DISPLAY[model],
            xlabel="Fraction of 14×14 patches perturbed or revealed",
        )
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Locked predicted-class probability")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    figure.suptitle("Locked-test attribution perturbation curves", y=1.02, fontsize=16)
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    outputs = []
    for suffix in ("png", "svg"):
        path = evidence_dir / f"faithfulness_perturbation_curves.{suffix}"
        figure.savefig(path, dpi=190 if suffix == "png" else None, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    return outputs


def gallery_rows(rows: pd.DataFrame) -> pd.DataFrame:
    correct = rows[rows["correct"].astype(bool)].sort_values(
        "confidence", ascending=False
    ).head(2)
    incorrect = rows[~rows["correct"].astype(bool)].sort_values(
        "confidence", ascending=False
    ).head(2)
    chosen = pd.concat([correct, incorrect])
    if len(chosen) < 4:
        extras = rows[~rows["image_id"].isin(chosen["image_id"])].sort_values(
            "confidence"
        )
        chosen = pd.concat([chosen, extras.head(4 - len(chosen))])
    return chosen


def render_galleries(evidence_dir: Path, manifest: pd.DataFrame) -> list[Path]:
    per_image = pd.read_csv(
        evidence_dir / "faithfulness_test_per_image.csv", dtype={"image_id": str}
    )
    manifest = manifest.copy()
    manifest["image_id"] = manifest["image_id"].astype(str)
    manifest_by_id = manifest.set_index("image_id")
    archive = np.load(evidence_dir / "faithfulness_test_maps.npz")
    outputs = []
    try:
        for model in ("convnext_small", "dinov2_small", "probability_blend"):
            chosen = gallery_rows(per_image[per_image["model"].eq(model)])
            figure, axes = plt.subplots(len(chosen), 2, figsize=(8.8, 3.9 * len(chosen)))
            if len(chosen) == 1:
                axes = np.asarray([axes])
            for row_index, record in enumerate(chosen.itertuples(index=False)):
                image_id = str(record.image_id)
                image_path = Path(manifest_by_id.loc[image_id, "resolved_image_path"])
                image = display_image(image_path)
                attribution = archive[f"{model}__{image_id}"]
                heatmap = attribution / max(float(attribution.max()), 1e-12)
                true_name = CLASS_DISPLAY[int(record.true_class)]
                target_name = CLASS_DISPLAY[int(record.target_class)]
                method_name = METHOD_DISPLAY.get(str(record.method), str(record.method))

                axes[row_index, 0].imshow(image)
                axes[row_index, 0].axis("off")
                axes[row_index, 0].set_title(
                    f"True: {true_name} · Predicted: {target_name}", fontsize=10.5
                )
                axes[row_index, 1].imshow(image)
                axes[row_index, 1].imshow(
                    heatmap, cmap="turbo", alpha=0.48, vmin=0.0, vmax=1.0
                )
                axes[row_index, 1].axis("off")
                axes[row_index, 1].set_title(
                    f"{method_name} · p={float(record.confidence):.3f}", fontsize=10.5
                )
            figure.suptitle(
                f"{MODEL_DISPLAY[model]} — locked-test attributions",
                fontsize=15,
                y=0.998,
            )
            figure.tight_layout(rect=(0, 0, 1, 0.988))
            path = evidence_dir / f"{model}_faithfulness_gallery.jpg"
            figure.savefig(
                path,
                dpi=145,
                format="jpeg",
                bbox_inches="tight",
                pil_kwargs={"quality": 90, "optimize": True},
            )
            plt.close(figure)
            outputs.append(path)
    finally:
        archive.close()
    return outputs


def refresh_provenance(evidence_dir: Path, outputs: list[Path]) -> None:
    provenance_path = evidence_dir / "faithfulness_provenance.json"
    with provenance_path.open(encoding="utf-8") as handle:
        provenance = json.load(handle)
    renderer = Path(__file__).resolve()
    provenance["rendering"] = {
        "status": "METRICS_UNCHANGED_PRESENTATION_RERENDER",
        "renderer": renderer.name,
        "renderer_sha256": sha256_file(renderer),
        "outputs": [path.name for path in outputs],
    }
    evidence_files = [
        path
        for path in evidence_dir.iterdir()
        if path.is_file()
        and path.suffix.casefold() in {".csv", ".jpg", ".json", ".png", ".svg"}
        and path.name != provenance_path.name
    ]
    provenance["evidence"] = {
        path.name: sha256_file(path) for path in sorted(evidence_files)
    }
    write_json(provenance_path, provenance)


def main() -> None:
    args = parse_args()
    evidence_dir = args.evidence_dir.resolve()
    manifest, _ = load_and_validate_manifest(args.manifest.resolve(), require_images=True)
    required = (
        "faithfulness_method_selection.csv",
        "faithfulness_test_curves.csv",
        "faithfulness_test_per_image.csv",
        "faithfulness_test_maps.npz",
        "faithfulness_provenance.json",
    )
    missing = [name for name in required if not (evidence_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Faithfulness rendering evidence is incomplete: {missing}")

    outputs = [
        *render_selection(evidence_dir),
        *render_curves(evidence_dir),
        *render_galleries(evidence_dir, manifest),
    ]
    refresh_provenance(evidence_dir, outputs)
    print(f"Rendered {len(outputs)} portfolio figures in {evidence_dir}")


if __name__ == "__main__":
    main()
