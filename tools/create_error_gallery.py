"""Create a compact post-hoc gallery of champion test errors."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

CLASS_NAMES = ["sitting", "standing", "walking/running"]
TENSOR_ID = re.compile(r"^tensor\(([^)]+)\)$")


def normalize_image_id(value: object) -> str:
    text = str(value).strip()
    match = TENSOR_ID.fullmatch(text)
    return match.group(1).strip() if match else text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--maximum-images", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    manifest = pd.read_csv(repository / "data" / "manifest.csv", dtype={"image_id": str})
    predictions = pd.read_csv(
        repository / "results" / "champion_test_predictions.csv", dtype={"image_id": str}
    )
    manifest["image_id"] = manifest["image_id"].map(normalize_image_id)
    predictions["image_id"] = predictions["image_id"].map(normalize_image_id)
    errors = predictions[predictions["y_true"] != predictions["y_pred"]].copy()
    errors = errors.merge(
        manifest[["image_id", "image_path", "label", "image_url"]],
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    if errors["image_path"].isna().any():
        missing = errors.loc[errors["image_path"].isna(), "image_id"].tolist()
        raise RuntimeError(f"Prediction IDs are absent from the manifest: {missing}")
    errors["true_class"] = errors["y_true"].map(dict(enumerate(CLASS_NAMES)))
    errors["predicted_class"] = errors["y_pred"].map(dict(enumerate(CLASS_NAMES)))
    errors = errors.sort_values("confidence", ascending=False).reset_index(drop=True)
    errors.drop(columns=["image_path"]).to_csv(
        repository / "results" / "champion_error_analysis.csv", index=False
    )

    shown = errors.head(int(args.maximum_images))
    if shown.empty:
        raise RuntimeError("The locked champion has no test errors to display.")
    if len(shown) <= 3:
        columns = len(shown)
    elif len(shown) == 4:
        columns = 2
    else:
        columns = 3
    rows = math.ceil(len(shown) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.6 * rows))
    axes = [axes] if rows == columns == 1 else list(getattr(axes, "flat", axes))
    for axis, record in zip(axes, shown.itertuples(index=False), strict=False):
        image_path = (repository / record.image_path).resolve()
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(
            f"True: {record.true_class}\nPred: {record.predicted_class} ({record.confidence:.2f})",
            fontsize=10,
        )
        axis.axis("off")
    for axis in axes[len(shown) :]:
        axis.axis("off")
    fig.suptitle("Highest-confidence champion errors", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(repository / "assets" / "champion_error_gallery.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {len(errors)} champion errors; displayed {len(shown)}")


if __name__ == "__main__":
    main()
