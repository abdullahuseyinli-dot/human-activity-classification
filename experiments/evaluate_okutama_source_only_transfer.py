"""Evaluate label-blind source-only predictions on Okutama development metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.vcoco_v3_models import CLASS_NAMES, locomotion_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", type=Path, default=Path("experiments/okutama_temporal_grid.json")
    )
    parser.add_argument(
        "--prediction-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/okutama/source_only/summary.json"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(".runs/vcoco_v3/okutama/source_only/predictions.npz"),
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--target-store", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".runs/vcoco_v3/okutama/source_only/evaluation"),
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def subgroup_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    return {
        "samples": int(len(labels)),
        "accuracy": float((predictions == labels).mean()),
        "macro_f1_all_classes": float(
            f1_score(
                labels,
                predictions,
                labels=np.arange(len(CLASS_NAMES)),
                average="macro",
                zero_division=0,
            )
        ),
        "locomotion_f1": float(
            f1_score(labels == 2, predictions == 2, average="binary", zero_division=0)
        ),
    }


def main() -> None:
    args = parse_args()
    grid_path = args.grid.resolve()
    summary_path = args.prediction_summary.resolve()
    predictions_path = args.predictions.resolve()
    metadata_path = args.metadata.resolve()
    store_path = args.target_store.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    source = json.loads(summary_path.read_text(encoding="utf-8"))
    if source.get("status") != "OKUTAMA_SOURCE_ONLY_TRANSFER_PREDICTIONS_COMPLETE":
        raise RuntimeError("Source-only transfer predictions are incomplete")
    if source.get("target_labels_read") != 0:
        raise RuntimeError("Target labels were read during source-only fitting")
    if source.get("target_partition") != "development":
        raise RuntimeError("Development evaluation requires development source-only predictions")
    if sha256_file(predictions_path) != source["artifact_sha256"][predictions_path.name]:
        raise RuntimeError("Source-only predictions changed after fitting")
    store = json.loads(store_path.read_text(encoding="utf-8"))
    metadata = pd.read_csv(
        metadata_path,
        dtype={
            "sample_id": str,
            "recording_id": str,
            "track_id": str,
            "scenario_id": str,
        },
    )
    with np.load(predictions_path, allow_pickle=False) as payload:
        indices = payload["target_feature_indices"].astype(int)
        probabilities = payload["probabilities"].astype(float)
        class_names = tuple(map(str, payload["class_names"].tolist()))
    if class_names != CLASS_NAMES:
        raise RuntimeError("Source-only prediction classes changed")
    if len(metadata) != int(store["samples"]) or len(probabilities) != len(metadata):
        raise RuntimeError("Okutama metadata and source-only predictions do not align")
    if not np.array_equal(metadata["feature_index"].to_numpy(dtype=int), indices):
        raise RuntimeError("Okutama feature indices are not aligned")
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = metadata["label"].map(mapping).to_numpy(dtype=int)
    metrics = classification_metrics(labels, probabilities)
    metrics["locomotion_f1"] = locomotion_f1(labels, probabilities)
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "class": CLASS_NAMES,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )

    subgroup_rows = []
    minimum_rows = int(grid["subgroup_definitions"]["minimum_reported_rows"])
    subgroup_values: dict[str, pd.Series] = {
        "walking_running_subtype": metadata["walking_running_subtype"].fillna("not_applicable"),
        "transition_window": metadata["transition_window"].astype(str).str.lower(),
        "center_occluded": metadata["center_occluded"].astype(str).str.lower(),
        "window_any_occluded": metadata["window_any_occluded"].astype(str).str.lower(),
        "drone_view": metadata["drone_view"].astype(str),
        "part_of_day": metadata["part_of_day"].astype(str),
        "scenario": metadata["recording_id"].astype(str),
    }
    subgroup_values["person_scale"] = pd.cut(
        metadata["bbox_area_fraction"].astype(float),
        bins=grid["subgroup_definitions"]["person_scale_area_fraction_edges"],
        labels=grid["subgroup_definitions"]["person_scale_names"],
        include_lowest=True,
        right=False,
    ).astype(str)
    for axis, values in subgroup_values.items():
        for value in sorted(values.unique()):
            mask = values.eq(value).to_numpy()
            if int(mask.sum()) < minimum_rows:
                continue
            subgroup_rows.append(
                {
                    "axis": axis,
                    "value": str(value),
                    **subgroup_metrics(labels[mask], probabilities[mask]),
                }
            )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    per_class_path = output_dir / "per_class.csv"
    subgroups_path = output_dir / "subgroups.csv"
    evaluated_path = output_dir / "predictions.csv"
    metric_rows = [{"regime": "source_only_static", "scope": "all_development", **metrics}]
    if "split" in metadata:
        for split in ("train", "validation", "calibration"):
            mask = metadata["split"].eq(split).to_numpy()
            if not np.any(mask):
                continue
            split_metrics = classification_metrics(labels[mask], probabilities[mask])
            split_metrics["locomotion_f1"] = locomotion_f1(
                labels[mask], probabilities[mask]
            )
            metric_rows.append(
                {"regime": "source_only_static", "scope": split, **split_metrics}
            )
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    per_class.to_csv(per_class_path, index=False)
    pd.DataFrame(subgroup_rows).to_csv(subgroups_path, index=False)
    pd.DataFrame(
        {
            "sample_id": metadata["sample_id"],
            "recording_id": metadata["recording_id"],
            "label": metadata["label"],
            "prediction": np.asarray(CLASS_NAMES)[predictions],
            **{
                f"probability_{name}": probabilities[:, index]
                for index, name in enumerate(CLASS_NAMES)
            },
        }
    ).to_csv(evaluated_path, index=False)
    summary = {
        "status": "OKUTAMA_SOURCE_ONLY_TRANSFER_EVALUATED",
        "regime": "source_only_static",
        "target_labels_used_for_fit_or_selection": False,
        "samples": len(metadata),
        "scenarios": int(metadata["recording_id"].nunique()),
        "metrics": metrics,
        "source_prediction_summary_sha256": sha256_file(summary_path),
        "source_sha256": {
            "grid": sha256_file(grid_path),
            "metadata": sha256_file(metadata_path),
            "target_store": sha256_file(store_path),
            "predictions": sha256_file(predictions_path),
        },
        "artifact_sha256": {
            path.name: sha256_file(path)
            for path in (metrics_path, per_class_path, subgroups_path, evaluated_path)
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
