"""Analyze scale, context, crowding, and source-tag mechanisms on V-COCO validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from hac.polar import sha256_file

CLASS_NAMES = ("sitting", "standing", "walking_running")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="name=npz_path:key",
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--tight", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_prediction_specs(values: list[str]) -> dict[str, tuple[Path, str]]:
    output = {}
    for value in values:
        if "=" not in value or ":" not in value.split("=", 1)[1]:
            raise ValueError("Predictions must use name=npz_path:key syntax")
        name, location = value.split("=", 1)
        path, key = location.rsplit(":", 1)
        if not name or name in output:
            raise ValueError(f"Invalid or duplicate prediction name: {name!r}")
        output[name] = (Path(path).resolve(), key)
    return output


def aligned_probabilities(path: Path, key: str, person_ids: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=True) as payload:
        if key not in payload.files or "person_ids" not in payload.files:
            raise RuntimeError(f"Prediction artifact is missing {key!r} or person_ids: {path}")
        index = {str(value): row for row, value in enumerate(payload["person_ids"])}
        try:
            order = np.asarray([index[value] for value in person_ids], dtype=int)
        except KeyError as error:
            raise RuntimeError(f"Prediction artifact lacks validation person {error}: {path}") from error
        values = np.asarray(payload[key][order], dtype=np.float64)
    if values.shape != (len(person_ids), len(CLASS_NAMES)) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid aligned probabilities: {path}:{key}")
    return values / values.sum(axis=1, keepdims=True)


def fixed_class_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
        zero_division=0,
    )
    output = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1_fixed_classes": float(f1.mean()),
    }
    for index, name in enumerate(CLASS_NAMES):
        output[f"{name}_precision"] = float(precision[index])
        output[f"{name}_recall"] = float(recall[index])
        output[f"{name}_f1"] = float(f1[index])
        output[f"{name}_support"] = int(support[index])
    return output


def add_strata(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    output["area_quartile"] = pd.qcut(
        output["bbox_area_fraction"], 4, labels=["Q1_small", "Q2", "Q3", "Q4_large"]
    ).astype(str)
    output["height_quartile"] = pd.qcut(
        output["person_pixel_height"], 4, labels=["Q1_short", "Q2", "Q3", "Q4_tall"]
    ).astype(str)
    occupancy = output.groupby("image_id")["person_id"].transform("size")
    output["scene_occupancy"] = np.where(occupancy.eq(1), "single_person", "multiple_people")
    truncated = (
        output["bbox_xmin"].le(1.0)
        | output["bbox_ymin"].le(1.0)
        | output["bbox_xmax"].ge(output["actual_width"] - 1.0)
        | output["bbox_ymax"].ge(output["actual_height"] - 1.0)
    )
    output["boundary_status"] = np.where(truncated, "touches_boundary", "interior_box")
    actions = output["source_actions"].astype(str)
    output["source_tag_group"] = np.select(
        [
            actions.eq("sit"),
            actions.eq("stand"),
            output["label_3"].eq("walking_running") & actions.str.contains("stand"),
        ],
        ["sit", "stand_only", "locomotion_with_stand_cotag"],
        default="locomotion_without_stand_cotag",
    )
    return output


def main() -> None:
    args = parse_args()
    lock_path = args.protocol_lock.resolve()
    protocol_hash = sha256_file(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("Mechanism analysis requires the locked V-COCO v2 protocol")
    manifest_path = args.val_manifest.resolve()
    if sha256_file(manifest_path) != lock["artifact_sha256"]["vcoco_val_clean.csv"]:
        raise RuntimeError("Locked validation manifest drift")
    rows = add_strata(
        pd.read_csv(manifest_path, dtype={"person_id": str, "image_id": str})
    )
    label_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = rows["label_3"].map(label_to_index).to_numpy(dtype=int)
    person_ids = rows["person_id"].astype(str).tolist()
    specs = parse_prediction_specs(args.prediction)
    probabilities = {
        name: aligned_probabilities(path, key, person_ids)
        for name, (path, key) in specs.items()
    }
    required = {args.reference, args.champion, args.tight, args.context}
    if not required.issubset(probabilities):
        raise RuntimeError(f"Missing declared analysis methods: {sorted(required - probabilities.keys())}")

    predictions = {name: values.argmax(axis=1) for name, values in probabilities.items()}
    stratum_columns = (
        "area_quartile",
        "height_quartile",
        "scene_occupancy",
        "boundary_status",
        "source_tag_group",
    )
    stratum_rows = []
    for column in stratum_columns:
        for value in sorted(rows[column].unique()):
            mask = rows[column].eq(value).to_numpy()
            for name, predicted in predictions.items():
                stratum_rows.append(
                    {
                        "stratum": column,
                        "value": str(value),
                        "method": name,
                        "people": int(mask.sum()),
                        "images": int(rows.loc[mask, "image_id"].nunique()),
                        **fixed_class_metrics(labels[mask], predicted[mask]),
                    }
                )
    stratum_frame = pd.DataFrame(stratum_rows)

    reference_correct = predictions[args.reference] == labels
    champion_correct = predictions[args.champion] == labels
    tight_correct = predictions[args.tight] == labels
    context_correct = predictions[args.context] == labels
    transition_rows = []
    for column in ("overall", *stratum_columns):
        groups = [("all", np.ones(len(rows), dtype=bool))]
        if column != "overall":
            groups = [
                (str(value), rows[column].eq(value).to_numpy())
                for value in sorted(rows[column].unique())
            ]
        for value, mask in groups:
            transition_rows.append(
                {
                    "stratum": column,
                    "value": value,
                    "people": int(mask.sum()),
                    "reference_errors": int((~reference_correct & mask).sum()),
                    "champion_errors": int((~champion_correct & mask).sum()),
                    "reference_to_champion_rescued": int(
                        (~reference_correct & champion_correct & mask).sum()
                    ),
                    "reference_to_champion_harmed": int(
                        (reference_correct & ~champion_correct & mask).sum()
                    ),
                    "context_rescues_tight": int((~tight_correct & context_correct & mask).sum()),
                    "context_harms_tight": int((tight_correct & ~context_correct & mask).sum()),
                }
            )
    transitions = pd.DataFrame(transition_rows)

    correlation_rows = []
    for name, values in probabilities.items():
        confidence = values.max(axis=1)
        correct = (predictions[name] == labels).astype(float)
        for variable in ("bbox_area_fraction", "person_pixel_height"):
            statistic, p_value = spearmanr(np.log(np.clip(rows[variable], 1e-8, None)), correct)
            confidence_statistic, confidence_p = spearmanr(
                np.log(np.clip(rows[variable], 1e-8, None)), confidence
            )
            correlation_rows.append(
                {
                    "method": name,
                    "variable": variable,
                    "correctness_spearman_rho": float(statistic),
                    "correctness_p_value_unadjusted": float(p_value),
                    "confidence_spearman_rho": float(confidence_statistic),
                    "confidence_p_value_unadjusted": float(confidence_p),
                }
            )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stratum_frame.to_csv(output_dir / "mechanism_strata.csv", index=False)
    transitions.to_csv(output_dir / "mechanism_error_transitions.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(output_dir / "mechanism_correlations.csv", index=False)
    result = {
        "status": "VCOCO_V2_DEVELOPMENT_MECHANISM_ANALYSIS_COMPLETE",
        "methods": sorted(probabilities),
        "reference": args.reference,
        "champion": args.champion,
        "tight_view": args.tight,
        "context_view": args.context,
        "correlation_scope": "exploratory_unadjusted_not_causal",
        "protocol_lock_sha256": protocol_hash,
        "test_rows_read": 0,
        "test_predictions_run": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
