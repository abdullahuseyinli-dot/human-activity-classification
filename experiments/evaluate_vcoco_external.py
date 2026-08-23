"""Evaluate the locked POLAR system on the independent V-COCO train/validation cohort."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evaluate_polar_final import (
    evaluate_neural_component,
    evaluate_probe_components,
    verify_final_fits,
)

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_analysis import (
    confusion_metrics,
    per_class_metrics,
    stratified_paired_bootstrap,
)
from hac.polar_training import TASK_LABELS, normalize_probability_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_external_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "person_id",
        "image_id",
        "image_path",
        "label_3",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "image_level_unambiguous",
        "eligible_person",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"External manifest is missing columns: {sorted(missing)}")
    output = frame.copy()
    output["person_id"] = output["person_id"].astype(str)
    output["image_id"] = output["image_id"].astype(str)
    if output["person_id"].duplicated().any():
        raise ValueError("External person identifiers must be unique")
    if not output["eligible_person"].astype(str).str.lower().eq("true").all():
        raise ValueError("External evaluation manifest contains an ineligible person")
    observed = set(output["label_3"].astype(str))
    if observed != set(TASK_LABELS["label_3"]):
        raise ValueError(f"Unexpected external labels: {sorted(observed)}")
    return output.sort_values("person_id", ignore_index=True)


def collapse_to_three(probabilities: np.ndarray) -> np.ndarray:
    return normalize_probability_rows(
        np.column_stack(
            [probabilities[:, 0], probabilities[:, 1], probabilities[:, 2] + probabilities[:, 3]]
        )
    )


def image_level_arrays(
    frame: pd.DataFrame, probabilities: np.ndarray, label_to_index: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible = frame["image_level_unambiguous"].astype(str).str.lower().eq("true").to_numpy()
    rows = frame[eligible].reset_index(drop=True)
    values = probabilities[eligible]
    image_ids = []
    labels = []
    averaged = []
    for image_id, indices in rows.groupby("image_id", sort=True).indices.items():
        group = rows.iloc[indices]
        group_labels = set(group["label_3"].astype(str))
        if len(group_labels) != 1:
            raise RuntimeError(f"Image-level V-COCO label is ambiguous: {image_id}")
        image_ids.append(str(image_id))
        labels.append(label_to_index[next(iter(group_labels))])
        averaged.append(values[indices].mean(axis=0))
    return (
        np.asarray(image_ids),
        np.asarray(labels, dtype=int),
        normalize_probability_rows(np.asarray(averaged)),
    )


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    lock_path = args.selection_lock.resolve()
    lock_hash = sha256_file(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "FINAL_SELECTION_LOCKED_PRE_TEST":
        raise RuntimeError("External evaluation requires the immutable final selection lock")
    if lock["external_validation"]["dataset"] != "V-COCO trainval":
        raise RuntimeError("Selection lock does not authorize this V-COCO cohort")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("status") == "LOCKED_EXTERNAL_EVALUATION_COMPLETE" and existing.get(
            "selection_lock_sha256"
        ) == lock_hash:
            print(json.dumps(existing, indent=2, sort_keys=True), flush=True)
            return

    final_root = args.final_root.resolve()
    resolved = verify_final_fits(lock, final_root, lock_hash)
    manifest_path = args.manifest.resolve()
    frame = validate_external_manifest(
        pd.read_csv(manifest_path, dtype={"person_id": str, "image_id": str})
    )
    expected_manifest_hash = lock["external_validation"]["manifest_sha256"]
    if sha256_file(manifest_path) != expected_manifest_hash:
        raise RuntimeError("V-COCO clean manifest differs from the final selection lock")

    inference_frame = frame.copy()
    inference_frame["coco_image_id"] = inference_frame["image_id"]
    inference_frame["image_id"] = inference_frame["person_id"]
    inference_frame["split"] = "external"
    inference_frame["label_4"] = inference_frame["label_3"].map(
        {
            "sitting": "sitting",
            "standing": "standing",
            "walking_running": "walking",
        }
    )

    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    four_class_components = {}
    for model_id, specification in lock["final_neural_fits"].items():
        probabilities, _ = evaluate_neural_component(
            model_id,
            specification,
            resolved["neural"][model_id],
            inference_frame,
            device,
        )
        four_class_components[model_id] = probabilities
    probe_components, direct_three, direct_three_names = evaluate_probe_components(
        lock, resolved["probes"], inference_frame, device
    )
    four_class_components.update(probe_components)
    weights = lock["ensemble"]["weights"]
    if set(weights) != set(four_class_components):
        raise RuntimeError("External component predictions differ from the locked ensemble")
    ensemble_four = normalize_probability_rows(
        sum(float(weights[name]) * four_class_components[name] for name in weights)
    )
    person_candidates = {
        **{name: collapse_to_three(values) for name, values in four_class_components.items()},
        "locked_ensemble_collapsed": collapse_to_three(ensemble_four),
        "direct_three_class_probe": direct_three,
    }
    class_names = list(TASK_LABELS["label_3"])
    if direct_three_names != class_names:
        raise RuntimeError("Direct three-class probe class order differs")
    label_to_index = {name: index for index, name in enumerate(class_names)}
    person_labels = frame["label_3"].map(label_to_index).to_numpy(dtype=int)

    person_rows = []
    image_rows = []
    per_class_rows = []
    confusions = {"person": {}, "image": {}}
    uncertainty = {}
    image_predictions = {}
    image_ids_reference = None
    image_labels_reference = None
    for name, probabilities in person_candidates.items():
        person_metrics = classification_metrics(person_labels, probabilities)
        person_rows.append({"candidate": name, **person_metrics})
        per_class_rows.extend(
            {"evaluation_unit": "person", "candidate": name, **record}
            for record in per_class_metrics(person_labels, probabilities, class_names)
        )
        confusions["person"][name] = confusion_metrics(
            person_labels, probabilities, class_names
        )
        image_ids, image_labels, image_probabilities = image_level_arrays(
            frame, probabilities, label_to_index
        )
        if image_ids_reference is None:
            image_ids_reference = image_ids
            image_labels_reference = image_labels
        elif not np.array_equal(image_ids, image_ids_reference) or not np.array_equal(
            image_labels, image_labels_reference
        ):
            raise RuntimeError("External image-level cohorts differ by candidate")
        image_predictions[name] = image_probabilities
        image_metrics = classification_metrics(image_labels, image_probabilities)
        image_rows.append({"candidate": name, **image_metrics})
        per_class_rows.extend(
            {"evaluation_unit": "image", "candidate": name, **record}
            for record in per_class_metrics(image_labels, image_probabilities, class_names)
        )
        confusions["image"][name] = confusion_metrics(
            image_labels, image_probabilities, class_names
        )
        uncertainty[name] = stratified_paired_bootstrap(
            image_labels,
            image_probabilities,
            resamples=lock["evaluation"]["bootstrap_resamples"],
            seed=lock["evaluation"]["bootstrap_seed"],
        )

    person_frame = pd.DataFrame(person_rows).sort_values(
        ["macro_f1", "log_loss", "candidate"], ascending=[False, True, True], ignore_index=True
    )
    image_frame = pd.DataFrame(image_rows).sort_values(
        ["macro_f1", "log_loss", "candidate"], ascending=[False, True, True], ignore_index=True
    )
    person_frame.to_csv(output_dir / "external_person_metrics.csv", index=False)
    image_frame.to_csv(output_dir / "external_image_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).sort_values(
        ["evaluation_unit", "candidate", "class"], ignore_index=True
    ).to_csv(output_dir / "external_per_class.csv", index=False)
    write_json(output_dir / "external_confusions.json", confusions)
    write_json(output_dir / "external_image_uncertainty.json", uncertainty)
    np.savez_compressed(
        output_dir / "external_predictions.npz",
        **{f"person_{name}": value for name, value in person_candidates.items()},
        **{f"image_{name}": value for name, value in image_predictions.items()},
        person_labels=person_labels,
        person_ids=frame["person_id"].to_numpy(),
        image_labels=image_labels_reference,
        image_ids=image_ids_reference,
        class_names=np.asarray(class_names),
    )
    summary = {
        "status": "LOCKED_EXTERNAL_EVALUATION_COMPLETE",
        "selection_role": "none",
        "selection_lock_sha256": lock_hash,
        "manifest_sha256": expected_manifest_hash,
        "person_rows": len(frame),
        "unique_images": frame["image_id"].nunique(),
        "image_level_rows": len(image_ids_reference),
        "best_person_metrics": person_frame.iloc[0].to_dict(),
        "best_image_metrics": image_frame.iloc[0].to_dict(),
        "ensemble_weights": weights,
        "predictions_sha256": sha256_file(output_dir / "external_predictions.npz"),
        "runtime_seconds": time.perf_counter() - started,
        "polar_test_rows_read": 0,
        "test_used_for_selection": False,
    }
    write_json(summary_path, summary)
    print(person_frame.to_string(index=False), flush=True)
    print(image_frame.to_string(index=False), flush=True)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
