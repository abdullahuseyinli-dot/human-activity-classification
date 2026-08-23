"""Run the one authorized V-COCO v2 official-test comparison after final lock."""

from __future__ import annotations

import argparse
import gc
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from analyze_vcoco_v2_mechanisms import add_strata, fixed_class_metrics
from evaluate_polar_final import model_config_from_final
from evaluate_vcoco_v2_factorized_fusion import geometry_features
from torch import nn

from hac.augmentations import build_eval_transform
from hac.data import make_loader
from hac.metrics import classification_metrics, selective_classification_metrics
from hac.polar import sha256_file
from hac.polar_analysis import confusion_metrics, per_class_metrics
from hac.polar_models import build_polar_model
from hac.polar_training import TASK_LABELS, evaluate_classifier, normalize_probability_rows
from hac.transfer import image_cluster_paired_bootstrap

CLASS_NAMES = ("sitting", "standing", "walking_running")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--final-fit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
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
        return float(value)
    return value


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def open_test_manifest_once(
    manifest_path: Path,
    output_dir: Path,
    selection: dict,
    selection_hash: str,
) -> tuple[pd.DataFrame, dict]:
    gate_path = output_dir / "test_access_gate.json"
    cached_path = output_dir / "opened_test_manifest.csv"
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if (
            gate.get("status") != "VCOCO_V2_OFFICIAL_TEST_GATE_OPEN"
            or gate.get("selection_lock_sha256") != selection_hash
            or gate.get("official_test_label_open_count") != 1
            or sha256_file(cached_path) != gate.get("opened_manifest_sha256")
        ):
            raise RuntimeError("Existing official-test access gate is invalid")
        return pd.read_csv(cached_path, dtype={"person_id": str, "image_id": str}), gate
    encoded = manifest_path.read_bytes()
    digest = sha256_file(manifest_path)
    if digest != selection["final_test"]["manifest_sha256"]:
        raise RuntimeError("Official-test manifest drift")
    frame = pd.read_csv(io.BytesIO(encoded), dtype={"person_id": str, "image_id": str})
    if len(frame) != selection["final_test"]["expected_people"]:
        raise RuntimeError("Official-test person count drift")
    temporary = cached_path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(cached_path)
    gate = {
        "status": "VCOCO_V2_OFFICIAL_TEST_GATE_OPEN",
        "selection_lock_sha256": selection_hash,
        "test_manifest_sha256": digest,
        "opened_manifest_sha256": sha256_file(cached_path),
        "official_test_label_open_count": 1,
        "test_rows_read": len(frame),
    }
    write_json(gate_path, gate)
    return frame, gate


def collapse_to_three(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Four-class probabilities must have shape (rows, 4)")
    return normalize_probability_rows(
        np.column_stack(
            [values[:, 0], values[:, 1], values[:, 2] + values[:, 3]]
        )
    )


def historical_baseline(
    selection: dict,
    frame: pd.DataFrame,
    device: torch.device,
    *,
    batch_size: int,
    workers: int,
) -> tuple[np.ndarray, list[dict]]:
    inference = frame.copy()
    inference["image_id"] = inference["person_id"].astype(str)
    inference["label_4"] = inference["label_4"].fillna("").replace("", "walking")
    inference["label"] = inference["label_4"]
    class_names = list(TASK_LABELS["label_4"])
    class_to_index = {name: index for index, name in enumerate(class_names)}
    loader = make_loader(
        inference,
        class_to_index,
        build_eval_transform(),
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        workers=workers,
        view="person_context_25",
    )
    criterion = nn.CrossEntropyLoss()
    predictions = []
    checkpoint_rows = []
    reference_configuration = None
    for record in selection["historical_baseline"]["checkpoints"]:
        checkpoint_path = Path(record["path"])
        if sha256_file(checkpoint_path) != record["sha256"]:
            raise RuntimeError("Historical baseline checkpoint drift")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        configuration = payload["configuration"]
        if reference_configuration is None:
            reference_configuration = configuration
        elif configuration != reference_configuration:
            raise RuntimeError("Historical baseline checkpoint configurations differ")
        model = build_polar_model(
            model_config_from_final(configuration), num_classes=len(class_names), pretrained=False
        ).to(device)
        model.load_state_dict(payload["model_state_dict"])
        evaluation = evaluate_classifier(model, loader, criterion, device)
        if evaluation["image_ids"] != inference["person_id"].astype(str).tolist():
            raise RuntimeError("Historical baseline prediction order drift")
        predictions.append(collapse_to_three(evaluation["probabilities"]))
        checkpoint_rows.append(
            {
                "path": str(checkpoint_path),
                "sha256": record["sha256"],
                "epoch": int(payload["epoch"]),
            }
        )
        del model, payload, evaluation
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return normalize_probability_rows(np.mean(predictions, axis=0)), checkpoint_rows


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid loader settings")
    protocol_path = args.protocol_lock.resolve()
    selection_path = args.selection_lock.resolve()
    manifest_path = args.test_manifest.resolve()
    protocol_hash = sha256_file(protocol_path)
    selection_hash = sha256_file(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection.get("status") != "VCOCO_V2_FINAL_SELECTION_LOCKED_PRE_TEST"
        or selection.get("protocol_lock_sha256") != protocol_hash
    ):
        raise RuntimeError("Official-test evaluation requires the matching final lock")
    if selection["test_access"]["model_predictions_before_lock"]:
        raise RuntimeError("Selection lock reports prior official-test predictions")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "VCOCO_V2_OFFICIAL_TEST_EVALUATION_COMPLETE"
            and existing.get("selection_lock_sha256") == selection_hash
        ):
            print(json.dumps(existing, indent=2, sort_keys=True), flush=True)
            return
        raise RuntimeError("Existing test output belongs to another final selection")

    feature_dir = args.test_features.resolve()
    feature_summary = json.loads((feature_dir / "summary.json").read_text(encoding="utf-8"))
    if (
        feature_summary.get("status") != "VCOCO_V2_LOCKED_TEST_FEATURES_COMPLETE"
        or feature_summary.get("selection_lock_sha256") != selection_hash
        or feature_summary.get("test_label_columns_read") != 0
    ):
        raise RuntimeError("Locked official-test features are invalid")
    rows_path = feature_dir / "rows.csv"
    tight_path = feature_dir / "tight_features.npy"
    context_path = feature_dir / "context_features.npy"
    for path in (rows_path, tight_path, context_path):
        if sha256_file(path) != feature_summary["artifact_sha256"][path.name]:
            raise RuntimeError(f"Official-test feature artifact drift: {path}")
    feature_rows = pd.read_csv(rows_path, dtype={"person_id": str, "image_id": str})
    tight = np.asarray(np.load(tight_path, mmap_mode="r"), dtype=np.float32)
    context = np.asarray(np.load(context_path, mmap_mode="r"), dtype=np.float32)
    if len(feature_rows) != len(tight) or tight.shape != context.shape:
        raise RuntimeError("Official-test feature arrays do not align")

    final_fit_path = args.final_fit.resolve()
    if sha256_file(final_fit_path) != selection["final_fit"]["sha256"]:
        raise RuntimeError("Final stack artifact drift")
    artifact = joblib.load(final_fit_path)
    if (
        artifact.get("status") != "VCOCO_V2_FINAL_DEVELOPMENT_STACK_FIT"
        or artifact.get("fit_scope") != "locked_train_plus_validation_after_selection"
        or artifact.get("development_people") != selection["final_fit"]["training_people"]
    ):
        raise RuntimeError("Unexpected final stack artifact")
    tight_probabilities = normalize_probability_rows(artifact["tight_model"].predict_proba(tight))
    context_probabilities = normalize_probability_rows(
        artifact["context_model"].predict_proba(context)
    )
    geometry = geometry_features(feature_rows)
    stack_features = np.concatenate(
        [
            np.log(np.clip(tight_probabilities, 1e-8, 1.0)),
            np.log(np.clip(context_probabilities, 1e-8, 1.0)),
            geometry,
        ],
        axis=1,
    )
    champion = normalize_probability_rows(artifact["stacker"].predict_proba(stack_features))

    frame, gate = open_test_manifest_once(manifest_path, output_dir, selection, selection_hash)
    if not frame["person_id"].astype(str).equals(feature_rows["person_id"].astype(str)):
        raise RuntimeError("Official-test label and feature person order differ")
    if not frame["image_id"].astype(str).equals(feature_rows["image_id"].astype(str)):
        raise RuntimeError("Official-test label and feature image order differ")
    label_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = frame["label_3"].map(label_to_index)
    if labels.isna().any():
        raise RuntimeError("Official-test manifest contains an unknown class")
    labels_array = labels.to_numpy(dtype=int)
    image_ids = frame["image_id"].astype(str).to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline, baseline_checkpoints = historical_baseline(
        selection,
        frame,
        device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    methods = {
        "scale_conditioned_stacking": champion,
        "historical_v1_dino": baseline,
    }
    metric_rows = []
    class_rows = []
    selective = {}
    confusions = {}
    for name, probabilities in methods.items():
        metrics = classification_metrics(labels_array, probabilities)
        metric_rows.append({"method": name, **metrics})
        class_rows.extend(
            {"method": name, **row}
            for row in per_class_metrics(labels_array, probabilities, CLASS_NAMES)
        )
        selective_rows, aurc = selective_classification_metrics(labels_array, probabilities)
        selective[name] = {"aurc": aurc, "coverage_points": selective_rows}
        confusions[name] = confusion_metrics(labels_array, probabilities, CLASS_NAMES)
    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        ["macro_f1", "log_loss", "method"], ascending=[False, True, True], ignore_index=True
    )
    uncertainty = image_cluster_paired_bootstrap(
        labels_array,
        champion,
        baseline,
        image_ids,
        resamples=int(selection["final_test"]["bootstrap_resamples"]),
    )
    confirmatory_success = bool(
        uncertainty["point_estimate"] >= 0.01 and uncertainty["ci_95_low"] > 0.0
    )

    strata_rows = []
    stratified = add_strata(frame)
    for column in (
        "area_quartile",
        "height_quartile",
        "scene_occupancy",
        "boundary_status",
        "source_tag_group",
    ):
        for value in sorted(stratified[column].unique()):
            mask = stratified[column].eq(value).to_numpy()
            for name, probabilities in methods.items():
                strata_rows.append(
                    {
                        "stratum": column,
                        "value": str(value),
                        "method": name,
                        "people": int(mask.sum()),
                        "images": int(stratified.loc[mask, "image_id"].nunique()),
                        **fixed_class_metrics(
                            labels_array[mask], probabilities[mask].argmax(axis=1)
                        ),
                    }
                )

    metrics_frame.to_csv(output_dir / "test_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output_dir / "test_per_class.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(output_dir / "test_strata.csv", index=False)
    write_json(output_dir / "test_confusions.json", confusions)
    write_json(output_dir / "test_selective_metrics.json", selective)
    write_json(output_dir / "test_paired_uncertainty.json", uncertainty)
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        person_ids=frame["person_id"].astype(str).to_numpy(),
        image_ids=image_ids,
        labels=labels_array,
        class_names=np.asarray(CLASS_NAMES),
        **methods,
    )
    result = {
        "status": "VCOCO_V2_OFFICIAL_TEST_EVALUATION_COMPLETE",
        "selection_lock_sha256": selection_hash,
        "protocol_lock_sha256": protocol_hash,
        "test_access_gate_sha256": sha256_file(output_dir / "test_access_gate.json"),
        "official_test_label_open_count": gate["official_test_label_open_count"],
        "primary_method": "scale_conditioned_stacking",
        "primary_metrics": metrics_frame.loc[
            metrics_frame["method"].eq("scale_conditioned_stacking")
        ].iloc[0].to_dict(),
        "baseline_metrics": metrics_frame.loc[
            metrics_frame["method"].eq("historical_v1_dino")
        ].iloc[0].to_dict(),
        "paired_macro_f1_difference": uncertainty,
        "confirmatory_success": confirmatory_success,
        "confirmation_rule": "gain_at_least_0.01_and_image_cluster_95pct_interval_above_zero",
        "ontology_scope": "source_tag_derived_legacy_three_class_not_human_harmonized",
        "historical_baseline_checkpoints": baseline_checkpoints,
        "test_used_for_selection": False,
        "predictions_sha256": sha256_file(output_dir / "test_predictions.npz"),
    }
    write_json(summary_path, result)
    print(json.dumps(json_safe(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
