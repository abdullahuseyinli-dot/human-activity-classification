"""Open the locked POLAR test manifest once and evaluate every completed final fit."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from hac.augmentations import build_eval_transform
from hac.config import ModelConfig
from hac.data import make_loader
from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_analysis import (
    confusion_metrics,
    per_class_metrics,
    stratified_paired_bootstrap,
)
from hac.polar_features import PinnedDinoFeatureModel
from hac.polar_models import build_polar_model
from hac.polar_training import TASK_LABELS, evaluate_classifier, normalize_probability_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
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
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def confined_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"Locked relative path escapes final root: {relative}") from error
    return candidate


def verify_final_fits(lock: dict, final_root: Path, lock_hash: str) -> dict:
    """Prove every final train+validation fit completed before the test gate opens."""

    resolved = {"neural": {}, "probes": {}}
    for model_id, specification in lock["final_neural_fits"].items():
        resolved["neural"][model_id] = {}
        for seed in specification["seeds"]:
            relative = specification["output_dir_pattern"].format(seed=seed)
            run_dir = confined_path(final_root, relative)
            summary_path = run_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            checkpoint = run_dir / "final_checkpoint.pt"
            expected = {
                "status": "COMPLETE",
                "stage": "LOCKED_FINAL_TRAIN_PLUS_VALIDATION_FIT",
                "model_id": model_id,
                "seed": seed,
                "selection_lock_sha256": lock_hash,
                "configuration": specification["configuration"],
            }
            if any(summary.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"Final neural fit does not satisfy its lock: {summary_path}")
            if summary.get("test_rows_read") != 0:
                raise RuntimeError(f"Final neural fit accessed test rows: {summary_path}")
            if sha256_file(checkpoint) != summary["final_checkpoint_sha256"]:
                raise RuntimeError(f"Final neural checkpoint hash drift: {checkpoint}")
            resolved["neural"][model_id][int(seed)] = {
                "run_dir": run_dir,
                "summary": summary,
                "checkpoint": checkpoint,
            }

    for probe_id, specification in lock["final_probe_fits"].items():
        run_dir = confined_path(final_root, specification["output_dir"])
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        pipeline = run_dir / "pipeline.joblib"
        expected = {
            "status": "COMPLETE",
            "stage": "LOCKED_FINAL_TRAIN_PLUS_VALIDATION_PROBE_FIT",
            "probe_id": probe_id,
            "selection_lock_sha256": lock_hash,
            "configuration": specification["configuration"],
        }
        if any(summary.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Final probe fit does not satisfy its lock: {summary_path}")
        if summary.get("test_rows_read") != 0:
            raise RuntimeError(f"Final probe fit accessed test rows: {summary_path}")
        if sha256_file(pipeline) != summary["pipeline_sha256"]:
            raise RuntimeError(f"Final probe pipeline hash drift: {pipeline}")
        resolved["probes"][probe_id] = {
            "run_dir": run_dir,
            "summary": summary,
            "pipeline": pipeline,
        }
    return resolved


def validate_test_frame(frame: pd.DataFrame, expected_rows: int) -> pd.DataFrame:
    required = {
        "image_id",
        "image_path",
        "split",
        "label_4",
        "label_3",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Test manifest is missing columns: {sorted(missing)}")
    output = frame.copy()
    output["image_id"] = output["image_id"].astype(str)
    if len(output) != expected_rows:
        raise ValueError(
            f"Locked test row count changed: expected {expected_rows}, found {len(output)}"
        )
    if set(output["split"].astype(str)) != {"test"}:
        raise ValueError("Locked test manifest must contain only the test split")
    if output["image_id"].duplicated().any():
        raise ValueError("Locked test image identifiers must be unique")
    for task, class_names in TASK_LABELS.items():
        observed = set(output[task].astype(str))
        if observed != set(class_names):
            raise ValueError(f"Unexpected {task} labels in locked test manifest: {observed}")
    return output.sort_values("image_id", ignore_index=True)


def open_test_manifest_once(
    path: Path,
    output_dir: Path,
    lock: dict,
    lock_hash: str,
) -> tuple[pd.DataFrame, dict]:
    gate_path = output_dir / "test_access_gate.json"
    cached_path = output_dir / "opened_test_manifest.csv"
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        expected = {
            "status": "POLAR_TEST_GATE_OPEN",
            "selection_lock_sha256": lock_hash,
            "test_manifest_sha256": lock["test_gate"]["test_manifest_sha256"],
            "official_test_manifest_open_count": 1,
        }
        if any(gate.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Existing POLAR test-access gate differs from the current lock")
        if sha256_file(cached_path) != gate["opened_manifest_cache_sha256"]:
            raise RuntimeError("Cached opened test manifest hash drift")
        cached = pd.read_csv(cached_path, dtype={"image_id": str})
        return validate_test_frame(cached, lock["test_gate"]["expected_rows"]), gate

    encoded = path.resolve().read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != lock["test_gate"]["test_manifest_sha256"]:
        raise RuntimeError("POLAR test manifest hash differs from the final selection lock")
    frame = validate_test_frame(
        pd.read_csv(io.BytesIO(encoded), dtype={"image_id": str}),
        lock["test_gate"]["expected_rows"],
    )
    temporary = cached_path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(cached_path)
    gate = {
        "status": "POLAR_TEST_GATE_OPEN",
        "selection_lock_sha256": lock_hash,
        "test_manifest_sha256": digest,
        "official_test_manifest_open_count": 1,
        "opened_manifest_cache_sha256": sha256_file(cached_path),
        "test_rows_read": len(frame),
    }
    write_json(gate_path, gate)
    return frame, gate


def model_config_from_final(configuration: dict) -> ModelConfig:
    return ModelConfig(
        model_kind=configuration["model_kind"],
        augmentation_strength=configuration["augmentation"],
        batch_size=configuration["batch_size"],
        head_lr=configuration["head_lr"],
        backbone_lr=configuration["backbone_lr"],
        weight_decay=configuration["weight_decay"],
        label_smoothing=configuration["label_smoothing"],
        dropout=configuration["dropout"],
        mixup_alpha=configuration["mixup_alpha"],
        unfreeze_strategy=configuration["unfreeze_strategy"],
        top_n_blocks=configuration["top_n_blocks"],
    )


def evaluate_neural_component(
    model_id: str,
    specification: dict,
    resolved_runs: dict[int, dict],
    frame: pd.DataFrame,
    device: torch.device,
) -> tuple[np.ndarray, list[dict]]:
    class_names = list(TASK_LABELS[specification["configuration"]["task"]])
    class_to_index = {name: index for index, name in enumerate(class_names)}
    evaluation_frame = frame.copy()
    evaluation_frame["label"] = evaluation_frame[specification["configuration"]["task"]]
    configuration = specification["configuration"]
    loader = make_loader(
        evaluation_frame,
        class_to_index,
        build_eval_transform(),
        batch_size=configuration["batch_size"],
        shuffle=False,
        seed=0,
        workers=configuration["workers"],
        view=configuration["view"],
    )
    probabilities = []
    seed_rows = []
    criterion = nn.CrossEntropyLoss()
    for seed in specification["seeds"]:
        model = build_polar_model(
            model_config_from_final(configuration), num_classes=len(class_names)
        ).to(device)
        payload = torch.load(
            resolved_runs[int(seed)]["checkpoint"], map_location=device, weights_only=False
        )
        model.load_state_dict(payload["model_state_dict"])
        evaluation = evaluate_classifier(model, loader, criterion, device)
        if evaluation["image_ids"] != frame["image_id"].astype(str).tolist():
            raise RuntimeError(f"Test prediction order drift for {model_id} seed {seed}")
        probabilities.append(evaluation["probabilities"])
        seed_rows.append({"candidate": model_id, "seed": seed, **evaluation["metrics"]})
        del model, payload, evaluation
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return normalize_probability_rows(np.mean(probabilities, axis=0)), seed_rows


@torch.inference_mode()
def extract_probe_features(
    model: nn.Module,
    frame: pd.DataFrame,
    view: str,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> np.ndarray:
    evaluation_frame = frame.copy()
    evaluation_frame["label"] = evaluation_frame["label_4"]
    class_to_index = {name: index for index, name in enumerate(TASK_LABELS["label_4"])}
    loader = make_loader(
        evaluation_frame,
        class_to_index,
        build_eval_transform(),
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        workers=workers,
        view=view,
    )
    output = []
    model.eval()
    for batch in loader:
        values = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            features = model(values)
        output.append(features.float().cpu().numpy())
    return np.concatenate(output)


def evaluate_probe_components(
    lock: dict,
    resolved_probes: dict,
    frame: pd.DataFrame,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    primary_id = lock["ensemble"]["frozen_probe_component"]
    primary_spec = lock["final_probe_fits"][primary_id]
    primary_config = primary_spec["configuration"]
    views = primary_config["views"]
    for probe_id, specification in lock["final_probe_fits"].items():
        config = specification["configuration"]
        representation_key = (config["model_kind"], config["representation"], config["views"])
        expected_key = (
            primary_config["model_kind"],
            primary_config["representation"],
            views,
        )
        if representation_key != expected_key:
            raise RuntimeError(
                f"Final probes do not share the locked representation and views: {probe_id}"
            )

    model = PinnedDinoFeatureModel(
        primary_config["model_kind"], primary_config["representation"]
    ).to(device)
    feature_views = [
        extract_probe_features(
            model,
            frame,
            view,
            lock["test_gate"]["probe_batch_size"],
            lock["test_gate"]["workers"],
            device,
        )
        for view in views
    ]
    features = np.concatenate(feature_views, axis=1)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    probabilities = {}
    direct_three = None
    direct_three_names: list[str] = []
    for probe_id, specification in lock["final_probe_fits"].items():
        pipeline = joblib.load(resolved_probes[probe_id]["pipeline"])
        expected_dimensions = resolved_probes[probe_id]["summary"]["feature_dimensions"]
        if features.shape[1] != expected_dimensions:
            raise RuntimeError(
                f"Frozen feature width differs for {probe_id}: "
                f"expected {expected_dimensions}, observed {features.shape[1]}"
            )
        predicted = normalize_probability_rows(pipeline.predict_proba(features))
        classes = np.asarray(pipeline.classes_, dtype=int)
        expected_classes = np.arange(predicted.shape[1])
        if not np.array_equal(classes, expected_classes):
            raise RuntimeError(f"Unexpected pipeline class order for {probe_id}: {classes}")
        if specification["configuration"]["task"] == "label_4":
            probabilities[probe_id] = predicted
        elif specification["configuration"]["task"] == "label_3":
            direct_three = predicted
            direct_three_names = list(TASK_LABELS["label_3"])
    if direct_three is None:
        raise RuntimeError("Final lock must contain a direct three-class probe")
    return probabilities, direct_three, direct_three_names


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    selection_lock_path = args.selection_lock.resolve()
    lock_hash = sha256_file(selection_lock_path)
    lock = json.loads(selection_lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "FINAL_SELECTION_LOCKED_PRE_TEST":
        raise RuntimeError("Final evaluation requires a FINAL_SELECTION_LOCKED_PRE_TEST lock")
    if lock.get("test_rows_read") != 0 or lock.get("test_used_for_selection"):
        raise RuntimeError("Final selection lock violates the test gate")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "LOCKED_FINAL_TEST_COMPLETE"
            and existing.get("selection_lock_sha256") == lock_hash
        ):
            print(json.dumps(existing, indent=2, sort_keys=True), flush=True)
            return

    final_root = args.final_root.resolve()
    resolved = verify_final_fits(lock, final_root, lock_hash)
    frame, gate = open_test_manifest_once(args.test_manifest, output_dir, lock, lock_hash)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    components: dict[str, np.ndarray] = {}
    seed_rows = []
    for model_id, specification in lock["final_neural_fits"].items():
        probabilities, records = evaluate_neural_component(
            model_id,
            specification,
            resolved["neural"][model_id],
            frame,
            device,
        )
        components[model_id] = probabilities
        seed_rows.extend(records)

    probe_components, direct_three, direct_three_names = evaluate_probe_components(
        lock, resolved["probes"], frame, device
    )
    components.update(probe_components)
    weights = lock["ensemble"]["weights"]
    if set(weights) != set(components):
        raise RuntimeError(
            f"Locked ensemble components differ: weights={set(weights)}, predictions={set(components)}"
        )
    ensemble = normalize_probability_rows(
        sum(float(weights[name]) * components[name] for name in components)
    )

    class_names = list(TASK_LABELS["label_4"])
    label_to_index = {name: index for index, name in enumerate(class_names)}
    labels = frame["label_4"].map(label_to_index).to_numpy(dtype=int)
    candidates = {**components, "locked_ensemble": ensemble}
    metric_rows = []
    per_class_rows = []
    confusions = {}
    uncertainty = {}
    evaluation = lock["evaluation"]
    for name, probabilities in candidates.items():
        metrics = classification_metrics(labels, probabilities)
        metric_rows.append({"candidate": name, **metrics})
        per_class_rows.extend(
            {"candidate": name, **record}
            for record in per_class_metrics(labels, probabilities, class_names)
        )
        confusions[name] = confusion_metrics(labels, probabilities, class_names)
        uncertainty[name] = stratified_paired_bootstrap(
            labels,
            probabilities,
            resamples=evaluation["bootstrap_resamples"],
            seed=evaluation["bootstrap_seed"],
        )
    uncertainty["locked_ensemble_paired_deltas"] = {
        name: stratified_paired_bootstrap(
            labels,
            ensemble,
            probabilities,
            resamples=evaluation["bootstrap_resamples"],
            seed=evaluation["bootstrap_seed"],
        )
        for name, probabilities in components.items()
    }

    secondary_names = list(TASK_LABELS["label_3"])
    secondary_to_index = {name: index for index, name in enumerate(secondary_names)}
    secondary_labels = frame["label_3"].map(secondary_to_index).to_numpy(dtype=int)
    collapsed_three = normalize_probability_rows(
        np.column_stack([ensemble[:, 0], ensemble[:, 1], ensemble[:, 2] + ensemble[:, 3]])
    )
    if direct_three_names != secondary_names:
        raise RuntimeError("Direct three-class pipeline class names differ from the protocol")
    secondary = {
        "collapsed_locked_four_class": collapsed_three,
        "direct_three_class_probe": direct_three,
    }
    secondary_rows = [
        {"candidate": name, **classification_metrics(secondary_labels, probabilities)}
        for name, probabilities in secondary.items()
    ]

    metric_frame = pd.DataFrame(metric_rows).sort_values(
        ["macro_f1", "log_loss", "candidate"],
        ascending=[False, True, True],
        ignore_index=True,
    )
    metric_frame.to_csv(output_dir / "test_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).sort_values(["candidate", "class"], ignore_index=True).to_csv(
        output_dir / "test_per_class.csv", index=False
    )
    pd.DataFrame(seed_rows).sort_values(["candidate", "seed"], ignore_index=True).to_csv(
        output_dir / "test_seed_metrics.csv", index=False
    )
    pd.DataFrame(secondary_rows).sort_values(
        ["macro_f1", "log_loss"], ascending=[False, True], ignore_index=True
    ).to_csv(output_dir / "test_secondary_metrics.csv", index=False)
    write_json(output_dir / "test_confusions.json", confusions)
    write_json(output_dir / "test_uncertainty.json", uncertainty)
    prediction_payload = {
        **{f"probabilities_{name}": value for name, value in candidates.items()},
        "probabilities_collapsed_three": collapsed_three,
        "probabilities_direct_three": direct_three,
        "labels_4": labels,
        "labels_3": secondary_labels,
        "image_ids": frame["image_id"].to_numpy(),
        "class_names_4": np.asarray(class_names),
        "class_names_3": np.asarray(secondary_names),
    }
    np.savez_compressed(output_dir / "test_predictions.npz", **prediction_payload)
    summary = {
        "status": "LOCKED_FINAL_TEST_COMPLETE",
        "selection_lock_sha256": lock_hash,
        "test_manifest_sha256": gate["test_manifest_sha256"],
        "official_test_manifest_open_count": gate["official_test_manifest_open_count"],
        "test_rows_read": len(frame),
        "ensemble_weights": weights,
        "primary_metrics": metric_frame.iloc[0].to_dict(),
        "all_primary_metrics": metric_frame.to_dict("records"),
        "secondary_metrics": secondary_rows,
        "bootstrap_resamples": evaluation["bootstrap_resamples"],
        "bootstrap_seed": evaluation["bootstrap_seed"],
        "predictions_sha256": sha256_file(output_dir / "test_predictions.npz"),
        "runtime_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "test_used_for_selection": False,
    }
    write_json(summary_path, summary)
    print(metric_frame.to_string(index=False), flush=True)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
