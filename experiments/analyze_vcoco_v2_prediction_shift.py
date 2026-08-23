"""Evaluate prediction-only prior and calibration interventions on V-COCO v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_analysis import confusion_metrics, per_class_metrics
from hac.polar_training import normalize_probability_rows
from hac.transfer import (
    apply_prior_ratio,
    apply_temperature,
    estimate_label_shift_em,
    fit_temperature,
    image_cluster_paired_bootstrap,
    probability_logits,
    softmax,
)

CLASS_NAMES = ("sitting", "standing", "walking_running")
DEFAULT_CANDIDATES = (
    "person_dinov2_base_top4",
    "person_locked_ensemble_collapsed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--polar-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--candidate", action="append", dest="candidates")
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


def validate_protocol(lock_path: Path, manifests: dict[str, Path]) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("Prediction-shift analysis requires the locked v2 protocol")
    for split, path in manifests.items():
        expected = lock["artifact_sha256"][f"vcoco_{split}_clean.csv"]
        if sha256_file(path) != expected:
            raise RuntimeError(f"Locked {split} manifest drift")
    if lock["test_access"]["model_predictions_run"]:
        raise RuntimeError("Protocol claims that target-test predictions already exist")
    return lock


def load_prediction_pool(path: Path, candidates: tuple[str, ...]) -> dict:
    with np.load(path, allow_pickle=True) as payload:
        class_names = tuple(str(value) for value in payload["class_names"])
        if class_names != CLASS_NAMES:
            raise RuntimeError(f"Unexpected class order: {class_names}")
        person_ids = np.asarray([str(value) for value in payload["person_ids"]])
        if len(np.unique(person_ids)) != len(person_ids):
            raise RuntimeError("Prediction pool contains duplicate people")
        missing = set(candidates) - set(payload.files)
        if missing:
            raise RuntimeError(f"Prediction pool lacks candidates: {sorted(missing)}")
        return {
            "person_ids": person_ids,
            "probabilities": {
                name: normalize_probability_rows(payload[name]) for name in candidates
            },
        }


def align_split(frame: pd.DataFrame, pool: dict) -> tuple[pd.DataFrame, np.ndarray]:
    frame = frame.copy()
    frame["person_id"] = frame["person_id"].astype(str)
    frame["image_id"] = frame["image_id"].astype(str)
    index = {person_id: row for row, person_id in enumerate(pool["person_ids"])}
    missing = set(frame["person_id"]) - set(index)
    if missing:
        raise RuntimeError(f"Existing external predictions miss {len(missing)} people")
    order = np.asarray([index[value] for value in frame["person_id"]], dtype=int)
    return frame.reset_index(drop=True), order


def label_indices(frame: pd.DataFrame) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = frame["label_3"].map(mapping)
    if labels.isna().any():
        raise RuntimeError("Manifest contains a non-legacy label")
    return labels.to_numpy(dtype=int)


def source_prior(polar_manifest: Path) -> np.ndarray:
    frame = pd.read_csv(polar_manifest, usecols=["split", "label_3"])
    train = frame[frame["split"].astype(str).eq("train")]
    counts = train["label_3"].value_counts()
    values = np.asarray([counts[name] for name in CLASS_NAMES], dtype=np.float64)
    return values / values.sum()


def target_prior(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    return counts / counts.sum()


def select_logistic_calibrator(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    balanced: bool,
) -> tuple[LogisticRegression, dict]:
    features = probability_logits(probabilities)
    candidates = (0.01, 0.1, 1.0, 10.0, 100.0)
    rows = []
    splitter = GroupKFold(n_splits=5)
    for c_value in candidates:
        out_of_fold = np.zeros_like(probabilities)
        for train_index, held_index in splitter.split(features, labels, groups):
            model = LogisticRegression(
                C=float(c_value),
                class_weight="balanced" if balanced else None,
                max_iter=2_000,
                random_state=42,
                solver="lbfgs",
            )
            model.fit(features[train_index], labels[train_index])
            out_of_fold[held_index] = model.predict_proba(features[held_index])
        metrics = classification_metrics(labels, out_of_fold)
        rows.append({"C": c_value, **metrics})
    ranking = sorted(rows, key=lambda row: (-row["macro_f1"], row["log_loss"], row["C"]))
    selected = ranking[0]
    final = LogisticRegression(
        C=float(selected["C"]),
        class_weight="balanced" if balanced else None,
        max_iter=2_000,
        random_state=42,
        solver="lbfgs",
    )
    final.fit(features, labels)
    return final, {"selected_C": selected["C"], "inner_cv": rows}


def select_decision_offsets(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logits = probability_logits(probabilities)
    grid = np.linspace(-2.5, 2.5, 101)
    best = None
    for sitting_offset in grid:
        for standing_offset in grid:
            offsets = np.asarray([sitting_offset, standing_offset, 0.0])
            predictions = (logits + offsets).argmax(axis=1)
            macro = float(f1_score(labels, predictions, average="macro"))
            rank = (-macro, float(np.square(offsets).sum()), sitting_offset, standing_offset)
            if best is None or rank < best[0]:
                best = (rank, offsets.copy())
    if best is None:  # pragma: no cover - grid is non-empty
        raise RuntimeError("Decision-offset search produced no candidate")
    return best[1]


def evaluate_method(
    candidate: str,
    method: str,
    fit_role: str,
    train_labels: np.ndarray,
    train_probabilities: np.ndarray,
    val_frame: pd.DataFrame,
    val_labels: np.ndarray,
    val_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    bootstrap_resamples: int,
) -> tuple[dict, list[dict], dict, dict]:
    train_metrics = classification_metrics(train_labels, train_probabilities)
    val_metrics = classification_metrics(val_labels, val_probabilities)
    row = {
        "candidate": candidate,
        "method": method,
        "fit_role": fit_role,
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"val_{key}": value for key, value in val_metrics.items()},
    }
    per_class = [
        {"candidate": candidate, "method": method, **item}
        for item in per_class_metrics(val_labels, val_probabilities, CLASS_NAMES)
    ]
    confusion = confusion_metrics(val_labels, val_probabilities, CLASS_NAMES)
    uncertainty = image_cluster_paired_bootstrap(
        val_labels,
        val_probabilities,
        baseline_probabilities,
        val_frame["image_id"].astype(str).to_numpy(),
        resamples=bootstrap_resamples,
    )
    return row, per_class, confusion, uncertainty


def main() -> None:
    args = parse_args()
    candidates = tuple(args.candidates or DEFAULT_CANDIDATES)
    train_path = args.train_manifest.resolve()
    val_path = args.val_manifest.resolve()
    protocol = validate_protocol(
        args.protocol_lock.resolve(), {"train": train_path, "val": val_path}
    )
    train = pd.read_csv(train_path, dtype={"person_id": str, "image_id": str})
    val = pd.read_csv(val_path, dtype={"person_id": str, "image_id": str})
    pool = load_prediction_pool(args.predictions.resolve(), candidates)
    train, train_order = align_split(train, pool)
    val, val_order = align_split(val, pool)
    train_labels = label_indices(train)
    val_labels = label_indices(val)
    polar_prior = source_prior(args.polar_manifest.resolve())
    observed_target_prior = target_prior(train_labels)

    summary_rows = []
    class_rows = []
    confusions = {}
    uncertainty = {}
    fit_details = {}
    adapted_probabilities = {}

    for candidate in candidates:
        train_base = pool["probabilities"][candidate][train_order]
        val_base = pool["probabilities"][candidate][val_order]
        methods = {
            "unchanged": (
                "source_only",
                train_base,
                val_base,
                {},
            )
        }

        temperature = fit_temperature(train_base, train_labels)
        methods["target_train_temperature"] = (
            "target_supervised_calibration",
            apply_temperature(train_base, temperature),
            apply_temperature(val_base, temperature),
            {"temperature": temperature},
        )

        em_prior, em_details = estimate_label_shift_em(train_base, polar_prior)
        methods["unlabeled_em_label_shift"] = (
            "unlabeled_target_train_adaptation",
            apply_prior_ratio(train_base, polar_prior, em_prior),
            apply_prior_ratio(val_base, polar_prior, em_prior),
            {"estimated_target_prior": em_prior, **em_details},
        )

        methods["supervised_prior_ratio"] = (
            "target_supervised_decision_adaptation",
            apply_prior_ratio(train_base, polar_prior, observed_target_prior),
            apply_prior_ratio(val_base, polar_prior, observed_target_prior),
            {"observed_target_train_prior": observed_target_prior},
        )

        offsets = select_decision_offsets(train_base, train_labels)
        methods["supervised_macro_f1_offsets"] = (
            "target_supervised_decision_adaptation",
            softmax(probability_logits(train_base) + offsets),
            softmax(probability_logits(val_base) + offsets),
            {"class_offsets": offsets},
        )

        for balanced in (False, True):
            calibrator, details = select_logistic_calibrator(
                train_base,
                train_labels,
                train["image_id"].astype(str).to_numpy(),
                balanced=balanced,
            )
            name = "logistic_calibration_balanced" if balanced else "logistic_calibration"
            methods[name] = (
                "target_supervised_calibration",
                calibrator.predict_proba(probability_logits(train_base)),
                calibrator.predict_proba(probability_logits(val_base)),
                details,
            )

        confusions[candidate] = {}
        uncertainty[candidate] = {}
        fit_details[candidate] = {}
        adapted_probabilities[candidate] = {}
        for method, (fit_role, train_values, val_values, details) in methods.items():
            row, classes, confusion, interval = evaluate_method(
                candidate,
                method,
                fit_role,
                train_labels,
                train_values,
                val,
                val_labels,
                val_values,
                val_base,
                args.bootstrap_resamples,
            )
            summary_rows.append(row)
            class_rows.extend(classes)
            confusions[candidate][method] = confusion
            uncertainty[candidate][method] = interval
            fit_details[candidate][method] = details
            adapted_probabilities[candidate][method] = val_values

    summary = pd.DataFrame(summary_rows).sort_values(
        ["val_macro_f1", "val_log_loss", "candidate", "method"],
        ascending=[False, True, True, True],
        ignore_index=True,
    )
    classes = pd.DataFrame(class_rows)
    best = summary.iloc[0].to_dict()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "prediction_shift_summary.csv"
    classes_path = output_dir / "prediction_shift_per_class.csv"
    summary.to_csv(summary_path, index=False)
    classes.to_csv(classes_path, index=False)
    write_json(output_dir / "prediction_shift_confusions.json", confusions)
    write_json(output_dir / "prediction_shift_uncertainty.json", uncertainty)
    write_json(output_dir / "prediction_shift_fit_details.json", fit_details)
    np.savez_compressed(
        output_dir / "prediction_shift_val_probabilities.npz",
        person_ids=val["person_id"].astype(str).to_numpy(),
        image_ids=val["image_id"].astype(str).to_numpy(),
        labels=val_labels,
        class_names=np.asarray(CLASS_NAMES),
        **{
            f"{candidate}__{method}": values
            for candidate, methods in adapted_probabilities.items()
            for method, values in methods.items()
        },
    )
    result = {
        "status": "VCOCO_V2_PREDICTION_SHIFT_DEVELOPMENT_COMPLETE",
        "selection_partition": "official_vcoco_val",
        "fit_partition": "official_vcoco_train",
        "test_rows_read": 0,
        "test_predictions_run": False,
        "source_prior": polar_prior,
        "observed_target_train_prior": observed_target_prior,
        "best_validation_result": best,
        "protocol_lock_sha256": sha256_file(args.protocol_lock.resolve()),
        "source_sha256": {
            "predictions": sha256_file(args.predictions.resolve()),
            "train_manifest": sha256_file(train_path),
            "val_manifest": sha256_file(val_path),
            "polar_manifest": sha256_file(args.polar_manifest.resolve()),
        },
        "artifact_sha256": {
            summary_path.name: sha256_file(summary_path),
            classes_path.name: sha256_file(classes_path),
        },
        "protocol_status": protocol["status"],
    }
    write_json(output_dir / "summary.json", result)
    print(json.dumps(json_safe(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
