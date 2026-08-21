"""Retrain locked configurations and evaluate the untouched course test set.

Configuration selection is intentionally external to this script. A lock file
created from five-fold OOF confirmation evidence must exist before any final
test inference is performed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from recover_experiment import CANDIDATES, initialise_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-notebook", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--cv-epochs", type=int, default=24)
    parser.add_argument("--cv-patience", type=int, default=5)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def calibrated_output(namespace: dict, output: dict, temperature: float) -> dict:
    logits = np.asarray(output["logits"], dtype=np.float64) / float(temperature)
    probs = softmax(logits)
    metrics, per_class, calibration = namespace["basic_metrics"](
        np.asarray(output["labels"], dtype=int), probs
    )
    return {
        "labels": np.asarray(output["labels"], dtype=int),
        "preds": probs.argmax(axis=1),
        "logits": logits,
        "probs": probs,
        "paths": list(output["paths"]),
        "image_ids": list(output["image_ids"]),
        "features": output.get("features"),
        "metrics": metrics,
        "per_class_df": per_class,
        "calibration_df": calibration,
        "attentions": [],
    }


def validate_lock(lock_path: Path, namespace: dict) -> dict:
    with lock_path.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    if lock.get("status") != "LOCKED_BEFORE_FINAL_TEST":
        raise RuntimeError("Selection lock does not have the required locked status.")
    if bool(lock.get("test_used_for_selection", True)):
        raise RuntimeError("Selection lock reports test-set use during selection.")

    expected_families = {"convnext_small", "dinov2_small"}
    if set(lock.get("selected", {})) != expected_families:
        raise RuntimeError("Selection lock must contain exactly ConvNeXt and DINOv2.")

    protocol_path = Path(lock["protocol_path"])
    if sha256_file(protocol_path) != lock["protocol_sha256"]:
        raise RuntimeError("Selection protocol hash no longer matches its lock.")
    with protocol_path.open("r", encoding="utf-8") as handle:
        locked_protocol = json.load(handle)
    final_protocol_path = namespace["DIRS"]["configs"] / "selection_protocol.json"
    with final_protocol_path.open("r", encoding="utf-8") as handle:
        final_protocol = json.load(handle)
    if locked_protocol != final_protocol:
        raise RuntimeError("Final data protocol differs from the selection protocol.")

    for family, entry in lock["selected"].items():
        candidate_id = entry["candidate_id"]
        if candidate_id not in CANDIDATES:
            raise RuntimeError(f"Unknown locked candidate: {candidate_id}")
        expected = dict(CANDIDATES[candidate_id])
        expected_family = expected.pop("model_kind")
        if expected_family != family or expected != entry["config"]:
            raise RuntimeError(f"Locked config drift detected for {candidate_id}")
        evidence_path = Path(entry["evidence_path"])
        if sha256_file(evidence_path) != entry["evidence_sha256"]:
            raise RuntimeError(f"Confirmation evidence changed for {candidate_id}")
    return lock


def save_oof_bundle(namespace: dict, out_dir: Path, output: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    namespace["save_prediction_bundle"](out_dir, "oof_raw", output)
    if output.get("pred_df") is not None:
        output["pred_df"].to_csv(out_dir / "oof_index.csv", index=False)


def derive_lr_schedule(cv_out: dict, config: dict) -> list[list[float]]:
    """Replay the median fold LR state instead of dropping the final scheduler."""
    final_epochs = int(cv_out["derived_final_epochs"])
    histories = cv_out["fold_histories"]
    has_backbone_group = any("lr_group_1" in frame.columns for frame in histories)
    initial = [float(config["head_lr"])]
    if has_backbone_group:
        initial.append(float(config.get("backbone_lr", config["head_lr"])))

    schedule: list[list[float]] = []
    for epoch in range(1, final_epochs + 1):
        if epoch == 1:
            schedule.append(initial.copy())
            continue
        row_values = []
        for group_index in range(len(initial)):
            column = f"lr_group_{group_index}"
            values = []
            for frame in histories:
                previous = frame.loc[frame["epoch"].astype(int) == epoch - 1, column]
                if len(previous):
                    values.append(float(previous.iloc[-1]))
                elif column in frame.columns and len(frame):
                    values.append(float(frame[column].iloc[-1]))
            if not values:
                raise RuntimeError(
                    f"Cannot derive LR schedule for epoch {epoch}, group {group_index}"
                )
            row_values.append(float(np.median(values)))
        schedule.append(row_values)
    return schedule


def run_or_load_cv(
    namespace: dict,
    family: str,
    candidate_id: str,
    config: dict,
    seed: int,
    cv_epochs: int,
    cv_patience: int,
) -> dict:
    out_dir = namespace["DIRS"]["predictions"] / family / f"seed_{seed}" / "cv"
    summary_path = out_dir / "cv_summary.json"
    required = (
        out_dir / "oof_raw_logits.npy",
        out_dir / "oof_raw_probs.npy",
        out_dir / "oof_raw_labels.npy",
        out_dir / "oof_raw_features.npy",
        out_dir / "oof_tta_calibrated_probs.npy",
        out_dir / "oof_image_ids.csv",
    )
    if summary_path.is_file() and all(path.is_file() for path in required):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        print(f"[resume] CV evidence {family} seed={seed}", flush=True)
        return summary

    print(f"[cv] {family} candidate={candidate_id} seed={seed}", flush=True)
    cv_out = namespace["run_cv_for_cfg"](
        model_kind=family,
        cfg=config,
        pool_df_=namespace["non_test_pool_df"],
        seed=int(seed),
        epochs=int(cv_epochs),
        patience=int(cv_patience),
        run_name_prefix=f"final_cv_{candidate_id}_seed_{seed}",
        n_splits=5,
        split_seed=42,
        save_fold_predictions=False,
    )
    oof = cv_out["oof_output"]
    oof_tta = cv_out["oof_tta_output"]
    if oof.get("features") is None:
        raise RuntimeError("OOF features are required for the locked downstream probe.")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_oof_bundle(namespace, out_dir, oof)
    pd.DataFrame({"image_id": oof["image_ids"], "image_path": oof["paths"]}).to_csv(
        out_dir / "oof_image_ids.csv", index=False
    )
    cv_out["fold_metrics_df"].to_csv(out_dir / "fold_metrics.csv", index=False)
    cv_out["fold_summary_df"].to_csv(out_dir / "fold_summary.csv", index=False)
    cv_out["history_df"].to_csv(out_dir / "aggregate_training_history.csv", index=False)

    scaler, calibration_loss = namespace["fit_temperature_scaler"](oof["logits"], oof["labels"])
    temperature = float(scaler.temperature.detach().cpu().item())
    calibrated = calibrated_output(namespace, oof, temperature)
    namespace["save_prediction_bundle"](out_dir, "oof_calibrated", calibrated)

    tta_scaler, tta_calibration_loss = namespace["fit_temperature_scaler"](
        oof_tta["logits"], oof_tta["labels"]
    )
    tta_temperature = float(tta_scaler.temperature.detach().cpu().item())
    tta_calibrated = calibrated_output(namespace, oof_tta, tta_temperature)
    namespace["save_prediction_bundle"](out_dir, "oof_tta_raw", oof_tta)
    namespace["save_prediction_bundle"](out_dir, "oof_tta_calibrated", tta_calibrated)

    summary = {
        "family": family,
        "candidate_id": candidate_id,
        "seed": int(seed),
        "cv_n_splits": 5,
        "cv_split_seed": 42,
        "derived_final_epochs": int(cv_out["derived_final_epochs"]),
        "fold_best_epochs": [int(row["best_epoch"]) for row in cv_out["fold_summaries"]],
        "full_pool_lr_schedule": derive_lr_schedule(cv_out, config),
        "raw_oof_metrics": oof["metrics"],
        "calibrated_oof_metrics": calibrated["metrics"],
        "tta_raw_oof_metrics": oof_tta["metrics"],
        "tta_calibrated_oof_metrics": tta_calibrated["metrics"],
        "temperature": temperature,
        "temperature_fit_loss": float(calibration_loss),
        "tta_temperature": tta_temperature,
        "tta_temperature_fit_loss": float(tta_calibration_loss),
        "test_evaluated": False,
        "config": config,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2)
    del cv_out, oof, oof_tta, calibrated, tta_calibrated, scaler, tta_scaler
    gc.collect()
    if namespace["torch"].cuda.is_available():
        namespace["torch"].cuda.empty_cache()
    return summary


def load_or_train_full_model(
    namespace: dict,
    family: str,
    candidate_id: str,
    config: dict,
    seed: int,
    final_epochs: int,
    lr_schedule: list[list[float]],
):
    run_name = f"final_full_{candidate_id}_seed_{seed}"
    checkpoint = namespace["DIRS"]["checkpoints"] / family / run_name / "full_pool_checkpoint.pt"
    if checkpoint.is_file():
        payload = namespace["torch"].load(
            checkpoint, map_location=namespace["DEVICE"], weights_only=False
        )
        if (
            payload.get("cfg") != config
            or int(payload.get("seed")) != int(seed)
            or int(payload.get("derived_final_epochs")) != int(final_epochs)
            or payload.get("lr_schedule") != lr_schedule
        ):
            raise RuntimeError(f"Checkpoint configuration drift: {checkpoint}")
        model = namespace["build_model_and_strategy"](family, config).to(namespace["DEVICE"])
        model.load_state_dict(payload["model_state_dict"])
        print(f"[resume] full-pool checkpoint {family} seed={seed}", flush=True)
        return model, checkpoint

    print(
        f"[train-full] {family} candidate={candidate_id} seed={seed} epochs={final_epochs}",
        flush=True,
    )
    trained = train_full_pool_with_schedule(
        namespace=namespace,
        family=family,
        config=config,
        seed=int(seed),
        final_epochs=int(final_epochs),
        lr_schedule=lr_schedule,
        run_name=run_name,
    )
    return trained["model"], trained["best_checkpoint_path"]


def train_full_pool_with_schedule(
    namespace: dict,
    family: str,
    config: dict,
    seed: int,
    final_epochs: int,
    lr_schedule: list[list[float]],
    run_name: str,
) -> dict:
    if len(lr_schedule) != int(final_epochs):
        raise RuntimeError("LR schedule length does not match the fixed epoch count.")
    namespace["seed_everything"](seed)
    transform = namespace["build_train_transform"](config["augmentation_strength"])
    loader = namespace["make_loader"](
        namespace["non_test_pool_df"],
        transform,
        int(config["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    model = namespace["build_model_and_strategy"](family, config).to(namespace["DEVICE"])
    total_params = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable_params = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    criterion = namespace["loss_fn_builder"](
        label_smoothing=float(config.get("label_smoothing", 0.0))
    )
    optimizer, _ = namespace["make_optimizer_and_scheduler"](
        model,
        lr_head=float(config["head_lr"]),
        lr_backbone=config.get("backbone_lr"),
        weight_decay=float(config["weight_decay"]),
    )
    if any(len(row) != len(optimizer.param_groups) for row in lr_schedule):
        raise RuntimeError("LR schedule parameter-group count does not match the optimizer.")
    scaler = namespace["torch"].cuda.amp.GradScaler(enabled=namespace["USE_AMP"])

    checkpoint_dir = namespace["DIRS"]["checkpoints"] / family / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch, epoch_lrs in enumerate(lr_schedule, start=1):
        for group, learning_rate in zip(optimizer.param_groups, epoch_lrs, strict=False):
            group["lr"] = float(learning_rate)
        if namespace["torch"].cuda.is_available():
            namespace["torch"].cuda.reset_peak_memory_stats()
        started = namespace["time"].perf_counter()
        train_loss, grad_norm = namespace["train_one_epoch"](
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            grad_accum_steps=namespace["GRAD_ACCUM_STEPS"],
        )
        elapsed = float(namespace["time"].perf_counter() - started)
        peak_memory = (
            float(namespace["torch"].cuda.max_memory_allocated() / 1024**2)
            if namespace["torch"].cuda.is_available()
            else np.nan
        )
        row = {
            "run_name": run_name,
            "model_kind": family,
            "seed": int(seed),
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "grad_norm": float(grad_norm),
            "epoch_seconds": elapsed,
            "peak_memory_mb": peak_memory,
            "derived_final_epochs": int(final_epochs),
            "lr_schedule_source": "median_cv_fold_replay",
            "total_params": total_params,
            "trainable_params": trainable_params,
        }
        for index, learning_rate in enumerate(epoch_lrs):
            row[f"lr_group_{index}"] = float(learning_rate)
        history.append(row)

    history_df = pd.DataFrame(history)
    history_df.to_csv(checkpoint_dir / "full_pool_training_history.csv", index=False)
    checkpoint_path = checkpoint_dir / "full_pool_checkpoint.pt"
    namespace["torch"].save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": config,
            "seed": int(seed),
            "model_kind": family,
            "derived_final_epochs": int(final_epochs),
            "lr_schedule": lr_schedule,
            "label_map": namespace["label_map"],
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
        checkpoint_path,
    )
    summary = {
        "model_kind": family,
        "run_name": run_name,
        "seed": int(seed),
        "epochs_ran": int(final_epochs),
        "lr_schedule_source": "median_cv_fold_replay",
        "mean_epoch_seconds": float(history_df["epoch_seconds"].mean()),
        "peak_memory_mb": float(history_df["peak_memory_mb"].max()),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "checkpoint_path": str(checkpoint_path.resolve()),
    }
    with (checkpoint_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2)
    return {
        "model": model,
        "history_df": history_df,
        "best_checkpoint_path": checkpoint_path,
        "summary": summary,
    }


def evaluate_locked_seed(
    namespace: dict,
    family: str,
    candidate_id: str,
    config: dict,
    seed: int,
    cv_summary: dict,
) -> dict:
    out_dir = namespace["DIRS"]["predictions"] / family / f"seed_{seed}" / "final"
    result_path = out_dir / "seed_result.json"
    if result_path.is_file():
        print(f"[skip] completed final result {family} seed={seed}", flush=True)
        with result_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    model, checkpoint_path = load_or_train_full_model(
        namespace,
        family,
        candidate_id,
        config,
        seed,
        int(cv_summary["derived_final_epochs"]),
        cv_summary["full_pool_lr_schedule"],
    )
    criterion = namespace["loss_fn_builder"](
        label_smoothing=float(config.get("label_smoothing", 0.0))
    )
    eval_transform = namespace["build_eval_transform"]()

    pool_loader = namespace["make_loader"](
        namespace["non_test_pool_df"],
        eval_transform,
        int(config["batch_size"]),
        shuffle=False,
        seed=int(seed),
    )
    pool_output = namespace["evaluate_model"](model, pool_loader, criterion, return_features=True)
    if pool_output.get("features") is None:
        raise RuntimeError("Full-pool embeddings were not produced.")

    test_loader = namespace["make_loader"](
        namespace["test_df"],
        eval_transform,
        int(config["batch_size"]),
        shuffle=False,
        seed=int(seed),
    )
    print(f"[test-once] {family} seed={seed}", flush=True)
    test_raw = namespace["evaluate_model"](model, test_loader, criterion, return_features=True)
    test_calibrated = calibrated_output(namespace, test_raw, float(cv_summary["temperature"]))

    flip_transform = namespace["transforms"].Compose(
        [
            namespace["build_eval_transform"](),
            namespace["transforms"].RandomHorizontalFlip(p=1.0),
        ]
    )
    flip_loader = namespace["make_loader"](
        namespace["test_df"],
        flip_transform,
        int(config["batch_size"]),
        shuffle=False,
        seed=int(seed),
    )
    test_flip = namespace["evaluate_model"](model, flip_loader, criterion, return_features=False)
    if test_raw["image_ids"] != test_flip["image_ids"] or not np.array_equal(
        test_raw["labels"], test_flip["labels"]
    ):
        raise RuntimeError("Final TTA views are not row-aligned.")
    tta_probs = 0.5 * (test_raw["probs"] + test_flip["probs"])
    tta_logits = np.log(np.clip(tta_probs, 1e-12, 1.0))
    tta_metrics, tta_per_class, tta_calibration = namespace["basic_metrics"](
        test_raw["labels"], tta_probs
    )
    test_tta_raw = {
        "labels": test_raw["labels"],
        "preds": tta_probs.argmax(axis=1),
        "logits": tta_logits,
        "probs": tta_probs,
        "paths": test_raw["paths"],
        "image_ids": test_raw["image_ids"],
        "features": test_raw["features"],
        "metrics": tta_metrics,
        "per_class_df": tta_per_class,
        "calibration_df": tta_calibration,
        "attentions": [],
    }
    test_tta_calibrated = calibrated_output(
        namespace, test_tta_raw, float(cv_summary["tta_temperature"])
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    namespace["save_prediction_bundle"](out_dir, "test_raw", test_raw)
    namespace["save_prediction_bundle"](out_dir, "test_calibrated", test_calibrated)
    namespace["save_prediction_bundle"](out_dir, "test_tta_raw", test_tta_raw)
    namespace["save_prediction_bundle"](out_dir, "test_tta_calibrated", test_tta_calibrated)
    np.save(out_dir / "full_pool_features.npy", pool_output["features"])
    np.save(out_dir / "full_pool_labels.npy", pool_output["labels"])
    pd.DataFrame({"image_id": pool_output["image_ids"], "image_path": pool_output["paths"]}).to_csv(
        out_dir / "full_pool_image_ids.csv", index=False
    )
    pd.DataFrame({"image_id": test_raw["image_ids"], "image_path": test_raw["paths"]}).to_csv(
        out_dir / "test_image_ids.csv", index=False
    )

    result = {
        "family": family,
        "candidate_id": candidate_id,
        "seed": int(seed),
        "config": config,
        "derived_final_epochs": int(cv_summary["derived_final_epochs"]),
        "full_pool_lr_schedule": cv_summary["full_pool_lr_schedule"],
        "temperature_from_oof": float(cv_summary["temperature"]),
        "raw_oof_metrics": cv_summary["raw_oof_metrics"],
        "calibrated_oof_metrics": cv_summary["calibrated_oof_metrics"],
        "tta_raw_oof_metrics": cv_summary["tta_raw_oof_metrics"],
        "tta_calibrated_oof_metrics": cv_summary["tta_calibrated_oof_metrics"],
        "raw_test_metrics": test_raw["metrics"],
        "calibrated_test_metrics": test_calibrated["metrics"],
        "tta_raw_test_metrics": test_tta_raw["metrics"],
        "tta_calibrated_test_metrics": test_tta_calibrated["metrics"],
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256_file(Path(checkpoint_path)),
        "test_evaluated_after_lock": True,
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(result), handle, indent=2)

    del model, pool_output, test_raw, test_calibrated, test_flip
    del test_tta_raw, test_tta_calibrated
    gc.collect()
    if namespace["torch"].cuda.is_available():
        namespace["torch"].cuda.empty_cache()
    return result


def write_run_registry(namespace: dict, records: list[dict]) -> None:
    rows = []
    for record in records:
        for calibration_state, key in (
            ("raw", "raw_test_metrics"),
            ("temperature_scaled", "calibrated_test_metrics"),
            ("flip_tta_raw", "tta_raw_test_metrics"),
            ("flip_tta_temperature_scaled", "tta_calibrated_test_metrics"),
        ):
            metrics = record[key]
            rows.append(
                {
                    "family": record["family"],
                    "candidate_id": record["candidate_id"],
                    "seed": record["seed"],
                    "calibration": calibration_state,
                    "derived_final_epochs": record["derived_final_epochs"],
                    **metrics,
                }
            )
    registry = pd.DataFrame(rows).sort_values(["family", "seed", "calibration"])
    registry.to_csv(namespace["ART"] / "final_seed_metrics.csv", index=False)


def main() -> None:
    args = parse_args()
    args.artifact_root = args.artifact_root.resolve()
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    namespace = initialise_pipeline(args)
    lock = validate_lock(args.selection_lock.resolve(), namespace)
    seeds = args.seeds or [42, 52, 62]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")

    execution_plan = {
        "status": "PREDECLARED_BEFORE_FINAL_TEST",
        "seeds": [int(seed) for seed in seeds],
        "cv_n_splits": 5,
        "cv_split_seed": 42,
        "cv_epochs": int(args.cv_epochs),
        "cv_patience": int(args.cv_patience),
        "final_epoch_rule": "median_best_epoch",
        "final_lr_schedule": "median_cv_fold_replay",
        "selection_lock": str(args.selection_lock.resolve()),
        "selection_lock_sha256": sha256_file(args.selection_lock.resolve()),
        "final_runner": Path(__file__).name,
        "final_runner_sha256": sha256_file(Path(__file__).resolve()),
        "test_role": "opened only after every predeclared CV and full-pool run completes",
    }
    plan_path = namespace["DIRS"]["configs"] / "final_execution_plan.json"
    if plan_path.exists():
        with plan_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous != execution_plan:
            raise RuntimeError("Existing final execution plan differs from this invocation.")
    else:
        with plan_path.open("w", encoding="utf-8") as handle:
            json.dump(execution_plan, handle, indent=2)
        with (namespace["DIRS"]["configs"] / "selection_lock.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(lock, handle, indent=2)

    planned_runs: list[tuple[str, str, dict, int, dict]] = []
    for family in ("convnext_small", "dinov2_small"):
        selected = lock["selected"][family]
        candidate_id = selected["candidate_id"]
        config = selected["config"]
        for seed in seeds:
            cv_summary = run_or_load_cv(
                namespace,
                family,
                candidate_id,
                config,
                int(seed),
                int(args.cv_epochs),
                int(args.cv_patience),
            )
            model, _ = load_or_train_full_model(
                namespace,
                family,
                candidate_id,
                config,
                int(seed),
                int(cv_summary["derived_final_epochs"]),
                cv_summary["full_pool_lr_schedule"],
            )
            del model
            gc.collect()
            if namespace["torch"].cuda.is_available():
                namespace["torch"].cuda.empty_cache()
            planned_runs.append((family, candidate_id, config, int(seed), cv_summary))

    print("[gate] all CV and full-pool training completed; opening locked test", flush=True)
    records: list[dict] = []
    for family, candidate_id, config, seed, cv_summary in planned_runs:
        records.append(
            evaluate_locked_seed(
                namespace,
                family,
                candidate_id,
                config,
                seed,
                cv_summary,
            )
        )
    write_run_registry(namespace, records)
    print(f"[done] final evidence: {namespace['ART']}", flush=True)


if __name__ == "__main__":
    main()
