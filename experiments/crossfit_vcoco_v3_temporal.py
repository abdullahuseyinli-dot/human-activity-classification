"""Generate recording-grouped out-of-fold static or temporal teacher predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from hac.polar import sha256_file
from hac.polar_training import warmup_cosine_scheduler
from hac.training import seed_everything
from hac.vcoco_v3_temporal import grouped_recording_splits
from hac.vcoco_v3_temporal_training import (
    build_temporal_development_model,
    evaluate_temporal_development,
    forward_temporal_development,
    hierarchical_class_weights,
    make_temporal_loader,
    temporal_development_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", type=Path, default=Path("experiments/vcoco_v3_temporal_grid.json")
    )
    parser.add_argument(
        "--temporal-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/temporal_grid_lock.json"),
    )
    parser.add_argument(
        "--teacher-selection",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/teacher_selection_lock.json"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--development-run-root",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/development"),
    )
    parser.add_argument("--model-role", choices=["static", "teacher"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def development_run_path(root: Path, role: str, candidate: str, seed: int) -> Path:
    if role == "static":
        return root / "static" / f"seed-{seed}"
    return root / "teacher" / candidate / f"seed-{seed}"


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    started = time.perf_counter()
    grid_path = args.grid.resolve()
    lock_path = args.temporal_lock.resolve()
    selection_path = args.teacher_selection.resolve()
    manifest_path = args.manifest.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_TEMPORAL_GRID_LOCKED_BEFORE_FIT":
        raise RuntimeError("The temporal grid is not locked")
    if selection.get("status") != "VCOCO_V3_TEMPORAL_TEACHER_SELECTED":
        raise RuntimeError("The temporal teacher is not selected")
    if selection["source_sha256"]["temporal_grid_lock"] != sha256_file(lock_path):
        raise RuntimeError("Teacher selection belongs to a different temporal grid")
    if args.seed not in lock["seeds"]:
        raise ValueError("The cross-fit seed is not declared")
    folds = int(grid["training"]["teacher_crossfit_folds"])
    if not 0 <= args.fold < folds:
        raise ValueError("The cross-fit fold is outside the locked range")
    teacher_candidate = selection["selected_teacher"]
    candidate = (
        {**lock["teacher_candidates"][0], "candidate_id": "static_center_frame"}
        if args.model_role == "static"
        else teacher_candidate
    )
    development_root = args.development_run_root.resolve()
    source_run = development_run_path(
        development_root, args.model_role, candidate["candidate_id"], args.seed
    )
    source_summary_path = source_run / "summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    source_key = source_run.relative_to(development_root).as_posix()
    if selection["run_artifact_sha256"][source_key]["summary"] != sha256_file(source_summary_path):
        raise RuntimeError("The fixed-epoch source run changed after teacher selection")
    fixed_epochs = int(source_summary["best_epoch"]) + 1
    if not 1 <= fixed_epochs <= int(grid["training"]["maximum_epochs"]):
        raise RuntimeError("The selected fixed epoch count is outside the declared schedule")

    frame = pd.read_csv(
        manifest_path,
        dtype={"sample_id": str, "recording_id": str, "track_id": str},
    )
    train_frame = frame[frame["split"].eq("train")].reset_index(drop=True)
    label_mapping = {"sitting": 0, "standing": 1, "walking_running": 2}
    labels = train_frame["label"].map(label_mapping).to_numpy(dtype=int)
    recordings = train_frame["recording_id"].astype(str).to_numpy(dtype=str)
    splits = grouped_recording_splits(
        labels,
        recordings,
        folds=folds,
        seed=int(grid["training"]["crossfit_seed"]),
    )
    fit_index, held_index = splits[args.fold]
    if set(recordings[fit_index]).intersection(recordings[held_index]):
        raise RuntimeError("A recording crossed the temporal cross-fit boundary")
    fit_frame = train_frame.iloc[fit_index].reset_index(drop=True)
    held_frame = train_frame.iloc[held_index].reset_index(drop=True)
    configuration = {
        "model_role": args.model_role,
        "candidate": candidate,
        "fold": args.fold,
        "seed": args.seed,
        "fixed_epochs": fixed_epochs,
    }
    request_core = {
        "status": "VCOCO_V3_TEMPORAL_CROSSFIT_REQUEST",
        "configuration": configuration,
        "fit_samples": len(fit_frame),
        "held_samples": len(held_frame),
        "validation_samples_read": 0,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "temporal_grid": sha256_file(grid_path),
            "temporal_grid_lock": sha256_file(lock_path),
            "teacher_selection": sha256_file(selection_path),
            "fixed_epoch_source_summary": sha256_file(source_summary_path),
            "manifest": sha256_file(manifest_path),
            "runner": sha256_file(Path(__file__).resolve()),
        },
    }
    request_hash = hashlib.sha256(
        json.dumps(request_core, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("status") == "VCOCO_V3_TEMPORAL_CROSSFIT_RUN_COMPLETE":
            if previous.get("request_sha256") != request_hash:
                raise RuntimeError("The cross-fit output contains a different completed request")
            print(json.dumps(previous, indent=2, sort_keys=True), flush=True)
            return
    write_json(output_dir / "request.json", {**request_core, "request_sha256": request_hash})

    if not torch.cuda.is_available():
        raise RuntimeError("Temporal cross-fitting requires CUDA; CPU fallback is disabled")
    seed_everything(args.seed)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    model = build_temporal_development_model(
        args.model_role,
        input_dimensions=int(lock["input_dimensions"]),
        architecture=grid["architecture"],
        maximum_length=max(int(item["uniform_samples"]) for item in lock["teacher_candidates"]),
    ).to(device)
    training = grid["training"]
    fit_loader = make_temporal_loader(
        fit_frame,
        candidate=candidate,
        manifest_directory=manifest_path.parent,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        seed=args.seed,
        workers=args.workers,
    )
    held_loader = make_temporal_loader(
        held_frame,
        candidate=candidate,
        manifest_directory=manifest_path.parent,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        seed=args.seed,
        workers=args.workers,
    )
    fit_labels = fit_frame["label"].map(label_mapping).to_numpy(dtype=int)
    posture_weight, motion_weight = hierarchical_class_weights(fit_labels, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=max(1, len(fit_loader) * fixed_epochs),
        warmup_fraction=float(training["warmup_fraction"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=4096.0)
    history = []
    for epoch in range(fixed_epochs):
        model.train()
        losses = []
        for batch in fit_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = forward_temporal_development(model, args.model_role, batch, device)
                loss = temporal_development_loss(
                    output,
                    args.model_role,
                    batch["label"].to(device, non_blocking=True),
                    label_smoothing=float(training["label_smoothing"]),
                    posture_weight=posture_weight,
                    motion_weight=motion_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(training["gradient_clip_norm"]),
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().item()))
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses))})
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    held = evaluate_temporal_development(model, args.model_role, held_loader, device)
    predictions_path = output_dir / "held_predictions.npz"
    checkpoint_path = output_dir / "checkpoint.pt"
    np.savez_compressed(
        predictions_path,
        sample_ids=held["sample_ids"],
        recording_ids=held["recording_ids"],
        track_ids=held["track_ids"],
        labels=held["labels"],
        probabilities=held["probabilities"],
    )
    torch.save(
        {
            "request_sha256": request_hash,
            "fixed_epochs": fixed_epochs,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    write_json(output_dir / "history.json", history)
    summary = {
        "status": "VCOCO_V3_TEMPORAL_CROSSFIT_RUN_COMPLETE",
        "model_role": args.model_role,
        "candidate_id": candidate["candidate_id"],
        "fold": args.fold,
        "seed": args.seed,
        "fixed_epochs": fixed_epochs,
        "fit_samples": len(fit_frame),
        "held_samples": len(held_frame),
        "training_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "held_metrics": held["metrics"],
        "validation_samples_read": 0,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "runtime_seconds": time.perf_counter() - started,
        "request_sha256": request_hash,
        "artifact_sha256": {
            predictions_path.name: sha256_file(predictions_path),
            checkpoint_path.name: sha256_file(checkpoint_path),
            "history.json": sha256_file(output_dir / "history.json"),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
