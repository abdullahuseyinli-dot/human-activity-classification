"""Train one locked static baseline or temporal-teacher development candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_training import is_better_validation, warmup_cosine_scheduler
from hac.training import seed_everything
from hac.vcoco_v3_models import locomotion_f1
from hac.vcoco_v3_temporal import (
    StaticIdentifiabilityStudent,
    TemporalFactorizedTeacher,
    TemporalFeatureDataset,
    static_student_supervised_loss,
    temporal_teacher_loss,
)

FAILURE_DIRECTORY: Path | None = None


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
        "--manifest-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/temporal_manifest_lock.json"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-role", choices=["static", "teacher"], required=True)
    parser.add_argument("--candidate-id")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_inputs(args: argparse.Namespace) -> tuple[dict, dict, dict, pd.DataFrame]:
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    grid_path = args.grid.resolve()
    lock_path = args.temporal_lock.resolve()
    manifest_lock_path = args.manifest_lock.resolve()
    manifest_path = args.manifest.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest_lock = json.loads(manifest_lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_TEMPORAL_GRID_LOCKED_BEFORE_FIT":
        raise RuntimeError("The temporal candidate grid is not locked")
    if manifest_lock.get("status") != "VCOCO_V3_TEMPORAL_MANIFEST_LOCKED":
        raise RuntimeError("The temporal manifest is not locked")
    if lock["source_sha256"]["temporal_grid"] != sha256_file(grid_path):
        raise RuntimeError("The temporal grid changed after locking")
    if lock["source_sha256"]["temporal_manifest_lock"] != sha256_file(manifest_lock_path):
        raise RuntimeError("The temporal grid belongs to a different manifest lock")
    if manifest_lock["source_sha256"]["manifest"] != sha256_file(manifest_path):
        raise RuntimeError("The temporal manifest changed after locking")
    if args.seed not in lock["seeds"]:
        raise ValueError("The temporal seed is not declared")
    candidates = {candidate["candidate_id"]: candidate for candidate in lock["teacher_candidates"]}
    if args.model_role == "teacher":
        if args.candidate_id not in candidates:
            raise ValueError("Teacher jobs require one locked temporal candidate")
        candidate = candidates[str(args.candidate_id)]
    else:
        if args.candidate_id is not None:
            raise ValueError("The static baseline does not select a temporal window")
        candidate = {**lock["teacher_candidates"][0], "candidate_id": "static_center_frame"}
    frame = pd.read_csv(
        manifest_path,
        dtype={"sample_id": str, "recording_id": str, "track_id": str},
    )
    expected_splits = {"train", "validation", "calibration"}
    if grid.get("dataset", {}).get("confirmation_partition") != "provider_test":
        expected_splits.add("confirmation")
    if set(frame["split"]) != expected_splits:
        raise RuntimeError(
            "The temporal manifest split set differs from the declared confirmation policy"
        )
    return grid, lock, candidate, frame


def make_loader(
    frame: pd.DataFrame,
    *,
    candidate: dict,
    manifest_directory: Path,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TemporalFeatureDataset(
            frame,
            uniform_samples=int(candidate["uniform_samples"]),
            window_seconds=float(candidate["window_seconds"]),
            manifest_directory=manifest_directory,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        generator=generator,
    )


def hierarchical_weights(labels: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    posture = (labels != 0).astype(int)
    motion = (labels[labels != 0] == 2).astype(int)

    def calculate(values: np.ndarray) -> torch.Tensor:
        counts = np.bincount(values, minlength=2).astype(float)
        if np.any(counts == 0):
            raise RuntimeError("A temporal fit partition is missing a hierarchical class")
        weights = counts.sum() / (2.0 * counts)
        weights /= weights.mean()
        return torch.as_tensor(weights, dtype=torch.float32, device=device)

    return calculate(posture), calculate(motion)


def forward_model(model: nn.Module, role: str, batch: dict, device: torch.device):
    if role == "teacher":
        return model(
            batch["clip_features"].to(device, non_blocking=True),
            batch["valid_mask"].to(device, non_blocking=True),
        )
    return model(batch["static_features"].to(device, non_blocking=True))


def supervised_loss(
    output,
    role: str,
    labels: torch.Tensor,
    *,
    label_smoothing: float,
    posture_weight: torch.Tensor,
    motion_weight: torch.Tensor,
) -> torch.Tensor:
    function = temporal_teacher_loss if role == "teacher" else static_student_supervised_loss
    return function(
        output,
        labels,
        label_smoothing=label_smoothing,
        posture_weight=posture_weight,
        motion_weight=motion_weight,
    )


@torch.inference_mode()
def evaluate(model: nn.Module, role: str, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    labels = []
    probabilities = []
    sample_ids = []
    recording_ids = []
    track_ids = []
    for batch in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = forward_model(model, role, batch, device)
        labels.append(batch["label"].numpy())
        probabilities.append(output.probabilities.float().cpu().numpy())
        sample_ids.extend(map(str, batch["sample_id"]))
        recording_ids.extend(map(str, batch["recording_id"]))
        track_ids.extend(map(str, batch["track_id"]))
    target = np.concatenate(labels)
    predicted = np.concatenate(probabilities)
    metrics = classification_metrics(target, predicted)
    metrics["locomotion_f1"] = locomotion_f1(target, predicted)
    return {
        "labels": target,
        "probabilities": predicted,
        "sample_ids": np.asarray(sample_ids),
        "recording_ids": np.asarray(recording_ids),
        "track_ids": np.asarray(track_ids),
        "metrics": metrics,
    }


def main() -> None:
    global FAILURE_DIRECTORY
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    FAILURE_DIRECTORY = output_dir
    started = time.perf_counter()
    grid, lock, candidate, frame = validate_inputs(args)
    train_frame = frame[frame["split"].eq("train")].reset_index(drop=True)
    validation_frame = frame[frame["split"].eq("validation")].reset_index(drop=True)
    configuration = {
        "model_role": args.model_role,
        "candidate": candidate,
        "seed": args.seed,
        "architecture": grid["architecture"],
        "training": grid["training"],
    }
    request_core = {
        "status": "VCOCO_V3_TEMPORAL_DEVELOPMENT_REQUEST",
        "configuration": configuration,
        "train_samples": len(train_frame),
        "validation_samples": len(validation_frame),
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "temporal_grid": sha256_file(args.grid.resolve()),
            "temporal_grid_lock": sha256_file(args.temporal_lock.resolve()),
            "temporal_manifest_lock": sha256_file(args.manifest_lock.resolve()),
            "manifest": sha256_file(args.manifest.resolve()),
            "runner": sha256_file(Path(__file__).resolve()),
        },
    }
    request_hash = hashlib.sha256(
        json.dumps(request_core, sort_keys=True).encode("utf-8")
    ).hexdigest()
    request = {**request_core, "request_sha256": request_hash}
    request_path = output_dir / "request.json"
    if request_path.is_file():
        previous = json.loads(request_path.read_text(encoding="utf-8"))
        if previous.get("request_sha256") != request_hash:
            raise RuntimeError("The temporal output directory contains a different request")
    else:
        write_json(request_path, request)
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("status") == "VCOCO_V3_TEMPORAL_DEVELOPMENT_RUN_COMPLETE":
            print(json.dumps(previous, indent=2, sort_keys=True), flush=True)
            return

    if not torch.cuda.is_available():
        raise RuntimeError("Temporal model training requires CUDA; CPU fallback is disabled")
    seed_everything(args.seed)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    architecture = grid["architecture"]
    input_dimensions = int(lock["input_dimensions"])
    if args.model_role == "teacher":
        model = TemporalFactorizedTeacher(
            input_dimensions,
            model_dim=int(architecture["temporal_model_dim"]),
            layers=int(architecture["temporal_layers"]),
            attention_heads=int(architecture["attention_heads"]),
            feedforward_dim=int(architecture["feedforward_dim"]),
            dropout=float(architecture["dropout"]),
            maximum_length=max(int(item["uniform_samples"]) for item in lock["teacher_candidates"]),
        ).to(device)
    else:
        model = StaticIdentifiabilityStudent(
            input_dimensions,
            hidden_dim=int(architecture["static_hidden_dim"]),
            dropout=float(architecture["dropout"]),
        ).to(device)
    training = grid["training"]
    train_loader = make_loader(
        train_frame,
        candidate=candidate,
        manifest_directory=args.manifest.resolve().parent,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        seed=args.seed,
        workers=args.workers,
    )
    validation_loader = make_loader(
        validation_frame,
        candidate=candidate,
        manifest_directory=args.manifest.resolve().parent,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        seed=args.seed,
        workers=args.workers,
    )
    mapping = {"sitting": 0, "standing": 1, "walking_running": 2}
    train_labels = train_frame["label"].map(mapping).to_numpy(dtype=int)
    posture_weight, motion_weight = hierarchical_weights(train_labels, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    steps_per_epoch = math.ceil(len(train_loader))
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=steps_per_epoch * int(training["maximum_epochs"]),
        warmup_fraction=float(training["warmup_fraction"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=4096.0)
    history = []
    best_metrics = None
    best_epoch = -1
    stale_epochs = 0
    best_path = output_dir / "best_checkpoint.pt"
    for epoch in range(int(training["maximum_epochs"])):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = forward_model(model, args.model_role, batch, device)
                loss = supervised_loss(
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
        validation = evaluate(model, args.model_role, validation_loader, device)
        row = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "validation": validation["metrics"],
        }
        history.append(row)
        if is_better_validation(validation["metrics"], best_metrics):
            best_metrics = validation["metrics"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "request_sha256": request_hash,
                    "epoch": epoch,
                    "metrics": best_metrics,
                    "model_state_dict": model.state_dict(),
                },
                best_path,
            )
        else:
            stale_epochs += 1
        write_json(output_dir / "history.json", history)
        print(json.dumps(row, sort_keys=True), flush=True)
        if epoch + 1 >= int(training["minimum_epochs"]) and stale_epochs >= int(
            training["early_stopping_patience"]
        ):
            break
    if not best_path.is_file():
        raise RuntimeError("Temporal training produced no valid checkpoint")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    if checkpoint["request_sha256"] != request_hash:
        raise RuntimeError("Temporal best checkpoint belongs to a different request")
    model.load_state_dict(checkpoint["model_state_dict"])
    validation = evaluate(model, args.model_role, validation_loader, device)
    probabilities_path = output_dir / "validation_predictions.npz"
    np.savez_compressed(
        probabilities_path,
        sample_ids=validation["sample_ids"],
        recording_ids=validation["recording_ids"],
        track_ids=validation["track_ids"],
        labels=validation["labels"],
        probabilities=validation["probabilities"],
    )
    summary = {
        "status": "VCOCO_V3_TEMPORAL_DEVELOPMENT_RUN_COMPLETE",
        "model_role": args.model_role,
        "candidate_id": candidate["candidate_id"],
        "seed": args.seed,
        "best_epoch": best_epoch,
        "validation_metrics": validation["metrics"],
        "train_samples": len(train_frame),
        "validation_samples": len(validation_frame),
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "training_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "runtime_seconds": time.perf_counter() - started,
        "request_sha256": request_hash,
        "artifact_sha256": {
            best_path.name: sha256_file(best_path),
            probabilities_path.name: sha256_file(probabilities_path),
            "history.json": sha256_file(output_dir / "history.json"),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if FAILURE_DIRECTORY is not None:
            write_json(
                FAILURE_DIRECTORY / "failure.json",
                {
                    "status": "VCOCO_V3_TEMPORAL_DEVELOPMENT_RUN_FAILED",
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
