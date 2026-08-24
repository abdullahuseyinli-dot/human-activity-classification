"""Train one locked distilled or identifiability-conditioned static student."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from hac.polar import sha256_file
from hac.polar_training import is_better_validation, warmup_cosine_scheduler
from hac.training import seed_everything
from hac.vcoco_v3_temporal import (
    StaticIdentifiabilityStudent,
    static_student_distillation_loss,
)
from hac.vcoco_v3_temporal_training import (
    evaluate_temporal_development,
    hierarchical_class_weights,
    make_temporal_loader,
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
    parser.add_argument(
        "--student-target-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/student_targets/summary.json"),
    )
    parser.add_argument(
        "--student-targets",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/student_targets/student_targets.npz"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def target_lookup(sample_ids: np.ndarray) -> dict[str, int]:
    values = list(map(str, sample_ids))
    if len(values) != len(set(values)):
        raise RuntimeError("Student target sample identifiers must be unique")
    return {value: index for index, value in enumerate(values)}


def batch_target_indices(sample_ids, lookup: dict[str, int]) -> np.ndarray:
    try:
        return np.asarray([lookup[str(value)] for value in sample_ids], dtype=int)
    except KeyError as error:
        raise RuntimeError(
            f"A data-loader sample is absent from student targets: {error}"
        ) from error


def identifiability_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return {
            "average_precision": None,
            "roc_auc": None,
            "positive_fraction": float(labels.mean()),
        }
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "positive_fraction": float(labels.mean()),
    }


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    started = time.perf_counter()
    grid_path = args.grid.resolve()
    lock_path = args.temporal_lock.resolve()
    selection_path = args.teacher_selection.resolve()
    target_summary_path = args.student_target_summary.resolve()
    targets_path = args.student_targets.resolve()
    manifest_path = args.manifest.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    target_summary = json.loads(target_summary_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_TEMPORAL_GRID_LOCKED_BEFORE_FIT":
        raise RuntimeError("The temporal grid is not locked")
    if selection.get("status") != "VCOCO_V3_TEMPORAL_TEACHER_SELECTED":
        raise RuntimeError("The temporal teacher is not selected")
    if target_summary.get("status") != "VCOCO_V3_TEMPORAL_STUDENT_TARGETS_LOCKED":
        raise RuntimeError("The recording-grouped student targets are not locked")
    if sha256_file(targets_path) != target_summary["artifact_sha256"][targets_path.name]:
        raise RuntimeError("The student targets changed after locking")
    candidates = {candidate["candidate_id"]: candidate for candidate in lock["student_candidates"]}
    if args.candidate_id not in candidates:
        raise ValueError("The requested temporal student candidate is not locked")
    if args.seed not in lock["seeds"]:
        raise ValueError("The temporal student seed is not declared")
    candidate = candidates[args.candidate_id]
    teacher_candidate = selection["selected_teacher"]
    targets = np.load(targets_path, allow_pickle=False)
    train_lookup = target_lookup(targets["train_sample_ids"])
    validation_lookup = target_lookup(targets["validation_sample_ids"])

    frame = pd.read_csv(
        manifest_path,
        dtype={"sample_id": str, "recording_id": str, "track_id": str},
    )
    train_frame = frame[frame["split"].eq("train")].reset_index(drop=True)
    validation_frame = frame[frame["split"].eq("validation")].reset_index(drop=True)
    if set(train_frame["sample_id"]) != set(train_lookup):
        raise RuntimeError("Training manifest rows do not match the student targets")
    if set(validation_frame["sample_id"]) != set(validation_lookup):
        raise RuntimeError("Validation manifest rows do not match the student targets")
    request_core = {
        "status": "VCOCO_V3_TEMPORAL_STUDENT_REQUEST",
        "candidate": candidate,
        "seed": args.seed,
        "train_samples": len(train_frame),
        "validation_samples": len(validation_frame),
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "temporal_grid": sha256_file(grid_path),
            "temporal_grid_lock": sha256_file(lock_path),
            "teacher_selection": sha256_file(selection_path),
            "student_target_summary": sha256_file(target_summary_path),
            "student_targets": sha256_file(targets_path),
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
        if previous.get("status") == "VCOCO_V3_TEMPORAL_STUDENT_RUN_COMPLETE":
            if previous.get("request_sha256") != request_hash:
                raise RuntimeError("The student output contains a different completed request")
            print(json.dumps(previous, indent=2, sort_keys=True), flush=True)
            return
    write_json(output_dir / "request.json", {**request_core, "request_sha256": request_hash})

    if not torch.cuda.is_available():
        raise RuntimeError("Temporal student training requires CUDA; CPU fallback is disabled")
    seed_everything(args.seed)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    model = StaticIdentifiabilityStudent(
        int(lock["input_dimensions"]),
        hidden_dim=int(grid["architecture"]["static_hidden_dim"]),
        dropout=float(grid["architecture"]["dropout"]),
    ).to(device)
    training = grid["training"]
    train_loader = make_temporal_loader(
        train_frame,
        candidate=teacher_candidate,
        manifest_directory=manifest_path.parent,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        seed=args.seed,
        workers=args.workers,
    )
    validation_loader = make_temporal_loader(
        validation_frame,
        candidate=teacher_candidate,
        manifest_directory=manifest_path.parent,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        seed=args.seed,
        workers=args.workers,
    )
    label_mapping = {"sitting": 0, "standing": 1, "walking_running": 2}
    train_labels = train_frame["label"].map(label_mapping).to_numpy(dtype=int)
    posture_weight, motion_weight = hierarchical_class_weights(train_labels, device)
    train_advantage = targets["train_advantage_targets"].astype(int)
    positives = int(train_advantage.sum())
    if candidate["identifiability_weight"] > 0.0 and not 0 < positives < len(train_advantage):
        raise RuntimeError("Identifiability training needs both advantage target classes")
    positive_weight = (
        torch.tensor(
            (len(train_advantage) - positives) / positives,
            dtype=torch.float32,
            device=device,
        )
        if positives
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=max(1, len(train_loader) * int(training["maximum_epochs"])),
        warmup_fraction=float(training["warmup_fraction"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=4096.0)
    best_metrics = None
    best_epoch = -1
    stale_epochs = 0
    history = []
    best_path = output_dir / "best_checkpoint.pt"
    for epoch in range(int(training["maximum_epochs"])):
        model.train()
        losses = []
        for batch in train_loader:
            positions = batch_target_indices(batch["sample_id"], train_lookup)
            teacher_probabilities = torch.as_tensor(
                targets["train_teacher_probabilities"][positions],
                dtype=torch.float32,
                device=device,
            )
            advantage = torch.as_tensor(
                targets["train_advantage_targets"][positions],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(batch["static_features"].to(device, non_blocking=True))
                loss_output = static_student_distillation_loss(
                    output,
                    batch["label"].to(device, non_blocking=True),
                    teacher_probabilities,
                    identifiability_targets=advantage,
                    supervised_weight=float(candidate["supervised_weight"]),
                    distillation_weight=float(candidate["teacher_distribution_weight"]),
                    identifiability_weight=float(candidate["identifiability_weight"]),
                    temperature=float(grid["distillation"]["temperature"]),
                    label_smoothing=float(training["label_smoothing"]),
                    identifiability_positive_weight=positive_weight,
                    posture_weight=posture_weight,
                    motion_weight=motion_weight,
                )
                loss = loss_output["loss"]
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
        validation = evaluate_temporal_development(model, "static", validation_loader, device)
        positions = batch_target_indices(validation["sample_ids"], validation_lookup)
        identification = identifiability_metrics(
            targets["validation_advantage_targets"][positions],
            validation["identifiability_scores"],
        )
        row = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "validation": validation["metrics"],
            "identifiability": identification,
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
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation = evaluate_temporal_development(model, "static", validation_loader, device)
    positions = batch_target_indices(validation["sample_ids"], validation_lookup)
    identification = identifiability_metrics(
        targets["validation_advantage_targets"][positions],
        validation["identifiability_scores"],
    )
    predictions_path = output_dir / "validation_predictions.npz"
    np.savez_compressed(
        predictions_path,
        sample_ids=validation["sample_ids"],
        recording_ids=validation["recording_ids"],
        track_ids=validation["track_ids"],
        labels=validation["labels"],
        probabilities=validation["probabilities"],
        identifiability_scores=validation["identifiability_scores"],
        advantage_targets=targets["validation_advantage_targets"][positions],
    )
    summary = {
        "status": "VCOCO_V3_TEMPORAL_STUDENT_RUN_COMPLETE",
        "candidate_id": args.candidate_id,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "validation_metrics": validation["metrics"],
        "identifiability_metrics": identification,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "training_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "runtime_seconds": time.perf_counter() - started,
        "request_sha256": request_hash,
        "artifact_sha256": {
            best_path.name: sha256_file(best_path),
            predictions_path.name: sha256_file(predictions_path),
            "history.json": sha256_file(output_dir / "history.json"),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
