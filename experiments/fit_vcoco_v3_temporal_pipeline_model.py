"""Fit one locked final temporal model and predict the calibration split only."""

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
from hac.vcoco_v3_temporal import (
    StaticIdentifiabilityStudent,
    static_student_distillation_loss,
)
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
        "--development-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/development_final/summary.json"),
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
    parser.add_argument("--model-role", choices=["static", "teacher", "student"], required=True)
    parser.add_argument("--student-candidate")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def lookup(sample_ids: np.ndarray) -> dict[str, int]:
    values = list(map(str, sample_ids))
    if len(values) != len(set(values)):
        raise RuntimeError("Final temporal target sample identifiers must be unique")
    return {value: index for index, value in enumerate(values)}


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    started = time.perf_counter()
    grid_path = args.grid.resolve()
    lock_path = args.temporal_lock.resolve()
    development_path = args.development_summary.resolve()
    target_summary_path = args.student_target_summary.resolve()
    targets_path = args.student_targets.resolve()
    manifest_path = args.manifest.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_TEMPORAL_GRID_LOCKED_BEFORE_FIT":
        raise RuntimeError("The temporal grid is not locked")
    if development.get("status") != "VCOCO_V3_TEMPORAL_DEVELOPMENT_COMPLETE":
        raise RuntimeError("Temporal development is incomplete")
    if args.seed not in lock["seeds"]:
        raise ValueError("The final temporal seed is not declared")
    student_candidates = {
        candidate["candidate_id"]: candidate for candidate in lock["student_candidates"]
    }
    if args.model_role == "student":
        if args.student_candidate not in student_candidates:
            raise ValueError("Final student fitting requires a locked student candidate")
        student_candidate = student_candidates[str(args.student_candidate)]
        target_summary = json.loads(target_summary_path.read_text(encoding="utf-8"))
        if target_summary.get("status") != "VCOCO_V3_TEMPORAL_STUDENT_TARGETS_LOCKED":
            raise RuntimeError("Student target evidence is incomplete")
        if sha256_file(targets_path) != target_summary["artifact_sha256"][targets_path.name]:
            raise RuntimeError("Student targets changed after locking")
        targets = np.load(targets_path, allow_pickle=False)
    else:
        if args.student_candidate is not None:
            raise ValueError("Only student models accept --student-candidate")
        student_candidate = None
        targets = None
    teacher_candidate = development["selected_teacher"]
    if args.model_role == "student":
        fixed_epochs = int(
            development["fixed_epochs"]["students"][str(args.student_candidate)][str(args.seed)]
        )
    else:
        fixed_epochs = int(development["fixed_epochs"][args.model_role][str(args.seed)])
    if not 1 <= fixed_epochs <= int(grid["training"]["maximum_epochs"]):
        raise RuntimeError("The locked final epoch count is outside the declared schedule")

    frame = pd.read_csv(
        manifest_path,
        dtype={"sample_id": str, "recording_id": str, "track_id": str},
    )
    fit_frame = frame[frame["split"].isin(["train", "validation"])].reset_index(drop=True)
    calibration_frame = frame[frame["split"].eq("calibration")].reset_index(drop=True)
    request_core = {
        "status": "VCOCO_V3_TEMPORAL_FINAL_MODEL_REQUEST",
        "model_role": args.model_role,
        "student_candidate": args.student_candidate,
        "teacher_candidate": teacher_candidate,
        "seed": args.seed,
        "fixed_epochs": fixed_epochs,
        "fit_samples": len(fit_frame),
        "calibration_samples": len(calibration_frame),
        "confirmation_samples_read": 0,
        "source_sha256": {
            "temporal_grid": sha256_file(grid_path),
            "temporal_grid_lock": sha256_file(lock_path),
            "development_summary": sha256_file(development_path),
            "manifest": sha256_file(manifest_path),
            "runner": sha256_file(Path(__file__).resolve()),
        },
    }
    if args.model_role == "student":
        request_core["source_sha256"].update(
            {
                "student_target_summary": sha256_file(target_summary_path),
                "student_targets": sha256_file(targets_path),
            }
        )
    request_hash = hashlib.sha256(
        json.dumps(request_core, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("status") == "VCOCO_V3_TEMPORAL_FINAL_MODEL_COMPLETE":
            if previous.get("request_sha256") != request_hash:
                raise RuntimeError("The final model output contains a different request")
            print(json.dumps(previous, indent=2, sort_keys=True), flush=True)
            return
    write_json(output_dir / "request.json", {**request_core, "request_sha256": request_hash})

    if not torch.cuda.is_available():
        raise RuntimeError("Final temporal fitting requires CUDA; CPU fallback is disabled")
    seed_everything(args.seed)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    if args.model_role == "student":
        model = StaticIdentifiabilityStudent(
            int(lock["input_dimensions"]),
            hidden_dim=int(grid["architecture"]["static_hidden_dim"]),
            dropout=float(grid["architecture"]["dropout"]),
        ).to(device)
        loader_role = "static"
    else:
        model = build_temporal_development_model(
            args.model_role,
            input_dimensions=int(lock["input_dimensions"]),
            architecture=grid["architecture"],
            maximum_length=max(int(item["uniform_samples"]) for item in lock["teacher_candidates"]),
        ).to(device)
        loader_role = args.model_role
    training = grid["training"]
    fit_loader = make_temporal_loader(
        fit_frame,
        candidate=teacher_candidate,
        manifest_directory=manifest_path.parent,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        seed=args.seed,
        workers=args.workers,
    )
    calibration_loader = make_temporal_loader(
        calibration_frame,
        candidate=teacher_candidate,
        manifest_directory=manifest_path.parent,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        seed=args.seed,
        workers=args.workers,
    )
    mapping = {"sitting": 0, "standing": 1, "walking_running": 2}
    fit_labels = fit_frame["label"].map(mapping).to_numpy(dtype=int)
    posture_weight, motion_weight = hierarchical_class_weights(fit_labels, device)
    if args.model_role == "student":
        combined_ids = np.concatenate(
            (targets["train_sample_ids"].astype(str), targets["validation_sample_ids"].astype(str))
        )
        combined_teacher = np.concatenate(
            (targets["train_teacher_probabilities"], targets["validation_teacher_probabilities"])
        )
        combined_advantage = np.concatenate(
            (targets["train_advantage_targets"], targets["validation_advantage_targets"])
        ).astype(int)
        target_lookup = lookup(combined_ids)
        if set(fit_frame["sample_id"].astype(str)) != set(target_lookup):
            raise RuntimeError("Final student fit rows do not match locked out-of-sample targets")
        positives = int(combined_advantage.sum())
        if student_candidate["identifiability_weight"] > 0.0 and not 0 < positives < len(
            combined_advantage
        ):
            raise RuntimeError("Final identifiability fit needs both target classes")
        positive_weight = (
            torch.tensor(
                (len(combined_advantage) - positives) / positives,
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
                if args.model_role == "student":
                    positions = np.asarray(
                        [target_lookup[str(value)] for value in batch["sample_id"]], dtype=int
                    )
                    output = model(batch["static_features"].to(device, non_blocking=True))
                    loss = static_student_distillation_loss(
                        output,
                        batch["label"].to(device, non_blocking=True),
                        torch.as_tensor(
                            combined_teacher[positions], dtype=torch.float32, device=device
                        ),
                        identifiability_targets=torch.as_tensor(
                            combined_advantage[positions], dtype=torch.float32, device=device
                        ),
                        supervised_weight=float(student_candidate["supervised_weight"]),
                        distillation_weight=float(student_candidate["teacher_distribution_weight"]),
                        identifiability_weight=float(student_candidate["identifiability_weight"]),
                        temperature=float(grid["distillation"]["temperature"]),
                        label_smoothing=float(training["label_smoothing"]),
                        identifiability_positive_weight=positive_weight,
                        posture_weight=posture_weight,
                        motion_weight=motion_weight,
                    )["loss"]
                else:
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
    calibration = evaluate_temporal_development(model, loader_role, calibration_loader, device)
    predictions_path = output_dir / "calibration_predictions.npz"
    checkpoint_path = output_dir / "checkpoint.pt"
    np.savez_compressed(
        predictions_path,
        sample_ids=calibration["sample_ids"],
        recording_ids=calibration["recording_ids"],
        track_ids=calibration["track_ids"],
        labels=calibration["labels"],
        probabilities=calibration["probabilities"],
        identifiability_scores=(
            calibration["identifiability_scores"]
            if calibration["identifiability_scores"] is not None
            else np.asarray([], dtype=float)
        ),
    )
    torch.save(
        {
            "request_sha256": request_hash,
            "model_role": args.model_role,
            "student_candidate": args.student_candidate,
            "teacher_candidate": teacher_candidate,
            "seed": args.seed,
            "fixed_epochs": fixed_epochs,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    write_json(output_dir / "history.json", history)
    summary = {
        "status": "VCOCO_V3_TEMPORAL_FINAL_MODEL_COMPLETE",
        "model_role": args.model_role,
        "student_candidate": args.student_candidate,
        "seed": args.seed,
        "fixed_epochs": fixed_epochs,
        "fit_samples": len(fit_frame),
        "calibration_samples": len(calibration_frame),
        "calibration_metrics": calibration["metrics"],
        "training_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
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
