"""Fit one locked CPTR grouped cross-fit fold on CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from hac.cptr import cptr_loss
from hac.cptr_features import (
    CPTRFeatureDataset,
    coherent_feature_augmentation,
    model_kwargs_from_batch,
)
from hac.cptr_training import build_cptr_model, classification_summary, evaluate_cptr
from hac.polar import sha256_file
from hac.polar_training import warmup_cosine_scheduler
from hac.training import seed_everything
from hac.vcoco_v3_temporal import (
    StaticIdentifiabilityStudent,
    TemporalFactorizedTeacher,
    grouped_recording_splits,
)
from hac.vcoco_v3_temporal_training import hierarchical_class_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiments/okutama_cptr_protocol.json")
    )
    parser.add_argument("--protocol-lock", type=Path, default=Path(".runs/cptr/protocol_lock.json"))
    parser.add_argument(
        "--plan", type=Path, default=Path("experiments/okutama_cptr_crossfit_plan.json")
    )
    parser.add_argument(
        "--plan-lock", type=Path, default=Path(".runs/cptr/crossfit_plan_lock.json")
    )
    parser.add_argument(
        "--candidate-grid",
        type=Path,
        default=Path("experiments/okutama_cptr_adaptive_grid.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/development_manifest.csv"),
    )
    parser.add_argument(
        "--v3-grid", type=Path, default=Path("experiments/okutama_temporal_grid.json")
    )
    parser.add_argument(
        "--v3-crossfit-root", type=Path, default=Path(".runs/vcoco_v3/temporal/crossfit")
    )
    parser.add_argument(
        "--part-store", type=Path, default=Path(".runs/cptr/part_features/store.json")
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_frozen_fold_baselines(
    *,
    fold: int,
    seed: int,
    input_dim: int,
    v3_grid: dict,
    root: Path,
    device: torch.device,
) -> tuple[StaticIdentifiabilityStudent, TemporalFactorizedTeacher, dict[str, str]]:
    architecture = v3_grid["architecture"]
    static = StaticIdentifiabilityStudent(
        input_dim,
        hidden_dim=int(architecture["static_hidden_dim"]),
        dropout=float(architecture["dropout"]),
    )
    teacher = TemporalFactorizedTeacher(
        input_dim,
        model_dim=int(architecture["temporal_model_dim"]),
        layers=int(architecture["temporal_layers"]),
        attention_heads=int(architecture["attention_heads"]),
        feedforward_dim=int(architecture["feedforward_dim"]),
        dropout=float(architecture["dropout"]),
        maximum_length=16,
    )
    paths = {
        "static": root / "static" / f"fold-{fold}" / f"seed-{seed}" / "checkpoint.pt",
        "teacher": root / "teacher" / f"fold-{fold}" / f"seed-{seed}" / "checkpoint.pt",
    }
    for name, model in (("static", static), ("teacher", teacher)):
        checkpoint = torch.load(paths[name], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval().requires_grad_(False)
    return static, teacher, {name: sha256_file(path) for name, path in paths.items()}


def make_loader(
    dataset: CPTRFeatureDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
    )


def kwargs_builder(candidate: dict):
    def build(batch: Mapping[str, object], device: torch.device) -> dict[str, object]:
        return model_kwargs_from_batch(
            batch,
            device,
            use_long=bool(candidate["use_long"]),
            use_trajectory=bool(candidate["use_trajectory"]),
            use_parts=bool(candidate["use_parts"]),
            use_pose=bool(candidate["use_pose"]),
            use_siglip=bool(candidate["use_siglip"]),
        )

    return build


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CPTR cross-fitting requires CUDA; CPU fallback is disabled")
    protocol_path = args.protocol.resolve()
    protocol_lock_path = args.protocol_lock.resolve()
    plan_path = args.plan.resolve()
    plan_lock_path = args.plan_lock.resolve()
    grid_path = args.candidate_grid.resolve()
    manifest_path = args.manifest.resolve()
    part_store_path = args.part_store.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_lock = json.loads(protocol_lock_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_lock = json.loads(plan_lock_path.read_text(encoding="utf-8"))
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    if protocol_lock.get("status") != "OKUTAMA_CPTR_PROTOCOL_LOCKED_BEFORE_FIT":
        raise RuntimeError("The CPTR protocol is not locked")
    if plan_lock.get("status") != "OKUTAMA_CPTR_CROSSFIT_PLAN_LOCKED_BEFORE_FIT":
        raise RuntimeError("The CPTR cross-fit plan is not locked")
    if plan_lock["source_sha256"]["plan"] != sha256_file(plan_path):
        raise RuntimeError("The cross-fit plan changed after locking")
    if plan_lock["source_sha256"]["runner"] != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("The cross-fit runner changed after locking")
    if plan_lock["source_sha256"]["manifest"] != sha256_file(manifest_path):
        raise RuntimeError("The development manifest changed after cross-fit locking")
    seed_specs = {int(item["seed"]): item for item in plan["seeds"]}
    if args.seed not in seed_specs or not 0 <= args.fold < int(plan["folds"]):
        raise ValueError("The requested seed or fold is outside the locked plan")
    fixed_epochs = int(seed_specs[args.seed]["fixed_epochs"])
    candidate = next(
        item for item in grid["candidates"] if item["candidate_id"] == plan["candidate_id"]
    )
    if not candidate["use_parts"] or any(
        candidate[key] for key in ("use_long", "use_trajectory", "use_pose", "use_siglip")
    ):
        raise RuntimeError("The locked cross-fit candidate contract changed")

    frame = pd.read_csv(
        manifest_path,
        dtype={"sample_id": str, "recording_id": str, "track_id": str},
    )
    train_frame = frame[frame["split"].eq("train")].reset_index(drop=True)
    label_mapping = {"sitting": 0, "standing": 1, "walking_running": 2}
    labels = train_frame["label"].map(label_mapping).to_numpy(dtype=int)
    groups = train_frame["recording_id"].astype(str).to_numpy()
    splits = grouped_recording_splits(
        labels,
        groups,
        folds=int(plan["folds"]),
        seed=int(plan["split_seed"]),
    )
    fit_indices, held_indices = splits[args.fold]
    if set(groups[fit_indices]).intersection(groups[held_indices]):
        raise RuntimeError("A recording crossed the CPTR fold boundary")
    fit_frame = train_frame.iloc[fit_indices].reset_index(drop=True)
    held_frame = train_frame.iloc[held_indices].reset_index(drop=True)
    base_store_path = Path(str(train_frame.iloc[0]["feature_path"])).resolve()
    base_store = json.loads(base_store_path.read_text(encoding="utf-8"))
    input_dim = 2 * int(base_store["feature_dimensions"]) + 6
    part_store = json.loads(part_store_path.read_text(encoding="utf-8"))
    if part_store.get("status") != "OKUTAMA_CPTR_FEATURE_STORE_COMPLETE":
        raise RuntimeError("The CPTR part store is incomplete")
    part_dim = int(part_store["part_token_dim"])
    device = torch.device("cuda")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats(device)
    v3_grid = json.loads(args.v3_grid.resolve().read_text(encoding="utf-8"))
    static, teacher, baseline_hashes = load_frozen_fold_baselines(
        fold=args.fold,
        seed=args.seed,
        input_dim=input_dim,
        v3_grid=v3_grid,
        root=args.v3_crossfit_root.resolve(),
        device=device,
    )
    model = build_cptr_model(
        candidate,
        protocol,
        input_dim=input_dim,
        part_input_dim=part_dim,
        pose_joint_count=1,
        siglip_dim=1,
        static=static,
        teacher=teacher,
    ).to(device)
    dataset_args = {
        "manifest_directory": manifest_path.parent,
        "part_store": part_store_path,
        "short_samples": 8,
        "short_seconds": 0.5,
        "long_samples": 8,
        "long_seconds": 1.0,
    }
    fit_dataset = CPTRFeatureDataset(fit_frame, **dataset_args)
    held_dataset = CPTRFeatureDataset(held_frame, **dataset_args)
    batch_size = int(protocol["training"]["batch_size"])
    fit_loader = make_loader(
        fit_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=args.seed,
        workers=args.workers,
    )
    held_loader = make_loader(
        held_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=args.seed,
        workers=args.workers,
    )
    builder = kwargs_builder(candidate)
    baseline_predictions_path = (
        args.v3_crossfit_root.resolve()
        / "teacher"
        / f"fold-{args.fold}"
        / f"seed-{args.seed}"
        / "held_predictions.npz"
    )
    baseline_predictions = np.load(baseline_predictions_path, allow_pickle=False)
    if set(baseline_predictions["sample_ids"].astype(str)) != set(
        held_frame["sample_id"].astype(str)
    ):
        raise RuntimeError("CPTR and baseline cross-fit held rows differ")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "status": "OKUTAMA_CPTR_CROSSFIT_REQUEST",
        "candidate_id": candidate["candidate_id"],
        "fold": int(args.fold),
        "seed": int(args.seed),
        "fixed_epochs": fixed_epochs,
        "fit_samples": int(len(fit_frame)),
        "held_samples": int(len(held_frame)),
        "validation_samples_read": 0,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "protocol": sha256_file(protocol_path),
            "protocol_lock": sha256_file(protocol_lock_path),
            "plan": sha256_file(plan_path),
            "plan_lock": sha256_file(plan_lock_path),
            "candidate_grid": sha256_file(grid_path),
            "manifest": sha256_file(manifest_path),
            "part_store": sha256_file(part_store_path),
            "base_store": sha256_file(base_store_path),
            "baseline_predictions": sha256_file(baseline_predictions_path),
            "runner": sha256_file(Path(__file__).resolve()),
            "model_module": sha256_file(Path(__file__).resolve().parents[1] / "src/hac/cptr.py"),
            "feature_module": sha256_file(
                Path(__file__).resolve().parents[1] / "src/hac/cptr_features.py"
            ),
            "training_module": sha256_file(
                Path(__file__).resolve().parents[1] / "src/hac/cptr_training.py"
            ),
        },
    }
    request_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()
    request_path = output_dir / "request.json"
    if request_path.is_file():
        previous = json.loads(request_path.read_text(encoding="utf-8"))
        if previous.get("request_sha256") != request_hash:
            raise RuntimeError("The cross-fit output contains a different request")
    else:
        write_json(request_path, {**request, "request_sha256": request_hash})
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("status") == "OKUTAMA_CPTR_CROSSFIT_RUN_COMPLETE":
            print(json.dumps(previous, indent=2, sort_keys=True))
            return

    training = protocol["training"]
    fit_labels = fit_frame["label"].map(label_mapping).to_numpy(dtype=int)
    posture_weight, motion_weight = hierarchical_class_weights(fit_labels, device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = (
        warmup_cosine_scheduler(
            optimizer,
            total_steps=max(1, len(fit_loader) * fixed_epochs),
            warmup_fraction=0.1,
        )
        if fixed_epochs
        else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=4096.0)
    history = []
    augmentation = protocol["augmentation"]
    loss_spec = protocol["counterfactual_training"]
    started = time.perf_counter()
    for epoch in range(fixed_epochs):
        model.train()
        epoch_losses = []
        for batch in fit_loader:
            optimizer.zero_grad(set_to_none=True)
            kwargs = coherent_feature_augmentation(
                builder(batch, device),
                feature_noise=float(augmentation["temporally_coherent_feature_noise"]),
                geometry_jitter=float(augmentation["temporally_coherent_geometry_jitter"]),
            )
            target = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda"):
                output = model(**kwargs)
                losses = cptr_loss(
                    output,
                    target,
                    transition_targets=batch["transition_target"].to(device, non_blocking=True),
                    gait_targets=batch["gait_target"].to(device, non_blocking=True),
                    occlusion_targets=batch["occlusion_target"].to(device, non_blocking=True),
                    label_smoothing=float(training["label_smoothing"]),
                    posture_weight=posture_weight,
                    motion_weight=motion_weight,
                    transition_weight=float(loss_spec["transition_auxiliary_weight"]),
                    gait_weight=float(loss_spec["gait_auxiliary_weight"]),
                    quality_weight=float(loss_spec["quality_auxiliary_weight"]),
                    motion_null_weight=float(loss_spec["motion_null_residual_weight"]),
                    reversal_weight=float(loss_spec["non_transition_reversal_consistency_weight"]),
                    camera_invariance_weight=float(loss_spec["camera_jitter_invariance_weight"]),
                )
                loss = losses["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(training["gradient_clip_norm"]),
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            epoch_losses.append(float(loss.detach()))
        row = {"epoch": epoch, "training_loss": float(np.mean(epoch_losses))}
        history.append(row)
        write_json(output_dir / "history.json", history)
        print(json.dumps(row, sort_keys=True), flush=True)

    held = evaluate_cptr(model, held_loader, device, kwargs_builder=builder)
    baseline_positions = {
        value: index for index, value in enumerate(baseline_predictions["sample_ids"].astype(str))
    }
    order = np.asarray([baseline_positions[value] for value in held["sample_ids"]])
    baseline_labels = baseline_predictions["labels"].astype(int)[order]
    if not np.array_equal(baseline_labels, held["labels"]):
        raise RuntimeError("CPTR and baseline held labels differ")
    baseline_probability = baseline_predictions["probabilities"].astype(float)[order]
    baseline_metrics = classification_summary(held["labels"], baseline_probability)
    predictions_path = output_dir / "held_predictions.npz"
    np.savez_compressed(
        predictions_path,
        sample_ids=held["sample_ids"],
        recording_ids=held["recording_ids"],
        track_ids=held["track_ids"],
        labels=held["labels"],
        probabilities=held["probabilities"],
        baseline_probabilities=baseline_probability,
        transition_targets=held["transition_targets"],
        occlusion_targets=held["occlusion_targets"],
    )
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "request_sha256": request_hash,
            "fixed_epochs": fixed_epochs,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    if not (output_dir / "history.json").is_file():
        write_json(output_dir / "history.json", history)
    summary = {
        "status": "OKUTAMA_CPTR_CROSSFIT_RUN_COMPLETE",
        "candidate_id": candidate["candidate_id"],
        "fold": int(args.fold),
        "seed": int(args.seed),
        "fixed_epochs": fixed_epochs,
        "fit_samples": int(len(fit_frame)),
        "held_samples": int(len(held_frame)),
        "held_metrics": held["metrics"],
        "baseline_held_metrics": baseline_metrics,
        "macro_f1_delta": float(held["metrics"]["macro_f1"] - baseline_metrics["macro_f1"]),
        "training_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "runtime_seconds": time.perf_counter() - started,
        "validation_samples_read": 0,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "request_sha256": request_hash,
        "baseline_checkpoint_sha256": baseline_hashes,
        "artifact_sha256": {
            "checkpoint.pt": sha256_file(checkpoint_path),
            "held_predictions.npz": sha256_file(predictions_path),
            "history.json": sha256_file(output_dir / "history.json"),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
