"""Run one leakage-safe inner-screen or outer-fit multiview neural job."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import subprocess
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
from hac.vcoco_v3_models import CLASS_NAMES, grouped_splits, locomotion_f1
from hac.vcoco_v3_neural import (
    PairedPersonDataset,
    PairedPersonSafeTransform,
    SharedMultiviewFactorizedModel,
    build_pinned_backbone,
    configure_parameter_efficient_backbone,
    multiview_factorized_loss,
    parameter_count_summary,
    trainable_parameter_groups,
)
from hac.vcoco_v3_representations import local_checkpoint_evidence

FAILURE_DIRECTORY: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=Path("experiments/vcoco_v3_neural_grid.json"))
    parser.add_argument(
        "--neural-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/neural/neural_grid_lock.json"),
    )
    parser.add_argument(
        "--v2-lock",
        type=Path,
        default=Path(".runs/polar_v2/locked_protocol/vcoco_v2_protocol_lock.json"),
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path(".runs/polar_v2/locked_protocol/vcoco_train_clean.csv"),
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path(".runs/polar_v2/locked_protocol/vcoco_val_clean.csv"),
    )
    parser.add_argument(
        "--selection-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/neural/inner_selection_lock.json"),
    )
    parser.add_argument("--role", choices=["inner_screen", "outer_fit"], required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--inner-fold", type=int)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def validate_and_load(args: argparse.Namespace) -> tuple[dict, dict, dict, pd.DataFrame]:
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    grid_path = args.grid.resolve()
    lock_path = args.neural_lock.resolve()
    v2_lock_path = args.v2_lock.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_NEURAL_GRID_LOCKED_BEFORE_FIT":
        raise RuntimeError("The neural grid is not eligible and locked")
    if sha256_file(grid_path) != lock["source_sha256"]["neural_grid"]:
        raise RuntimeError("The neural grid changed after locking")
    root = Path(__file__).resolve().parents[1]
    locked_sources = {
        "neural_runner_source": Path(__file__).resolve(),
        "neural_queue_source": root / "experiments/run_vcoco_v3_neural_queue.py",
        "neural_module_source": root / "src/hac/vcoco_v3_neural.py",
    }
    for name, source_path in locked_sources.items():
        if sha256_file(source_path) != lock["source_sha256"].get(name):
            raise RuntimeError(f"Locked neural source drift: {name}")
    if not grid.get("execution_backend", {}).get("cuda_required"):
        raise RuntimeError("The neural grid does not require CUDA")
    candidates = {candidate["candidate_id"]: candidate for candidate in lock["candidates"]}
    if args.candidate_id not in candidates:
        raise ValueError("The requested neural candidate is not locked")
    candidate = candidates[args.candidate_id]
    cross_validation = grid["cross_validation"]
    if not 0 <= args.outer_fold < int(cross_validation["outer_folds"]):
        raise ValueError("outer-fold is outside the locked grid")
    if args.role == "inner_screen":
        if args.inner_fold is None or not 0 <= args.inner_fold < int(
            cross_validation["inner_folds"]
        ):
            raise ValueError("inner-screen jobs require a valid inner-fold")
        if args.seed not in cross_validation["screening_seeds"]:
            raise ValueError("The inner-screen seed is not declared")
    else:
        if args.inner_fold is not None:
            raise ValueError("outer-fit jobs cannot declare an inner-fold")
        if args.seed not in cross_validation["outer_fit_seeds"]:
            raise ValueError("The outer-fit seed is not declared")
        selection_path = args.selection_lock.resolve()
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("status") != "VCOCO_V3_NEURAL_INNER_SELECTION_LOCKED":
            raise RuntimeError("Outer fitting requires the completed inner selection lock")
        if selection.get("source_sha256", {}).get("neural_grid_lock") != sha256_file(lock_path):
            raise RuntimeError("The inner selection belongs to a different neural grid")
        selected = selection["selected_by_outer_fold"][str(args.outer_fold)]["candidate_id"]
        if selected != args.candidate_id:
            raise RuntimeError("The requested candidate was not selected for this outer fold")

    v2_lock = json.loads(v2_lock_path.read_text(encoding="utf-8"))
    frames = []
    for split, path in (("train", args.train_manifest), ("val", args.val_manifest)):
        resolved = path.resolve()
        if sha256_file(resolved) != v2_lock["artifact_sha256"][f"vcoco_{split}_clean.csv"]:
            raise RuntimeError(f"Locked {split} manifest drift")
        frame = pd.read_csv(resolved, dtype={"person_id": str, "image_id": str})
        frame["split"] = split
        frames.append(frame)
    development = pd.concat(frames, ignore_index=True)
    if development["person_id"].duplicated().any():
        raise RuntimeError("Development person identifiers must be unique")
    if set(development["label_3"].astype(str)) != set(CLASS_NAMES):
        raise RuntimeError("The neural manifest has an unexpected source-tag ontology")
    missing = development[~development["image_path"].map(lambda value: Path(str(value)).is_file())]
    if len(missing):
        raise FileNotFoundError(f"Development cohort is missing {len(missing)} source images")
    return grid, lock, candidate, development


def split_indices(
    frame: pd.DataFrame,
    grid: dict,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = frame["label_3"].map(mapping).to_numpy(dtype=int)
    groups = frame["image_id"].astype(str).to_numpy(dtype=str)
    cross_validation = grid["cross_validation"]
    outer = grouped_splits(
        labels,
        groups,
        folds=int(cross_validation["outer_folds"]),
        seed=int(cross_validation["outer_seed"]),
    )
    outer_train, outer_held = outer[args.outer_fold]
    if args.role == "outer_fit":
        fit_pool, held = outer_train, outer_held
        split_evidence = {
            "outer_fold": args.outer_fold,
            "inner_fold": None,
            "fit_pool_rows": len(fit_pool),
            "held_rows": len(held),
        }
    else:
        inner = grouped_splits(
            labels[outer_train],
            groups[outer_train],
            folds=int(cross_validation["inner_folds"]),
            seed=int(cross_validation["inner_seed"]) + args.outer_fold,
        )
        relative_fit, relative_held = inner[int(args.inner_fold)]
        fit_pool, held = outer_train[relative_fit], outer_train[relative_held]
        split_evidence = {
            "outer_fold": args.outer_fold,
            "inner_fold": args.inner_fold,
            "fit_pool_rows": len(fit_pool),
            "held_rows": len(held),
        }
    early = grouped_splits(
        labels[fit_pool],
        groups[fit_pool],
        folds=int(cross_validation["early_stopping_folds"]),
        seed=int(cross_validation["inner_seed"]) + 100 * args.outer_fold + args.seed,
    )
    relative_train, relative_stop = early[int(cross_validation["early_stopping_fold"])]
    train, stopping = fit_pool[relative_train], fit_pool[relative_stop]
    for left, right, name in (
        (train, stopping, "train/early-stop"),
        (train, held, "train/held"),
        (stopping, held, "early-stop/held"),
    ):
        if set(groups[left]).intersection(groups[right]):
            raise RuntimeError(f"A source image crossed the {name} boundary")
    split_evidence.update(
        {
            "train_rows": len(train),
            "early_stopping_rows": len(stopping),
            "train_images": int(pd.Series(groups[train]).nunique()),
            "early_stopping_images": int(pd.Series(groups[stopping]).nunique()),
            "held_images": int(pd.Series(groups[held]).nunique()),
        }
    )
    return train, stopping, held, split_evidence


def make_loader(
    frame: pd.DataFrame,
    *,
    transform: PairedPersonSafeTransform,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        PairedPersonDataset(frame, transform),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        generator=generator,
    )


def hierarchical_weights(labels: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    posture = (np.asarray(labels) != 0).astype(int)
    upright = np.asarray(labels) != 0
    motion = (np.asarray(labels)[upright] == 2).astype(int)

    def weights(values: np.ndarray) -> torch.Tensor:
        counts = np.bincount(values, minlength=2).astype(float)
        if np.any(counts == 0):
            raise ValueError("Every hierarchical target needs both classes in the fit partition")
        result = counts.sum() / (2.0 * counts)
        result /= result.mean()
        return torch.as_tensor(result, dtype=torch.float32, device=device)

    return weights(posture), weights(motion)


@torch.inference_mode()
def evaluate(
    model: SharedMultiviewFactorizedModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    labels = []
    probabilities = []
    person_ids = []
    image_ids = []
    gate_weights = []
    for batch in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(
                batch["tight_pixels"].to(device, non_blocking=True),
                batch["context_pixels"].to(device, non_blocking=True),
                batch["geometry"].to(device, non_blocking=True),
            )
        labels.append(batch["label"].numpy())
        probabilities.append(output.probabilities.float().cpu().numpy())
        gate_weights.append(output.gate_weights.float().cpu().numpy())
        person_ids.extend(map(str, batch["person_id"]))
        image_ids.extend(map(str, batch["image_id"]))
    labels_array = np.concatenate(labels)
    probabilities_array = np.concatenate(probabilities)
    metrics = classification_metrics(labels_array, probabilities_array)
    metrics["locomotion_f1"] = locomotion_f1(labels_array, probabilities_array)
    return {
        "labels": labels_array,
        "probabilities": probabilities_array,
        "gate_weights": np.concatenate(gate_weights),
        "person_ids": np.asarray(person_ids),
        "image_ids": np.asarray(image_ids),
        "metrics": metrics,
    }


def learned_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    learned_names = {
        name
        for name, _ in model.named_parameters()
        if not name.startswith("backbone.") or ".lora_a." in name or ".lora_b." in name
    }
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in learned_names
    }


def load_learned_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected learned checkpoint keys: {result.unexpected_keys[:3]}")
    required = set(state)
    if not required:
        raise RuntimeError("Learned checkpoint is empty")


def set_lora_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        if ".lora_a." in name or ".lora_b." in name:
            parameter.requires_grad = bool(trainable)


def build_optimizer_and_scheduler(
    model: SharedMultiviewFactorizedModel,
    training: dict,
    *,
    optimizer_steps_per_epoch: int,
    phase_epochs: int,
) -> tuple[torch.optim.Optimizer, object, list[dict]]:
    groups = trainable_parameter_groups(
        model,
        head_lr=float(training["head_learning_rate"]),
        adapter_lr=float(training["adapter_learning_rate"]),
        lora_lr=float(training["lora_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    evidence = [
        {
            "group_name": group["group_name"],
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
            "parameters": int(sum(parameter.numel() for parameter in group["params"])),
        }
        for group in groups
    ]
    optimizer = torch.optim.AdamW(groups)
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=max(1, optimizer_steps_per_epoch * phase_epochs),
        warmup_fraction=float(training["warmup_fraction"]),
    )
    return optimizer, scheduler, evidence


def train_epoch(
    model: SharedMultiviewFactorizedModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    *,
    grad_accumulation: int,
    gradient_clip: float,
    label_smoothing: float,
    auxiliary_weight: float,
    posture_weight: torch.Tensor,
    motion_weight: torch.Tensor,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    optimizer_steps = 0
    started = time.perf_counter()
    samples = 0
    for step, batch in enumerate(loader):
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(
                batch["tight_pixels"].to(device, non_blocking=True),
                batch["context_pixels"].to(device, non_blocking=True),
                batch["geometry"].to(device, non_blocking=True),
            )
            loss_output = multiview_factorized_loss(
                output,
                batch["label"].to(device, non_blocking=True),
                label_smoothing=label_smoothing,
                auxiliary_view_weight=auxiliary_weight,
                posture_weight=posture_weight,
                motion_weight=motion_weight,
            )
            raw_loss = loss_output["loss"]
            window_start = (step // grad_accumulation) * grad_accumulation
            window_size = min(grad_accumulation, len(loader) - window_start)
            loss = raw_loss / window_size
        scaler.scale(loss).backward()
        losses.append(float(raw_loss.detach().item()))
        samples += len(batch["label"])
        if (step + 1) % grad_accumulation == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=gradient_clip,
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_steps += 1
    elapsed = time.perf_counter() - started
    if not optimizer_steps:
        raise RuntimeError("Training epoch completed without an optimizer update")
    return {
        "loss": float(np.mean(losses)),
        "optimizer_steps": optimizer_steps,
        "seconds": elapsed,
        "people_per_second": float(samples / elapsed),
    }


def checkpoint_evidence_cached(
    model_kind: str,
    root: Path,
) -> tuple[dict, Path]:
    path = root / ".runs" / "vcoco_v3" / "neural" / "checkpoint_evidence" / f"{model_kind}.json"
    if path.is_file():
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("model_kind") != model_kind:
            raise RuntimeError("Cached checkpoint evidence belongs to a different model")
        return evidence, path
    evidence = {"model_kind": model_kind, **local_checkpoint_evidence(model_kind)}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, evidence)
    return evidence, path


def main() -> None:
    global FAILURE_DIRECTORY
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    FAILURE_DIRECTORY = output_dir
    repository_root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    grid, lock, candidate, development = validate_and_load(args)
    train_index, stopping_index, held_index, split_evidence = split_indices(development, grid, args)
    configuration = {
        "role": args.role,
        "candidate": candidate,
        "outer_fold": args.outer_fold,
        "inner_fold": args.inner_fold,
        "seed": args.seed,
        "workers": args.workers,
        "training": grid["training"],
        "architecture": grid["architecture"],
        "views": grid["views"],
    }
    request_core = {
        "status": "VCOCO_V3_NEURAL_RUN_REQUEST",
        "configuration": configuration,
        "split_evidence": split_evidence,
        "source_sha256": {
            "neural_grid": sha256_file(args.grid.resolve()),
            "neural_grid_lock": sha256_file(args.neural_lock.resolve()),
            "train_manifest": sha256_file(args.train_manifest.resolve()),
            "val_manifest": sha256_file(args.val_manifest.resolve()),
            "runner": sha256_file(Path(__file__).resolve()),
            "neural_module": sha256_file(repository_root / "src/hac/vcoco_v3_neural.py"),
        },
        "human_pilot_labels_used_for_selection": False,
        "official_v2_test_rows_read": 0,
        "official_v2_test_predictions_run": False,
    }
    if args.role == "outer_fit":
        request_core["source_sha256"]["inner_selection_lock"] = sha256_file(
            args.selection_lock.resolve()
        )
    request_hash = hashlib.sha256(
        json.dumps(request_core, sort_keys=True).encode("utf-8")
    ).hexdigest()
    request = {
        **request_core,
        "request_sha256": request_hash,
        "git_revision_at_start": git_revision(repository_root),
    }
    request_path = output_dir / "request.json"
    if request_path.is_file():
        previous = json.loads(request_path.read_text(encoding="utf-8"))
        if previous.get("request_sha256") != request_hash:
            raise RuntimeError("The existing output directory contains a different run request")
    else:
        write_json(request_path, request)
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("status") == "VCOCO_V3_NEURAL_RUN_COMPLETE":
            print(json.dumps(previous, indent=2, sort_keys=True), flush=True)
            return

    training = grid["training"]
    architecture = grid["architecture"]
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("V-COCO v3 neural training requires CUDA")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    backbone, backbone_dim = build_pinned_backbone(candidate["model_kind"])
    replacements = configure_parameter_efficient_backbone(
        backbone,
        strategy=candidate["strategy"],
        top_blocks=int(candidate["top_blocks"]),
        lora_rank=max(1, int(candidate["lora_rank"])),
        lora_alpha=max(1.0, float(candidate["lora_alpha"])),
        lora_dropout=float(candidate["lora_dropout"]),
    )
    if replacements and training["gradient_checkpointing"]:
        enable = getattr(backbone, "gradient_checkpointing_enable", None)
        if enable is not None:
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = SharedMultiviewFactorizedModel(
        backbone,
        backbone_dim=backbone_dim,
        adapter_dim=int(architecture["adapter_dim"]),
        geometry_dim=int(architecture["geometry_features"]),
        dropout=float(candidate["dropout"]),
    ).to(device)
    checkpoint_evidence, checkpoint_evidence_path = checkpoint_evidence_cached(
        candidate["model_kind"], repository_root
    )

    training_transform = PairedPersonSafeTransform(
        image_size=int(grid["views"]["image_size"]),
        random_erasing_probability=float(candidate["random_erasing_probability"]),
        training=True,
    )
    evaluation_transform = PairedPersonSafeTransform(
        image_size=int(grid["views"]["image_size"]),
        random_erasing_probability=0.0,
        training=False,
    )
    batch_size = int(training["batch_size"])
    train_loader = make_loader(
        development.iloc[train_index],
        transform=training_transform,
        batch_size=batch_size,
        shuffle=True,
        seed=args.seed,
        workers=args.workers,
    )
    stopping_loader = make_loader(
        development.iloc[stopping_index],
        transform=evaluation_transform,
        batch_size=batch_size,
        shuffle=False,
        seed=args.seed,
        workers=args.workers,
    )
    held_loader = make_loader(
        development.iloc[held_index],
        transform=evaluation_transform,
        batch_size=batch_size,
        shuffle=False,
        seed=args.seed,
        workers=args.workers,
    )
    label_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    training_labels = development.iloc[train_index]["label_3"].map(label_mapping).to_numpy(int)
    posture_weight, motion_weight = hierarchical_weights(training_labels, device)
    accumulation = int(training["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    maximum_epochs = int(training["maximum_epochs"])
    probe_epochs = int(training["probe_epochs"]) if replacements else 0
    current_phase = "probe" if probe_epochs else "adapt"
    set_lora_trainable(model, current_phase == "adapt")
    phase_epochs = probe_epochs if current_phase == "probe" else maximum_epochs
    optimizer, scheduler, optimizer_evidence = build_optimizer_and_scheduler(
        model,
        training,
        optimizer_steps_per_epoch=steps_per_epoch,
        phase_epochs=phase_epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=4096.0)
    start_epoch = 0
    history = []
    best_metrics = None
    best_epoch = -1
    best_phase = None
    stale_epochs = 0
    last_checkpoint = output_dir / "last_checkpoint.pt"
    best_checkpoint = output_dir / "best_checkpoint.pt"
    if last_checkpoint.is_file():
        payload = torch.load(last_checkpoint, map_location=device, weights_only=False)
        if payload.get("request_sha256") != request_hash:
            raise RuntimeError("Last checkpoint belongs to a different neural request")
        load_learned_state(model, payload["learned_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        history = payload["history"]
        best_metrics = payload["best_metrics"]
        best_epoch = int(payload["best_epoch"])
        best_phase = payload["best_phase"]
        stale_epochs = int(payload["stale_epochs"])
        desired_phase = "probe" if replacements and start_epoch < probe_epochs else "adapt"
        if desired_phase != current_phase:
            current_phase = desired_phase
            set_lora_trainable(model, current_phase == "adapt")
            phase_epochs = (
                probe_epochs if current_phase == "probe" else maximum_epochs - probe_epochs
            )
            optimizer, scheduler, optimizer_evidence = build_optimizer_and_scheduler(
                model,
                training,
                optimizer_steps_per_epoch=steps_per_epoch,
                phase_epochs=phase_epochs,
            )
        if payload["phase"] == current_phase:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            scheduler.load_state_dict(payload["scheduler_state_dict"])
            scaler.load_state_dict(payload["scaler_state_dict"])
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if device.type == "cuda" and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        train_loader.generator.set_state(payload["loader_generator_state"])

    for epoch in range(start_epoch, maximum_epochs):
        if replacements and epoch == probe_epochs and current_phase == "probe":
            current_phase = "adapt"
            set_lora_trainable(model, True)
            optimizer, scheduler, optimizer_evidence = build_optimizer_and_scheduler(
                model,
                training,
                optimizer_steps_per_epoch=steps_per_epoch,
                phase_epochs=maximum_epochs - probe_epochs,
            )
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=4096.0)
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            grad_accumulation=accumulation,
            gradient_clip=float(training["gradient_clip_norm"]),
            label_smoothing=float(training["label_smoothing"]),
            auxiliary_weight=float(architecture["auxiliary_view_loss_weight"]),
            posture_weight=posture_weight,
            motion_weight=motion_weight,
        )
        stopping = evaluate(model, stopping_loader, device)
        epoch_record = {
            "epoch": epoch,
            "phase": current_phase,
            "training": train_metrics,
            "early_stopping": stopping["metrics"],
        }
        history.append(epoch_record)
        improved = is_better_validation(stopping["metrics"], best_metrics)
        if improved:
            best_metrics = stopping["metrics"]
            best_epoch = epoch
            best_phase = current_phase
            stale_epochs = 0
            torch.save(
                {
                    "request_sha256": request_hash,
                    "epoch": epoch,
                    "phase": current_phase,
                    "metrics": best_metrics,
                    "learned_state_dict": learned_state_dict(model),
                },
                best_checkpoint,
            )
        else:
            stale_epochs += 1
        torch.save(
            {
                "request_sha256": request_hash,
                "epoch": epoch,
                "phase": current_phase,
                "learned_state_dict": learned_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "history": history,
                "best_metrics": best_metrics,
                "best_epoch": best_epoch,
                "best_phase": best_phase,
                "stale_epochs": stale_epochs,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
                "loader_generator_state": train_loader.generator.get_state(),
            },
            last_checkpoint,
        )
        write_json(output_dir / "history.json", history)
        print(json.dumps(epoch_record, sort_keys=True), flush=True)
        minimum_epochs = int(training["minimum_epochs"])
        if epoch + 1 >= minimum_epochs and stale_epochs >= int(training["early_stopping_patience"]):
            break

    if not best_checkpoint.is_file():
        raise RuntimeError("Neural training completed without a best checkpoint")
    best_payload = torch.load(best_checkpoint, map_location=device, weights_only=False)
    load_learned_state(model, best_payload["learned_state_dict"])
    held = evaluate(model, held_loader, device)
    predictions_path = output_dir / "held_predictions.npz"
    np.savez_compressed(
        predictions_path,
        person_ids=held["person_ids"],
        image_ids=held["image_ids"],
        labels=held["labels"],
        probabilities=held["probabilities"],
        gate_weights=held["gate_weights"],
        class_names=np.asarray(CLASS_NAMES),
    )
    learned_hash = sha256_file(best_checkpoint)
    summary = {
        "status": "VCOCO_V3_NEURAL_RUN_COMPLETE",
        "role": args.role,
        "candidate_id": args.candidate_id,
        "model_kind": candidate["model_kind"],
        "outer_fold": args.outer_fold,
        "inner_fold": args.inner_fold,
        "seed": args.seed,
        "best_epoch": int(best_payload["epoch"]),
        "best_phase": best_payload["phase"],
        "early_stopping_metrics": best_payload["metrics"],
        "held_metrics": held["metrics"],
        "split_evidence": split_evidence,
        "parameter_counts": parameter_count_summary(model),
        "lora_replacements": replacements,
        "optimizer_groups_last_phase": optimizer_evidence,
        "checkpoint_evidence_sha256": sha256_file(checkpoint_evidence_path),
        "checkpoint_total_bytes": checkpoint_evidence["total_bytes"],
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "training_device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "runtime_seconds": time.perf_counter() - started,
        "request_sha256": request_hash,
        "human_pilot_labels_used_for_selection": False,
        "official_v2_test_rows_read": 0,
        "official_v2_test_predictions_run": False,
        "artifact_sha256": {
            predictions_path.name: sha256_file(predictions_path),
            best_checkpoint.name: learned_hash,
            "history.json": sha256_file(output_dir / "history.json"),
        },
    }
    write_json(summary_path, summary)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if FAILURE_DIRECTORY is not None:
            write_json(
                FAILURE_DIRECTORY / "failure.json",
                {
                    "status": "VCOCO_V3_NEURAL_RUN_FAILED",
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
