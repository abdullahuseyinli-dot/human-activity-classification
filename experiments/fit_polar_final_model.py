"""Fit one locked POLAR neural model on all clean development rows without test access."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from train_polar_candidate import checkpoint_evidence, implementation_hashes, train_epoch

from hac.augmentations import build_train_transform
from hac.config import ModelConfig
from hac.data import make_loader
from hac.models import parameter_counts
from hac.polar import sha256_file
from hac.polar_models import build_polar_model
from hac.polar_training import (
    TASK_LABELS,
    inverse_frequency_weights,
    optimizer_parameter_groups,
    validate_development_manifest,
    warmup_cosine_scheduler,
)
from hac.training import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-kind", choices=["convnext_small", "dinov2_small", "dinov2_base"], required=True
    )
    parser.add_argument("--task", choices=sorted(TASK_LABELS), default="label_4")
    parser.add_argument(
        "--view",
        choices=["full_frame", "person_context_10", "person_context_25"],
        required=True,
    )
    parser.add_argument(
        "--unfreeze-strategy",
        choices=["head_only", "last_stage", "probe_only", "top_blocks", "full_backbone"],
        required=True,
    )
    parser.add_argument("--top-n-blocks", type=int)
    parser.add_argument("--augmentation", choices=["mild", "moderate"], default="mild")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--layer-decay", type=float)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--class-balance", choices=["unweighted", "inverse_frequency"], default="unweighted"
    )
    parser.add_argument("--fixed-epochs", type=int, required=True)
    parser.add_argument("--warmup-fraction", type=float, default=0.10)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, required=True)
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


def configuration(args: argparse.Namespace) -> dict:
    return {
        "model_kind": args.model_kind,
        "task": args.task,
        "view": args.view,
        "unfreeze_strategy": args.unfreeze_strategy,
        "top_n_blocks": args.top_n_blocks,
        "augmentation": args.augmentation,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "workers": args.workers,
        "head_lr": args.head_lr,
        "backbone_lr": args.backbone_lr,
        "weight_decay": args.weight_decay,
        "layer_decay": args.layer_decay,
        "dropout": args.dropout,
        "mixup_alpha": args.mixup_alpha,
        "label_smoothing": args.label_smoothing,
        "class_balance": args.class_balance,
        "fixed_epochs": args.fixed_epochs,
        "warmup_fraction": args.warmup_fraction,
        "gradient_clip": args.gradient_clip,
    }


def validate_lock(args: argparse.Namespace, config: dict) -> tuple[dict, str]:
    lock_path = args.selection_lock.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "FINAL_SELECTION_LOCKED_PRE_TEST":
        raise RuntimeError("Final fitting requires a FINAL_SELECTION_LOCKED_PRE_TEST lock")
    if lock.get("test_rows_read") != 0 or lock.get("test_used_for_selection"):
        raise RuntimeError("Selection lock violates the test gate")
    try:
        specification = lock["final_neural_fits"][args.model_id]
    except KeyError as error:
        raise RuntimeError(f"Model id is absent from selection lock: {args.model_id}") from error
    if int(args.seed) not in specification["seeds"]:
        raise RuntimeError(f"Seed {args.seed} is not locked for {args.model_id}")
    expected = specification["configuration"]
    if config != expected:
        raise RuntimeError(
            "Final-fit arguments differ from the selection lock: "
            f"expected={expected!r}, observed={config!r}"
        )
    return lock, sha256_file(lock_path)


def validate_arguments(args: argparse.Namespace) -> None:
    if args.fixed_epochs < 1 or args.batch_size < 1 or args.grad_accum_steps < 1:
        raise ValueError("fixed-epochs, batch-size, and grad-accum-steps must be positive")
    if args.gradient_clip <= 0.0:
        raise ValueError("gradient-clip must be positive")
    if args.layer_decay is not None and not 0.0 < args.layer_decay <= 1.0:
        raise ValueError("layer-decay must be in (0, 1]")
    if args.layer_decay is not None and args.unfreeze_strategy != "full_backbone":
        raise ValueError("layer-decay is reserved for full-backbone adaptation")


def main() -> None:
    args = parse_args()
    validate_arguments(args)
    torch.set_float32_matmul_precision("high")
    config = configuration(args)
    _, selection_lock_hash = validate_lock(args, config)
    manifest_path = args.manifest.resolve()
    repository_root = Path(__file__).resolve().parents[1]
    request_core = {
        "status": "LOCKED_FINAL_DEVELOPMENT_FIT_REQUEST",
        "model_id": args.model_id,
        "seed": args.seed,
        "configuration": config,
        "selection_lock_sha256": selection_lock_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "implementation_sha256": implementation_hashes(repository_root),
        "test_rows_read": 0,
    }
    request_hash = hashlib.sha256(
        json.dumps(request_core, sort_keys=True).encode("utf-8")
    ).hexdigest()
    request = {**request_core, "request_sha256": request_hash}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "request.json"
    if request_path.is_file():
        existing = json.loads(request_path.read_text(encoding="utf-8"))
        if existing.get("request_sha256") != request_hash:
            raise RuntimeError(f"Existing final-fit request differs: {request_path}")
    else:
        write_json(request_path, request)

    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETE" and existing.get("request_sha256") == request_hash:
            print(json.dumps(existing, indent=2, sort_keys=True), flush=True)
            return

    frame = validate_development_manifest(
        pd.read_csv(manifest_path, dtype={"image_id": str}), args.task
    )
    frame = frame.copy()
    class_names = list(TASK_LABELS[args.task])
    class_to_index = {name: index for index, name in enumerate(class_names)}
    numeric_labels = frame["label"].map(class_to_index).to_numpy(dtype=int)

    model_config = ModelConfig(
        model_kind=args.model_kind,
        augmentation_strength=args.augmentation,
        batch_size=args.batch_size,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        mixup_alpha=args.mixup_alpha,
        unfreeze_strategy=args.unfreeze_strategy,
        top_n_blocks=args.top_n_blocks,
    )
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_polar_model(model_config, num_classes=len(class_names)).to(device)
    groups = optimizer_parameter_groups(
        model,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        model_kind=args.model_kind,
        layer_decay=args.layer_decay,
    )
    optimizer_group_spec = [
        {
            "group_name": group["group_name"],
            "initial_lr": float(group["lr"]),
            "lr_scale": float(group["lr_scale"]),
            "weight_decay": float(group["weight_decay"]),
            "parameter_count": int(sum(parameter.numel() for parameter in group["params"])),
        }
        for group in groups
    ]
    optimizer = torch.optim.AdamW(groups)
    batches_per_epoch = math.ceil(len(frame) / args.batch_size)
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum_steps)
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=optimizer_steps_per_epoch * args.fixed_epochs,
        warmup_fraction=args.warmup_fraction,
    )
    class_weights = None
    if args.class_balance == "inverse_frequency":
        class_weights = inverse_frequency_weights(numeric_labels, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=4096.0
    )

    last_checkpoint = output_dir / "last_checkpoint.pt"
    start_epoch = 1
    history: list[dict] = []
    if last_checkpoint.is_file():
        payload = torch.load(last_checkpoint, map_location=device, weights_only=False)
        if payload.get("request_sha256") != request_hash:
            raise RuntimeError("Final-fit checkpoint request hash differs")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        scaler.load_state_dict(payload["scaler_state_dict"])
        history = list(payload["history"])
        start_epoch = int(payload["epoch"]) + 1
        print(f"[resume] epoch={start_epoch}", flush=True)

    started = time.perf_counter()
    for epoch in range(start_epoch, args.fixed_epochs + 1):
        epoch_seed = args.seed * 10_000 + epoch
        seed_everything(epoch_seed)
        loader = make_loader(
            frame,
            class_to_index,
            build_train_transform(args.augmentation),
            batch_size=args.batch_size,
            shuffle=True,
            seed=epoch_seed,
            workers=args.workers,
            view=args.view,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        statistics = train_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            criterion,
            device,
            scaler,
            mixup_alpha=args.mixup_alpha,
            grad_accum_steps=args.grad_accum_steps,
            gradient_clip=args.gradient_clip,
            frozen_backbone=args.unfreeze_strategy in {"head_only", "probe_only"},
        )
        record = {
            "epoch": epoch,
            "train_loss": statistics["loss"],
            "train_seconds": statistics["seconds"],
            "images_per_second": statistics["images_per_second"],
            "optimizer_updates": statistics["optimizer_updates"],
            "skipped_overflow_updates": statistics["skipped_overflow_updates"],
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "peak_gpu_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / 1024**2)
                if device.type == "cuda"
                else 0.0
            ),
        }
        history.append(record)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "history": history,
                "request_sha256": request_hash,
            },
            last_checkpoint,
        )
        print(
            f"[epoch {epoch:02d}/{args.fixed_epochs:02d}] train_loss={statistics['loss']:.4f}",
            flush=True,
        )

    final_checkpoint = output_dir / "final_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": args.fixed_epochs,
            "request_sha256": request_hash,
            "class_names": class_names,
            "configuration": config,
        },
        final_checkpoint,
    )
    summary = {
        "status": "COMPLETE",
        "stage": "LOCKED_FINAL_TRAIN_PLUS_VALIDATION_FIT",
        "model_id": args.model_id,
        "seed": args.seed,
        "request_sha256": request_hash,
        "selection_lock_sha256": selection_lock_hash,
        "manifest_sha256": request["manifest_sha256"],
        "configuration": config,
        "development_rows": len(frame),
        "class_counts": frame["label"].value_counts().sort_index().to_dict(),
        "epochs_completed": len(history),
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "optimizer_parameter_groups": optimizer_group_spec,
        "parameter_counts": parameter_counts(model),
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "pretrained_checkpoint": checkpoint_evidence(args.model_kind),
        "runtime_seconds_this_invocation": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "test_rows_read": 0,
        "test_used_for_selection": False,
    }
    write_json(summary_path, summary)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True), flush=True)
    del model, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
