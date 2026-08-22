"""Train one validation-screened POLAR candidate without touching the test split."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from torch import nn

from hac.augmentations import build_eval_transform, build_train_transform
from hac.config import ModelConfig
from hac.data import make_loader
from hac.models import parameter_counts
from hac.polar import sha256_file
from hac.polar_models import DINO_MODEL_SPECS, build_polar_model
from hac.polar_training import (
    TASK_LABELS,
    evaluate_classifier,
    inverse_frequency_weights,
    is_better_validation,
    nested_stratified_subset,
    optimizer_parameter_groups,
    validate_development_manifest,
    warmup_cosine_scheduler,
)
from hac.training import mixup_batch, seed_everything

FAILURE_DIRECTORY: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-role",
        choices=[
            "engineering_smoke",
            "adaptation_screen",
            "regularization_screen",
            "confirmation",
            "scale",
        ],
        default="adaptation_screen",
    )
    parser.add_argument(
        "--model-kind",
        choices=["convnext_small", "dinov2_small", "dinov2_base"],
        required=True,
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
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--warmup-fraction", type=float, default=0.10)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", default="all")
    return parser.parse_args()


def parse_train_size(value: str) -> int | None:
    if value == "all":
        return None
    size = int(value)
    if size < 1:
        raise ValueError("train-size must be positive or 'all'")
    return size


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


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
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


def implementation_hashes(root: Path) -> dict[str, str]:
    relative_paths = [
        "experiments/train_polar_candidate.py",
        "src/hac/augmentations.py",
        "src/hac/config.py",
        "src/hac/data.py",
        "src/hac/metrics.py",
        "src/hac/models.py",
        "src/hac/polar.py",
        "src/hac/polar_models.py",
        "src/hac/polar_training.py",
        "src/hac/training.py",
    ]
    return {value: sha256_file(root / value) for value in relative_paths}


def checkpoint_evidence(model_kind: str) -> dict:
    if model_kind in DINO_MODEL_SPECS:
        specification = DINO_MODEL_SPECS[model_kind]
        root = Path(
            snapshot_download(
                repo_id=specification["model_id"],
                revision=specification["revision"],
                local_files_only=True,
            )
        )
        files = []
        for filename in ("model.safetensors", "pytorch_model.bin", "config.json"):
            path = root / filename
            if path.is_file():
                files.append(
                    {
                        "path": filename,
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        return {**specification, "files": files}
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "convnext_small-0c510722.pth"
    return {
        "model_id": "torchvision/convnext_small",
        "revision": "ConvNeXt_Small_Weights.IMAGENET1K_V1",
        "files": [
            {
                "path": checkpoint.name,
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
            }
        ],
    }


def configuration(args: argparse.Namespace) -> dict:
    return {
        "model_kind": args.model_kind,
        "run_role": args.run_role,
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
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "warmup_fraction": args.warmup_fraction,
        "gradient_clip": args.gradient_clip,
        "seed": args.seed,
        "train_size": args.train_size,
    }


def validate_arguments(args: argparse.Namespace) -> None:
    positive = {
        "batch-size": args.batch_size,
        "grad-accum-steps": args.grad_accum_steps,
        "max-epochs": args.max_epochs,
        "min-epochs": args.min_epochs,
        "patience": args.patience,
    }
    for name, value in positive.items():
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
    if args.min_epochs > args.max_epochs:
        raise ValueError("min-epochs cannot exceed max-epochs")
    if args.gradient_clip <= 0.0:
        raise ValueError("gradient-clip must be positive")
    if args.layer_decay is not None and not 0.0 < args.layer_decay <= 1.0:
        raise ValueError("layer-decay must be in (0, 1]")
    if args.layer_decay is not None and args.unfreeze_strategy != "full_backbone":
        raise ValueError("layer-decay is reserved for full-backbone adaptation")


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    scaler,
    *,
    mixup_alpha: float,
    grad_accum_steps: int,
    gradient_clip: float,
    frozen_backbone: bool,
) -> dict[str, float]:
    model.train()
    if frozen_backbone:
        model.backbone.eval()
        model.classifier.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    samples = 0
    optimizer_updates = 0
    skipped_overflow_updates = 0
    started = time.perf_counter()
    for step, batch in enumerate(loader):
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        mixed, labels_a, labels_b, weight = mixup_batch(inputs, labels, mixup_alpha)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(mixed)
            raw_loss = weight * criterion(logits, labels_a) + (1.0 - weight) * criterion(
                logits, labels_b
            )
            window_start = (step // grad_accum_steps) * grad_accum_steps
            window_size = min(grad_accum_steps, len(loader) - window_start)
            loss = raw_loss / window_size
        scaler.scale(loss).backward()
        losses.append(float(raw_loss.detach().item()))
        samples += len(labels)
        should_step = (step + 1) % grad_accum_steps == 0 or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            try:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip,
                    error_if_nonfinite=True,
                )
            except RuntimeError as error:
                if "non-finite" not in str(error).lower() or not scaler.is_enabled():
                    raise
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                skipped_overflow_updates += 1
                continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_updates += 1
    elapsed = time.perf_counter() - started
    if optimizer_updates == 0:
        raise FloatingPointError("Epoch completed without a finite optimizer update")
    return {
        "loss": float(np.mean(losses)),
        "seconds": float(elapsed),
        "images_per_second": float(samples / elapsed),
        "optimizer_updates": int(optimizer_updates),
        "skipped_overflow_updates": int(skipped_overflow_updates),
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_epoch: int,
    best_metrics: dict | None,
    stale_epochs: int,
    history: list[dict],
    request_hash: str,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_metrics": best_metrics,
            "stale_epochs": int(stale_epochs),
            "history": history,
            "request_sha256": request_hash,
        },
        path,
    )


def main() -> None:
    global FAILURE_DIRECTORY
    args = parse_args()
    validate_arguments(args)
    torch.set_float32_matmul_precision("high")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    FAILURE_DIRECTORY = output_dir
    manifest_path = args.manifest.resolve()
    repository_root = Path(__file__).resolve().parents[1]

    config_values = configuration(args)
    request_core = {
        "status": "DEVELOPMENT_TRAINING_REQUEST",
        "configuration": config_values,
        "manifest_sha256": sha256_file(manifest_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "implementation_sha256": implementation_hashes(repository_root),
        "test_rows_read": 0,
        "test_used_for_selection": False,
    }
    request_encoded = json.dumps(request_core, sort_keys=True).encode("utf-8")
    request_hash = hashlib.sha256(request_encoded).hexdigest()
    request = {
        **request_core,
        "git_revision_at_start": git_revision(repository_root),
        "request_sha256": request_hash,
    }
    request_path = output_dir / "request.json"
    if request_path.is_file():
        previous = json.loads(request_path.read_text(encoding="utf-8"))
        if previous.get("request_sha256") != request_hash:
            raise RuntimeError(f"Existing run request differs: {request_path}")
        request = previous
    else:
        write_json(request_path, request)

    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("status") == "COMPLETE" and previous.get("request_sha256") == request_hash:
            print(json.dumps(previous, indent=2, sort_keys=True), flush=True)
            return

    frame = validate_development_manifest(
        pd.read_csv(manifest_path, dtype={"image_id": str}), args.task
    )
    train_frame = frame[frame["split"].eq("train")].copy()
    validation_frame = frame[frame["split"].eq("val")].copy()
    requested_train_size = parse_train_size(args.train_size)
    train_frame = nested_stratified_subset(
        train_frame,
        requested_train_size,
        label_column="label",
        seed=20260822,
    )
    class_names = list(TASK_LABELS[args.task])
    class_to_index = {name: index for index, name in enumerate(class_names)}
    train_labels = train_frame["label"].map(class_to_index).to_numpy(dtype=int)

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
    optimizer = torch.optim.AdamW(groups)
    batches_per_epoch = math.ceil(len(train_frame) / args.batch_size)
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum_steps)
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=optimizer_steps_per_epoch * args.max_epochs,
        warmup_fraction=args.warmup_fraction,
    )
    class_weights = None
    if args.class_balance == "inverse_frequency":
        class_weights = inverse_frequency_weights(train_labels, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
        init_scale=4096.0,
    )
    validation_loader = make_loader(
        validation_frame,
        class_to_index,
        build_eval_transform(),
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        workers=args.workers,
        view=args.view,
    )

    last_checkpoint = output_dir / "last_checkpoint.pt"
    best_checkpoint = output_dir / "best_checkpoint.pt"
    start_epoch = 1
    best_epoch = 0
    best_metrics = None
    stale_epochs = 0
    history: list[dict] = []
    if last_checkpoint.is_file():
        payload = torch.load(last_checkpoint, map_location=device, weights_only=False)
        if payload.get("request_sha256") != request_hash:
            raise RuntimeError("Checkpoint request hash differs")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        scaler.load_state_dict(payload["scaler_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        best_epoch = int(payload["best_epoch"])
        best_metrics = payload["best_metrics"]
        stale_epochs = int(payload["stale_epochs"])
        history = list(payload["history"])
        print(f"[resume] epoch={start_epoch}", flush=True)

    run_started = time.perf_counter()
    stopped_early = False
    for epoch in range(start_epoch, args.max_epochs + 1):
        seed_everything(args.seed * 10_000 + epoch)
        train_loader = make_loader(
            train_frame,
            class_to_index,
            build_train_transform(args.augmentation),
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed * 10_000 + epoch,
            workers=args.workers,
            view=args.view,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_stats = train_epoch(
            model,
            train_loader,
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
        validation = evaluate_classifier(model, validation_loader, criterion, device)
        metrics = validation["metrics"]
        improved = is_better_validation(metrics, best_metrics)
        if improved:
            best_epoch = epoch
            best_metrics = metrics
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "request_sha256": request_hash,
                    "class_names": class_names,
                },
                best_checkpoint,
            )
        else:
            stale_epochs += 1
        record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_seconds": train_stats["seconds"],
            "images_per_second": train_stats["images_per_second"],
            "optimizer_updates": train_stats["optimizer_updates"],
            "skipped_overflow_updates": train_stats["skipped_overflow_updates"],
            "validation_loss": validation["loss"],
            **{f"validation_{key}": value for key, value in metrics.items()},
            "improved": improved,
            "stale_epochs": stale_epochs,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "peak_gpu_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / 1024**2)
                if device.type == "cuda"
                else 0.0
            ),
        }
        history.append(record)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        save_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            stale_epochs=stale_epochs,
            history=history,
            request_hash=request_hash,
        )
        print(
            f"[epoch {epoch:02d}] train_loss={train_stats['loss']:.4f} "
            f"val_macro_f1={metrics['macro_f1']:.4f} val_log_loss={metrics['log_loss']:.4f} "
            f"best={best_epoch:02d} stale={stale_epochs}",
            flush=True,
        )
        if epoch >= args.min_epochs and stale_epochs >= args.patience:
            stopped_early = True
            break

    if not best_checkpoint.is_file():
        raise RuntimeError("Training produced no best checkpoint")
    best_payload = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state_dict"])
    final_validation = evaluate_classifier(
        model,
        validation_loader,
        criterion,
        device,
        return_features=True,
    )
    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        logits=final_validation["logits"],
        probabilities=final_validation["probabilities"],
        predictions=final_validation["predictions"],
        labels=final_validation["labels"],
        image_ids=np.asarray(final_validation["image_ids"]),
        class_names=np.asarray(class_names),
    )
    write_json(output_dir / "validation_metrics.json", final_validation["metrics"])
    checkpoint_hash = sha256_file(best_checkpoint)
    summary = {
        "status": "COMPLETE",
        "stage": "DEVELOPMENT_ONLY_CANDIDATE",
        "request_sha256": request_hash,
        "manifest_sha256": request["manifest_sha256"],
        "configuration": config_values,
        "train_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "test_rows_read": 0,
        "test_used_for_selection": False,
        "class_names": class_names,
        "train_class_counts": train_frame["label"].value_counts().sort_index().to_dict(),
        "best_epoch": int(best_payload["epoch"]),
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "best_validation_metrics": final_validation["metrics"],
        "parameter_counts": parameter_counts(model),
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "checkpoint_sha256": checkpoint_hash,
        "pretrained_checkpoint": checkpoint_evidence(args.model_kind),
        "runtime_seconds_this_invocation": time.perf_counter() - run_started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True), flush=True)
    del model, optimizer, scheduler, validation_loader, final_validation
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if FAILURE_DIRECTORY is not None:
            write_json(
                FAILURE_DIRECTORY / "failure.json",
                {
                    "status": "FAILED",
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "test_rows_read": 0,
                },
            )
        raise
