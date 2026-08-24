"""Train one locked CPTR development candidate on CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from hac.cptr import cptr_loss, cptr_primary_loss_per_sample
from hac.cptr_features import (
    CPTRFeatureDataset,
    coherent_feature_augmentation,
    jittered_camera_kwargs,
    model_kwargs_from_batch,
    motion_null_kwargs,
    reversed_kwargs,
)
from hac.cptr_training import (
    better_metrics,
    build_cptr_model,
    evaluate_cptr,
    load_frozen_v3_baselines,
    model_evidence,
)
from hac.polar import sha256_file
from hac.polar_training import warmup_cosine_scheduler
from hac.training import seed_everything
from hac.vcoco_v3_temporal_training import hierarchical_class_weights

FAILURE_DIRECTORY: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiments/okutama_cptr_protocol.json")
    )
    parser.add_argument(
        "--protocol-lock", type=Path, default=Path(".runs/cptr/protocol_lock.json")
    )
    parser.add_argument("--grid", type=Path, default=Path("experiments/okutama_cptr_grid.json"))
    parser.add_argument("--grid-lock", type=Path, default=Path(".runs/cptr/grid_lock.json"))
    parser.add_argument(
        "--v3-grid", type=Path, default=Path("experiments/okutama_temporal_grid.json")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/development_manifest.csv"),
    )
    parser.add_argument(
        "--v3-root", type=Path, default=Path(".runs/vcoco_v3/temporal/development")
    )
    parser.add_argument(
        "--motion-store", type=Path, default=Path(".runs/cptr/motion_features/store.json")
    )
    parser.add_argument("--part-store", type=Path)
    parser.add_argument("--pose-store", type=Path)
    parser.add_argument("--siglip-store", type=Path)
    parser.add_argument("--masked-checkpoint", type=Path)
    parser.add_argument("--candidate-id", required=True)
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


def validate_configuration(args: argparse.Namespace) -> tuple[dict, dict, dict, pd.DataFrame]:
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    protocol_path = args.protocol.resolve()
    protocol_lock_path = args.protocol_lock.resolve()
    grid_path = args.grid.resolve()
    grid_lock_path = args.grid_lock.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_lock = json.loads(protocol_lock_path.read_text(encoding="utf-8"))
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    grid_lock = json.loads(grid_lock_path.read_text(encoding="utf-8"))
    if protocol_lock.get("status") != "OKUTAMA_CPTR_PROTOCOL_LOCKED_BEFORE_FIT":
        raise RuntimeError("The CPTR protocol is not locked")
    accepted_grid_statuses = {
        "OKUTAMA_CPTR_GRID_LOCKED_BEFORE_FIT",
        "OKUTAMA_CPTR_ADAPTIVE_GRID_LOCKED_BEFORE_FIT",
        "OKUTAMA_CPTR_STAGE2_GRID_LOCKED_BEFORE_FIT",
        "OKUTAMA_CPTR_STAGE3_GRID_LOCKED_BEFORE_FIT",
        "OKUTAMA_CPTR_STAGE4_GRID_LOCKED_BEFORE_FIT",
    }
    if grid_lock.get("status") not in accepted_grid_statuses:
        raise RuntimeError("The CPTR candidate grid is not locked")
    if protocol_lock["source_sha256"]["protocol"] != sha256_file(protocol_path):
        raise RuntimeError("The CPTR protocol changed after locking")
    locked_grid_hash = grid_lock["source_sha256"].get(
        "grid",
        grid_lock["source_sha256"].get(
            "adaptive_grid",
            grid_lock["source_sha256"].get(
                "stage2_grid",
                grid_lock["source_sha256"].get(
                    "stage3_grid", grid_lock["source_sha256"].get("stage4_grid")
                ),
            ),
        ),
    )
    if locked_grid_hash != sha256_file(grid_path):
        raise RuntimeError("The CPTR grid changed after locking")
    if grid_lock["source_sha256"]["protocol_lock"] != sha256_file(protocol_lock_path):
        raise RuntimeError("The CPTR grid belongs to a different protocol lock")
    candidates = {item["candidate_id"]: item for item in grid["candidates"]}
    if args.candidate_id not in candidates:
        raise ValueError(f"Unknown CPTR candidate: {args.candidate_id}")
    candidate = candidates[args.candidate_id]
    if args.seed not in set(map(int, protocol["training"]["promotion_seeds"])):
        raise ValueError("The CPTR seed is not declared")
    if candidate["use_trajectory"] and not args.motion_store.resolve().is_file():
        raise FileNotFoundError("The trajectory candidate requires the locked motion store")
    if candidate["use_parts"] and (args.part_store is None or not args.part_store.is_file()):
        raise FileNotFoundError("The part candidate requires a part-token store")
    if candidate["use_pose"] and (args.pose_store is None or not args.pose_store.is_file()):
        raise FileNotFoundError("The pose candidate requires a pose store")
    if candidate["use_siglip"] and (
        args.siglip_store is None or not args.siglip_store.is_file()
    ):
        raise FileNotFoundError("The SigLIP candidate requires a specialist store")
    if candidate["use_masked_initialisation"] and (
        args.masked_checkpoint is None or not args.masked_checkpoint.is_file()
    ):
        raise FileNotFoundError("The masked-adapted candidate requires its pretraining checkpoint")
    manifest_path = args.manifest.resolve()
    if protocol_lock["source_sha256"]["development_manifest"] != sha256_file(manifest_path):
        raise RuntimeError("The development manifest changed after locking")
    frame = pd.read_csv(
        manifest_path,
        dtype={"sample_id": str, "recording_id": str, "track_id": str},
    )
    if set(frame["split"].astype(str)) != {"train", "validation", "calibration"}:
        raise RuntimeError("The CPTR manifest split set changed")
    return protocol, candidate, grid_lock, frame


def store_dimensions(path: Path | None, array_name: str, default: int) -> int:
    if path is None:
        return int(default)
    declaration = json.loads(path.resolve().read_text(encoding="utf-8"))
    if declaration.get("status") != "OKUTAMA_CPTR_FEATURE_STORE_COMPLETE":
        raise RuntimeError(f"The CPTR auxiliary store is incomplete: {path}")
    if array_name in declaration:
        return int(declaration[array_name])
    item = declaration["arrays"][array_name]
    array = np.load((path.resolve().parent / item["path"]).resolve(), mmap_mode="r")
    return int(array.shape[-1])


def build_dataset(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    candidate: dict,
) -> CPTRFeatureDataset:
    return CPTRFeatureDataset(
        frame,
        manifest_directory=args.manifest.resolve().parent,
        motion_store=args.motion_store.resolve() if candidate["use_trajectory"] else None,
        part_store=args.part_store.resolve() if candidate["use_parts"] else None,
        pose_store=args.pose_store.resolve() if candidate["use_pose"] else None,
        siglip_store=args.siglip_store.resolve() if candidate["use_siglip"] else None,
        use_compensated_trajectory=bool(candidate["use_compensated_trajectory"]),
        short_samples=8,
        short_seconds=0.5,
        long_samples=8,
        long_seconds=1.0,
    )


def make_loader(
    dataset: CPTRFeatureDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
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


def load_masked_initialisation(model: nn.Module, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("encoder_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("The masked pretraining checkpoint lacks encoder_state_dict")
    loaded = []
    for name in ("short_encoder", "long_encoder"):
        module = getattr(model, name, None)
        if module is not None:
            incompatible = module.load_state_dict(state, strict=False)
            if incompatible.unexpected_keys:
                raise RuntimeError("Masked pretraining exposed unexpected encoder parameters")
            loaded.append(name)
    if not loaded:
        raise RuntimeError("The masked checkpoint has no compatible CPTR temporal encoder")
    return {"loaded_encoders": loaded, "checkpoint_sha256": sha256_file(checkpoint_path)}


class GroupDROState:
    def __init__(self, group_names: list[str], *, step_size: float = 0.05) -> None:
        self.names = tuple(group_names)
        self.index = {name: value for value, name in enumerate(self.names)}
        self.weights = torch.full((len(self.names),), 1.0 / len(self.names), device="cuda")
        self.step_size = float(step_size)

    def objective(
        self,
        per_sample_loss: torch.Tensor,
        group_values: list[str],
    ) -> torch.Tensor:
        indices = torch.as_tensor(
            [self.index[str(value)] for value in group_values],
            device=per_sample_loss.device,
            dtype=torch.long,
        )
        present = torch.unique(indices)
        group_losses = []
        selected_weights = []
        with torch.no_grad():
            for group in present:
                loss = per_sample_loss[indices == group].mean().detach()
                self.weights[group] *= torch.exp(self.step_size * loss.clamp(max=20.0))
            self.weights /= self.weights.sum()
        for group in present:
            group_losses.append(per_sample_loss[indices == group].mean())
            selected_weights.append(self.weights[group])
        weights = torch.stack(selected_weights)
        weights = weights / weights.sum().clamp_min(1e-8)
        return torch.sum(weights * torch.stack(group_losses))

    def summary(self) -> dict[str, float]:
        return {
            name: float(self.weights[index].detach().cpu())
            for index, name in enumerate(self.names)
        }


def main() -> None:
    global FAILURE_DIRECTORY
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    FAILURE_DIRECTORY = output_dir
    protocol, candidate, grid_lock, frame = validate_configuration(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CPTR model fitting requires CUDA; CPU fallback is disabled")
    seed_everything(args.seed)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats(device)
    train_frame = frame[frame["split"].eq("train")].reset_index(drop=True)
    validation_frame = frame[frame["split"].eq("validation")].reset_index(drop=True)
    training = protocol["training"]
    base_store_path = Path(str(train_frame.iloc[0]["feature_path"])).resolve()
    base_store = json.loads(base_store_path.read_text(encoding="utf-8"))
    input_dim = 2 * int(base_store["feature_dimensions"]) + 6
    part_dim = store_dimensions(args.part_store, "part_token_dim", 1)
    pose_joints = store_dimensions(args.pose_store, "pose_joint_count", 1)
    siglip_dim = store_dimensions(args.siglip_store, "feature_dim", 1)
    v3_grid = json.loads(args.v3_grid.resolve().read_text(encoding="utf-8"))
    static, teacher, baseline_hashes = load_frozen_v3_baselines(
        args.seed,
        input_dim=input_dim,
        v3_grid=v3_grid,
        v3_root=args.v3_root.resolve(),
        device=device,
    )
    model = build_cptr_model(
        candidate,
        protocol,
        input_dim=input_dim,
        part_input_dim=part_dim,
        pose_joint_count=pose_joints,
        siglip_dim=siglip_dim,
        static=static,
        teacher=teacher,
    ).to(device)
    masked_evidence = None
    if candidate["use_masked_initialisation"]:
        masked_evidence = load_masked_initialisation(model, args.masked_checkpoint.resolve())
    train_dataset = build_dataset(train_frame, args, candidate)
    validation_dataset = build_dataset(validation_frame, args, candidate)
    train_loader = make_loader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        seed=args.seed,
        workers=args.workers,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        seed=args.seed,
        workers=args.workers,
    )
    builder = kwargs_builder(candidate)
    request_core = {
        "status": "OKUTAMA_CPTR_CANDIDATE_REQUEST",
        "candidate": candidate,
        "seed": int(args.seed),
        "train_samples": int(len(train_frame)),
        "validation_samples": int(len(validation_frame)),
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "protocol": sha256_file(args.protocol.resolve()),
            "protocol_lock": sha256_file(args.protocol_lock.resolve()),
            "grid": sha256_file(args.grid.resolve()),
            "grid_lock": sha256_file(args.grid_lock.resolve()),
            "manifest": sha256_file(args.manifest.resolve()),
            "base_store": sha256_file(base_store_path),
            "motion_store": (
                sha256_file(args.motion_store.resolve()) if candidate["use_trajectory"] else None
            ),
            "part_store": (
                sha256_file(args.part_store.resolve()) if candidate["use_parts"] else None
            ),
            "pose_store": (
                sha256_file(args.pose_store.resolve()) if candidate["use_pose"] else None
            ),
            "siglip_store": (
                sha256_file(args.siglip_store.resolve()) if candidate["use_siglip"] else None
            ),
            "masked_checkpoint": (
                sha256_file(args.masked_checkpoint.resolve())
                if candidate["use_masked_initialisation"]
                else None
            ),
            "runner": sha256_file(Path(__file__).resolve()),
            "model_module": sha256_file(
                Path(__file__).resolve().parents[1] / "src/hac/cptr.py"
            ),
            "feature_module": sha256_file(
                Path(__file__).resolve().parents[1] / "src/hac/cptr_features.py"
            ),
            "training_module": sha256_file(
                Path(__file__).resolve().parents[1] / "src/hac/cptr_training.py"
            ),
        },
    }
    request_hash = hashlib.sha256(
        json.dumps(request_core, sort_keys=True).encode("utf-8")
    ).hexdigest()
    request_path = output_dir / "request.json"
    if request_path.is_file():
        previous = json.loads(request_path.read_text(encoding="utf-8"))
        if previous.get("request_sha256") != request_hash:
            raise RuntimeError("The CPTR output directory contains a different request")
    else:
        write_json(request_path, {**request_core, "request_sha256": request_hash})
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("status") == "OKUTAMA_CPTR_CANDIDATE_COMPLETE":
            print(json.dumps(previous, indent=2, sort_keys=True))
            return

    mapping = {"sitting": 0, "standing": 1, "walking_running": 2}
    train_labels = train_frame["label"].map(mapping).to_numpy(dtype=int)
    posture_weight, motion_weight = hierarchical_class_weights(train_labels, device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    total_steps = math.ceil(len(train_loader)) * int(training["maximum_epochs"])
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_fraction=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=4096.0)
    group_dro = (
        GroupDROState(sorted(train_frame["scenario_id"].astype(str).unique()))
        if candidate["group_objective"] == "group_dro"
        else None
    )
    history: list[dict] = []
    best_metrics: dict[str, float] | None = None
    stale_epochs = 0
    best_path = output_dir / "best_checkpoint.pt"
    started = time.perf_counter()

    initial = evaluate_cptr(model, validation_loader, device, kwargs_builder=builder)
    best_metrics = initial["metrics"]
    torch.save(
        {
            "request_sha256": request_hash,
            "epoch": -1,
            "metrics": best_metrics,
            "model_state_dict": model.state_dict(),
        },
        best_path,
    )
    history.append({"epoch": -1, "training_loss": None, "validation": best_metrics})
    write_json(output_dir / "history.json", history)

    loss_spec = protocol["counterfactual_training"]
    augmentation = protocol["augmentation"]
    for epoch in range(int(training["maximum_epochs"])):
        model.train()
        epoch_losses: list[float] = []
        component_totals: dict[str, list[float]] = {}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            kwargs = builder(batch, device)
            kwargs = coherent_feature_augmentation(
                kwargs,
                feature_noise=float(augmentation["temporally_coherent_feature_noise"]),
                geometry_jitter=float(augmentation["temporally_coherent_geometry_jitter"]),
            )
            labels = batch["label"].to(device, non_blocking=True)
            transitions = batch["transition_target"].to(device, non_blocking=True)
            gait = batch["gait_target"].to(device, non_blocking=True)
            occlusion = batch["occlusion_target"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda"):
                output = model(**kwargs)
                null_output = reversed_output = jittered_output = None
                if candidate["use_counterfactuals"]:
                    null_output = model(**motion_null_kwargs(kwargs))
                    reversed_output = model(**reversed_kwargs(kwargs))
                    jittered_output = model(**jittered_camera_kwargs(kwargs))
                losses = cptr_loss(
                    output,
                    labels,
                    transition_targets=transitions,
                    gait_targets=gait,
                    occlusion_targets=occlusion,
                    null_output=null_output,
                    reversed_output=reversed_output,
                    jittered_output=jittered_output,
                    label_smoothing=float(training["label_smoothing"]),
                    posture_weight=posture_weight,
                    motion_weight=motion_weight,
                    transition_weight=float(loss_spec["transition_auxiliary_weight"]),
                    gait_weight=float(loss_spec["gait_auxiliary_weight"]),
                    quality_weight=float(loss_spec["quality_auxiliary_weight"]),
                    motion_null_weight=float(loss_spec["motion_null_residual_weight"]),
                    reversal_weight=float(
                        loss_spec["non_transition_reversal_consistency_weight"]
                    ),
                    camera_invariance_weight=float(
                        loss_spec["camera_jitter_invariance_weight"]
                    ),
                )
                loss = losses["loss"]
                if group_dro is not None:
                    per_sample = cptr_primary_loss_per_sample(
                        output,
                        labels,
                        label_smoothing=float(training["label_smoothing"]),
                        posture_weight=posture_weight,
                        motion_weight=motion_weight,
                    )
                    robust_primary = group_dro.objective(
                        per_sample,
                        list(map(str, batch["scenario_id"])),
                    )
                    loss = loss - losses["primary_loss"] + robust_primary
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=float(training["gradient_clip_norm"]),
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_losses.append(float(loss.detach().item()))
            for name, value in losses.items():
                component_totals.setdefault(name, []).append(float(value.detach().item()))
        validation = evaluate_cptr(model, validation_loader, device, kwargs_builder=builder)
        row = {
            "epoch": epoch,
            "training_loss": float(np.mean(epoch_losses)),
            "loss_components": {
                name: float(np.mean(values)) for name, values in component_totals.items()
            },
            "validation": validation["metrics"],
            "transition_validation": validation["subgroups"]["transition"],
            "group_dro_weights": group_dro.summary() if group_dro is not None else None,
        }
        history.append(row)
        if better_metrics(validation["metrics"], best_metrics):
            best_metrics = validation["metrics"]
            stale_epochs = 0
            torch.save(
                {
                    "request_sha256": request_hash,
                    "epoch": epoch,
                    "metrics": best_metrics,
                    "model_state_dict": model.state_dict(),
                    "group_dro_weights": group_dro.summary() if group_dro is not None else None,
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
    if checkpoint["request_sha256"] != request_hash:
        raise RuntimeError("The CPTR best checkpoint belongs to a different request")
    model.load_state_dict(checkpoint["model_state_dict"])
    validation = evaluate_cptr(model, validation_loader, device, kwargs_builder=builder)
    predictions_path = output_dir / "validation_predictions.npz"
    np.savez_compressed(
        predictions_path,
        sample_ids=validation["sample_ids"],
        recording_ids=validation["recording_ids"],
        track_ids=validation["track_ids"],
        labels=validation["labels"],
        probabilities=validation["probabilities"],
        static_probabilities=validation["static_probabilities"],
        posture_gates=validation["posture_gates"],
        motion_gates=validation["motion_gates"],
        expert_reliability=validation["expert_reliability"],
        unknown_scores=validation["unknown_scores"],
        transition_targets=validation["transition_targets"],
        occlusion_targets=validation["occlusion_targets"],
    )
    baseline_summary_path = (
        args.v3_root.resolve()
        / "teacher"
        / "temporal_8f_050s"
        / f"seed-{args.seed}"
        / "summary.json"
    )
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    baseline_metrics = baseline_summary["validation_metrics"]
    summary = {
        "status": "OKUTAMA_CPTR_CANDIDATE_COMPLETE",
        "candidate_id": candidate["candidate_id"],
        "candidate": candidate,
        "seed": int(args.seed),
        "best_epoch": int(checkpoint["epoch"]),
        "validation_metrics": validation["metrics"],
        "validation_subgroups": validation["subgroups"],
        "same_seed_legacy_temporal_metrics": baseline_metrics,
        "macro_f1_delta_vs_same_seed_legacy": float(
            validation["metrics"]["macro_f1"] - baseline_metrics["macro_f1"]
        ),
        "model": model_evidence(model),
        "masked_initialisation": masked_evidence,
        "train_samples": int(len(train_frame)),
        "validation_samples": int(len(validation_frame)),
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "training_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "runtime_seconds": time.perf_counter() - started,
        "request_sha256": request_hash,
        "baseline_checkpoint_sha256": baseline_hashes,
        "grid_lock_sha256": sha256_file(args.grid_lock.resolve()),
        "artifact_sha256": {
            best_path.name: sha256_file(best_path),
            predictions_path.name: sha256_file(predictions_path),
            "history.json": sha256_file(output_dir / "history.json"),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if FAILURE_DIRECTORY is not None:
            write_json(
                FAILURE_DIRECTORY / "failure.json",
                {
                    "status": "OKUTAMA_CPTR_CANDIDATE_FAILED",
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
