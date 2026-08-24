"""Replay the locked v3 baseline and run CPTR temporal-faithfulness diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, log_loss
from torch.utils.data import DataLoader

from hac.cptr_features import CPTRFeatureDataset
from hac.polar import sha256_file
from hac.training import seed_everything
from hac.vcoco_v3_models import locomotion_f1
from hac.vcoco_v3_temporal import StaticIdentifiabilityStudent, TemporalFactorizedTeacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiments/okutama_cptr_protocol.json")
    )
    parser.add_argument(
        "--protocol-lock", type=Path, default=Path(".runs/cptr/protocol_lock.json")
    )
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
    parser.add_argument("--output-dir", type=Path, default=Path(".runs/cptr/baseline"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    class_f1 = f1_score(labels, predictions, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "macro_f1": float(class_f1.mean()),
        "accuracy": float((predictions == labels).mean()),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1, 2])),
        "sitting_f1": float(class_f1[0]),
        "standing_f1": float(class_f1[1]),
        "walking_running_f1": float(class_f1[2]),
        "locomotion_f1": float(locomotion_f1(labels, probabilities)),
    }


def intervention_batch(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    centre_index: int,
    intervention: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = features.clone()
    mask = valid_mask.clone()
    if intervention == "real_all_frames":
        mask.fill_(True)
    elif intervention == "occlusion_masked":
        pass
    elif intervention == "repeat_centre_frame":
        values = values[:, centre_index : centre_index + 1].expand_as(values).clone()
        mask.fill_(True)
    elif intervention == "reverse_temporal_order":
        values = torch.flip(values, dims=(1,))
        mask = torch.flip(mask, dims=(1,))
    elif intervention == "deterministic_temporal_shuffle":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260912)
        order = torch.randperm(values.shape[1], generator=generator)
        values = values[:, order]
        mask = mask[:, order]
    elif intervention == "zero_geometry":
        values[:, :, -6:] = 0.0
        mask.fill_(True)
    elif intervention == "zero_appearance_dynamics":
        centre_visual = values[:, centre_index : centre_index + 1, :-6]
        values[:, :, :-6] = centre_visual
        mask.fill_(True)
    elif intervention == "geometry_only":
        values[:, :, :-6] = 0.0
        mask.fill_(True)
    elif intervention == "coherent_camera_jitter":
        values[:, :, -4] += 0.01
        values[:, :, -3] -= 0.01
        mask.fill_(True)
    else:
        raise ValueError(f"Unknown diagnostic intervention: {intervention}")
    if not torch.all(mask.any(dim=1)):
        mask = mask.clone()
        mask[~mask.any(dim=1), centre_index] = True
    return values, mask


def load_models(
    seed: int,
    *,
    input_dim: int,
    grid: dict,
    root: Path,
    device: torch.device,
) -> tuple[StaticIdentifiabilityStudent, TemporalFactorizedTeacher, dict[str, str]]:
    architecture = grid["architecture"]
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
    static_path = root / "static" / f"seed-{seed}" / "best_checkpoint.pt"
    teacher_path = (
        root
        / "teacher"
        / "temporal_8f_050s"
        / f"seed-{seed}"
        / "best_checkpoint.pt"
    )
    for model, path in ((static, static_path), (teacher, teacher_path)):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval().requires_grad_(False)
    return static, teacher, {
        "static": sha256_file(static_path),
        "teacher": sha256_file(teacher_path),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Batch size and workers are invalid")
    if not torch.cuda.is_available():
        raise RuntimeError("The CPTR baseline replay requires CUDA")
    protocol_path = args.protocol.resolve()
    lock_path = args.protocol_lock.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "OKUTAMA_CPTR_PROTOCOL_LOCKED_BEFORE_FIT":
        raise RuntimeError("The CPTR protocol is not locked")
    if lock["source_sha256"]["protocol"] != sha256_file(protocol_path):
        raise RuntimeError("The CPTR protocol changed after locking")
    manifest_path = args.manifest.resolve()
    if lock["source_sha256"]["development_manifest"] != sha256_file(manifest_path):
        raise RuntimeError("The development manifest changed after locking")
    grid = json.loads(args.v3_grid.resolve().read_text(encoding="utf-8"))
    frame = pd.read_csv(
        manifest_path,
        dtype={"sample_id": str, "recording_id": str, "track_id": str},
    )
    validation = frame[frame["split"].eq("validation")].reset_index(drop=True)
    dataset = CPTRFeatureDataset(
        validation,
        manifest_directory=manifest_path.parent,
        short_samples=8,
        short_seconds=0.5,
        long_samples=8,
        long_seconds=1.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    seed_everything(42)
    interventions = (
        "real_all_frames",
        "occlusion_masked",
        "repeat_centre_frame",
        "reverse_temporal_order",
        "deterministic_temporal_shuffle",
        "zero_geometry",
        "zero_appearance_dynamics",
        "geometry_only",
        "coherent_camera_jitter",
    )
    seed_static: list[np.ndarray] = []
    seed_teacher: dict[str, list[np.ndarray]] = {name: [] for name in interventions}
    labels_reference: np.ndarray | None = None
    sample_ids_reference: np.ndarray | None = None
    checkpoint_hashes: dict[str, dict[str, str]] = {}
    started = time.perf_counter()
    base_store_path = Path(str(validation.iloc[0]["feature_path"])).resolve()
    base_store = json.loads(base_store_path.read_text(encoding="utf-8"))
    input_dim = 2 * int(base_store["feature_dimensions"]) + 6
    for seed in protocol["training"]["promotion_seeds"]:
        static, teacher, hashes = load_models(
            int(seed),
            input_dim=input_dim,
            grid=grid,
            root=args.v3_root.resolve(),
            device=device,
        )
        checkpoint_hashes[str(seed)] = hashes
        labels_batches: list[np.ndarray] = []
        sample_batches: list[np.ndarray] = []
        static_batches: list[np.ndarray] = []
        teacher_batches: dict[str, list[np.ndarray]] = {name: [] for name in interventions}
        for batch in loader:
            static_features = batch["static_features"].to(device, non_blocking=True)
            short_features = batch["short_features"].to(device, non_blocking=True)
            valid_mask = batch["short_valid_mask"].to(device, non_blocking=True)
            centre_index = int(batch["short_centre_index"][0])
            with torch.autocast(device_type="cuda"):
                static_output = static(static_features)
                for name in interventions:
                    values, mask = intervention_batch(
                        short_features,
                        valid_mask,
                        centre_index=centre_index,
                        intervention=name,
                    )
                    teacher_batches[name].append(
                        teacher(values, mask).probabilities.float().cpu().numpy()
                    )
            static_batches.append(static_output.probabilities.float().cpu().numpy())
            labels_batches.append(batch["label"].numpy())
            sample_batches.append(np.asarray(batch["sample_id"], dtype=str))
        labels = np.concatenate(labels_batches)
        sample_ids = np.concatenate(sample_batches)
        if labels_reference is None:
            labels_reference = labels
            sample_ids_reference = sample_ids
        elif not np.array_equal(labels_reference, labels) or not np.array_equal(
            sample_ids_reference, sample_ids
        ):
            raise RuntimeError("Baseline seed evaluation row order changed")
        seed_static.append(np.concatenate(static_batches))
        for name in interventions:
            seed_teacher[name].append(np.concatenate(teacher_batches[name]))
        del static, teacher
        torch.cuda.empty_cache()

    if labels_reference is None or sample_ids_reference is None:
        raise RuntimeError("Baseline replay produced no validation predictions")
    static_probabilities = np.stack(seed_static).mean(axis=0)
    teacher_probabilities = {
        name: np.stack(values).mean(axis=0) for name, values in seed_teacher.items()
    }
    rows = [
        {
            "model": "static_ensemble",
            "intervention": "centre_frame",
            **metrics(labels_reference, static_probabilities),
        }
    ]
    for name, probabilities in teacher_probabilities.items():
        rows.append(
            {
                "model": "temporal_ensemble",
                "intervention": name,
                **metrics(labels_reference, probabilities),
            }
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "baseline_and_diagnostic_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    prediction_path = output_dir / "baseline_and_diagnostic_predictions.npz"
    np.savez_compressed(
        prediction_path,
        sample_ids=sample_ids_reference,
        labels=labels_reference,
        static_probabilities=static_probabilities,
        **{f"teacher__{name}": values for name, values in teacher_probabilities.items()},
    )
    real = teacher_probabilities["real_all_frames"]
    repeated = teacher_probabilities["repeat_centre_frame"]
    rows_index = np.arange(len(labels_reference))
    faithfulness = {
        "mean_true_class_log_probability_gain_real_over_repeat": float(
            np.mean(
                np.log(np.clip(real[rows_index, labels_reference], 1e-8, 1.0))
                - np.log(np.clip(repeated[rows_index, labels_reference], 1e-8, 1.0))
            )
        ),
        "prediction_change_fraction_real_vs_repeat": float(
            np.mean(real.argmax(axis=1) != repeated.argmax(axis=1))
        ),
        "real_rescues_repeat": int(
            np.sum(
                (real.argmax(axis=1) == labels_reference)
                & (repeated.argmax(axis=1) != labels_reference)
            )
        ),
        "real_harms_repeat": int(
            np.sum(
                (real.argmax(axis=1) != labels_reference)
                & (repeated.argmax(axis=1) == labels_reference)
            )
        ),
    }
    summary = {
        "status": "OKUTAMA_CPTR_BASELINE_AND_DIAGNOSTICS_COMPLETE",
        "validation_samples": int(len(labels_reference)),
        "seeds": list(map(int, protocol["training"]["promotion_seeds"])),
        "baseline_metrics": rows[0],
        "temporal_metrics": next(
            row for row in rows if row["intervention"] == "real_all_frames"
        ),
        "faithfulness": faithfulness,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "training_device": torch.cuda.get_device_name(0),
        "runtime_seconds": time.perf_counter() - started,
        "checkpoint_sha256": checkpoint_hashes,
        "source_sha256": {
            "protocol": sha256_file(protocol_path),
            "protocol_lock": sha256_file(lock_path),
            "manifest": sha256_file(manifest_path),
            "runner": sha256_file(Path(__file__).resolve()),
        },
        "artifact_sha256": {
            metrics_path.name: sha256_file(metrics_path),
            prediction_path.name: sha256_file(prediction_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
