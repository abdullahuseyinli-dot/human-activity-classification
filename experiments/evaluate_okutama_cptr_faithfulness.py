"""Evaluate CPTR intervention faithfulness and cached-feature inference cost."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hac.cptr_features import (
    BASE_GEOMETRY_DIM,
    CPTRFeatureDataset,
    jittered_camera_kwargs,
    model_kwargs_from_batch,
    motion_null_kwargs,
    reversed_kwargs,
)
from hac.cptr_training import (
    build_cptr_model,
    classification_summary,
    load_frozen_v3_baselines,
)
from hac.polar import sha256_file
from hac.training import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiments/okutama_cptr_protocol.json")
    )
    parser.add_argument(
        "--protocol-lock", type=Path, default=Path(".runs/cptr/protocol_lock.json")
    )
    parser.add_argument(
        "--grid", type=Path, default=Path("experiments/okutama_cptr_adaptive_grid.json")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/development_manifest.csv"),
    )
    parser.add_argument(
        "--part-store", type=Path, default=Path(".runs/cptr/part_features/store.json")
    )
    parser.add_argument(
        "--v3-grid", type=Path, default=Path("experiments/okutama_temporal_grid.json")
    )
    parser.add_argument(
        "--v3-root",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/development"),
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path(".runs/cptr/promotion/centre_short_parts"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".runs/cptr/development_final/faithfulness")
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-repetitions", type=int, default=100)
    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def clone_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    return {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in kwargs.items()
    }


def repeat_stream(
    kwargs: Mapping[str, object],
    *,
    value_name: str,
    centre_name: str,
    companion_names: tuple[str, ...] = (),
) -> dict[str, object]:
    output = clone_kwargs(kwargs)
    values = output[value_name]
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{value_name} is not a tensor")
    centre = int(output[centre_name])
    for name in (value_name, *companion_names):
        tensor = output[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} is not a tensor")
        output[name] = tensor[:, centre : centre + 1].expand_as(tensor).clone()
    return output


def missing_parts_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    output = clone_kwargs(kwargs)
    output["part_tokens"] = torch.zeros_like(output["part_tokens"])
    output["part_confidence"] = torch.zeros_like(output["part_confidence"])
    quality = output["quality_features"]
    if not isinstance(quality, torch.Tensor):
        raise TypeError("quality_features is not a tensor")
    quality[:, 6] = 0.0
    return output


def zero_geometry_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    output = clone_kwargs(kwargs)
    for name in ("static_features", "short_features"):
        values = output[name]
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"{name} is not a tensor")
        values[..., -BASE_GEOMETRY_DIM:] = 0.0
    return output


def zero_appearance_dynamics_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    output = clone_kwargs(kwargs)
    short = output["short_features"]
    if not isinstance(short, torch.Tensor):
        raise TypeError("short_features is not a tensor")
    centre = int(output["short_centre_index"])
    visual_width = short.shape[-1] - BASE_GEOMETRY_DIM
    short[..., :visual_width] = short[:, centre : centre + 1, :visual_width]
    part = output["part_tokens"]
    confidence = output["part_confidence"]
    if not isinstance(part, torch.Tensor) or not isinstance(confidence, torch.Tensor):
        raise TypeError("Part inputs are not tensors")
    part_centre = int(output["part_centre_index"])
    part[:] = part[:, part_centre : part_centre + 1].expand_as(part)
    confidence[:] = confidence[:, part_centre : part_centre + 1].expand_as(confidence)
    return output


def geometry_only_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    output = missing_parts_kwargs(kwargs)
    for name in ("static_features", "short_features"):
        values = output[name]
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"{name} is not a tensor")
        values[..., :-BASE_GEOMETRY_DIM] = 0.0
    return output


def deterministic_shuffle_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    output = clone_kwargs(kwargs)
    stream_specs = (
        ("short_features", "short_centre_index", ("short_valid_mask",)),
        (
            "part_tokens",
            "part_centre_index",
            ("part_confidence", "part_valid_mask"),
        ),
    )
    for value_name, centre_name, companions in stream_specs:
        values = output[value_name]
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"{value_name} is not a tensor")
        steps = values.shape[1]
        permutation = torch.arange(steps, device=values.device)
        permutation = torch.cat((permutation[::2], permutation[1::2])).flip(0)
        for name in (value_name, *companions):
            tensor = output[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} is not a tensor")
            output[name] = tensor.index_select(1, permutation)
        old_centre = int(output[centre_name])
        output[centre_name] = int(torch.nonzero(permutation == old_centre)[0, 0])
    return output


def unmask_occlusion_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    output = clone_kwargs(kwargs)
    for name in ("short_valid_mask", "part_valid_mask"):
        output[name] = torch.ones_like(output[name], dtype=torch.bool)
    return output


def intervention_summary(
    labels: np.ndarray,
    real: np.ndarray,
    intervention: np.ndarray,
) -> dict[str, object]:
    positions = np.arange(len(labels))
    real_prediction = real.argmax(axis=1)
    intervention_prediction = intervention.argmax(axis=1)
    true_log_gain = np.log(real[positions, labels].clip(min=1e-8)) - np.log(
        intervention[positions, labels].clip(min=1e-8)
    )
    real_metrics = classification_summary(labels, real)
    metrics = classification_summary(labels, intervention)
    return {
        "metrics": metrics,
        "macro_f1_delta_real_minus_intervention": real_metrics["macro_f1"]
        - metrics["macro_f1"],
        "mean_true_class_log_probability_gain_real_minus_intervention": float(
            true_log_gain.mean()
        ),
        "prediction_change_fraction": float(np.mean(real_prediction != intervention_prediction)),
        "real_rescues_intervention": int(
            np.sum((real_prediction == labels) & (intervention_prediction != labels))
        ),
        "real_harms_intervention": int(
            np.sum((real_prediction != labels) & (intervention_prediction == labels))
        ),
    }


def benchmark_forward(model, kwargs: dict[str, object], *, warmup: int, repetitions: int) -> dict:
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(warmup):
            model(**kwargs)
        torch.cuda.synchronize()
        timings = np.empty(repetitions, dtype=float)
        for index in range(repetitions):
            started = time.perf_counter()
            model(**kwargs)
            torch.cuda.synchronize()
            timings[index] = (time.perf_counter() - started) * 1000.0
    batch = len(kwargs["static_features"])
    return {
        "batch_size": int(batch),
        "warmup_iterations": int(warmup),
        "measured_iterations": int(repetitions),
        "batch_latency_ms_mean": float(timings.mean()),
        "batch_latency_ms_p50": float(np.quantile(timings, 0.50)),
        "batch_latency_ms_p95": float(np.quantile(timings, 0.95)),
        "per_sample_latency_ms_mean": float(timings.mean() / batch),
        "scope": "classifier_forward_from_cached_DINOv2_and_part_features",
    }


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.latency_warmup, args.latency_repetitions) < 1:
        raise ValueError("Batch size and latency iteration counts must be positive")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CPTR faithfulness evaluation requires CUDA")
    protocol_path = args.protocol.resolve()
    protocol_lock_path = args.protocol_lock.resolve()
    grid_path = args.grid.resolve()
    manifest_path = args.manifest.resolve()
    part_store_path = args.part_store.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_lock = json.loads(protocol_lock_path.read_text(encoding="utf-8"))
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    if protocol_lock.get("status") != "OKUTAMA_CPTR_PROTOCOL_LOCKED_BEFORE_FIT":
        raise RuntimeError("The CPTR protocol lock is invalid")
    if protocol_lock["source_sha256"]["protocol"] != sha256_file(protocol_path):
        raise RuntimeError("The CPTR protocol changed after locking")
    candidate = next(
        item for item in grid["candidates"] if item["candidate_id"] == "centre_short_parts"
    )
    frame = pd.read_csv(
        manifest_path,
        dtype={"sample_id": str, "recording_id": str, "track_id": str},
    )
    validation = frame[frame["split"].eq("validation")].reset_index(drop=True)
    dataset = CPTRFeatureDataset(
        validation,
        manifest_directory=manifest_path.parent,
        part_store=part_store_path,
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
        persistent_workers=args.workers > 0,
    )
    base_store_path = Path(str(validation.iloc[0]["feature_path"])).resolve()
    base_store = json.loads(base_store_path.read_text(encoding="utf-8"))
    part_store = json.loads(part_store_path.read_text(encoding="utf-8"))
    input_dim = 2 * int(base_store["feature_dimensions"]) + BASE_GEOMETRY_DIM
    part_dim = int(part_store["part_token_dim"])
    v3_grid = json.loads(args.v3_grid.resolve().read_text(encoding="utf-8"))
    seeds = list(map(int, protocol["training"]["promotion_seeds"]))
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    seed_probabilities: dict[str, list[np.ndarray]] = {}
    gate_values = []
    reliability_values = []
    labels = None
    sample_ids = None
    transition_targets = None
    occlusion_targets = None
    latency_rows = []
    checkpoint_hashes = {}
    intervention_builders = {
        "motion_null": motion_null_kwargs,
        "short_repeated": lambda values: repeat_stream(
            values,
            value_name="short_features",
            centre_name="short_centre_index",
        ),
        "parts_repeated": lambda values: repeat_stream(
            values,
            value_name="part_tokens",
            centre_name="part_centre_index",
            companion_names=("part_confidence",),
        ),
        "parts_missing": missing_parts_kwargs,
        "reverse_temporal_order": reversed_kwargs,
        "deterministic_temporal_shuffle": deterministic_shuffle_kwargs,
        "zero_geometry": zero_geometry_kwargs,
        "zero_appearance_dynamics": zero_appearance_dynamics_kwargs,
        "geometry_only": geometry_only_kwargs,
        "occlusion_unmasked": unmask_occlusion_kwargs,
        "background_camera_jitter": lambda values: jittered_camera_kwargs(
            values, maximum_shift=0.01
        ),
    }
    for seed in seeds:
        seed_everything(seed)
        static, teacher, _ = load_frozen_v3_baselines(
            seed,
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
            pose_joint_count=1,
            siglip_dim=1,
            static=static,
            teacher=teacher,
        ).to(device)
        checkpoint_path = (
            args.candidate_root.resolve() / f"seed-{seed}" / "best_checkpoint.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        checkpoint_hashes[str(seed)] = sha256_file(checkpoint_path)
        collected: dict[str, list[np.ndarray]] = {
            "real": [],
            "legacy_temporal": [],
            **{name: [] for name in intervention_builders},
        }
        seed_labels = []
        seed_sample_ids = []
        seed_transitions = []
        seed_occlusions = []
        seed_gates = []
        seed_reliability = []
        latency_kwargs = None
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                kwargs = model_kwargs_from_batch(
                    batch,
                    device,
                    use_long=False,
                    use_trajectory=False,
                    use_parts=True,
                    use_pose=False,
                    use_siglip=False,
                )
                if latency_kwargs is None:
                    latency_kwargs = clone_kwargs(kwargs)
                with torch.autocast(device_type="cuda"):
                    real = model(**kwargs)
                    collected["real"].append(real.probabilities.float().cpu().numpy())
                    collected["legacy_temporal"].append(
                        real.legacy_probabilities.float().cpu().numpy()
                    )
                    seed_gates.append(
                        torch.stack((real.posture_gates, real.motion_gates), dim=1)
                        .float()
                        .cpu()
                        .numpy()
                    )
                    seed_reliability.append(real.expert_reliability.float().cpu().numpy())
                    for name, builder in intervention_builders.items():
                        if name == "background_camera_jitter":
                            seed_everything(seed + batch_index + 100_000)
                        intervention = model(**builder(kwargs))
                        collected[name].append(
                            intervention.probabilities.float().cpu().numpy()
                        )
                seed_labels.append(batch["label"].numpy())
                seed_sample_ids.extend(map(str, batch["sample_id"]))
                seed_transitions.append(batch["transition_target"].numpy())
                seed_occlusions.append(batch["occlusion_target"].numpy())
        if latency_kwargs is None:
            raise RuntimeError("The validation loader is empty")
        seed_latency = benchmark_forward(
            model,
            latency_kwargs,
            warmup=args.latency_warmup,
            repetitions=args.latency_repetitions,
        )
        latency_rows.append({"seed": seed, **seed_latency})
        current_labels = np.concatenate(seed_labels).astype(int)
        current_sample_ids = np.asarray(seed_sample_ids)
        current_transitions = np.concatenate(seed_transitions).astype(bool)
        current_occlusions = np.concatenate(seed_occlusions).astype(bool)
        if labels is None:
            labels = current_labels
            sample_ids = current_sample_ids
            transition_targets = current_transitions
            occlusion_targets = current_occlusions
        elif not (
            np.array_equal(labels, current_labels)
            and np.array_equal(sample_ids, current_sample_ids)
            and np.array_equal(transition_targets, current_transitions)
            and np.array_equal(occlusion_targets, current_occlusions)
        ):
            raise RuntimeError("Faithfulness seed evaluation rows differ")
        for name, blocks in collected.items():
            seed_probabilities.setdefault(name, []).append(np.concatenate(blocks))
        gate_values.append(np.concatenate(seed_gates))
        reliability_values.append(np.concatenate(seed_reliability))

    if labels is None or sample_ids is None:
        raise RuntimeError("No faithfulness predictions were produced")
    ensemble = {
        name: np.stack(values).mean(axis=0) for name, values in seed_probabilities.items()
    }
    real = ensemble["real"]
    diagnostics = {
        name: intervention_summary(labels, real, probabilities)
        for name, probabilities in ensemble.items()
    }
    transitions = np.asarray(transition_targets, dtype=bool)
    occlusions = np.asarray(occlusion_targets, dtype=bool)
    for name, probabilities in ensemble.items():
        diagnostics[name]["transition_metrics"] = classification_summary(
            labels[transitions], probabilities[transitions]
        )
        diagnostics[name]["occluded_metrics"] = classification_summary(
            labels[occlusions], probabilities[occlusions]
        )
    gates = np.stack(gate_values).mean(axis=0)
    reliability = np.stack(reliability_values).mean(axis=0)
    expert_names = ("legacy_short", "centre_short", "parts")
    gate_summary = {
        "posture": {
            name: float(gates[:, 0, index].mean()) for index, name in enumerate(expert_names)
        },
        "motion": {
            name: float(gates[:, 1, index].mean()) for index, name in enumerate(expert_names)
        },
        "reliability": {
            name: float(reliability[:, index].mean()) for index, name in enumerate(expert_names)
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "intervention_predictions.npz"
    metrics_path = output_dir / "intervention_metrics.csv"
    np.savez_compressed(
        predictions_path,
        sample_ids=sample_ids,
        labels=labels,
        transition_targets=transitions,
        occlusion_targets=occlusions,
        **{f"{name}_probabilities": values for name, values in ensemble.items()},
    )
    pd.DataFrame(
        [
            {
                "intervention": name,
                **values["metrics"],
                "macro_f1_delta_real_minus_intervention": values[
                    "macro_f1_delta_real_minus_intervention"
                ],
                "mean_true_class_log_probability_gain_real_minus_intervention": values[
                    "mean_true_class_log_probability_gain_real_minus_intervention"
                ],
                "prediction_change_fraction": values["prediction_change_fraction"],
                "real_rescues_intervention": values["real_rescues_intervention"],
                "real_harms_intervention": values["real_harms_intervention"],
            }
            for name, values in diagnostics.items()
        ]
    ).to_csv(metrics_path, index=False)
    summary = {
        "status": "OKUTAMA_CPTR_FAITHFULNESS_COMPLETE",
        "candidate_id": "centre_short_parts",
        "seeds": seeds,
        "samples": int(len(labels)),
        "diagnostics": diagnostics,
        "gate_summary": gate_summary,
        "latency": latency_rows,
        "latency_feature_extraction_excluded": True,
        "validation_samples_read": int(len(labels)),
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "checkpoint_sha256": checkpoint_hashes,
        "source_sha256": {
            "protocol": sha256_file(protocol_path),
            "protocol_lock": sha256_file(protocol_lock_path),
            "grid": sha256_file(grid_path),
            "manifest": sha256_file(manifest_path),
            "part_store": sha256_file(part_store_path),
            "runner": sha256_file(Path(__file__).resolve()),
        },
        "artifact_sha256": {
            predictions_path.name: sha256_file(predictions_path),
            metrics_path.name: sha256_file(metrics_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
