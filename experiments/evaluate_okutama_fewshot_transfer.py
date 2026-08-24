"""Measure CUDA few-shot adaptation on the grouped Okutama development split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.vcoco_v3_cuda_heads import (
    cuda_logistic_fit_audit,
    fit_probability_head_cuda,
    predict_probability_head_cuda,
    reset_cuda_logistic_fit_audit,
)
from hac.vcoco_v3_models import CLASS_NAMES, locomotion_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", type=Path, default=Path("experiments/okutama_temporal_grid.json")
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-store", type=Path, required=True)
    parser.add_argument(
        "--source-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/okutama/source_only/summary.json"),
    )
    parser.add_argument(
        "--source-predictions",
        type=Path,
        default=Path(".runs/vcoco_v3/okutama/source_only/predictions.npz"),
    )
    parser.add_argument(
        "--protocol-amendment",
        type=Path,
        default=Path(".runs/vcoco_v3/protocol/external_cuda_amendment_lock.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".runs/vcoco_v3/okutama/fewshot"),
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def round_robin_class_sample(
    frame: pd.DataFrame,
    indices: np.ndarray,
    *,
    budget: int,
    rng: np.random.Generator,
) -> np.ndarray:
    subset = frame.loc[indices, ["recording_id", "track_id"]].copy()
    subset["row_index"] = indices
    scenario_order = np.asarray(
        subset["recording_id"].astype(str).unique(), dtype=str
    ).copy()
    rng.shuffle(scenario_order)
    queues: dict[str, list[int]] = {}
    remainders: dict[str, list[int]] = {}
    for scenario in scenario_order:
        scenario_rows = subset[subset["recording_id"].astype(str).eq(str(scenario))]
        track_order = np.asarray(
            scenario_rows["track_id"].astype(str).unique(), dtype=str
        ).copy()
        rng.shuffle(track_order)
        first_pass = []
        remainder = []
        for track in track_order:
            track_rows = scenario_rows[
                scenario_rows["track_id"].astype(str).eq(str(track))
            ]["row_index"].to_numpy(dtype=int, copy=True)
            rng.shuffle(track_rows)
            first_pass.append(int(track_rows[0]))
            remainder.extend(map(int, track_rows[1:]))
        rng.shuffle(remainder)
        queues[str(scenario)] = first_pass
        remainders[str(scenario)] = remainder

    selected = []
    while len(selected) < budget and any(queues.values()):
        for scenario in map(str, scenario_order):
            if queues[scenario] and len(selected) < budget:
                selected.append(queues[scenario].pop(0))
    while len(selected) < budget and any(remainders.values()):
        for scenario in map(str, scenario_order):
            if remainders[scenario] and len(selected) < budget:
                selected.append(remainders[scenario].pop(0))
    if len(selected) != budget:
        raise RuntimeError("The declared few-shot budget exceeds an available class")
    return np.asarray(selected, dtype=int)


def sampled_rows(
    frame: pd.DataFrame,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    *,
    budget: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for class_index in range(len(CLASS_NAMES)):
        candidates = fit_indices[labels[fit_indices] == class_index]
        selected.append(
            round_robin_class_sample(frame, candidates, budget=budget, rng=rng)
        )
    output = np.concatenate(selected)
    rng.shuffle(output)
    return output


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Few-shot transfer fitting requires CUDA")
    started = time.perf_counter()
    grid_path = args.grid.resolve()
    manifest_path = args.manifest.resolve()
    store_path = args.target_store.resolve()
    source_summary_path = args.source_summary.resolve()
    source_predictions_path = args.source_predictions.resolve()
    amendment_path = args.protocol_amendment.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if (
        amendment.get("status")
        != "VCOCO_V3_EXTERNAL_CUDA_AMENDMENT_LOCKED_BEFORE_TARGET_FITTING"
    ):
        raise RuntimeError("The external CUDA protocol amendment is not locked")
    if amendment["source_sha256"].get("fewshot_transfer_source") != sha256_file(
        Path(__file__).resolve()
    ):
        raise RuntimeError("The amended few-shot implementation changed")
    if source_summary.get("source_sha256", {}).get(
        "external_cuda_amendment"
    ) != sha256_file(amendment_path):
        raise RuntimeError("Source-only probabilities belong to a different amendment")
    if source_summary.get("status") != "OKUTAMA_SOURCE_ONLY_TRANSFER_PREDICTIONS_COMPLETE":
        raise RuntimeError("Source-only probabilities are incomplete")
    if source_summary.get("target_labels_read") != 0:
        raise RuntimeError("Source-only fitting read target labels")
    if source_summary.get("target_partition") != "development":
        raise RuntimeError("Few-shot evaluation requires development source-only predictions")
    if (
        sha256_file(source_predictions_path)
        != source_summary["artifact_sha256"][source_predictions_path.name]
    ):
        raise RuntimeError("Source-only probabilities changed")
    frame = pd.read_csv(
        manifest_path,
        dtype={
            "sample_id": str,
            "recording_id": str,
            "track_id": str,
            "scenario_id": str,
        },
    )
    if set(frame["split"]) != {"train", "validation", "calibration"}:
        raise RuntimeError("Few-shot evaluation requires the locked three-way development split")
    label_map = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = frame["label"].map(label_map).to_numpy(dtype=int)
    fit_indices = np.flatnonzero(frame["split"].eq(grid["transfer"]["few_shot_fit_split"]))
    held_indices = np.flatnonzero(
        frame["split"].eq(grid["transfer"]["few_shot_evaluation_split"])
    )
    if set(frame.loc[fit_indices, "recording_id"]).intersection(
        frame.loc[held_indices, "recording_id"]
    ):
        raise RuntimeError("A scenario crossed the few-shot fit/evaluation boundary")

    store = json.loads(store_path.read_text(encoding="utf-8"))
    if store.get("status") != "VCOCO_V3_PACKED_TEMPORAL_FEATURE_STORE_COMPLETE":
        raise RuntimeError("The packed Okutama feature store is incomplete")
    arrays = {}
    for name in ("tight", "context", "geometry"):
        declaration = store["arrays"][name]
        path = (store_path.parent / str(declaration["path"])).resolve()
        if sha256_file(path) != declaration["sha256"]:
            raise RuntimeError(f"Packed Okutama {name} features changed")
        arrays[name] = np.load(path, mmap_mode="r")
    center = int(store["center_frame_index"])
    visual_features = np.concatenate(
        (
            arrays["tight"][:, center],
            arrays["context"][:, center],
            arrays["geometry"][:, center],
        ),
        axis=1,
    ).astype(np.float32)
    with np.load(source_predictions_path, allow_pickle=False) as payload:
        source_indices = payload["target_feature_indices"].astype(int)
        source_probabilities = payload["probabilities"].astype(float)
    if not np.array_equal(frame["feature_index"].to_numpy(dtype=int), source_indices):
        raise RuntimeError("Source-only probabilities and the target manifest are not aligned")
    recalibration_features = np.concatenate(
        (
            np.log(np.clip(source_probabilities, 1e-8, 1.0)),
            arrays["geometry"][:, center],
        ),
        axis=1,
    ).astype(np.float32)

    transfer = grid["transfer"]
    methods = {
        "target_only_factorized_linear": visual_features,
        "source_probability_recalibration": recalibration_features,
    }
    if list(methods) != list(transfer["few_shot_methods"]):
        raise RuntimeError("Implemented few-shot methods differ from the declaration")
    c_value = float(transfer["few_shot_head_C"])
    if transfer["few_shot_solver"] != "pytorch_cuda_lbfgs_logistic":
        raise RuntimeError("The declared few-shot solver is not implemented")
    maximum_iterations = int(transfer["few_shot_maximum_iterations"])
    tolerance = float(transfer["few_shot_gradient_tolerance"])
    metric_rows = []
    sample_rows = []
    prediction_blocks = []
    prediction_keys = []
    reset_cuda_logistic_fit_audit()
    source_metrics = classification_metrics(
        labels[held_indices], source_probabilities[held_indices]
    )
    source_metrics["locomotion_f1"] = locomotion_f1(
        labels[held_indices], source_probabilities[held_indices]
    )
    metric_rows.append(
        {
            "method": "source_only_static",
            "budget_per_class": 0,
            "fit_samples": 0,
            "seed": -1,
            **source_metrics,
        }
    )
    for budget in map(int, transfer["few_shot_per_class_budgets"]):
        for seed in map(int, transfer["few_shot_seeds"]):
            selected = sampled_rows(
                frame,
                labels,
                fit_indices,
                budget=budget,
                seed=seed,
            )
            for index in selected:
                sample_rows.append(
                    {
                        "budget_per_class": budget,
                        "seed": seed,
                        "sample_id": frame.iloc[index]["sample_id"],
                        "recording_id": frame.iloc[index]["recording_id"],
                        "track_id": frame.iloc[index]["track_id"],
                        "label": frame.iloc[index]["label"],
                    }
                )
            for method_index, (method, features) in enumerate(methods.items()):
                model = fit_probability_head_cuda(
                    features[selected],
                    labels[selected],
                    factorized=True,
                    c_value=c_value,
                    class_weight="none",
                    seed=seed + 10_000 * method_index,
                    maximum_iterations=maximum_iterations,
                    tolerance=tolerance,
                )
                probabilities = predict_probability_head_cuda(model, features[held_indices])
                metrics = classification_metrics(labels[held_indices], probabilities)
                metrics["locomotion_f1"] = locomotion_f1(
                    labels[held_indices], probabilities
                )
                metric_rows.append(
                    {
                        "method": method,
                        "budget_per_class": budget,
                        "fit_samples": len(selected),
                        "seed": seed,
                        **metrics,
                    }
                )
                prediction_keys.append(f"{method}__budget-{budget}__seed-{seed}")
                prediction_blocks.append(probabilities)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    samples_path = output_dir / "sampled_rows.csv"
    predictions_path = output_dir / "validation_predictions.npz"
    optimization_path = output_dir / "cuda_optimization.json"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(sample_rows).to_csv(samples_path, index=False)
    np.savez_compressed(
        predictions_path,
        keys=np.asarray(prediction_keys),
        sample_ids=frame.iloc[held_indices]["sample_id"].astype(str).to_numpy(),
        labels=labels[held_indices],
        probabilities=np.stack(prediction_blocks),
    )
    audit = cuda_logistic_fit_audit()
    write_json(
        optimization_path,
        {
            "solver": "pytorch_cuda_lbfgs_logistic",
            "device": torch.cuda.get_device_name(0),
            "fits": len(audit),
            "iteration_limit_reached_fits": sum(
                bool(record["iteration_limit_reached"]) for record in audit
            ),
            "records": audit,
        },
    )
    summary = {
        "status": "OKUTAMA_FEWSHOT_TRANSFER_DEVELOPMENT_COMPLETE",
        "fit_split": transfer["few_shot_fit_split"],
        "evaluation_split": transfer["few_shot_evaluation_split"],
        "budgets_per_class": transfer["few_shot_per_class_budgets"],
        "seeds": transfer["few_shot_seeds"],
        "methods": transfer["few_shot_methods"],
        "fit_hyperparameter_selection": transfer["few_shot_hyperparameter_selection"],
        "target_confirmation_rows_read": 0,
        "training_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "runtime_seconds": time.perf_counter() - started,
        "source_sha256": {
            "grid": sha256_file(grid_path),
            "manifest": sha256_file(manifest_path),
            "target_store": sha256_file(store_path),
            "source_summary": sha256_file(source_summary_path),
            "source_predictions": sha256_file(source_predictions_path),
            "external_cuda_amendment": sha256_file(amendment_path),
            "runner": sha256_file(Path(__file__).resolve()),
        },
        "artifact_sha256": {
            path.name: sha256_file(path)
            for path in (metrics_path, samples_path, predictions_path, optimization_path)
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
