"""Measure input and quantized-head bit-flip robustness for locked POLAR models."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evaluate_polar_final import model_config_from_final, verify_final_fits
from PIL import Image

from hac.augmentations import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform
from hac.metrics import classification_metrics
from hac.polar import image_view, sha256_file
from hac.polar_faithfulness import (
    flip_uint8_bits_exact,
    quantize_and_flip_parameter_bits,
    stable_seed,
)
from hac.polar_models import build_polar_model
from hac.polar_training import TASK_LABELS, normalize_probability_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-evaluation-dir", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
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
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_view_tensors(frame: pd.DataFrame, view: str) -> torch.Tensor:
    transform = build_eval_transform()
    values = []
    for _, row in frame.iterrows():
        with Image.open(row["image_path"]) as image:
            source = image.convert("RGB")
            values.append(transform(image_view(source, row, view)))
    return torch.stack(values)


def input_bit_faults(inputs: torch.Tensor, rate: float, seed: int) -> tuple[torch.Tensor, int]:
    mean = torch.tensor(IMAGENET_MEAN, dtype=inputs.dtype)[None, :, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=inputs.dtype)[None, :, None, None]
    encoded = torch.clamp(inputs * std + mean, 0.0, 1.0).mul(255.0).round().to(torch.uint8)
    bit_flips = int(round(encoded.numel() * 8 * float(rate)))
    corrupted = flip_uint8_bits_exact(encoded.numpy(), bit_flips=bit_flips, seed=seed)
    decoded = torch.from_numpy(corrupted).float().div(255.0)
    return (decoded - mean) / std, bit_flips


def predict_batches(
    model, inputs: torch.Tensor, *, batch_size: int, device: torch.device
) -> np.ndarray:
    output = []
    model.eval()
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size].to(device, non_blocking=True)
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type, enabled=device.type == "cuda"),
        ):
            logits = model(batch)
        output.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(output)


def component_predictions(models, inputs, *, batch_size: int, device: torch.device) -> np.ndarray:
    return normalize_probability_rows(
        np.mean(
            [
                predict_batches(model, inputs, batch_size=batch_size, device=device)
                for model in models
            ],
            axis=0,
        )
    )


def head_module(model, family: str):
    return model.backbone.classifier[2] if family == "convnext_small_full" else model.classifier


def result_row(
    *,
    family: str,
    condition: str,
    level: float | int,
    fault_seed: int | str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    clean: np.ndarray,
    realized_bit_flips: int,
) -> dict:
    predictions = np.argmax(probabilities, axis=1)
    clean_predictions = np.argmax(clean, axis=1)
    return {
        "family": family,
        "condition": condition,
        "level": level,
        "fault_seed": fault_seed,
        "realized_bit_flips": realized_bit_flips,
        **classification_metrics(labels, probabilities),
        "prediction_agreement_with_clean": float(np.mean(predictions == clean_predictions)),
        "mean_absolute_probability_drift": float(np.mean(np.abs(probabilities - clean))),
        "max_absolute_probability_drift": float(np.max(np.abs(probabilities - clean))),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    lock_path = args.selection_lock.resolve()
    lock_hash = sha256_file(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    implementation_hash = sha256_file(Path(__file__).resolve())
    evaluation_dir = args.test_evaluation_dir.resolve()
    evaluation_summary = json.loads((evaluation_dir / "summary.json").read_text(encoding="utf-8"))
    if (
        evaluation_summary.get("status") != "LOCKED_FINAL_TEST_COMPLETE"
        or evaluation_summary.get("selection_lock_sha256") != lock_hash
    ):
        raise RuntimeError("Fault evaluation requires the completed locked POLAR test run")
    protocol = lock["fault_robustness"]
    if not protocol.get("reported_separately_from_faithfulness"):
        raise RuntimeError("Bit-flip evidence must remain separate from faithfulness")

    cohort_path = args.cohort.resolve()
    test_predictions_path = evaluation_dir / "test_predictions.npz"
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics_path = output_dir / "fault_robustness_metrics.csv"
        predictions_path = output_dir / "fault_robustness_predictions.npz"
        if (
            existing.get("status") == "LOCKED_POLAR_FAULT_ROBUSTNESS_COMPLETE"
            and existing.get("selection_lock_sha256") == lock_hash
            and existing.get("cohort_sha256") == sha256_file(cohort_path)
            and existing.get("implementation_sha256") == implementation_hash
            and existing.get("test_predictions_sha256")
            == sha256_file(test_predictions_path)
            and metrics_path.is_file()
            and sha256_file(metrics_path) == existing.get("metrics_sha256")
            and predictions_path.is_file()
            and sha256_file(predictions_path) == existing.get("predictions_sha256")
        ):
            print(json.dumps(existing, indent=2, sort_keys=True), flush=True)
            return
    cohort = pd.read_csv(cohort_path, dtype={"image_id": str})
    if len(cohort) != int(lock["faithfulness"]["cohort_rows"]):
        raise RuntimeError("Fault cohort differs from the locked faithfulness cohort")
    class_names = list(TASK_LABELS["label_4"])
    class_to_index = {name: index for index, name in enumerate(class_names)}
    labels = cohort["label_4"].map(class_to_index).to_numpy(dtype=int)

    prediction_artifact = np.load(test_predictions_path, allow_pickle=False)
    all_ids = [str(value) for value in prediction_artifact["image_ids"]]
    index_by_id = {value: index for index, value in enumerate(all_ids)}
    order = np.asarray([index_by_id[value] for value in cohort["image_id"].astype(str)])
    final_root = args.final_root.resolve()
    resolved = verify_final_fits(lock, final_root, lock_hash)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    rows = []
    probability_artifacts = {}
    for family in lock["faithfulness"]["families"]:
        specification = lock["final_neural_fits"][family]
        configuration = specification["configuration"]
        inputs = load_view_tensors(cohort, configuration["view"])
        models = []
        for seed in specification["seeds"]:
            model = build_polar_model(
                model_config_from_final(configuration), num_classes=len(class_names)
            ).to(device)
            payload = torch.load(
                resolved["neural"][family][int(seed)]["checkpoint"],
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(payload["model_state_dict"])
            model.eval()
            models.append(model)

        clean = component_predictions(models, inputs, batch_size=args.batch_size, device=device)
        locked = np.asarray(prediction_artifact[f"probabilities_{family}"][order], dtype=np.float64)
        rows.append(
            result_row(
                family=family,
                condition="clean_float",
                level=0,
                fault_seed="none",
                labels=labels,
                probabilities=clean,
                clean=clean,
                realized_bit_flips=0,
            )
        )
        rows[-1]["mean_absolute_locked_probability_error"] = float(np.mean(np.abs(clean - locked)))
        probability_artifacts[f"{family}__clean_float"] = clean

        for rate in protocol["input_pixel_bit_flip_rates"]:
            fault_probabilities = []
            realized = None
            for fault_seed in protocol["seeds"]:
                corrupted, realized = input_bit_faults(
                    inputs,
                    float(rate),
                    stable_seed("input-bit-flip", family, fault_seed, rate),
                )
                probabilities = component_predictions(
                    models, corrupted, batch_size=args.batch_size, device=device
                )
                fault_probabilities.append(probabilities)
                rows.append(
                    result_row(
                        family=family,
                        condition="uint8_input_bit_flip_rate",
                        level=float(rate),
                        fault_seed=int(fault_seed),
                        labels=labels,
                        probabilities=probabilities,
                        clean=clean,
                        realized_bit_flips=int(realized),
                    )
                )
            aggregate = normalize_probability_rows(np.mean(fault_probabilities, axis=0))
            rows.append(
                result_row(
                    family=family,
                    condition="uint8_input_bit_flip_rate",
                    level=float(rate),
                    fault_seed="aggregate",
                    labels=labels,
                    probabilities=aggregate,
                    clean=clean,
                    realized_bit_flips=int(realized),
                )
            )
            probability_artifacts[f"{family}__input_rate_{rate}"] = aggregate
            print(f"[{family}] input bit-flip rate {rate} complete", flush=True)

        for bit_flips in protocol["head_parameter_bit_flips"]:
            fault_probabilities = []
            scales_by_seed = []
            for fault_seed in protocol["seeds"]:
                model_probabilities = []
                for model_index, model in enumerate(models):
                    head = head_module(model, family)
                    original = head.weight.detach().clone()
                    corrupted, scale = quantize_and_flip_parameter_bits(
                        original,
                        bit_flips=int(bit_flips),
                        seed=stable_seed(
                            "head-bit-flip", family, fault_seed, model_index, bit_flips
                        ),
                    )
                    scales_by_seed.append(scale)
                    with torch.no_grad():
                        head.weight.copy_(corrupted.to(device))
                    model_probabilities.append(
                        predict_batches(model, inputs, batch_size=args.batch_size, device=device)
                    )
                    with torch.no_grad():
                        head.weight.copy_(original)
                probabilities = normalize_probability_rows(np.mean(model_probabilities, axis=0))
                fault_probabilities.append(probabilities)
                row = result_row(
                    family=family,
                    condition="symmetric_int8_head_weight_bit_flips",
                    level=int(bit_flips),
                    fault_seed=int(fault_seed),
                    labels=labels,
                    probabilities=probabilities,
                    clean=clean,
                    realized_bit_flips=int(bit_flips) * len(models),
                )
                row["quantization_scale_mean"] = float(np.mean(scales_by_seed[-len(models) :]))
                rows.append(row)
            aggregate = normalize_probability_rows(np.mean(fault_probabilities, axis=0))
            row = result_row(
                family=family,
                condition="symmetric_int8_head_weight_bit_flips",
                level=int(bit_flips),
                fault_seed="aggregate",
                labels=labels,
                probabilities=aggregate,
                clean=clean,
                realized_bit_flips=int(bit_flips) * len(models),
            )
            row["quantization_scale_mean"] = float(np.mean(scales_by_seed))
            rows.append(row)
            probability_artifacts[f"{family}__head_flips_{bit_flips}"] = aggregate
            print(f"[{family}] head bit flips {bit_flips} complete", flush=True)

        del models, inputs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(output_dir / "fault_robustness_metrics.csv", index=False)
    np.savez_compressed(
        output_dir / "fault_robustness_predictions.npz",
        **probability_artifacts,
        labels=labels,
        image_ids=cohort["image_id"].astype(str).to_numpy(),
        class_names=np.asarray(class_names),
    )
    aggregate_rows = result_frame[
        result_frame["fault_seed"].astype(str).isin({"none", "aggregate"})
    ]
    summary = {
        "status": "LOCKED_POLAR_FAULT_ROBUSTNESS_COMPLETE",
        "selection_role": "none",
        "reported_separately_from_faithfulness": True,
        "selection_lock_sha256": lock_hash,
        "cohort_sha256": sha256_file(cohort_path),
        "cohort_rows": len(cohort),
        "test_predictions_sha256": sha256_file(test_predictions_path),
        "protocol": protocol,
        "parameter_fault_scope": "per-model symmetric-int8 classifier weight matrix",
        "input_fault_scope": "post-resize uint8 RGB tensor before ImageNet normalization",
        "aggregate_results": aggregate_rows.to_dict("records"),
        "metrics_sha256": sha256_file(output_dir / "fault_robustness_metrics.csv"),
        "predictions_sha256": sha256_file(output_dir / "fault_robustness_predictions.npz"),
        "implementation_sha256": implementation_hash,
        "runtime_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "test_rows_read": len(cohort),
        "test_used_for_selection": False,
    }
    write_json(summary_path, summary)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
