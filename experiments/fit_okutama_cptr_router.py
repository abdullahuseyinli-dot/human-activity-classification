"""Fit and evaluate the CPTR continuous utility router with grouped OOF predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from torch import nn
from torch.nn import functional as F

from hac.cptr_training import classification_summary
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
        "--manifest",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/development_manifest.csv"),
    )
    parser.add_argument(
        "--legacy-predictions",
        type=Path,
        default=Path(
            ".runs/vcoco_v3/temporal/development/teacher/temporal_8f_050s/seed-42/validation_predictions.npz"
        ),
    )
    parser.add_argument(
        "--parts-predictions",
        type=Path,
        default=Path(
            ".runs/cptr/adaptive/centre_short_parts/seed-42/validation_predictions.npz"
        ),
    )
    parser.add_argument(
        "--trajectory-predictions",
        type=Path,
        default=Path(
            ".runs/cptr/adaptive/centre_short_trajectory/seed-42/validation_predictions.npz"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output-dir", type=Path, default=Path(".runs/cptr/router"))
    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalized(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    return values / values.sum(axis=1, keepdims=True).clip(min=1e-12)


def prediction_diagnostics(probabilities: np.ndarray) -> np.ndarray:
    values = normalized(probabilities)
    ordered = np.sort(values, axis=1)
    entropy = -(values * np.log(values.clip(min=1e-12))).sum(axis=1)
    return np.column_stack(
        (
            values,
            1.0 - values.max(axis=1),
            entropy,
            1.0 - (ordered[:, -1] - ordered[:, -2]),
        )
    )


def geometry_features(frame: pd.DataFrame) -> np.ndarray:
    width = (frame["bbox_xmax"] - frame["bbox_xmin"]).to_numpy(dtype=float)
    height = (frame["bbox_ymax"] - frame["bbox_ymin"]).to_numpy(dtype=float)
    centre_x = ((frame["bbox_xmin"] + frame["bbox_xmax"]) / 2.0 / 1280.0).to_numpy()
    centre_y = ((frame["bbox_ymin"] + frame["bbox_ymax"]) / 2.0 / 720.0).to_numpy()
    edge = np.minimum.reduce((centre_x, 1.0 - centre_x, centre_y, 1.0 - centre_y))
    return np.column_stack(
        (
            centre_x,
            centre_y,
            np.log(np.clip(width * height / (1280.0 * 720.0), 1e-8, None)),
            np.log(np.clip(width / np.clip(height, 1e-6, None), 1e-8, None)),
            edge,
        )
    )


class UtilityRouter(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(24, 12),
            nn.GELU(),
            nn.Linear(12, 3),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def fit_router(
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    seed_everything(seed)
    train_x = features[train_indices]
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0).clip(min=1e-5)
    scaled_train = torch.as_tensor(
        (train_x - mean) / scale,
        dtype=torch.float32,
        device=device,
    )
    scaled_predict = torch.as_tensor(
        (features[predict_indices] - mean) / scale,
        dtype=torch.float32,
        device=device,
    )
    target = torch.as_tensor(targets[train_indices], dtype=torch.float32, device=device)
    model = UtilityRouter(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    rescue_positive = target[:, 1].sum().clamp_min(1.0)
    harm_positive = target[:, 2].sum().clamp_min(1.0)
    rescue_weight = (len(target) - rescue_positive) / rescue_positive
    harm_weight = (len(target) - harm_positive) / harm_positive
    final_loss = math.nan
    model.train()
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        output = model(scaled_train)
        gain_loss = F.smooth_l1_loss(output[:, 0], target[:, 0], beta=0.25)
        rescue_loss = F.binary_cross_entropy_with_logits(
            output[:, 1], target[:, 1], pos_weight=rescue_weight
        )
        harm_loss = F.binary_cross_entropy_with_logits(
            output[:, 2], target[:, 2], pos_weight=harm_weight
        )
        loss = gain_loss + 0.15 * rescue_loss + 0.15 * harm_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        final_loss = float(loss.detach())
    model.eval()
    with torch.inference_mode():
        predicted = model(scaled_predict)
        predicted = torch.column_stack(
            (predicted[:, 0], predicted[:, 1].sigmoid(), predicted[:, 2].sigmoid())
        )
    state = {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "final_loss": final_loss,
    }
    return predicted.cpu().numpy(), state


def safe_binary_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(target)) < 2:
        auc = None
    else:
        auc = float(roc_auc_score(target, probability))
    return {
        "roc_auc": auc,
        "brier": float(brier_score_loss(target, probability)),
        "prevalence": float(np.mean(target)),
    }


def routed_probabilities(
    static: np.ndarray,
    experts: np.ndarray,
    predicted_gain: np.ndarray,
    budget: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(round(len(static) * budget))
    best_expert = predicted_gain.argmax(axis=1)
    best_gain = predicted_gain[np.arange(len(static)), best_expert]
    selected = np.zeros(len(static), dtype=bool)
    if count:
        chosen = np.argsort(best_gain, kind="stable")[-count:]
        selected[chosen] = True
    output = static.copy()
    output[selected] = experts[best_expert[selected], np.flatnonzero(selected)]
    return normalized(output), selected, best_expert


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CPTR router fitting requires CUDA; CPU fallback is disabled")
    protocol_path = args.protocol.resolve()
    protocol_lock_path = args.protocol_lock.resolve()
    manifest_path = args.manifest.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_lock = json.loads(protocol_lock_path.read_text(encoding="utf-8"))
    if protocol_lock.get("status") != "OKUTAMA_CPTR_PROTOCOL_LOCKED_BEFORE_FIT":
        raise RuntimeError("The CPTR protocol is not locked")
    if protocol_lock["source_sha256"]["protocol"] != sha256_file(protocol_path):
        raise RuntimeError("The protocol changed after locking")
    if protocol_lock["source_sha256"]["development_manifest"] != sha256_file(
        manifest_path
    ):
        raise RuntimeError("The development manifest changed after locking")

    paths = {
        "legacy_temporal": args.legacy_predictions.resolve(),
        "centre_short_parts": args.parts_predictions.resolve(),
        "centre_short_trajectory": args.trajectory_predictions.resolve(),
    }
    arrays = {name: np.load(path, allow_pickle=False) for name, path in paths.items()}
    reference = arrays["centre_short_parts"]
    sample_ids = reference["sample_ids"].astype(str)
    labels = reference["labels"].astype(int)
    static = normalized(reference["static_probabilities"])
    expert_probabilities = {
        "legacy_temporal": normalized(arrays["legacy_temporal"]["probabilities"]),
        "centre_short_parts": normalized(reference["probabilities"]),
        "centre_short_trajectory": normalized(
            arrays["centre_short_trajectory"]["probabilities"]
        ),
    }
    for name, array in arrays.items():
        if not np.array_equal(array["sample_ids"].astype(str), sample_ids):
            raise RuntimeError(f"Router prediction order differs for {name}")
        if not np.array_equal(array["labels"].astype(int), labels):
            raise RuntimeError(f"Router labels differ for {name}")

    manifest = pd.read_csv(
        manifest_path,
        dtype={"sample_id": str, "recording_id": str, "track_id": str},
    )
    validation = manifest[manifest["split"].eq("validation")].set_index("sample_id")
    validation = validation.loc[sample_ids].reset_index()
    if len(validation) != len(sample_ids):
        raise RuntimeError("Router samples do not align with the locked validation split")
    groups = validation["scenario_id"].astype(str).to_numpy()
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 3:
        raise RuntimeError("Grouped router evaluation requires at least three scenarios")

    feature_names = [
        "static_p_sitting",
        "static_p_standing",
        "static_p_locomotion",
        "one_minus_static_max_probability",
        "static_entropy",
        "inverse_static_top_two_margin",
        "bbox_centre_x",
        "bbox_centre_y",
        "bbox_log_area_fraction",
        "bbox_log_aspect_ratio",
        "bbox_edge_distance",
    ]
    features = np.column_stack((prediction_diagnostics(static), geometry_features(validation)))
    if features.shape[1] != len(feature_names):
        raise RuntimeError("Router feature declaration changed")
    expert_names = tuple(expert_probabilities)
    stacked_experts = np.stack([expert_probabilities[name] for name in expert_names])
    targets: dict[str, np.ndarray] = {}
    static_prediction = static.argmax(axis=1)
    for name in expert_names:
        probability = expert_probabilities[name]
        expert_prediction = probability.argmax(axis=1)
        gain = np.log(probability[np.arange(len(labels)), labels].clip(min=1e-8)) - np.log(
            static[np.arange(len(labels)), labels].clip(min=1e-8)
        )
        rescue = ((static_prediction != labels) & (expert_prediction == labels)).astype(float)
        harm = ((static_prediction == labels) & (expert_prediction != labels)).astype(float)
        targets[name] = np.column_stack((np.clip(gain, -4.0, 4.0), rescue, harm))

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(args.seed)
    oof = {name: np.zeros((len(labels), 3), dtype=np.float32) for name in expert_names}
    fold_records: list[dict] = []
    splitter = LeaveOneGroupOut()
    started = time.perf_counter()
    for fold, (train_indices, held_indices) in enumerate(splitter.split(features, labels, groups)):
        held_groups = sorted(set(groups[held_indices]))
        for expert_index, name in enumerate(expert_names):
            predicted, state = fit_router(
                features,
                targets[name],
                train_indices,
                held_indices,
                seed=args.seed + fold * 10 + expert_index,
                device=device,
            )
            oof[name][held_indices] = predicted
            fold_records.append(
                {
                    "fold": fold,
                    "expert": name,
                    "held_scenarios": held_groups,
                    "train_samples": int(len(train_indices)),
                    "held_samples": int(len(held_indices)),
                    "final_training_loss": state["final_loss"],
                }
            )

    predicted_gain = np.column_stack([oof[name][:, 0] for name in expert_names])
    budgets = list(map(float, protocol["router"]["clip_budget_fractions"]))
    budget_curve: list[dict] = []
    for budget in budgets:
        probabilities, selected, selected_expert = routed_probabilities(
            static,
            stacked_experts,
            predicted_gain,
            budget,
        )
        counts = {
            name: int(np.sum(selected & (selected_expert == index)))
            for index, name in enumerate(expert_names)
        }
        budget_curve.append(
            {
                "budget_fraction": budget,
                "selected_samples": int(selected.sum()),
                "selected_expert_counts": counts,
                "metrics": classification_summary(labels, probabilities),
            }
        )

    mandatory_baselines: dict[str, list[dict]] = {}
    diagnostics = prediction_diagnostics(static)
    uncertainty_scores = {
        "one_minus_static_max_probability": diagnostics[:, 3],
        "static_entropy": diagnostics[:, 4],
        "inverse_static_top_two_margin": diagnostics[:, 5],
    }
    strongest = expert_probabilities["centre_short_parts"]
    for name, scores in uncertainty_scores.items():
        rows = []
        for budget in budgets:
            count = int(round(len(labels) * budget))
            selected = np.zeros(len(labels), dtype=bool)
            if count:
                selected[np.argsort(scores, kind="stable")[-count:]] = True
            probabilities = static.copy()
            probabilities[selected] = strongest[selected]
            rows.append(
                {
                    "budget_fraction": budget,
                    "metrics": classification_summary(labels, probabilities),
                }
            )
        mandatory_baselines[name] = rows

    router_target_metrics = {}
    for name in expert_names:
        router_target_metrics[name] = {
            "gain_mae": float(np.mean(np.abs(oof[name][:, 0] - targets[name][:, 0]))),
            "gain_correlation": float(
                np.corrcoef(oof[name][:, 0], targets[name][:, 0])[0, 1]
            ),
            "rescue": safe_binary_metrics(targets[name][:, 1], oof[name][:, 1]),
            "harm": safe_binary_metrics(targets[name][:, 2], oof[name][:, 2]),
        }

    final_models = {}
    all_indices = np.arange(len(labels))
    for expert_index, name in enumerate(expert_names):
        _, state = fit_router(
            features,
            targets[name],
            all_indices,
            all_indices[:1],
            seed=args.seed + 100 + expert_index,
            device=device,
        )
        final_models[name] = state
    checkpoint_path = args.output_dir.resolve() / "router_checkpoint.pt"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "status": "OKUTAMA_CPTR_UTILITY_ROUTER_COMPLETE",
            "feature_names": feature_names,
            "expert_names": expert_names,
            "models": final_models,
        },
        checkpoint_path,
    )
    predictions_path = output_dir / "oof_router_predictions.npz"
    np.savez_compressed(
        predictions_path,
        sample_ids=sample_ids,
        labels=labels,
        scenario_ids=groups,
        static_probabilities=static,
        expert_names=np.asarray(expert_names),
        expert_probabilities=stacked_experts,
        predicted_gain=predicted_gain,
        predicted_rescue=np.column_stack([oof[name][:, 1] for name in expert_names]),
        predicted_harm=np.column_stack([oof[name][:, 2] for name in expert_names]),
        true_gain=np.column_stack([targets[name][:, 0] for name in expert_names]),
    )
    request = {
        "status": "OKUTAMA_CPTR_ROUTER_REQUEST",
        "seed": int(args.seed),
        "samples": int(len(labels)),
        "scenarios": unique_groups,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "protocol": sha256_file(protocol_path),
            "protocol_lock": sha256_file(protocol_lock_path),
            "manifest": sha256_file(manifest_path),
            **{f"{name}_predictions": sha256_file(path) for name, path in paths.items()},
            "runner": sha256_file(Path(__file__).resolve()),
        },
    }
    request_hash = hashlib.sha256(
        json.dumps(request, sort_keys=True).encode("utf-8")
    ).hexdigest()
    write_json(output_dir / "request.json", {**request, "request_sha256": request_hash})
    summary = {
        "status": "OKUTAMA_CPTR_UTILITY_ROUTER_COMPLETE",
        "evaluation": "leave_one_scenario_out_oof_on_model_held_validation_predictions",
        "fold_count": int(len(unique_groups)),
        "folds": fold_records,
        "feature_names": feature_names,
        "expert_names": list(expert_names),
        "always_static": classification_summary(labels, static),
        "always_expert": {
            name: classification_summary(labels, probability)
            for name, probability in expert_probabilities.items()
        },
        "router_target_metrics": router_target_metrics,
        "budget_curve": budget_curve,
        "mandatory_uncertainty_baselines": mandatory_baselines,
        "training_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "runtime_seconds": time.perf_counter() - started,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "request_sha256": request_hash,
        "artifact_sha256": {
            "router_checkpoint.pt": sha256_file(checkpoint_path),
            "oof_router_predictions.npz": sha256_file(predictions_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
