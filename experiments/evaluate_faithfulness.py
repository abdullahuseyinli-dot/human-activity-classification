"""Select, lock, and evaluate faithful image attributions for the final models.

The final test split is never used to choose an attribution method. Candidate
methods are compared on a deterministic, class-balanced subset of out-of-fold
predictions. The resulting lock is written before the test explanations are
generated. Bulky maps and perturbation traces remain in the local output root;
only compact, path-sanitized evidence is intended for portfolio export.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import random
import re
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam.metrics.road import NoisyLinearImputer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hac.augmentations import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform  # noqa: E402
from hac.config import ModelConfig  # noqa: E402
from hac.explainability import (  # noqa: E402
    attribution_spearman,
    calibrated_probabilities,
    conv_cam_attributions,
    curve_auc,
    dino_attention_attributions,
    gaussian_baseline,
    integrated_gradients_attribution,
    normalize_attribution,
    patch_keep_mask,
    patch_scores,
)
from hac.models import build_model  # noqa: E402
from hac.protocol import load_and_validate_manifest, sha256_file  # noqa: E402

FAMILIES = ("convnext_small", "dinov2_small")
SEEDS = (42, 52, 62)
DISPLAY_NAMES = {
    "convnext_small": "ConvNeXt-Small",
    "dinov2_small": "DINOv2-Small",
    "probability_blend": "0.1 ConvNeXt + 0.9 DINOv2",
}
CANDIDATES = {
    "convnext_small": (
        "gradcam",
        "hirescam",
        "integrated_gradients",
    ),
    "dinov2_small": (
        "attention_rollout",
        "gradient_attention_rollout",
        "integrated_gradients",
    ),
}
ELIGIBLE = {
    "convnext_small": set(CANDIDATES["convnext_small"]),
    "dinov2_small": {"gradient_attention_rollout", "integrated_gradients"},
}
ROAD_FRACTIONS = np.arange(0.1, 1.0, 0.1, dtype=np.float64)
CURVE_FRACTIONS = np.arange(0.0, 1.01, 0.1, dtype=np.float64)
PATCH_GRID_SIZE = 16
TENSOR_ID = re.compile(r"^tensor\(([^)]+)\)$")
REPLAY_TOLERANCE = {
    "convnext_small": 1e-3,
    "dinov2_small": 1e-3,
    "probability_blend": 1e-3,
}
OOF_REPLAY_TOLERANCE = 1e-3


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    family: str
    seed: int
    checkpoint: Path
    config: dict
    temperature: float
    policy: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-per-class", type=int, default=12)
    parser.add_argument("--integrated-gradient-steps", type=int, default=16)
    parser.add_argument("--faithfulness-subsets", type=int, default=16)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def sha256_text(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def stable_seed(*values: object) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def normalize_image_id(value: object) -> str:
    text = str(value).strip()
    match = TENSOR_ID.fullmatch(text)
    return match.group(1).strip() if match else text


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def policy_for_family(lock: dict, family: str) -> str:
    policy = str(lock["evaluation_policy"][family])
    if policy not in {"center_crop", "center_plus_horizontal_flip"}:
        raise RuntimeError(f"Unsupported locked policy for {family}: {policy}")
    return policy


def checkpoint_specs(
    final_root: Path,
    downstream_lock: dict,
    family: str,
    *,
    fold_id: int | None,
) -> list[CheckpointSpec]:
    selection_lock = read_json(final_root / "configs" / "selection_lock.json")
    candidate = selection_lock["selected"][family]
    candidate_id = str(candidate["candidate_id"])
    config = dict(candidate["config"])
    policy = policy_for_family(downstream_lock, family)
    specs = []
    for seed in SEEDS:
        cv_dir = final_root / "predictions" / family / f"seed_{seed}" / "cv"
        cv_summary = read_json(cv_dir / "cv_summary.json")
        temperature_key = "tta_temperature" if policy == "center_plus_horizontal_flip" else "temperature"
        if fold_id is None:
            checkpoint = (
                final_root
                / "checkpoints"
                / family
                / f"final_full_{candidate_id}_seed_{seed}"
                / "full_pool_checkpoint.pt"
            )
        else:
            checkpoint = (
                final_root
                / "checkpoints"
                / family
                / f"final_cv_{candidate_id}_seed_{seed}_fold{int(fold_id):02d}"
                / "best_checkpoint.pt"
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        specs.append(
            CheckpointSpec(
                family=family,
                seed=int(seed),
                checkpoint=checkpoint,
                config=config,
                temperature=float(cv_summary[temperature_key]),
                policy=policy,
            )
        )
    return specs


def load_model(spec: CheckpointSpec, device: torch.device) -> torch.nn.Module:
    payload = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
    if str(payload.get("model_kind")) != spec.family:
        raise RuntimeError(f"Checkpoint family mismatch: {spec.checkpoint}")
    if dict(payload.get("cfg", {})) != spec.config:
        raise RuntimeError(f"Checkpoint configuration drift: {spec.checkpoint}")
    with torch.device("meta"):
        model = build_model(
            ModelConfig.from_mapping(spec.family, spec.config), pretrained=False
        )
    model.load_state_dict(payload["model_state_dict"], strict=True, assign=True)
    return model.to(device).eval()


class FamilyEnsemble:
    """Three-seed ensemble that preserves the locked per-seed score policy."""

    def __init__(self, specs: list[CheckpointSpec], device: torch.device) -> None:
        if len(specs) != len(SEEDS):
            raise ValueError("The locked ensemble requires exactly three seeds")
        self.specs = list(specs)
        self.device = device
        self.family = specs[0].family
        self.models = [load_model(spec, device) for spec in specs]

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        probabilities = []
        for model, spec in zip(self.models, self.specs, strict=True):
            probabilities.append(
                calibrated_probabilities(
                    model,
                    inputs,
                    temperature=spec.temperature,
                    policy=spec.policy,
                )
            )
        return torch.stack(probabilities).mean(dim=0)

    def close(self) -> None:
        self.models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class BlendEnsemble:
    def __init__(
        self,
        convnext: FamilyEnsemble,
        dinov2: FamilyEnsemble,
        convnext_weight: float,
    ) -> None:
        self.convnext = convnext
        self.dinov2 = dinov2
        self.convnext_weight = float(convnext_weight)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.convnext_weight * self.convnext.predict(inputs) + (
            1.0 - self.convnext_weight
        ) * self.dinov2.predict(inputs)


def predict_batches(
    predictor: FamilyEnsemble | BlendEnsemble,
    inputs: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    rows = []
    with torch.no_grad():
        for start in range(0, len(inputs), int(batch_size)):
            batch = inputs[start : start + int(batch_size)].to(device, non_blocking=True)
            rows.append(predictor.predict(batch).detach().cpu().numpy())
    return np.concatenate(rows, axis=0)


def preprocess_image(path: Path, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    with Image.open(path) as image:
        tensor = build_eval_transform()(image.convert("RGB"))
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype)[:, None, None]
    display = torch.clamp(tensor * std + mean, 0.0, 1.0).permute(1, 2, 0).numpy()
    return tensor.unsqueeze(0).to(device), display


def aggregate_maps(seed_maps: list[np.ndarray]) -> np.ndarray:
    if len(seed_maps) != len(SEEDS):
        raise RuntimeError("An attribution must cover every locked seed")
    return normalize_attribution(np.stack(seed_maps).mean(axis=0))


def candidate_maps(
    ensemble: FamilyEnsemble,
    inputs: torch.Tensor,
    target_class: int,
    *,
    integrated_gradient_steps: int,
    requested: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]], dict[str, list[float]]]:
    requested = requested or set(CANDIDATES[ensemble.family])
    maps_by_method = {method: [] for method in requested}
    view_similarity: dict[str, list[float]] = {method: [] for method in requested}
    baseline = gaussian_baseline(inputs)

    for model, spec in zip(ensemble.models, ensemble.specs, strict=True):
        if ensemble.family == "convnext_small":
            cam_methods = requested.intersection({"gradcam", "hirescam"})
            if cam_methods:
                cams = conv_cam_attributions(
                    model,
                    inputs,
                    target_class,
                    temperature=spec.temperature,
                    policy=spec.policy,
                )
                if "gradcam" in cam_methods:
                    maps_by_method["gradcam"].append(cams.gradcam)
                    if len(cams.gradcam_views) == 2:
                        view_similarity["gradcam"].append(
                            attribution_spearman(
                                patch_scores(cams.gradcam_views[0]),
                                patch_scores(cams.gradcam_views[1]),
                            )
                        )
                if "hirescam" in cam_methods:
                    maps_by_method["hirescam"].append(cams.hirescam)
                    if len(cams.hirescam_views) == 2:
                        view_similarity["hirescam"].append(
                            attribution_spearman(
                                patch_scores(cams.hirescam_views[0]),
                                patch_scores(cams.hirescam_views[1]),
                            )
                        )
            if "integrated_gradients" in requested:
                maps_by_method["integrated_gradients"].append(
                    integrated_gradients_attribution(
                        model,
                        inputs,
                        baseline,
                        target_class,
                        temperature=spec.temperature,
                        policy=spec.policy,
                        steps=integrated_gradient_steps,
                    )
                )
        else:
            attention_methods = requested.intersection(
                {"attention_rollout", "gradient_attention_rollout"}
            )
            if attention_methods:
                attentions = dino_attention_attributions(
                    model,
                    inputs,
                    target_class,
                    temperature=spec.temperature,
                )
                if "attention_rollout" in attention_methods:
                    maps_by_method["attention_rollout"].append(
                        attentions.attention_rollout
                    )
                if "gradient_attention_rollout" in attention_methods:
                    maps_by_method["gradient_attention_rollout"].append(
                        attentions.gradient_attention_rollout
                    )
            if "integrated_gradients" in requested:
                maps_by_method["integrated_gradients"].append(
                    integrated_gradients_attribution(
                        model,
                        inputs,
                        baseline,
                        target_class,
                        temperature=spec.temperature,
                        policy=spec.policy,
                        steps=integrated_gradient_steps,
                    )
                )

    aggregate = {method: aggregate_maps(values) for method, values in maps_by_method.items()}
    return aggregate, maps_by_method, view_similarity


def mean_seed_agreement(seed_maps: list[np.ndarray]) -> float:
    values = [
        attribution_spearman(patch_scores(left), patch_scores(right))
        for left, right in combinations(seed_maps, 2)
    ]
    return float(np.mean(values))


def road_imputation(
    inputs: torch.Tensor,
    keep_mask: torch.Tensor,
    *,
    random_seed: int,
) -> torch.Tensor:
    image = inputs.detach().cpu()[0]
    mask = keep_mask.detach().cpu()
    if float(mask.sum()) <= 0.0:
        raise ValueError("ROAD requires at least one observed pixel")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(random_seed))
        return NoisyLinearImputer(noise=0.01)(image, mask)


def perturbation_batch(
    inputs: torch.Tensor,
    attribution: np.ndarray,
    *,
    image_id: str,
    subset_count: int,
) -> tuple[torch.Tensor, list[dict]]:
    scores = patch_scores(attribution, PATCH_GRID_SIZE)
    input_cpu = inputs.detach().cpu()[0]
    baseline_cpu = gaussian_baseline(inputs).detach().cpu()[0]
    tensors: list[torch.Tensor] = []
    metadata: list[dict] = []

    for fraction in ROAD_FRACTIONS:
        for order in ("most", "least", "random"):
            random_key = stable_seed("road-order", image_id, fraction)
            keep = patch_keep_mask(
                scores,
                float(fraction),
                order=order,
                random_seed=random_key,
            )
            tensors.append(
                road_imputation(
                    inputs,
                    keep,
                    random_seed=stable_seed("road-noise", image_id, order, fraction),
                )
            )
            metadata.append(
                {"metric": f"road_{order}", "fraction": float(fraction)}
            )

    for fraction in CURVE_FRACTIONS:
        most_keep = patch_keep_mask(scores, float(fraction), order="most")
        random_keep = patch_keep_mask(
            scores,
            float(fraction),
            order="random",
            random_seed=stable_seed("deletion-random", image_id),
        )
        reveal = 1.0 - most_keep
        tensors.extend(
            [
                input_cpu * most_keep + baseline_cpu * (1.0 - most_keep),
                baseline_cpu * (1.0 - reveal) + input_cpu * reveal,
                input_cpu * random_keep + baseline_cpu * (1.0 - random_keep),
            ]
        )
        metadata.extend(
            [
                {"metric": "deletion", "fraction": float(fraction)},
                {"metric": "insertion", "fraction": float(fraction)},
                {"metric": "random_deletion", "fraction": float(fraction)},
            ]
        )

    rng = np.random.default_rng(stable_seed("faithfulness-correlation", image_id))
    flat_scores = scores.reshape(-1)
    subset_size = max(1, int(round(0.125 * flat_scores.size)))
    for subset_index in range(int(subset_count)):
        removed = rng.choice(flat_scores.size, size=subset_size, replace=False)
        keep_values = np.ones(flat_scores.size, dtype=np.float32)
        keep_values[removed] = 0.0
        keep = torch.from_numpy(keep_values.reshape(scores.shape))[None, None]
        keep = torch.nn.functional.interpolate(
            keep, size=tuple(input_cpu.shape[-2:]), mode="nearest"
        )[0, 0]
        tensors.append(input_cpu * keep + baseline_cpu * (1.0 - keep))
        metadata.append(
            {
                "metric": "faithfulness_subset",
                "subset": int(subset_index),
                "attribution_mass": float(flat_scores[removed].sum()),
            }
        )
    return torch.stack(tensors), metadata


def evaluate_attribution(
    predictor: FamilyEnsemble | BlendEnsemble,
    inputs: torch.Tensor,
    attribution: np.ndarray,
    target_class: int,
    *,
    image_id: str,
    subset_count: int,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, list[dict]]:
    original = float(
        predictor.predict(inputs)[0, int(target_class)].detach().cpu().item()
    )
    batch, metadata = perturbation_batch(
        inputs, attribution, image_id=image_id, subset_count=subset_count
    )
    probabilities = predict_batches(
        predictor, batch, device=device, batch_size=batch_size
    )[:, int(target_class)]
    for row, probability in zip(metadata, probabilities, strict=True):
        row["target_probability"] = float(probability)

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    curve_rows = []
    for metric in (
        "road_most",
        "road_least",
        "road_random",
        "deletion",
        "insertion",
        "random_deletion",
    ):
        selected = [row for row in metadata if row["metric"] == metric]
        fractions = np.asarray([row["fraction"] for row in selected], dtype=np.float64)
        values = np.asarray(
            [row["target_probability"] for row in selected], dtype=np.float64
        )
        curves[metric] = fractions, values
        curve_rows.extend(
            {
                "metric": metric,
                "fraction": float(fraction),
                "target_probability": float(value),
            }
            for fraction, value in zip(fractions, values, strict=True)
        )

    subset_rows = [row for row in metadata if row["metric"] == "faithfulness_subset"]
    masses = np.asarray([row["attribution_mass"] for row in subset_rows])
    drops = original - np.asarray([row["target_probability"] for row in subset_rows])
    faithfulness_correlation = attribution_spearman(masses[None, :], drops[None, :])
    road_morf = curves["road_most"][1]
    road_lerf = curves["road_least"][1]
    metrics = {
        "original_target_probability": original,
        "road_combined": float(np.mean((road_lerf - road_morf) / 2.0)),
        "road_morf_auc": curve_auc(*curves["road_most"]),
        "road_lerf_auc": curve_auc(*curves["road_least"]),
        "road_random_auc": curve_auc(*curves["road_random"]),
        "road_gap_auc": float(
            curve_auc(*curves["road_least"]) - curve_auc(*curves["road_most"])
        ),
        "deletion_auc": curve_auc(*curves["deletion"]),
        "insertion_auc": curve_auc(*curves["insertion"]),
        "random_deletion_auc": curve_auc(*curves["random_deletion"]),
        "selectivity_gap": float(
            curve_auc(*curves["random_deletion"]) - curve_auc(*curves["deletion"])
        ),
        "faithfulness_spearman": float(faithfulness_correlation),
    }
    return metrics, curve_rows


def deterministic_selection_cohort(
    assignment: pd.DataFrame,
    *,
    per_class: int,
) -> pd.DataFrame:
    if per_class < 1:
        raise ValueError("selection-per-class must be positive")
    frame = assignment.copy()
    frame["selection_key"] = frame["image_id"].astype(str).map(
        lambda value: hashlib.sha256(f"20260822|{value}".encode()).hexdigest()
    )
    pieces = []
    for _, group in frame.groupby("label", sort=True):
        if len(group) < per_class:
            raise RuntimeError("OOF selection cohort exceeds a class count")
        pieces.append(group.sort_values("selection_key").head(per_class))
    return pd.concat(pieces).sort_values(["fold_id", "selection_key"]).reset_index(drop=True)


def load_oof_probabilities(final_root: Path, family: str, policy: str) -> tuple[list[str], np.ndarray]:
    rows = []
    ids_ref = None
    filename = (
        "oof_tta_calibrated_probs.npy"
        if policy == "center_plus_horizontal_flip"
        else "oof_calibrated_probs.npy"
    )
    for seed in SEEDS:
        base = final_root / "predictions" / family / f"seed_{seed}" / "cv"
        ids = pd.read_csv(base / "oof_image_ids.csv", dtype={"image_id": str})[
            "image_id"
        ].map(normalize_image_id).tolist()
        if ids_ref is None:
            ids_ref = ids
        elif ids != ids_ref:
            raise RuntimeError(f"OOF ID drift across {family} seeds")
        rows.append(np.load(base / filename))
    assert ids_ref is not None
    return ids_ref, np.stack(rows).mean(axis=0)


def select_methods(
    *,
    final_root: Path,
    downstream_lock: dict,
    manifest: pd.DataFrame,
    output_dir: Path,
    per_class: int,
    integrated_gradient_steps: int,
    subset_count: int,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, pd.DataFrame]:
    assignment = pd.read_csv(
        final_root / "tables" / "T03c_cv_fold_assignment.csv",
        dtype={"image_id": str},
    )
    path_map = manifest.set_index("image_id")["resolved_image_path"].astype(str).to_dict()
    assignment["resolved_image_path"] = assignment["image_id"].map(path_map)
    if assignment["resolved_image_path"].isna().any():
        raise RuntimeError("OOF assignment contains image IDs absent from the manifest")
    cohort = deterministic_selection_cohort(assignment, per_class=per_class)
    cohort.to_csv(output_dir / "oof_selection_cohort.csv", index=False)

    detail_rows: list[dict] = []
    curve_rows: list[dict] = []
    replay_rows: list[dict] = []
    for family in FAMILIES:
        policy = policy_for_family(downstream_lock, family)
        oof_ids, oof_probs = load_oof_probabilities(final_root, family, policy)
        probability_by_id = dict(zip(oof_ids, oof_probs, strict=True))
        for fold_id, fold_frame in cohort.groupby("fold_id", sort=True):
            ensemble = FamilyEnsemble(
                checkpoint_specs(
                    final_root,
                    downstream_lock,
                    family,
                    fold_id=int(fold_id),
                ),
                device,
            )
            try:
                for record in fold_frame.itertuples(index=False):
                    image_id = str(record.image_id)
                    completed = len(
                        {
                            (row["family"], row["image_id"])
                            for row in detail_rows
                            if row["family"] == family
                        }
                    )
                    print(
                        f"[oof] {family} image {completed + 1:02d}/{len(cohort)} "
                        f"fold={int(fold_id)} id={image_id}",
                        flush=True,
                    )
                    inputs, _ = preprocess_image(Path(record.resolved_image_path), device)
                    expected = np.asarray(probability_by_id[image_id], dtype=np.float64)
                    observed = ensemble.predict(inputs).detach().cpu().numpy()[0]
                    tolerance = OOF_REPLAY_TOLERANCE
                    max_absolute = float(np.max(np.abs(observed - expected)))
                    class_match = bool(observed.argmax() == expected.argmax())
                    replay_rows.append(
                        {
                            "family": family,
                            "image_id": image_id,
                            "fold_id": int(fold_id),
                            "max_absolute_probability_difference": max_absolute,
                            "mean_absolute_probability_difference": float(
                                np.mean(np.abs(observed - expected))
                            ),
                            "class_match": class_match,
                            "acceptance_tolerance": float(tolerance),
                            "passed": bool(
                                class_match
                                and np.allclose(
                                    observed,
                                    expected,
                                    atol=tolerance,
                                    rtol=tolerance,
                                )
                            ),
                        }
                    )
                    if not np.allclose(
                        observed, expected, atol=tolerance, rtol=tolerance
                    ) or not class_match:
                        raise RuntimeError(
                            f"OOF prediction replay drift for {family} image {image_id}: "
                            f"max_abs={np.max(np.abs(observed - expected)):.3e}"
                        )
                    target = int(expected.argmax())
                    maps, seed_maps, view_similarity = candidate_maps(
                        ensemble,
                        inputs,
                        target,
                        integrated_gradient_steps=integrated_gradient_steps,
                    )
                    for method, attribution in maps.items():
                        metrics, curves = evaluate_attribution(
                            ensemble,
                            inputs,
                            attribution,
                            target,
                            image_id=image_id,
                            subset_count=subset_count,
                            device=device,
                            batch_size=batch_size,
                        )
                        detail_rows.append(
                            {
                                "family": family,
                                "method": method,
                                "eligible": method in ELIGIBLE[family],
                                "image_id": image_id,
                                "fold_id": int(fold_id),
                                "target_class": target,
                                "seed_agreement": mean_seed_agreement(seed_maps[method]),
                                "tta_view_agreement": (
                                    float(np.mean(view_similarity[method]))
                                    if view_similarity[method]
                                    else np.nan
                                ),
                                **metrics,
                            }
                        )
                        curve_rows.extend(
                            {
                                "stage": "oof_selection",
                                "family": family,
                                "method": method,
                                "image_id": image_id,
                                **row,
                            }
                            for row in curves
                        )
            finally:
                ensemble.close()

    detail = pd.DataFrame(detail_rows)
    pd.DataFrame(replay_rows).to_csv(
        output_dir / "oof_replay_validation.csv", index=False
    )
    detail.to_csv(output_dir / "oof_method_selection_per_image.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output_dir / "oof_method_selection_curves.csv", index=False)
    metric_columns = [
        "road_combined",
        "road_gap_auc",
        "deletion_auc",
        "insertion_auc",
        "random_deletion_auc",
        "selectivity_gap",
        "faithfulness_spearman",
        "seed_agreement",
        "tta_view_agreement",
    ]
    summary = (
        detail.groupby(["family", "method", "eligible"], as_index=False)[metric_columns]
        .agg(["mean", "std", "count"])
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary = summary.rename(
        columns={"family_": "family", "method_": "method", "eligible_": "eligible"}
    )
    if "family" not in summary.columns:
        summary = summary.rename(
            columns={"family": "family", "method": "method", "eligible": "eligible"}
        )

    selected = {}
    ranked_pieces = []
    for family in FAMILIES:
        family_rows = summary[
            summary["family"].eq(family) & summary["eligible"].astype(bool)
        ].copy()
        family_rows = family_rows.sort_values(
            [
                "road_combined_mean",
                "selectivity_gap_mean",
                "insertion_auc_mean",
                "method",
            ],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        family_rows["eligible_rank"] = np.arange(1, len(family_rows) + 1)
        winner = family_rows.iloc[0]
        selected[family] = {
            "method": str(winner["method"]),
            "oof_road_combined": float(winner["road_combined_mean"]),
            "oof_selectivity_gap": float(winner["selectivity_gap_mean"]),
            "oof_insertion_auc": float(winner["insertion_auc_mean"]),
        }
        ranked_pieces.append(family_rows)
    ranked = pd.concat(ranked_pieces, ignore_index=True)
    summary = summary.merge(
        ranked[["family", "method", "eligible_rank"]],
        on=["family", "method"],
        how="left",
    )
    summary["selected"] = summary.apply(
        lambda row: selected.get(row["family"], {}).get("method") == row["method"], axis=1
    )
    summary = summary.sort_values(
        ["family", "eligible", "eligible_rank", "method"],
        ascending=[True, False, True, True],
        na_position="last",
    )
    summary.to_csv(output_dir / "faithfulness_method_selection.csv", index=False)

    cohort_ids = cohort["image_id"].astype(str).tolist()
    lock = {
        "status": "LOCKED_FROM_OOF_BEFORE_FAITHFULNESS_TEST_EVALUATION",
        "test_used_for_selection": False,
        "selection_partition": "deterministic class-balanced OOF audit cohort",
        "selection_rows": int(len(cohort)),
        "selection_rows_per_class": int(per_class),
        "selection_image_ids_sha256": sha256_text(sorted(cohort_ids)),
        "primary_metric": "mean ROADCombined across 10%-90% removed patches",
        "tie_breakers": [
            "higher random-vs-MoRF selectivity gap",
            "higher insertion AUC",
            "method name",
        ],
        "ineligible_diagnostic_methods": {
            "dinov2_small": {
                "attention_rollout": "class-agnostic diagnostic baseline"
            }
        },
        "selected": selected,
        "perturbation_unit": "16x16 grid of 14x14 image patches",
        "road_fractions": ROAD_FRACTIONS.tolist(),
        "curve_fractions": CURVE_FRACTIONS.tolist(),
        "integrated_gradient_steps": int(integrated_gradient_steps),
        "faithfulness_random_subsets": int(subset_count),
    }
    lock_path = output_dir / "faithfulness_selection_lock.json"
    write_json(lock_path, lock)
    lock["lock_sha256"] = sha256_file(lock_path)
    return lock, summary


def load_test_probabilities(
    analysis_dir: Path,
) -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
    prediction_path = analysis_dir / "convnext_small_test_predictions.csv"
    predictions = pd.read_csv(prediction_path, dtype={"image_id": str})
    ids = predictions["image_id"].map(normalize_image_id).tolist()
    probabilities = {
        method: np.load(analysis_dir / f"{method}_test_probs.npy")
        for method in (*FAMILIES, "probability_blend")
    }
    labels = predictions["y_true"].to_numpy(dtype=int)
    for method, values in probabilities.items():
        if values.shape != (len(ids), 3):
            raise RuntimeError(f"Unexpected locked probability shape for {method}")
    return ids, probabilities, labels


def selected_map(
    ensemble: FamilyEnsemble,
    inputs: torch.Tensor,
    target_class: int,
    method: str,
    *,
    integrated_gradient_steps: int,
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    maps, seed_maps, view_similarity = candidate_maps(
        ensemble,
        inputs,
        target_class,
        integrated_gradient_steps=integrated_gradient_steps,
        requested={method},
    )
    return maps[method], seed_maps[method], view_similarity[method]


def evaluate_locked_test(
    *,
    final_root: Path,
    analysis_dir: Path,
    downstream_lock: dict,
    faithfulness_lock: dict,
    manifest: pd.DataFrame,
    output_dir: Path,
    integrated_gradient_steps: int,
    subset_count: int,
    device: torch.device,
    batch_size: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, FamilyEnsemble],
]:
    test_ids, expected_probabilities, expected_labels = load_test_probabilities(analysis_dir)
    manifest_by_id = manifest.set_index("image_id")
    missing = sorted(set(test_ids) - set(manifest_by_id.index.astype(str)))
    if missing:
        raise RuntimeError(f"Locked test IDs are absent from the manifest: {missing}")
    downstream_weight = float(downstream_lock["probability_blend"]["convnext_weight"])
    ensembles = {
        family: FamilyEnsemble(
            checkpoint_specs(final_root, downstream_lock, family, fold_id=None), device
        )
        for family in FAMILIES
    }
    blend = BlendEnsemble(
        ensembles["convnext_small"], ensembles["dinov2_small"], downstream_weight
    )
    selected = {
        family: str(faithfulness_lock["selected"][family]["method"])
        for family in FAMILIES
    }

    per_image_rows: list[dict] = []
    curve_rows: list[dict] = []
    maps_by_model: dict[str, dict[str, np.ndarray]] = {
        "convnext_small": {},
        "dinov2_small": {},
        "probability_blend": {},
    }
    input_cache: dict[str, torch.Tensor] = {}
    selected_cache: dict[tuple[str, str, int], tuple[np.ndarray, list[np.ndarray], list[float]]] = {}
    probability_replay: dict[str, list[np.ndarray]] = {method: [] for method in maps_by_model}

    for index, image_id in enumerate(test_ids):
        record = manifest_by_id.loc[str(image_id)]
        inputs, _ = preprocess_image(Path(record["resolved_image_path"]), device)
        input_cache[str(image_id)] = inputs.detach().cpu()
        print(f"[test] image {index + 1:02d}/{len(test_ids)} id={image_id}", flush=True)

        for family in FAMILIES:
            observed = ensembles[family].predict(inputs).detach().cpu().numpy()[0]
            expected = expected_probabilities[family][index]
            probability_replay[family].append(observed)
            tolerance = REPLAY_TOLERANCE[family]
            if not np.allclose(observed, expected, atol=tolerance, rtol=tolerance):
                raise RuntimeError(
                    f"Locked test replay drift for {family} image {image_id}: "
                    f"max_abs={np.max(np.abs(observed - expected)):.3e}"
                )
            target = int(expected.argmax())
            cache_key = (family, str(image_id), target)
            attribution, seed_maps, view_similarity = selected_map(
                ensembles[family],
                inputs,
                target,
                selected[family],
                integrated_gradient_steps=integrated_gradient_steps,
            )
            selected_cache[cache_key] = (attribution, seed_maps, view_similarity)
            maps_by_model[family][str(image_id)] = attribution
            metrics, curves = evaluate_attribution(
                ensembles[family],
                inputs,
                attribution,
                target,
                image_id=str(image_id),
                subset_count=subset_count,
                device=device,
                batch_size=batch_size,
            )
            per_image_rows.append(
                {
                    "model": family,
                    "method": selected[family],
                    "image_id": str(image_id),
                    "true_class": int(expected_labels[index]),
                    "target_class": target,
                    "correct": bool(target == int(expected_labels[index])),
                    "confidence": float(expected[target]),
                    "seed_agreement": mean_seed_agreement(seed_maps),
                    "tta_view_agreement": (
                        float(np.mean(view_similarity)) if view_similarity else np.nan
                    ),
                    **metrics,
                }
            )
            curve_rows.extend(
                {
                    "stage": "locked_test",
                    "model": family,
                    "method": selected[family],
                    "image_id": str(image_id),
                    **row,
                }
                for row in curves
            )

        observed_blend = blend.predict(inputs).detach().cpu().numpy()[0]
        expected_blend = expected_probabilities["probability_blend"][index]
        probability_replay["probability_blend"].append(observed_blend)
        blend_tolerance = REPLAY_TOLERANCE["probability_blend"]
        if not np.allclose(
            observed_blend,
            expected_blend,
            atol=blend_tolerance,
            rtol=blend_tolerance,
        ):
            raise RuntimeError(
                f"Locked blend replay drift for image {image_id}: "
                f"max_abs={np.max(np.abs(observed_blend - expected_blend)):.3e}"
            )
        blend_target = int(expected_blend.argmax())
        component_maps = {}
        for family in FAMILIES:
            cache_key = (family, str(image_id), blend_target)
            if cache_key not in selected_cache:
                selected_cache[cache_key] = selected_map(
                    ensembles[family],
                    inputs,
                    blend_target,
                    selected[family],
                    integrated_gradient_steps=integrated_gradient_steps,
                )
            component_maps[family] = selected_cache[cache_key][0]
        blend_map = normalize_attribution(
            downstream_weight * component_maps["convnext_small"]
            + (1.0 - downstream_weight) * component_maps["dinov2_small"]
        )
        maps_by_model["probability_blend"][str(image_id)] = blend_map
        metrics, curves = evaluate_attribution(
            blend,
            inputs,
            blend_map,
            blend_target,
            image_id=str(image_id),
            subset_count=subset_count,
            device=device,
            batch_size=batch_size,
        )
        per_image_rows.append(
            {
                "model": "probability_blend",
                "method": (
                    f"weighted_{selected['convnext_small']}+{selected['dinov2_small']}"
                ),
                "image_id": str(image_id),
                "true_class": int(expected_labels[index]),
                "target_class": blend_target,
                "correct": bool(blend_target == int(expected_labels[index])),
                "confidence": float(expected_blend[blend_target]),
                "seed_agreement": np.nan,
                "tta_view_agreement": np.nan,
                **metrics,
            }
        )
        curve_rows.extend(
            {
                "stage": "locked_test",
                "model": "probability_blend",
                "method": (
                    f"weighted_{selected['convnext_small']}+{selected['dinov2_small']}"
                ),
                "image_id": str(image_id),
                **row,
            }
            for row in curves
        )

    per_image = pd.DataFrame(per_image_rows)
    curves = pd.DataFrame(curve_rows)
    per_image.to_csv(output_dir / "faithfulness_test_per_image.csv", index=False)
    curves.to_csv(output_dir / "faithfulness_test_curves_per_image.csv", index=False)
    np.savez_compressed(
        output_dir / "faithfulness_test_maps.npz",
        **{
            f"{model}__{image_id}": attribution
            for model, values in maps_by_model.items()
            for image_id, attribution in values.items()
        },
    )
    np.savez_compressed(
        output_dir / "faithfulness_probability_replay.npz",
        **{
            model: np.asarray(values, dtype=np.float32)
            for model, values in probability_replay.items()
        },
    )
    replay_rows = []
    for model, values in probability_replay.items():
        replay = np.asarray(values, dtype=np.float64)
        reference = np.asarray(expected_probabilities[model], dtype=np.float64)
        absolute = np.abs(replay - reference)
        tolerance = REPLAY_TOLERANCE[model]
        replay_rows.append(
            {
                "model": model,
                "max_absolute_probability_difference": float(absolute.max()),
                "mean_absolute_probability_difference": float(absolute.mean()),
                "acceptance_tolerance": float(tolerance),
                "passed": bool(np.allclose(replay, reference, atol=tolerance, rtol=tolerance)),
            }
        )
    pd.DataFrame(replay_rows).to_csv(
        output_dir / "faithfulness_replay_validation.csv", index=False
    )
    return per_image, curves, maps_by_model, input_cache, ensembles


def stratified_bootstrap_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pieces = []
    for class_id in np.unique(labels):
        members = np.flatnonzero(labels == class_id)
        pieces.append(rng.choice(members, size=len(members), replace=True))
    return np.concatenate(pieces)


def summarize_test(per_image: pd.DataFrame, *, resamples: int) -> pd.DataFrame:
    metrics = [
        "road_combined",
        "road_gap_auc",
        "deletion_auc",
        "insertion_auc",
        "random_deletion_auc",
        "selectivity_gap",
        "faithfulness_spearman",
        "seed_agreement",
        "tta_view_agreement",
    ]
    rows = []
    for model, frame in per_image.groupby("model", sort=False):
        labels = frame["true_class"].to_numpy(dtype=int)
        rng = np.random.default_rng(stable_seed("bootstrap", model))
        bootstrap = {metric: [] for metric in metrics}
        for _ in range(int(resamples)):
            indices = stratified_bootstrap_indices(labels, rng)
            for metric in metrics:
                values = frame[metric].to_numpy(dtype=float)[indices]
                finite = values[np.isfinite(values)]
                bootstrap[metric].append(float(finite.mean()) if len(finite) else np.nan)
        row = {
            "model": model,
            "display_name": DISPLAY_NAMES[model],
            "method": str(frame.iloc[0]["method"]),
            "test_rows": int(len(frame)),
        }
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            boot = np.asarray(bootstrap[metric], dtype=float)
            boot = boot[np.isfinite(boot)]
            row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else np.nan
            row[f"{metric}_std"] = float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
            row[f"{metric}_ci_2_5"] = float(np.quantile(boot, 0.025)) if len(boot) else np.nan
            row[f"{metric}_ci_97_5"] = float(np.quantile(boot, 0.975)) if len(boot) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def reset_parameters(module: torch.nn.Module) -> None:
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


def randomize_ensemble(ensemble: FamilyEnsemble, condition: str) -> None:
    for model, spec in zip(ensemble.models, ensemble.specs, strict=True):
        seed_everything(stable_seed("randomization", ensemble.family, spec.seed, condition))
        reset_parameters(model.classifier)
        if condition == "head_plus_final_stage":
            if ensemble.family == "convnext_small":
                reset_parameters(model.backbone.features[-1])
            else:
                reset_parameters(model.backbone.encoder.layer[-1])
        elif condition != "head":
            raise ValueError(f"Unknown randomization condition: {condition}")
        model.eval()


def sanity_cohort(per_image: pd.DataFrame, per_class: int = 3) -> list[str]:
    base = per_image[per_image["model"].eq("probability_blend")].copy()
    base["key"] = base["image_id"].map(
        lambda value: hashlib.sha256(f"sanity|{value}".encode()).hexdigest()
    )
    pieces = [
        group.sort_values("key").head(per_class)
        for _, group in base.groupby("true_class", sort=True)
    ]
    return pd.concat(pieces)["image_id"].astype(str).tolist()


def run_sanity_and_stability(
    *,
    final_root: Path,
    downstream_lock: dict,
    faithfulness_lock: dict,
    per_image: pd.DataFrame,
    maps_by_model: dict[str, dict[str, np.ndarray]],
    input_cache: dict[str, torch.Tensor],
    ensembles: dict[str, FamilyEnsemble],
    expected_probabilities: dict[str, np.ndarray],
    test_ids: list[str],
    integrated_gradient_steps: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = {
        family: str(faithfulness_lock["selected"][family]["method"])
        for family in FAMILIES
    }
    cohort_ids = sanity_cohort(per_image)
    id_to_index = {image_id: index for index, image_id in enumerate(test_ids)}
    sanity_rows: list[dict] = []
    stability_rows: list[dict] = []

    for family in FAMILIES:
        for image_id in cohort_ids:
            index = id_to_index[image_id]
            inputs = input_cache[image_id].to(device)
            probabilities = expected_probabilities[family][index]
            target = int(probabilities.argmax())
            alternative = int(np.argsort(probabilities)[-2])
            original_map = maps_by_model[family][image_id]

            alternative_map, _, _ = selected_map(
                ensembles[family],
                inputs,
                alternative,
                selected[family],
                integrated_gradient_steps=integrated_gradient_steps,
            )
            flipped_map, _, _ = selected_map(
                ensembles[family],
                torch.flip(inputs, dims=(-1,)),
                target,
                selected[family],
                integrated_gradient_steps=integrated_gradient_steps,
            )
            flipped_map = np.flip(flipped_map, axis=1).copy()
            flipped_probability = float(
                ensembles[family]
                .predict(torch.flip(inputs, dims=(-1,)))[0, target]
                .detach()
                .cpu()
                .item()
            )
            stability_rows.extend(
                [
                    {
                        "family": family,
                        "method": selected[family],
                        "image_id": image_id,
                        "check": "target_class_change",
                        "spearman": attribution_spearman(
                            patch_scores(original_map), patch_scores(alternative_map)
                        ),
                        "probability_change": np.nan,
                    },
                    {
                        "family": family,
                        "method": selected[family],
                        "image_id": image_id,
                        "check": "horizontal_flip_equivariance",
                        "spearman": attribution_spearman(
                            patch_scores(original_map), patch_scores(flipped_map)
                        ),
                        "probability_change": float(
                            abs(flipped_probability - probabilities[target])
                        ),
                    },
                ]
            )

            if family == "dinov2_small":
                raw_target, _, _ = selected_map(
                    ensembles[family],
                    inputs,
                    target,
                    "attention_rollout",
                    integrated_gradient_steps=integrated_gradient_steps,
                )
                raw_alternative, _, _ = selected_map(
                    ensembles[family],
                    inputs,
                    alternative,
                    "attention_rollout",
                    integrated_gradient_steps=integrated_gradient_steps,
                )
                stability_rows.append(
                    {
                        "family": family,
                        "method": "attention_rollout",
                        "image_id": image_id,
                        "check": "target_class_change_diagnostic",
                        "spearman": attribution_spearman(
                            patch_scores(raw_target), patch_scores(raw_alternative)
                        ),
                        "probability_change": np.nan,
                    }
                )

        for condition in ("head", "head_plus_final_stage"):
            randomized = FamilyEnsemble(
                checkpoint_specs(final_root, downstream_lock, family, fold_id=None), device
            )
            randomize_ensemble(randomized, condition)
            try:
                for image_id in cohort_ids:
                    index = id_to_index[image_id]
                    target = int(expected_probabilities[family][index].argmax())
                    inputs = input_cache[image_id].to(device)
                    randomized_map, _, _ = selected_map(
                        randomized,
                        inputs,
                        target,
                        selected[family],
                        integrated_gradient_steps=integrated_gradient_steps,
                    )
                    sanity_rows.append(
                        {
                            "family": family,
                            "method": selected[family],
                            "image_id": image_id,
                            "condition": condition,
                            "spearman_with_trained_map": attribution_spearman(
                                patch_scores(maps_by_model[family][image_id]),
                                patch_scores(randomized_map),
                            ),
                        }
                    )
                    if family == "dinov2_small" and condition == "head":
                        raw_original, _, _ = selected_map(
                            ensembles[family],
                            inputs,
                            target,
                            "attention_rollout",
                            integrated_gradient_steps=integrated_gradient_steps,
                        )
                        raw_randomized, _, _ = selected_map(
                            randomized,
                            inputs,
                            target,
                            "attention_rollout",
                            integrated_gradient_steps=integrated_gradient_steps,
                        )
                        sanity_rows.append(
                            {
                                "family": family,
                                "method": "attention_rollout",
                                "image_id": image_id,
                                "condition": "head_diagnostic",
                                "spearman_with_trained_map": attribution_spearman(
                                    patch_scores(raw_original),
                                    patch_scores(raw_randomized),
                                ),
                            }
                        )
            finally:
                randomized.close()

    sanity = pd.DataFrame(sanity_rows)
    stability = pd.DataFrame(stability_rows)
    sanity.to_csv(output_dir / "faithfulness_sanity_checks.csv", index=False)
    stability.to_csv(output_dir / "faithfulness_stability_checks.csv", index=False)
    return sanity, stability


def aggregate_curves(curves: pd.DataFrame) -> pd.DataFrame:
    return (
        curves.groupby(["model", "method", "metric", "fraction"], as_index=False)[
            "target_probability"
        ]
        .agg(["mean", "std", "count"])
    )


def create_curve_plot(curves: pd.DataFrame, output_dir: Path) -> None:
    aggregated = aggregate_curves(curves)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
    styles = {
        "road_most": ("#DC2626", "ROAD most relevant"),
        "road_least": ("#2563EB", "ROAD least relevant"),
        "road_random": ("#64748B", "ROAD random"),
        "deletion": ("#EA580C", "Blur deletion"),
        "insertion": ("#16A34A", "Blur insertion"),
    }
    for axis, model in zip(axes, (*FAMILIES, "probability_blend"), strict=True):
        subset = aggregated[aggregated["model"].eq(model)]
        for metric, (color, label) in styles.items():
            rows = subset[subset["metric"].eq(metric)].sort_values("fraction")
            if rows.empty:
                continue
            axis.plot(
                rows["fraction"],
                rows["mean"],
                marker="o",
                linewidth=2,
                markersize=4,
                color=color,
                label=label,
            )
        axis.set_title(DISPLAY_NAMES[model])
        axis.set_xlabel("Fraction of 14×14 patches perturbed/revealed")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Predicted-class probability")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Locked-test attribution perturbation curves", y=1.02, fontsize=16)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(
            output_dir / f"faithfulness_perturbation_curves.{suffix}",
            dpi=190 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def create_selection_plot(selection: pd.DataFrame, output_dir: Path) -> None:
    frame = selection.copy()
    frame["label"] = frame["family"].map(DISPLAY_NAMES) + " — " + frame["method"]
    frame = frame.sort_values(["family", "road_combined_mean"], ascending=[True, True])
    colors = ["#0F766E" if value else "#94A3B8" for value in frame["selected"]]
    fig, axis = plt.subplots(figsize=(10, 5.8))
    axis.barh(frame["label"], frame["road_combined_mean"], color=colors)
    axis.axvline(0.0, color="#0F172A", linewidth=0.8)
    axis.set_xlabel("OOF ROADCombined (higher is better)")
    axis.set_ylabel("")
    axis.set_title("Attribution method selection before test evaluation")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(
            output_dir / f"faithfulness_method_selection.{suffix}",
            dpi=190 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def create_galleries(
    *,
    per_image: pd.DataFrame,
    maps_by_model: dict[str, dict[str, np.ndarray]],
    manifest: pd.DataFrame,
    label_map: dict[int, str],
    output_dir: Path,
) -> None:
    manifest_by_id = manifest.set_index("image_id")
    for model in (*FAMILIES, "probability_blend"):
        rows = per_image[per_image["model"].eq(model)].copy()
        correct = rows[rows["correct"]].sort_values("confidence", ascending=False).head(2)
        wrong = rows[~rows["correct"]].sort_values("confidence", ascending=False).head(2)
        chosen = pd.concat([correct, wrong])
        if len(chosen) < 4:
            extras = rows[~rows["image_id"].isin(chosen["image_id"])].sort_values(
                "confidence"
            )
            chosen = pd.concat([chosen, extras.head(4 - len(chosen))])
        fig, axes = plt.subplots(len(chosen), 2, figsize=(9, 4.1 * len(chosen)))
        if len(chosen) == 1:
            axes = np.asarray([axes])
        for row_index, record in enumerate(chosen.itertuples(index=False)):
            image_id = str(record.image_id)
            path = Path(manifest_by_id.loc[image_id]["resolved_image_path"])
            _, display = preprocess_image(path, torch.device("cpu"))
            attribution = maps_by_model[model][image_id]
            heatmap = attribution / max(float(attribution.max()), 1e-12)
            axes[row_index, 0].imshow(display)
            axes[row_index, 0].axis("off")
            axes[row_index, 0].set_title(
                f"true={label_map[int(record.true_class)]}  "
                f"pred={label_map[int(record.target_class)]}"
            )
            axes[row_index, 1].imshow(display)
            axes[row_index, 1].imshow(heatmap, cmap="turbo", alpha=0.48, vmin=0.0, vmax=1.0)
            axes[row_index, 1].axis("off")
            axes[row_index, 1].set_title(
                f"{record.method}  confidence={float(record.confidence):.3f}"
            )
        fig.suptitle(f"{DISPLAY_NAMES[model]} — locked-test attributions", fontsize=15)
        fig.tight_layout()
        stem = f"{model}_faithfulness_gallery"
        for suffix in ("png", "svg"):
            fig.savefig(
                output_dir / f"{stem}.{suffix}",
                dpi=180 if suffix == "png" else None,
                bbox_inches="tight",
            )
        plt.close(fig)


def checkpoint_manifest(
    final_root: Path, downstream_lock: dict, output_dir: Path
) -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        for fold_id in (*range(1, 6), None):
            for spec in checkpoint_specs(
                final_root, downstream_lock, family, fold_id=fold_id
            ):
                relative = spec.checkpoint.relative_to(final_root).as_posix()
                rows.append(
                    {
                        "family": family,
                        "seed": spec.seed,
                        "role": "locked_test" if fold_id is None else "oof_selection",
                        "fold_id": "" if fold_id is None else int(fold_id),
                        "artifact_relative_path": relative,
                        "checkpoint_sha256": sha256_file(spec.checkpoint),
                        "temperature": spec.temperature,
                        "evaluation_policy": spec.policy,
                    }
                )
    frame = pd.DataFrame(rows).drop_duplicates(
        ["family", "seed", "role", "fold_id", "artifact_relative_path"]
    )
    frame.to_csv(output_dir / "faithfulness_checkpoint_manifest.csv", index=False)
    return frame


def summarize_checks(
    sanity: pd.DataFrame, stability: pd.DataFrame, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sanity_summary = (
        sanity.groupby(["family", "method", "condition"], as_index=False)[
            "spearman_with_trained_map"
        ]
        .agg(["mean", "std", "count"])
    )
    stability_summary = (
        stability.groupby(["family", "method", "check"], as_index=False)[
            ["spearman", "probability_change"]
        ]
        .agg(["mean", "std", "count"])
    )
    if isinstance(stability_summary.columns, pd.MultiIndex):
        stability_summary.columns = [
            "_".join(str(part) for part in column if part).rstrip("_")
            for column in stability_summary.columns
        ]
    sanity_summary.to_csv(output_dir / "faithfulness_sanity_summary.csv", index=False)
    stability_summary.to_csv(output_dir / "faithfulness_stability_summary.csv", index=False)
    return sanity_summary, stability_summary


def build_provenance(
    *,
    args: argparse.Namespace,
    manifest_path: Path,
    final_root: Path,
    analysis_dir: Path,
    output_dir: Path,
    protocol,
    faithfulness_lock: dict,
    checkpoint_rows: pd.DataFrame,
    test_ids: list[str],
) -> dict:
    evidence_files = [
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".png", ".svg"}
        and path.name != "faithfulness_provenance.json"
    ]
    return {
        "status": "LOCKED_TEST_EVALUATED_AFTER_OOF_ATTRIBUTION_SELECTION",
        "runtime": {
            "runner": Path(__file__).name,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(args.device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "inputs": {
            "manifest_sha256": sha256_file(manifest_path),
            "dataset_manifest_sha256": protocol.manifest_sha256,
            "fixed_test_image_ids_sha256": protocol.test_image_ids_sha256,
            "model_selection_lock_sha256": sha256_file(
                final_root / "configs" / "selection_lock.json"
            ),
            "downstream_selection_lock_sha256": sha256_file(
                analysis_dir / "downstream_selection_lock.json"
            ),
            "faithfulness_selection_lock_sha256": faithfulness_lock["lock_sha256"],
            "checkpoint_manifest_sha256": sha256_file(
                output_dir / "faithfulness_checkpoint_manifest.csv"
            ),
            "checkpoint_count": int(len(checkpoint_rows)),
        },
        "protocol": {
            "test_used_for_attribution_selection": False,
            "test_rows": int(len(test_ids)),
            "test_image_ids_sha256": sha256_text(sorted(test_ids)),
            "target": "locked predicted class probability",
            "patch_grid": [PATCH_GRID_SIZE, PATCH_GRID_SIZE],
            "patch_pixels": [14, 14],
            "road_imputation": "ROAD noisy linear imputer, deterministic 0.01 noise",
            "road_fractions": ROAD_FRACTIONS.tolist(),
            "deletion_insertion_fractions": CURVE_FRACTIONS.tolist(),
            "integrated_gradient_steps": int(args.integrated_gradient_steps),
            "faithfulness_random_subsets": int(args.faithfulness_subsets),
            "bootstrap_resamples": int(args.bootstrap_resamples),
            "blend_attribution": (
                "unit-mass component maps combined with the locked 0.1/0.9 probability weights"
            ),
        },
        "evidence": {
            path.name: sha256_file(path) for path in sorted(evidence_files)
        },
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    final_root = args.final_root.resolve()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(20260822)

    manifest, protocol = load_and_validate_manifest(manifest_path, require_images=True)
    manifest["image_id"] = manifest["image_id"].astype(str)
    downstream_lock = read_json(analysis_dir / "downstream_selection_lock.json")
    if downstream_lock.get("status") != "LOCKED_FROM_OOF_BEFORE_DOWNSTREAM_TEST_EVALUATION":
        raise RuntimeError("The downstream model policy is not locked")

    print("[stage] OOF attribution method selection", flush=True)
    faithfulness_lock, selection_summary = select_methods(
        final_root=final_root,
        downstream_lock=downstream_lock,
        manifest=manifest,
        output_dir=output_dir,
        per_class=int(args.selection_per_class),
        integrated_gradient_steps=int(args.integrated_gradient_steps),
        subset_count=int(args.faithfulness_subsets),
        device=device,
        batch_size=int(args.inference_batch_size),
    )
    print(
        "[locked] "
        + ", ".join(
            f"{family}={row['method']}"
            for family, row in faithfulness_lock["selected"].items()
        ),
        flush=True,
    )

    print("[stage] locked-test faithfulness evaluation", flush=True)
    per_image, curves, maps_by_model, input_cache, ensembles = evaluate_locked_test(
        final_root=final_root,
        analysis_dir=analysis_dir,
        downstream_lock=downstream_lock,
        faithfulness_lock=faithfulness_lock,
        manifest=manifest,
        output_dir=output_dir,
        integrated_gradient_steps=int(args.integrated_gradient_steps),
        subset_count=int(args.faithfulness_subsets),
        device=device,
        batch_size=int(args.inference_batch_size),
    )
    summary = summarize_test(per_image, resamples=int(args.bootstrap_resamples))
    summary.to_csv(output_dir / "faithfulness_test_summary.csv", index=False)
    aggregate_curves(curves).to_csv(output_dir / "faithfulness_test_curves.csv", index=False)

    test_ids, expected_probabilities, _ = load_test_probabilities(analysis_dir)
    print("[stage] parameter randomization and stability checks", flush=True)
    sanity, stability = run_sanity_and_stability(
        final_root=final_root,
        downstream_lock=downstream_lock,
        faithfulness_lock=faithfulness_lock,
        per_image=per_image,
        maps_by_model=maps_by_model,
        input_cache=input_cache,
        ensembles=ensembles,
        expected_probabilities=expected_probabilities,
        test_ids=test_ids,
        integrated_gradient_steps=int(args.integrated_gradient_steps),
        device=device,
        output_dir=output_dir,
    )
    summarize_checks(sanity, stability, output_dir)

    label_map_raw = read_json(final_root / "configs" / "label_map.json")
    label_map = {int(key): str(value) for key, value in label_map_raw.items()}
    create_selection_plot(selection_summary, output_dir)
    create_curve_plot(curves, output_dir)
    create_galleries(
        per_image=per_image,
        maps_by_model=maps_by_model,
        manifest=manifest,
        label_map=label_map,
        output_dir=output_dir,
    )

    for ensemble in ensembles.values():
        ensemble.close()
    print("[stage] artifact fingerprints", flush=True)
    checkpoint_rows = checkpoint_manifest(final_root, downstream_lock, output_dir)
    provenance = build_provenance(
        args=args,
        manifest_path=manifest_path,
        final_root=final_root,
        analysis_dir=analysis_dir,
        output_dir=output_dir,
        protocol=protocol,
        faithfulness_lock=faithfulness_lock,
        checkpoint_rows=checkpoint_rows,
        test_ids=test_ids,
    )
    write_json(output_dir / "faithfulness_provenance.json", provenance)
    print(summary.to_string(index=False), flush=True)
    print(f"[done] faithfulness evidence: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
