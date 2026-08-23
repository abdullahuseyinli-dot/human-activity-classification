"""Evaluate locked neural POLAR components with bbox-aware faithfulness checks."""

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

from hac.augmentations import build_eval_transform
from hac.explainability import (
    attribution_spearman,
    conv_cam_attributions,
    curve_auc,
    gaussian_baseline,
    integrated_gradients_attribution,
    normalize_attribution,
    patch_keep_mask,
    patch_scores,
)
from hac.polar import image_view, sha256_file
from hac.polar_faithfulness import (
    area_matched_occlusion_masks,
    attribution_localization,
    box_mask,
    projected_person_box,
    select_bbox_stratified_cohort,
    stable_seed,
)
from hac.polar_models import build_polar_model
from hac.polar_training import TASK_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-evaluation-dir", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
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


def load_tensor(row: pd.Series, view: str, transform) -> torch.Tensor:
    with Image.open(row["image_path"]) as image:
        source = image.convert("RGB")
        return transform(image_view(source, row, view))


def predict_probabilities(model, batch: torch.Tensor) -> np.ndarray:
    with (
        torch.inference_mode(),
        torch.autocast(device_type=batch.device.type, enabled=batch.device.type == "cuda"),
    ):
        logits = model(batch)
    return torch.softmax(logits.float(), dim=-1).cpu().numpy()


def model_attribution(
    model,
    family: str,
    inputs: torch.Tensor,
    target_class: int,
    *,
    steps: int,
) -> np.ndarray:
    if family == "convnext_small_full":
        return conv_cam_attributions(
            model,
            inputs,
            target_class,
            temperature=1.0,
            policy="center_crop",
        ).gradcam
    baseline = gaussian_baseline(inputs)
    return integrated_gradients_attribution(
        model,
        inputs,
        baseline,
        target_class,
        temperature=1.0,
        policy="center_crop",
        steps=steps,
        internal_batch_size=4,
    )


def perturbation_batch(
    inputs: torch.Tensor,
    attribution: np.ndarray,
    fractions: np.ndarray,
    *,
    image_id: str,
    grid_size: int,
) -> tuple[torch.Tensor, list[tuple[str, float]]]:
    baseline = gaussian_baseline(inputs)
    scores = patch_scores(attribution, grid_size)
    samples = []
    keys = []
    for fraction in fractions:
        most_keep = patch_keep_mask(
            scores,
            float(fraction),
            order="most",
            output_size=tuple(inputs.shape[-2:]),
        ).to(inputs.device)
        random_keep = patch_keep_mask(
            scores,
            float(fraction),
            order="random",
            # Reuse one permutation so larger fractions strictly extend the
            # same random-deletion path instead of sampling unrelated masks.
            random_seed=stable_seed("random-deletion", image_id),
            output_size=tuple(inputs.shape[-2:]),
        ).to(inputs.device)
        most_keep = most_keep[None, None]
        random_keep = random_keep[None, None]
        deleted = inputs * most_keep + baseline * (1.0 - most_keep)
        inserted = inputs * (1.0 - most_keep) + baseline * most_keep
        random_deleted = inputs * random_keep + baseline * (1.0 - random_keep)
        for name, sample in (
            ("deletion", deleted),
            ("insertion", inserted),
            ("random_deletion", random_deleted),
        ):
            samples.append(sample[0])
            keys.append((name, float(fraction)))
    return torch.stack(samples), keys


def occluded_input(
    inputs: torch.Tensor, baseline: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    values = mask.to(inputs.device, dtype=inputs.dtype)[None, None]
    return inputs * (1.0 - values) + baseline * values


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    left = np.clip(np.asarray(left, dtype=np.float64), 1e-12, 1.0)
    right = np.clip(np.asarray(right, dtype=np.float64), 1e-12, 1.0)
    middle = 0.5 * (left + right)
    return float(
        0.5 * np.sum(left * np.log(left / middle)) + 0.5 * np.sum(right * np.log(right / middle))
    )


def stratified_bootstrap_mean(
    frame: pd.DataFrame,
    column: str,
    *,
    strata: tuple[str, ...],
    resamples: int,
    seed: int,
) -> dict:
    if column not in frame or any(name not in frame for name in strata):
        raise ValueError("Bootstrap column or stratum is missing")
    values = frame[column].to_numpy(dtype=float)
    finite = np.isfinite(values)
    excluded_rows = int((~finite).sum())
    frame = frame.loc[finite]
    array = frame[column].to_numpy(dtype=float)
    if len(array) < 1:
        raise ValueError("Bootstrap values must contain at least one finite observation")
    generator = np.random.default_rng(seed)
    draws = np.zeros(resamples, dtype=float)
    for _, group in frame.groupby(list(strata), sort=True, observed=True):
        values = group[column].to_numpy(dtype=float)
        weight = len(values) / len(frame)
        draws += weight * generator.choice(
            values, size=(resamples, len(values)), replace=True
        ).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "rows": len(array),
        "excluded_rows": excluded_rows,
    }


def classifier_head(model, family: str):
    return model.backbone.classifier[2] if family == "convnext_small_full" else model.classifier


def reset_leaf_modules(module: torch.nn.Module) -> None:
    for child in module.modules():
        if any(child.children()):
            continue
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


def randomized_head(model, family: str, seed: int) -> None:
    head = model.backbone.classifier[2] if family == "convnext_small_full" else model.classifier
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    head.reset_parameters()


def randomized_adapted_cascade(model, family: str, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    reset_leaf_modules(classifier_head(model, family))
    if family == "convnext_small_full":
        adapted = model.backbone.features[-1]
    else:
        adapted = model.backbone.encoder.layer[-4:]
    reset_leaf_modules(adapted)


def evaluate_family(
    family: str,
    specification: dict,
    resolved_runs: dict[int, dict],
    cohort: pd.DataFrame,
    locked_probabilities: np.ndarray,
    faithfulness: dict,
    device: torch.device,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fractions = np.asarray(faithfulness["deletion_insertion_fractions"], dtype=float)
    grid_size = int(faithfulness["perturbation_grid"][0])
    steps = int(faithfulness["integrated_gradient_steps"])
    transform = build_eval_transform()
    row_count = len(cohort)
    shape = (row_count, 224, 224)
    attribution_sum = np.zeros(shape, dtype=np.float32)
    alternative_sum = np.zeros(shape, dtype=np.float32)
    curve_sum = {
        name: np.zeros((row_count, len(fractions)), dtype=np.float64)
        for name in ("deletion", "insertion", "random_deletion")
    }
    original_sum = np.zeros(row_count, dtype=np.float64)
    person_occluded_sum = np.zeros(row_count, dtype=np.float64)
    context_occluded_sum = np.zeros(row_count, dtype=np.float64)
    full_sum = np.zeros((row_count, 4), dtype=np.float64)
    crop_sum = np.zeros((row_count, 4), dtype=np.float64)
    match_fraction = np.full(row_count, np.nan, dtype=np.float64)
    matched_occlusion_available = np.ones(row_count, dtype=bool)
    sanity_indices = np.concatenate(
        [group.index.to_numpy()[:4] for _, group in cohort.groupby("label_4", sort=True)]
    )
    randomized_sum = np.zeros((len(sanity_indices), 224, 224), dtype=np.float32)
    cascade_randomized_sum = np.zeros((len(sanity_indices), 224, 224), dtype=np.float32)
    targets = np.argmax(locked_probabilities, axis=1)
    alternatives = np.argsort(locked_probabilities, axis=1)[:, -2]
    class_names = list(TASK_LABELS["label_4"])
    configuration = specification["configuration"]

    for seed_index, seed in enumerate(specification["seeds"], start=1):
        model = build_polar_model(
            model_config_from_final(configuration), num_classes=len(class_names)
        ).to(device)
        payload = torch.load(
            resolved_runs[int(seed)]["checkpoint"], map_location=device, weights_only=False
        )
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        for position, row in cohort.iterrows():
            selected = load_tensor(row, configuration["view"], transform)[None].to(device)
            attribution = model_attribution(
                model, family, selected, int(targets[position]), steps=steps
            )
            alternative = model_attribution(
                model, family, selected, int(alternatives[position]), steps=steps
            )
            attribution_sum[position] += attribution
            alternative_sum[position] += alternative

            person_box = projected_person_box(row, configuration["view"])
            person_mask = box_mask(person_box)
            try:
                person_match, context_match, fraction = area_matched_occlusion_masks(
                    person_mask,
                    seed=stable_seed("matched-context", family, row["image_id"]),
                )
                match_fraction[position] = fraction
            except ValueError as error:
                if str(error) != "Both person and context regions must contain pixels":
                    raise
                # A full-frame projected box has no context comparison. Keep the
                # sample for every other metric and mark only this contrast absent.
                matched_occlusion_available[position] = False
                person_match = torch.zeros_like(person_mask)
                context_match = torch.zeros_like(person_mask)
            baseline = gaussian_baseline(selected)
            person_occluded = occluded_input(selected, baseline, person_match)
            context_occluded = occluded_input(selected, baseline, context_match)
            full = load_tensor(row, "full_frame", transform).to(device)
            crop = load_tensor(row, "person_context_25", transform).to(device)
            perturbations, keys = perturbation_batch(
                selected,
                attribution,
                fractions,
                image_id=str(row["image_id"]),
                grid_size=grid_size,
            )
            inference = torch.cat(
                [
                    selected,
                    full[None],
                    crop[None],
                    person_occluded,
                    context_occluded,
                    perturbations,
                ]
            )
            probabilities = predict_probabilities(model, inference)
            target = int(targets[position])
            original_sum[position] += probabilities[0, target]
            full_sum[position] += probabilities[1]
            crop_sum[position] += probabilities[2]
            person_occluded_sum[position] += probabilities[3, target]
            context_occluded_sum[position] += probabilities[4, target]
            for prediction, (curve_name, fraction) in zip(probabilities[5:], keys, strict=True):
                fraction_index = int(np.flatnonzero(np.isclose(fractions, fraction))[0])
                curve_sum[curve_name][position, fraction_index] += prediction[target]
            if (position + 1) % 16 == 0 or position + 1 == row_count:
                print(
                    f"[{family} seed={seed}] {position + 1}/{row_count} attributions",
                    flush=True,
                )

        randomized_head(model, family, stable_seed("head-randomization", family, seed))
        model.eval()
        for sanity_position, cohort_position in enumerate(sanity_indices):
            row = cohort.iloc[int(cohort_position)]
            selected = load_tensor(row, configuration["view"], transform)[None].to(device)
            randomized_sum[sanity_position] += model_attribution(
                model,
                family,
                selected,
                int(targets[int(cohort_position)]),
                steps=steps,
            )
        randomized_adapted_cascade(
            model, family, stable_seed("adapted-cascade-randomization", family, seed)
        )
        model.eval()
        for sanity_position, cohort_position in enumerate(sanity_indices):
            row = cohort.iloc[int(cohort_position)]
            selected = load_tensor(row, configuration["view"], transform)[None].to(device)
            cascade_randomized_sum[sanity_position] += model_attribution(
                model,
                family,
                selected,
                int(targets[int(cohort_position)]),
                steps=steps,
            )
        del model, payload
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[{family}] completed seed {seed} ({seed_index}/{len(specification['seeds'])})",
            flush=True,
        )

    seed_count = len(specification["seeds"])
    attribution_maps = np.stack(
        [normalize_attribution(values / seed_count) for values in attribution_sum]
    )
    alternative_maps = np.stack(
        [normalize_attribution(values / seed_count) for values in alternative_sum]
    )
    randomized_maps = np.stack(
        [normalize_attribution(values / seed_count) for values in randomized_sum]
    )
    cascade_randomized_maps = np.stack(
        [normalize_attribution(values / seed_count) for values in cascade_randomized_sum]
    )
    original = original_sum / seed_count
    person_occluded = person_occluded_sum / seed_count
    context_occluded = context_occluded_sum / seed_count
    full = full_sum / seed_count
    crop = crop_sum / seed_count
    curves = {name: values / seed_count for name, values in curve_sum.items()}

    per_image = []
    curve_rows = []
    for position, row in cohort.iterrows():
        target = int(targets[position])
        person_mask = box_mask(projected_person_box(row, configuration["view"]))
        localization = attribution_localization(attribution_maps[position], person_mask)
        target_sensitivity = attribution_spearman(
            patch_scores(attribution_maps[position], grid_size),
            patch_scores(alternative_maps[position], grid_size),
        )
        curve_metrics = {
            f"{name}_auc": curve_auc(fractions, values[position]) for name, values in curves.items()
        }
        matched_available = bool(matched_occlusion_available[position])
        record = {
            "family": family,
            "image_id": str(row["image_id"]),
            "true_label": str(row["label_4"]),
            "target_class": class_names[target],
            "target_correct": class_names[target] == str(row["label_4"]),
            "bbox_area_quartile": str(row["bbox_area_quartile"]),
            "bbox_area_fraction_source": float(row["bbox_area_fraction"]),
            **localization,
            **curve_metrics,
            "deletion_selectivity_gap": (
                curve_metrics["random_deletion_auc"] - curve_metrics["deletion_auc"]
            ),
            "locked_target_probability": float(locked_probabilities[position, target]),
            "recomputed_target_probability": float(original[position]),
            "probability_parity_absolute_error": float(
                abs(locked_probabilities[position, target] - original[position])
            ),
            "matched_occlusion_available": matched_available,
            "person_occlusion_drop": (
                float(original[position] - person_occluded[position])
                if matched_available
                else np.nan
            ),
            "matched_context_occlusion_drop": (
                float(original[position] - context_occluded[position])
                if matched_available
                else np.nan
            ),
            "person_minus_context_occlusion_drop": (
                float(context_occluded[position] - person_occluded[position])
                if matched_available
                else np.nan
            ),
            "matched_person_fraction": (
                float(match_fraction[position]) if matched_available else np.nan
            ),
            "full_crop_js_divergence": jensen_shannon(full[position], crop[position]),
            "full_crop_prediction_agreement": bool(
                np.argmax(full[position]) == np.argmax(crop[position])
            ),
            "full_crop_target_probability_difference": float(
                abs(full[position, target] - crop[position, target])
            ),
            "target_vs_alternative_attribution_spearman": target_sensitivity,
        }
        per_image.append(record)
        for name, values in curves.items():
            curve_rows.extend(
                {
                    "family": family,
                    "image_id": str(row["image_id"]),
                    "curve": name,
                    "fraction": float(fraction),
                    "target_probability": float(probability),
                }
                for fraction, probability in zip(fractions, values[position], strict=True)
            )

    sanity_rows = []
    for sanity_position, cohort_position in enumerate(sanity_indices):
        row = cohort.iloc[int(cohort_position)]
        sanity_rows.append(
            {
                "family": family,
                "image_id": str(row["image_id"]),
                "true_label": str(row["label_4"]),
                "trained_vs_randomized_head_spearman": attribution_spearman(
                    patch_scores(attribution_maps[int(cohort_position)], grid_size),
                    patch_scores(randomized_maps[sanity_position], grid_size),
                ),
                "trained_vs_randomized_adapted_cascade_spearman": attribution_spearman(
                    patch_scores(attribution_maps[int(cohort_position)], grid_size),
                    patch_scores(cascade_randomized_maps[sanity_position], grid_size),
                ),
            }
        )

    np.savez_compressed(
        output_dir / f"{family}_attribution_maps.npz",
        image_ids=cohort["image_id"].astype(str).to_numpy(dtype=str),
        attributions=attribution_maps,
        alternative_target_attributions=alternative_maps,
        randomized_head_attributions=randomized_maps,
        randomized_adapted_cascade_attributions=cascade_randomized_maps,
        randomized_head_image_ids=cohort.iloc[sanity_indices]["image_id"]
        .astype(str)
        .to_numpy(dtype=str),
    )
    return pd.DataFrame(per_image), pd.DataFrame(curve_rows), pd.DataFrame(sanity_rows)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    lock_path = args.selection_lock.resolve()
    lock_hash = sha256_file(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "FINAL_SELECTION_LOCKED_PRE_TEST":
        raise RuntimeError("Faithfulness evaluation requires the immutable final lock")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    implementation_hash = sha256_file(Path(__file__).resolve())
    summary_path = output_dir / "summary.json"
    evaluation_dir = args.test_evaluation_dir.resolve()
    evaluation_summary_path = evaluation_dir / "summary.json"
    evaluation_summary = json.loads(evaluation_summary_path.read_text(encoding="utf-8"))
    if (
        evaluation_summary.get("status") != "LOCKED_FINAL_TEST_COMPLETE"
        or evaluation_summary.get("selection_lock_sha256") != lock_hash
    ):
        raise RuntimeError("Faithfulness evaluation requires the completed locked test run")
    opened_manifest = evaluation_dir / "opened_test_manifest.csv"
    gate = json.loads((evaluation_dir / "test_access_gate.json").read_text(encoding="utf-8"))
    if sha256_file(opened_manifest) != gate["opened_manifest_cache_sha256"]:
        raise RuntimeError("Opened test-manifest cache hash drift")
    prediction_path = evaluation_dir / "test_predictions.npz"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        artifacts = existing.get("artifacts", {})
        artifacts_valid = bool(artifacts) and all(
            Path(name).name == name
            and (output_dir / name).is_file()
            and sha256_file(output_dir / name) == digest
            for name, digest in artifacts.items()
        )
        if (
            existing.get("status") == "LOCKED_POLAR_FAITHFULNESS_COMPLETE"
            and existing.get("selection_lock_sha256") == lock_hash
            and existing.get("implementation_sha256") == implementation_hash
            and existing.get("test_evaluation_summary_sha256")
            == sha256_file(evaluation_summary_path)
            and existing.get("test_predictions_sha256") == sha256_file(prediction_path)
            and existing.get("opened_test_manifest_cache_sha256")
            == sha256_file(opened_manifest)
            and artifacts_valid
        ):
            print(json.dumps(existing, indent=2, sort_keys=True), flush=True)
            return
    frame = pd.read_csv(opened_manifest, dtype={"image_id": str})
    faithfulness = lock["faithfulness"]
    cohort = select_bbox_stratified_cohort(
        frame,
        rows=int(faithfulness["cohort_rows"]),
        seed=int(faithfulness["cohort_seed"]),
    )
    cohort_path = output_dir / "faithfulness_cohort.csv"
    cohort.to_csv(cohort_path, index=False)

    final_root = args.final_root.resolve()
    resolved = verify_final_fits(lock, final_root, lock_hash)
    predictions = np.load(prediction_path, allow_pickle=False)
    # The opened manifest is the authoritative row order. Early locked archives
    # encoded image_ids as an object array, so do not deserialize that field.
    # Numeric labels provide an independent alignment check before attribution.
    prediction_ids = frame["image_id"].astype(str).tolist()
    expected_labels = frame["label_4"].map(
        {name: index for index, name in enumerate(TASK_LABELS["label_4"])}
    )
    if expected_labels.isna().any() or not np.array_equal(
        predictions["labels_4"], expected_labels.to_numpy(dtype=int)
    ):
        raise RuntimeError("Locked predictions do not align with the opened test manifest")
    index_by_id = {image_id: index for index, image_id in enumerate(prediction_ids)}
    order = np.asarray([index_by_id[value] for value in cohort["image_id"].astype(str)])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    per_image_frames = []
    curve_frames = []
    sanity_frames = []
    for family in faithfulness["families"]:
        specification = lock["final_neural_fits"][family]
        locked_probabilities = np.asarray(
            predictions[f"probabilities_{family}"][order], dtype=np.float64
        )
        per_image, curves, sanity = evaluate_family(
            family,
            specification,
            resolved["neural"][family],
            cohort,
            locked_probabilities,
            faithfulness,
            device,
            output_dir,
        )
        per_image_frames.append(per_image)
        curve_frames.append(curves)
        sanity_frames.append(sanity)

    per_image = pd.concat(per_image_frames, ignore_index=True)
    curves = pd.concat(curve_frames, ignore_index=True)
    sanity = pd.concat(sanity_frames, ignore_index=True)
    per_image.to_csv(output_dir / "faithfulness_per_image.csv", index=False)
    curves.to_csv(output_dir / "faithfulness_curves.csv", index=False)
    sanity.to_csv(output_dir / "faithfulness_randomization.csv", index=False)

    metric_columns = [
        "deletion_auc",
        "insertion_auc",
        "random_deletion_auc",
        "deletion_selectivity_gap",
        "person_attribution_mass",
        "person_attribution_mass_lift",
        "person_minus_context_occlusion_drop",
        "full_crop_js_divergence",
        "target_vs_alternative_attribution_spearman",
    ]
    aggregate = {}
    for family, family_frame in per_image.groupby("family", sort=True):
        aggregate[family] = {
            column: stratified_bootstrap_mean(
                family_frame,
                column,
                strata=("true_label", "bbox_area_quartile"),
                resamples=10_000,
                seed=stable_seed("faithfulness-bootstrap", family, column),
            )
            for column in metric_columns
        }
        aggregate[family]["pointing_game_rate"] = stratified_bootstrap_mean(
            family_frame,
            "pointing_game",
            strata=("true_label", "bbox_area_quartile"),
            resamples=10_000,
            seed=stable_seed("faithfulness-bootstrap", family, "pointing-game"),
        )
        aggregate[family]["full_crop_agreement_rate"] = stratified_bootstrap_mean(
            family_frame,
            "full_crop_prediction_agreement",
            strata=("true_label", "bbox_area_quartile"),
            resamples=10_000,
            seed=stable_seed("faithfulness-bootstrap", family, "full-crop-agreement"),
        )
        family_sanity = sanity[sanity["family"].eq(family)]
        aggregate[family]["randomized_head_spearman"] = stratified_bootstrap_mean(
            family_sanity,
            "trained_vs_randomized_head_spearman",
            strata=("true_label",),
            resamples=10_000,
            seed=stable_seed("faithfulness-bootstrap", family, "randomized-head"),
        )
        aggregate[family]["randomized_adapted_cascade_spearman"] = (
            stratified_bootstrap_mean(
                family_sanity,
                "trained_vs_randomized_adapted_cascade_spearman",
                strata=("true_label",),
                resamples=10_000,
                seed=stable_seed("faithfulness-bootstrap", family, "randomized-cascade"),
            )
        )

    stratum_summary = (
        per_image.groupby(["family", "true_label", "bbox_area_quartile"], as_index=False)[
            [
                "person_attribution_mass",
                "person_attribution_mass_lift",
                "person_minus_context_occlusion_drop",
                "deletion_selectivity_gap",
            ]
        ]
        .mean()
        .sort_values(["family", "true_label", "bbox_area_quartile"], ignore_index=True)
    )
    stratum_summary.to_csv(output_dir / "faithfulness_strata.csv", index=False)
    summary = {
        "status": "LOCKED_POLAR_FAITHFULNESS_COMPLETE",
        "selection_role": "none",
        "selection_lock_sha256": lock_hash,
        "test_evaluation_summary_sha256": sha256_file(evaluation_summary_path),
        "test_predictions_sha256": sha256_file(prediction_path),
        "opened_test_manifest_cache_sha256": sha256_file(opened_manifest),
        "cohort_rows": len(cohort),
        "cohort_sha256": sha256_file(cohort_path),
        "protocol": faithfulness,
        "parameter_randomization_scopes": {
            "convnext_small_full": [
                "classifier_head",
                "classifier_head_plus_last_convnext_stage",
            ],
            "dinov2_base_top4": [
                "classifier_head",
                "classifier_head_plus_top_four_transformer_blocks",
            ],
        },
        "parameter_randomization_rows_per_family": int(
            sanity.groupby("family", sort=True).size().min()
        ),
        "matched_occlusion_unavailable_rows_per_family": {
            str(family): int((~group["matched_occlusion_available"]).sum())
            for family, group in per_image.groupby("family", sort=True)
        },
        "bootstrap": {
            "resamples": 10_000,
            "per_image_strata": ["true_label", "bbox_area_quartile"],
            "randomization_strata": ["true_label"],
            "seed_policy": "sha256_stable_metric_specific",
        },
        "aggregate": aggregate,
        "max_probability_parity_absolute_error": float(
            per_image["probability_parity_absolute_error"].max()
        ),
        "artifacts": {
            path.name: sha256_file(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "summary.json"
        },
        "implementation_sha256": implementation_hash,
        "runtime_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "test_rows_read": len(cohort),
        "test_used_for_attribution_selection": False,
        "test_used_for_model_selection": False,
    }
    write_json(summary_path, summary)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
