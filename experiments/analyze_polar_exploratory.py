"""Run reproducible post-hoc analyses over the completed locked POLAR study.

The analyses in this module are descriptive and hypothesis-generating. They do not
reopen model selection, change the locked primary result, or convert test-set findings
into confirmatory evidence. Dense prediction files and clean local manifests are read
from ``.runs``; only aggregate, portable outputs are written to ``results`` and
``assets``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SEED = 20260823
CLASS_NAMES_4 = ("sitting", "standing", "walking", "running")
CLASS_NAMES_3 = ("sitting", "standing", "walking_running")
COMPONENTS = (
    "convnext_small_full",
    "dinov2_small_moderate",
    "dinov2_base_top4",
    "dinov2_base_multilayer_logistic",
    "dinov2_base_multilayer_rbf",
)
LOCKED_WEIGHTS = {
    "convnext_small_full": 0.20,
    "dinov2_small_moderate": 0.15,
    "dinov2_base_top4": 0.25,
    "dinov2_base_multilayer_logistic": 0.20,
    "dinov2_base_multilayer_rbf": 0.20,
}
DISPLAY = {
    "convnext_small_full": "ConvNeXt-S",
    "dinov2_small_moderate": "DINOv2-S",
    "dinov2_base_top4": "DINOv2-B adapted",
    "dinov2_base_multilayer_logistic": "DINOv2-B logistic",
    "dinov2_base_multilayer_rbf": "DINOv2-B RBF",
    "locked_ensemble": "Locked ensemble",
    "locked_ensemble_collapsed": "Locked ensemble",
    "direct_three_class_probe": "Direct 3-class probe",
}
PALETTE = {
    "convnext_small_full": "#0EA5E9",
    "dinov2_small_moderate": "#8B5CF6",
    "dinov2_base_top4": "#F97316",
    "dinov2_base_multilayer_logistic": "#10B981",
    "dinov2_base_multilayer_rbf": "#EF4444",
    "locked_ensemble": "#0F172A",
    "locked_ensemble_collapsed": "#0F172A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / ".runs")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--asset-dir", type=Path, default=ROOT / "assets")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_svg(path: Path) -> None:
    """Keep generated SVGs diff-clean without changing their rendered content."""
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def save_figure_pair(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    figure.savefig(png_path, dpi=220)
    figure.savefig(svg_path, dpi=220)
    normalize_svg(svg_path)


def macro_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "log_loss": float(log_loss(labels, probabilities)),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }


def stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate(
        [
            rng.choice(indices, size=len(indices), replace=True)
            for value in np.unique(labels)
            if len(indices := np.flatnonzero(labels == value))
        ]
    )


def bootstrap_macro_interval(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = stratified_indices(labels, rng)
        values[index] = f1_score(labels[sampled], predictions[sampled], average="macro")
    return {
        "point_estimate": float(f1_score(labels, predictions, average="macro")),
        "ci_95_low": float(np.quantile(values, 0.025)),
        "ci_95_high": float(np.quantile(values, 0.975)),
        "resamples": int(resamples),
        "seed": int(seed),
    }


def bootstrap_macro_delta(
    labels: np.ndarray,
    left_predictions: np.ndarray,
    right_predictions: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = stratified_indices(labels, rng)
        left = f1_score(labels[sampled], left_predictions[sampled], average="macro")
        right = f1_score(labels[sampled], right_predictions[sampled], average="macro")
        values[index] = left - right
    point = f1_score(labels, left_predictions, average="macro") - f1_score(
        labels, right_predictions, average="macro"
    )
    return {
        "point_estimate": float(point),
        "ci_95_low": float(np.quantile(values, 0.025)),
        "ci_95_high": float(np.quantile(values, 0.975)),
        "probability_difference_gt_zero": float(np.mean(values > 0)),
        "resamples": int(resamples),
        "seed": int(seed),
    }


def bootstrap_auroc_interval(
    binary_labels: np.ndarray,
    scores: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = stratified_indices(binary_labels, rng)
        values[index] = roc_auc_score(binary_labels[sampled], scores[sampled])
    return {
        "auroc": float(roc_auc_score(binary_labels, scores)),
        "ci_95_low": float(np.quantile(values, 0.025)),
        "ci_95_high": float(np.quantile(values, 0.975)),
        "resamples": int(resamples),
        "seed": int(seed),
    }


def cluster_bootstrap_macro_delta(
    labels: np.ndarray,
    left_predictions: np.ndarray,
    right_predictions: np.ndarray,
    clusters: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap a macro-F1 delta while resampling complete image clusters."""
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    rows_by_cluster = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters}
    values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        sampled_rows = np.concatenate([rows_by_cluster[cluster] for cluster in sampled_clusters])
        left = f1_score(labels[sampled_rows], left_predictions[sampled_rows], average="macro")
        right = f1_score(labels[sampled_rows], right_predictions[sampled_rows], average="macro")
        values[index] = left - right
    point = f1_score(labels, left_predictions, average="macro") - f1_score(
        labels, right_predictions, average="macro"
    )
    return {
        "point_estimate": float(point),
        "ci_95_low": float(np.quantile(values, 0.025)),
        "ci_95_high": float(np.quantile(values, 0.975)),
        "probability_difference_gt_zero": float(np.mean(values > 0)),
        "resamples": int(resamples),
        "seed": int(seed),
        "resampling_unit": "image_cluster",
    }


def box_features(frame: pd.DataFrame) -> pd.DataFrame:
    width = (frame["bbox_xmax"] - frame["bbox_xmin"]) / frame["actual_width"]
    height = (frame["bbox_ymax"] - frame["bbox_ymin"]) / frame["actual_height"]
    return pd.DataFrame(
        {
            "log_bbox_area": np.log(np.clip(frame["bbox_area_fraction"], 1e-6, None)),
            "log_bbox_aspect": np.log(np.clip(width / height, 1e-6, None)),
            "bbox_center_x": (
                (frame["bbox_xmin"] + frame["bbox_xmax"]) / 2 / frame["actual_width"]
            ),
            "bbox_center_y": (
                (frame["bbox_ymin"] + frame["bbox_ymax"]) / 2 / frame["actual_height"]
            ),
            "log_image_aspect": np.log(frame["actual_width"] / frame["actual_height"]),
        },
        index=frame.index,
    )


def box_quartiles(values: pd.Series) -> pd.Series:
    return pd.qcut(values, 4, labels=("Q1", "Q2", "Q3", "Q4"))


def add_box_strata(
    rows: list[dict],
    *,
    domain: str,
    evaluation_unit: str,
    labels: np.ndarray,
    frame: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
) -> None:
    quartiles = box_quartiles(frame["bbox_area_fraction"])
    for candidate, candidate_probabilities in probabilities.items():
        predictions = candidate_probabilities.argmax(axis=1)
        correct = predictions == labels
        area_correct = spearmanr(frame["bbox_area_fraction"], correct.astype(float))
        area_confidence = spearmanr(
            frame["bbox_area_fraction"], candidate_probabilities.max(axis=1)
        )
        for quartile in ("Q1", "Q2", "Q3", "Q4"):
            mask = quartiles.eq(quartile).to_numpy()
            metrics = macro_metrics(labels[mask], candidate_probabilities[mask])
            rows.append(
                {
                    "domain": domain,
                    "evaluation_unit": evaluation_unit,
                    "candidate": candidate,
                    "bbox_quartile": quartile,
                    "rows": int(mask.sum()),
                    **metrics,
                    "spearman_bbox_area_correctness": float(area_correct.statistic),
                    "spearman_bbox_area_confidence": float(area_confidence.statistic),
                }
            )


def selective_rows(
    *,
    domain: str,
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for candidate, candidate_probabilities in probabilities.items():
        predictions = candidate_probabilities.argmax(axis=1)
        confidence = candidate_probabilities.max(axis=1)
        correct = predictions == labels
        order = np.argsort(-confidence)
        correctness_auroc = roc_auc_score(correct.astype(int), confidence)
        for coverage in (0.50, 0.75, 0.90, 0.95, 1.00):
            selected_rows = max(1, int(np.floor(coverage * len(labels))))
            selected = order[:selected_rows]
            rows.append(
                {
                    "domain": domain,
                    "candidate": candidate,
                    "coverage": coverage,
                    "selected_rows": selected_rows,
                    "selective_accuracy": float(correct[selected].mean()),
                    "mean_confidence": float(confidence[selected].mean()),
                    "correctness_auroc": float(correctness_auroc),
                }
            )
    return rows


def run_geometry_baseline(
    development: pd.DataFrame,
    test: pd.DataFrame,
    test_labels: np.ndarray,
    external: pd.DataFrame,
    external_labels: np.ndarray,
    *,
    resamples: int,
) -> tuple[pd.DataFrame, dict]:
    label_map = {name: index for index, name in enumerate(CLASS_NAMES_4)}
    development_labels = development["label_4"].map(label_map).to_numpy()
    development_features = box_features(development)
    test_features = box_features(test)
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2_000,
            random_state=ANALYSIS_SEED,
        ),
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=ANALYSIS_SEED)
    oof_probabilities = cross_val_predict(
        estimator,
        development_features,
        development_labels,
        cv=splitter,
        method="predict_proba",
    )
    estimator.fit(development_features, development_labels)
    test_probabilities = estimator.predict_proba(test_features)
    test_predictions = test_probabilities.argmax(axis=1)
    external_probabilities_4 = estimator.predict_proba(box_features(external))
    external_probabilities = np.column_stack(
        (
            external_probabilities_4[:, 0],
            external_probabilities_4[:, 1],
            external_probabilities_4[:, 2] + external_probabilities_4[:, 3],
        )
    )
    external_predictions = external_probabilities.argmax(axis=1)
    interval = bootstrap_macro_interval(
        test_labels,
        test_predictions,
        resamples=resamples,
        seed=ANALYSIS_SEED + 30,
    )
    majority_class = int(pd.Series(development_labels).mode().iloc[0])
    majority_predictions = np.full(len(test_labels), majority_class, dtype=np.int64)
    metrics = pd.DataFrame(
        [
            {
                "candidate": "bbox_geometry_logistic_posthoc",
                "evaluation_domain": "POLAR",
                "evaluation_unit": "image",
                "development_oof_macro_f1": f1_score(
                    development_labels, oof_probabilities.argmax(axis=1), average="macro"
                ),
                "test_accuracy": accuracy_score(test_labels, test_predictions),
                "test_macro_f1": interval["point_estimate"],
                "test_macro_f1_ci_95_low": interval["ci_95_low"],
                "test_macro_f1_ci_95_high": interval["ci_95_high"],
                "test_log_loss": log_loss(test_labels, test_probabilities),
                "test_rows": len(test_labels),
                "selection_role": "posthoc_exploratory_fixed_specification",
            },
            {
                "candidate": "development_majority_class",
                "evaluation_domain": "POLAR",
                "evaluation_unit": "image",
                "development_oof_macro_f1": np.nan,
                "test_accuracy": accuracy_score(test_labels, majority_predictions),
                "test_macro_f1": f1_score(test_labels, majority_predictions, average="macro"),
                "test_macro_f1_ci_95_low": np.nan,
                "test_macro_f1_ci_95_high": np.nan,
                "test_log_loss": np.nan,
                "test_rows": len(test_labels),
                "selection_role": "descriptive_reference",
            },
            {
                "candidate": "bbox_geometry_logistic_posthoc",
                "evaluation_domain": "V-COCO",
                "evaluation_unit": "person",
                "development_oof_macro_f1": np.nan,
                "test_accuracy": accuracy_score(external_labels, external_predictions),
                "test_macro_f1": f1_score(external_labels, external_predictions, average="macro"),
                "test_macro_f1_ci_95_low": np.nan,
                "test_macro_f1_ci_95_high": np.nan,
                "test_log_loss": log_loss(external_labels, external_probabilities),
                "test_rows": len(external_labels),
                "selection_role": "posthoc_external_diagnostic_no_retuning",
            },
        ]
    )
    geometry = box_features(test)
    geometry["bbox_area_fraction"] = test["bbox_area_fraction"].to_numpy()
    geometry["_label"] = [CLASS_NAMES_4[index] for index in test_labels]
    class_medians = (
        geometry.groupby("_label", sort=False)[
            [
                "bbox_area_fraction",
                "log_bbox_aspect",
                "bbox_center_x",
                "bbox_center_y",
                "log_image_aspect",
            ]
        ]
        .median()
        .reset_index()
        .rename(columns={"_label": "class"})
    )
    payload = {
        "fixed_specification": {
            "features": list(development_features.columns),
            "classifier": "standardized balanced multinomial logistic regression",
            "C": 1.0,
            "development_cross_validation": "five-fold stratified, shuffled once",
        },
        "test_interval": interval,
        "external_person_metrics": {
            "accuracy": float(accuracy_score(external_labels, external_predictions)),
            "macro_f1": float(f1_score(external_labels, external_predictions, average="macro")),
            "predicted_class_proportions": {
                CLASS_NAMES_3[index]: float(np.mean(external_predictions == index))
                for index in range(3)
            },
        },
        "class_medians": class_medians.to_dict("records"),
    }
    return metrics, payload


def regularization_summary(training: pd.DataFrame) -> pd.DataFrame:
    training = training[training["model_kind"].eq("dinov2_small")]
    variants = {
        "baseline_mild_dropout_010": "full_ld075_seed",
        "moderate_augmentation": "full_moderate_seed",
        "dropout_020": "full_dropout02_seed",
        "label_smoothing_005": "full_ls005_seed",
        "weight_decay_0p01": "full_wd001_seed",
    }
    rows = []
    for variant, pattern in variants.items():
        selected = training[training["run_id"].str.contains(pattern, regex=False)]
        if sorted(selected["seed"].astype(int).tolist()) != [42, 52, 62]:
            raise RuntimeError(f"Expected seeds 42, 52, 62 for {variant}")
        if variant == "weight_decay_0p01" and not np.allclose(
            selected["weight_decay"], 0.01
        ):
            raise RuntimeError("The wd001 run IDs do not contain weight_decay=0.01")
        rows.append(
            {
                "variant": variant,
                "seeds": 3,
                "macro_f1_mean": selected["macro_f1"].mean(),
                "macro_f1_sd": selected["macro_f1"].std(ddof=1),
                "log_loss_mean": selected["log_loss"].mean(),
                "ece_mean": selected["ece"].mean(),
                "best_epoch_mean": selected["best_epoch"].mean(),
                "runtime_seconds_mean": selected["runtime_seconds"].mean(),
            }
        )
    return pd.DataFrame(rows)


def scale_summary(scale: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    grouped = (
        scale.groupby("actual_train_size", as_index=False)
        .agg(
            macro_f1=("macro_f1", "first"),
            fit_seconds=("fit_seconds", "first"),
            recorded_rows=("seed", "size"),
            unique_subset_hashes=("subset_image_ids_sha256", "nunique"),
            unique_macro_f1=("macro_f1", "nunique"),
        )
        .sort_values("actual_train_size")
    )
    log_size = np.log2(grouped["actual_train_size"].to_numpy(dtype=float))
    macro_f1 = grouped["macro_f1"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(log_size, macro_f1, 1)
    fitted = slope * log_size + intercept
    residual = np.square(macro_f1 - fitted).sum()
    total = np.square(macro_f1 - macro_f1.mean()).sum()
    payload = {
        "unique_experimental_points": int(len(grouped)),
        "recorded_rows": int(len(scale)),
        "all_repeated_rows_identical_within_size": bool(
            grouped["unique_subset_hashes"].eq(1).all() and grouped["unique_macro_f1"].eq(1).all()
        ),
        "descriptive_macro_f1_points_per_doubling": float(slope),
        "descriptive_log2_linear_r_squared": float(1 - residual / total),
        "claim_boundary": (
            "Five fixed subset points; repeated seed labels do not represent independent "
            "subset draws or stochastic fits. This is not a fitted scaling law."
        ),
    }
    return grouped, payload


def plot_bbox_effects(frame: pd.DataFrame, output_dir: Path) -> None:
    selected = (
        "convnext_small_full",
        "dinov2_base_top4",
        "dinov2_base_multilayer_rbf",
        "locked_ensemble",
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
    panels = (
        (axes[0], "POLAR", "image", "In-domain POLAR test"),
        (axes[1], "V-COCO", "person", "External V-COCO persons"),
    )
    for axis, domain, unit, title in panels:
        subset = frame[frame["domain"].eq(domain) & frame["evaluation_unit"].eq(unit)]
        for candidate in selected:
            external_name = (
                "locked_ensemble_collapsed" if candidate == "locked_ensemble" else candidate
            )
            candidate_name = external_name if domain == "V-COCO" else candidate
            rows = subset[subset["candidate"].eq(candidate_name)].sort_values("bbox_quartile")
            axis.plot(
                rows["bbox_quartile"],
                rows["macro_f1"],
                marker="o",
                linewidth=2.2,
                label=DISPLAY[candidate_name],
                color=PALETTE[candidate_name],
            )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Person-box area quartile")
        axis.set_ylabel("Macro-F1")
        axis.grid(alpha=0.25)
        axis.set_ylim(0.47 if domain == "V-COCO" else 0.84, 0.98)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle(
        "Person scale is associated with performance in both domains",
        fontsize=14,
        fontweight="bold",
    )
    save_figure_pair(figure, output_dir, "polar_exploratory_bbox_effects")
    plt.close(figure)


def plot_selective_shift(frame: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
    ensemble = frame[frame["candidate"].isin(("locked_ensemble", "locked_ensemble_collapsed"))]
    for domain, rows in ensemble.groupby("domain", sort=False):
        axes[0].plot(
            rows["coverage"],
            rows["selective_accuracy"],
            marker="o",
            linewidth=2.5,
            label=domain,
        )
    axes[0].set_title("Locked ensemble confidence ranking", fontweight="bold")
    axes[0].set_xlabel("Retained coverage")
    axes[0].set_ylabel("Accuracy among retained rows")
    axes[0].set_xlim(0.48, 1.02)
    axes[0].set_ylim(0.62, 1.01)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    names = (
        "convnext_small_full",
        "dinov2_base_top4",
        "dinov2_base_multilayer_logistic",
        "dinov2_base_multilayer_rbf",
        "locked_ensemble",
    )
    in_values = []
    out_values = []
    for candidate in names:
        in_row = frame[frame["domain"].eq("POLAR") & frame["candidate"].eq(candidate)].iloc[0]
        external_candidate = (
            "locked_ensemble_collapsed" if candidate == "locked_ensemble" else candidate
        )
        out_row = frame[
            frame["domain"].eq("V-COCO") & frame["candidate"].eq(external_candidate)
        ].iloc[0]
        in_values.append(in_row["correctness_auroc"])
        out_values.append(out_row["correctness_auroc"])
    positions = np.arange(len(names))
    width = 0.36
    axes[1].bar(positions - width / 2, in_values, width, label="POLAR", color="#2563EB")
    axes[1].bar(positions + width / 2, out_values, width, label="V-COCO", color="#F97316")
    axes[1].axhline(0.5, color="#64748B", linestyle="--", linewidth=1)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(("ConvNeXt", "DINO-B", "Logistic", "RBF", "Ensemble"), rotation=20)
    axes[1].set_ylim(0.35, 0.96)
    axes[1].set_ylabel("AUROC: confidence predicts correctness")
    axes[1].set_title("Confidence transfer degrades", fontweight="bold")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "In-domain selective prediction does not transfer unchanged",
        fontsize=14,
        fontweight="bold",
    )
    save_figure_pair(figure, output_dir, "polar_exploratory_selective_shift")
    plt.close(figure)


def plot_faithfulness_scale(frame: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
    for family in ("convnext_small_full", "dinov2_base_top4"):
        rows = frame[frame["family"].eq(family)].sort_values("bbox_area_quartile")
        label = DISPLAY[family]
        color = PALETTE[family]
        axes[0].plot(
            rows["bbox_area_quartile"],
            rows["person_attribution_mass_lift"],
            marker="o",
            linewidth=2.4,
            label=label,
            color=color,
        )
        axes[1].plot(
            rows["bbox_area_quartile"],
            rows["person_minus_context_occlusion_drop"],
            marker="o",
            linewidth=2.4,
            label=label,
            color=color,
        )
    axes[0].axhline(1.0, color="#64748B", linestyle="--", linewidth=1)
    axes[0].set_title("Area-normalized person attribution", fontweight="bold")
    axes[0].set_ylabel("Person mass / transformed box area")
    axes[1].axhline(0.0, color="#64748B", linestyle="--", linewidth=1)
    axes[1].set_title("Matched person-versus-context evidence", fontweight="bold")
    axes[1].set_ylabel("Target-probability drop")
    for axis in axes:
        axis.set_xlabel("Source person-box area quartile")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle(
        "Raw localization and class-specific evidence are different properties",
        fontsize=14,
        fontweight="bold",
    )
    save_figure_pair(figure, output_dir, "polar_exploratory_faithfulness_scale")
    plt.close(figure)


def plot_error_structure(rows: pd.DataFrame, output_dir: Path) -> None:
    class_rows = rows[rows["scope"].eq("class")]
    order = list(CLASS_NAMES_4)
    class_rows = class_rows.set_index("class").loc[order].reset_index()
    positions = np.arange(len(order))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
    axes[0].bar(
        positions - width / 2,
        class_rows["ensemble_accuracy"],
        width,
        label="Locked ensemble",
        color="#2563EB",
    )
    axes[0].bar(
        positions + width / 2,
        class_rows["oracle_accuracy"],
        width,
        label="Any component correct",
        color="#A78BFA",
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([name.title() for name in order])
    axes[0].set_ylim(0.84, 1.005)
    axes[0].set_ylabel("Class recall / oracle correctness")
    axes[0].set_title("Residual ensemble headroom", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    error_rows = rows[rows["scope"].eq("error_family")].sort_values("error_count", ascending=True)
    axes[1].barh(
        error_rows["class"],
        error_rows["error_count"],
        color=("#14B8A6", "#F59E0B", "#EF4444", "#94A3B8"),
    )
    axes[1].set_xlabel("Locked-ensemble errors")
    axes[1].set_title("Error concentration", fontweight="bold")
    axes[1].grid(axis="x", alpha=0.25)
    figure.suptitle(
        "Walking drives residual error and oracle headroom",
        fontsize=14,
        fontweight="bold",
    )
    save_figure_pair(figure, output_dir, "polar_exploratory_error_structure")
    plt.close(figure)


def plot_regularization_tradeoff(frame: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.8, 5.0), constrained_layout=True)
    labels = {
        "baseline_mild_dropout_010": "baseline: mild, dropout 0.10",
        "moderate_augmentation": "moderate augmentation",
        "dropout_020": "dropout 0.20",
        "label_smoothing_005": "label smoothing 0.05",
        "weight_decay_0p01": "weight decay 0.01",
    }
    offsets = {
        "baseline_mild_dropout_010": (5, -13),
        "weight_decay_0p01": (5, 7),
    }
    for row in frame.itertuples(index=False):
        axis.scatter(row.ece_mean, row.macro_f1_mean, s=75, color="#2563EB")
        axis.annotate(
            labels[row.variant],
            (row.ece_mean, row.macro_f1_mean),
            xytext=offsets.get(row.variant, (5, 4)),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Mean validation ECE across seeds (lower is better)")
    axis.set_ylabel("Mean validation macro-F1 across seeds")
    axis.set_title(
        "Regularization changed calibration and classification differently", fontweight="bold"
    )
    axis.grid(alpha=0.25)
    save_figure_pair(figure, output_dir, "polar_exploratory_regularization_tradeoff")
    plt.close(figure)


def plot_scale_class_curves(frame: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    colors = ("#2563EB", "#10B981", "#F59E0B", "#EF4444")
    for class_name, color in zip(CLASS_NAMES_4, colors, strict=True):
        rows = frame[frame["class"].eq(class_name)].sort_values("training_rows")
        axis.plot(
            rows["training_rows"],
            rows["class_f1"],
            marker="o",
            linewidth=2.4,
            label=class_name.title(),
            color=color,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks((242, 500, 1_000, 3_000, 9_958))
    axis.set_xticklabels(("242", "500", "1,000", "3,000", "9,958"))
    axis.set_xlabel("Training images in fixed nested subset")
    axis.set_ylabel("Validation class F1")
    axis.set_title("Additional data helps the standing-walking boundary most", fontweight="bold")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    save_figure_pair(figure, output_dir, "polar_exploratory_scale_per_class")
    plt.close(figure)


def plot_shift_diagnostics(
    disagreement: pd.DataFrame,
    mixed_scene: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), constrained_layout=True)
    positions = np.arange(len(disagreement))
    width = 0.36
    axes[0].bar(
        positions - width / 2,
        disagreement["ensemble_error_rate_when_unanimous"],
        width,
        label="Members unanimous",
        color="#2563EB",
    )
    axes[0].bar(
        positions + width / 2,
        disagreement["ensemble_error_rate_when_disagreeing"],
        width,
        label="Members disagree",
        color="#F97316",
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(disagreement["domain"])
    axes[0].set_ylabel("Locked-ensemble error rate")
    axes[0].set_title("Consensus loses reliability under shift", fontweight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    selected = mixed_scene[
        mixed_scene["candidate"].isin(
            ("convnext_small_full", "dinov2_base_top4", "locked_ensemble_collapsed")
        )
    ].copy()
    selected["display"] = selected["candidate"].map(DISPLAY)
    mixed_positions = np.arange(len(selected))
    axes[1].bar(
        mixed_positions - width / 2,
        selected["person_macro_f1"],
        width,
        label="Person macro-F1",
        color="#10B981",
    )
    axes[1].bar(
        mixed_positions + width / 2,
        selected["image_variation_rate"],
        width,
        label="Images with person-specific predictions",
        color="#A78BFA",
    )
    axes[1].set_xticks(mixed_positions)
    axes[1].set_xticklabels(selected["display"], rotation=15)
    axes[1].set_ylim(0.0, 0.72)
    axes[1].set_title("Instance conditioning matters in mixed scenes", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "External errors expose shared bias and view limitations",
        fontsize=14,
        fontweight="bold",
    )
    save_figure_pair(figure, output_dir, "polar_exploratory_shift_diagnostics")
    plt.close(figure)


def ensure_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Required local evidence is missing:\n{joined}")


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    result_dir = args.result_dir.resolve()
    asset_dir = args.asset_dir.resolve()
    asset_dir.mkdir(parents=True, exist_ok=True)

    test_npz_path = run_root / "polar_final" / "test_evaluation" / "test_predictions.npz"
    test_manifest_path = run_root / "polar_final" / "test_evaluation" / "opened_test_manifest.csv"
    external_npz_path = run_root / "polar_final" / "external_vcoco" / "external_predictions.npz"
    external_manifest_path = (
        run_root / "external" / "vcoco" / "audit" / "vcoco_person_manifest_clean.csv"
    )
    development_manifest_path = run_root / "polar_study" / "data" / "polar_development_manifest.csv"
    clean_manifest_path = run_root / "polar_study" / "data" / "polar_clean_manifest.csv"
    fault_npz_path = (
        run_root / "polar_final" / "fault_robustness" / "fault_robustness_predictions.npz"
    )
    scale_prediction_paths = {
        242: run_root
        / "polar_study"
        / "scale"
        / "dinov2_base_full_context10"
        / "size_242_seed_42_predictions.npz",
        500: run_root
        / "polar_study"
        / "scale"
        / "dinov2_base_full_context10"
        / "size_500_seed_42_predictions.npz",
        1_000: run_root
        / "polar_study"
        / "scale"
        / "dinov2_base_full_context10"
        / "size_1000_seed_42_predictions.npz",
        3_000: run_root
        / "polar_study"
        / "scale"
        / "dinov2_base_full_context10"
        / "size_3000_seed_42_predictions.npz",
        9_958: run_root
        / "polar_study"
        / "scale"
        / "dinov2_base_full_context10"
        / "size_all_seed_42_predictions.npz",
    }
    required = (
        test_npz_path,
        test_manifest_path,
        external_npz_path,
        external_manifest_path,
        development_manifest_path,
        clean_manifest_path,
        fault_npz_path,
        result_dir / "polar_faithfulness_per_image.csv",
        result_dir / "polar_training_runs.csv",
        result_dir / "polar_scale_runs.csv",
        *scale_prediction_paths.values(),
    )
    ensure_files(required)

    test_npz = np.load(test_npz_path, allow_pickle=True)
    test_manifest = pd.read_csv(test_manifest_path)
    test_labels = test_npz["labels_4"]
    if not np.array_equal(
        test_manifest["image_id"].astype(str).to_numpy(), test_npz["image_ids"].astype(str)
    ):
        raise RuntimeError("POLAR test manifest and predictions are not aligned")

    test_probabilities = {
        candidate: test_npz[f"probabilities_{candidate}"]
        for candidate in (*COMPONENTS, "locked_ensemble")
    }
    bbox_rows: list[dict] = []
    add_box_strata(
        bbox_rows,
        domain="POLAR",
        evaluation_unit="image",
        labels=test_labels,
        frame=test_manifest,
        probabilities=test_probabilities,
    )

    external_npz = np.load(external_npz_path, allow_pickle=True)
    external_manifest = pd.read_csv(external_manifest_path)
    external_manifest = external_manifest[external_manifest["eligible_person"]].copy()
    external_manifest.index = external_manifest["person_id"].astype(str)
    external_manifest = external_manifest.loc[external_npz["person_ids"].astype(str)].reset_index(
        drop=True
    )
    external_labels_person = external_npz["person_labels"]
    if not np.array_equal(
        external_manifest["label_3"].to_numpy(),
        external_npz["class_names"][external_labels_person],
    ):
        raise RuntimeError("V-COCO person manifest and predictions are not aligned")
    external_person_probabilities = {
        candidate: external_npz[f"person_{candidate}"]
        for candidate in (
            "convnext_small_full",
            "dinov2_small_moderate",
            "dinov2_base_top4",
            "dinov2_base_multilayer_logistic",
            "dinov2_base_multilayer_rbf",
            "locked_ensemble_collapsed",
            "direct_three_class_probe",
        )
    }
    add_box_strata(
        bbox_rows,
        domain="V-COCO",
        evaluation_unit="person",
        labels=external_labels_person,
        frame=external_manifest,
        probabilities=external_person_probabilities,
    )
    bbox_frame = pd.DataFrame(bbox_rows)
    write_csv(bbox_frame, result_dir / "polar_exploratory_bbox_strata.csv")

    pairwise_rows = []
    ensemble_predictions = test_probabilities["locked_ensemble"].argmax(axis=1)
    rbf_predictions = test_probabilities["dinov2_base_multilayer_rbf"].argmax(axis=1)
    logistic_predictions = test_probabilities["dinov2_base_multilayer_logistic"].argmax(axis=1)
    quartiles = box_quartiles(test_manifest["bbox_area_fraction"])
    for quartile_index, quartile in enumerate(("Q1", "Q2", "Q3", "Q4")):
        mask = quartiles.eq(quartile).to_numpy()
        interval = bootstrap_macro_delta(
            test_labels[mask],
            ensemble_predictions[mask],
            rbf_predictions[mask],
            resamples=args.bootstrap_resamples,
            seed=ANALYSIS_SEED + quartile_index,
        )
        pairwise_rows.append(
            {
                "domain": "POLAR",
                "stratum": f"bbox_{quartile}",
                "left": "locked_ensemble",
                "right": "dinov2_base_multilayer_rbf",
                **interval,
                "analysis_role": "posthoc_exploratory",
            }
        )
    pairwise_rows.append(
        {
            "domain": "POLAR",
            "stratum": "all",
            "left": "dinov2_base_multilayer_rbf",
            "right": "dinov2_base_multilayer_logistic",
            **bootstrap_macro_delta(
                test_labels,
                rbf_predictions,
                logistic_predictions,
                resamples=args.bootstrap_resamples,
                seed=ANALYSIS_SEED + 10,
            ),
            "analysis_role": "posthoc_exploratory",
        }
    )
    external_labels_image = external_npz["image_labels"]
    for comparison_index, reference in enumerate(
        ("dinov2_base_multilayer_logistic", "dinov2_base_multilayer_rbf")
    ):
        pairwise_rows.append(
            {
                "domain": "POLAR",
                "stratum": "all",
                "left": "dinov2_base_top4",
                "right": reference,
                **bootstrap_macro_delta(
                    test_labels,
                    test_probabilities["dinov2_base_top4"].argmax(axis=1),
                    test_probabilities[reference].argmax(axis=1),
                    resamples=args.bootstrap_resamples,
                    seed=ANALYSIS_SEED + 13 + comparison_index,
                ),
                "analysis_role": "posthoc_exploratory_pipeline_comparison",
            }
        )
        pairwise_rows.append(
            {
                "domain": "V-COCO",
                "stratum": "image_all",
                "left": "dinov2_base_top4",
                "right": reference,
                **bootstrap_macro_delta(
                    external_labels_image,
                    external_npz["image_dinov2_base_top4"].argmax(axis=1),
                    external_npz[f"image_{reference}"].argmax(axis=1),
                    resamples=args.bootstrap_resamples,
                    seed=ANALYSIS_SEED + 15 + comparison_index,
                ),
                "analysis_role": "posthoc_exploratory_pipeline_comparison",
            }
        )
    equal_probabilities = np.mean(
        [test_probabilities[candidate] for candidate in COMPONENTS], axis=0
    )
    equal_predictions = equal_probabilities.argmax(axis=1)
    pairwise_rows.append(
        {
            "domain": "POLAR",
            "stratum": "all",
            "left": "locked_ensemble",
            "right": "equal_weight_sensitivity_blend",
            **bootstrap_macro_delta(
                test_labels,
                ensemble_predictions,
                equal_predictions,
                resamples=args.bootstrap_resamples,
                seed=ANALYSIS_SEED + 11,
            ),
            "analysis_role": "posthoc_sensitivity",
        }
    )
    external_dino_predictions = external_npz["image_dinov2_base_top4"].argmax(axis=1)
    external_ensemble_predictions = external_npz["image_locked_ensemble_collapsed"].argmax(axis=1)
    pairwise_rows.append(
        {
            "domain": "V-COCO",
            "stratum": "image_all",
            "left": "dinov2_base_top4",
            "right": "locked_ensemble_collapsed",
            **bootstrap_macro_delta(
                external_labels_image,
                external_dino_predictions,
                external_ensemble_predictions,
                resamples=args.bootstrap_resamples,
                seed=ANALYSIS_SEED + 12,
            ),
            "analysis_role": "posthoc_exploratory",
        }
    )
    pairwise_frame = pd.DataFrame(pairwise_rows)
    write_csv(pairwise_frame, result_dir / "polar_exploratory_pairwise.csv")

    component_predictions = np.column_stack(
        [test_probabilities[candidate].argmax(axis=1) for candidate in COMPONENTS]
    )
    any_component_correct = (component_predictions == test_labels[:, None]).any(axis=1)
    component_disagreement = np.any(component_predictions != component_predictions[:, [0]], axis=1)
    ensemble_rows = [
        {
            "scope": "overall",
            "class": "all",
            "rows": len(test_labels),
            "ensemble_accuracy": float((ensemble_predictions == test_labels).mean()),
            "oracle_accuracy": float(any_component_correct.mean()),
            "component_disagreement_rate": float(component_disagreement.mean()),
            "error_count": int((ensemble_predictions != test_labels).sum()),
        }
    ]
    for class_index, class_name in enumerate(CLASS_NAMES_4):
        mask = test_labels == class_index
        ensemble_rows.append(
            {
                "scope": "class",
                "class": class_name,
                "rows": int(mask.sum()),
                "ensemble_accuracy": float(
                    (ensemble_predictions[mask] == test_labels[mask]).mean()
                ),
                "oracle_accuracy": float(any_component_correct[mask].mean()),
                "component_disagreement_rate": float(component_disagreement[mask].mean()),
                "error_count": int((ensemble_predictions[mask] != test_labels[mask]).sum()),
            }
        )
    confusion = pd.crosstab(
        pd.Series(test_labels, name="true"),
        pd.Series(ensemble_predictions, name="predicted"),
    ).reindex(index=range(4), columns=range(4), fill_value=0)
    error_families = {
        "standing <-> walking": int(confusion.loc[1, 2] + confusion.loc[2, 1]),
        "walking <-> running": int(confusion.loc[2, 3] + confusion.loc[3, 2]),
        "sitting <-> standing": int(confusion.loc[0, 1] + confusion.loc[1, 0]),
    }
    assigned = sum(error_families.values())
    total_errors = int((ensemble_predictions != test_labels).sum())
    error_families["other"] = total_errors - assigned
    for name, count in error_families.items():
        ensemble_rows.append(
            {
                "scope": "error_family",
                "class": name,
                "rows": len(test_labels),
                "ensemble_accuracy": np.nan,
                "oracle_accuracy": np.nan,
                "component_disagreement_rate": np.nan,
                "error_count": count,
            }
        )
    ensemble_frame = pd.DataFrame(ensemble_rows)
    write_csv(ensemble_frame, result_dir / "polar_exploratory_ensemble_structure.csv")

    sensitivity_rows = [
        {
            "condition": "locked_weights",
            **macro_metrics(test_labels, test_probabilities["locked_ensemble"]),
            "prediction_agreement_with_locked": 1.0,
        },
        {
            "condition": "equal_weights",
            **macro_metrics(test_labels, equal_probabilities),
            "prediction_agreement_with_locked": float(
                (equal_predictions == ensemble_predictions).mean()
            ),
        },
    ]
    for omitted in COMPONENTS:
        remaining_weight = 1.0 - LOCKED_WEIGHTS[omitted]
        probabilities = sum(
            LOCKED_WEIGHTS[candidate] / remaining_weight * test_probabilities[candidate]
            for candidate in COMPONENTS
            if candidate != omitted
        )
        sensitivity_rows.append(
            {
                "condition": f"leave_out_{omitted}",
                **macro_metrics(test_labels, probabilities),
                "prediction_agreement_with_locked": float(
                    (probabilities.argmax(axis=1) == ensemble_predictions).mean()
                ),
            }
        )
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    write_csv(sensitivity_frame, result_dir / "polar_exploratory_blend_sensitivity.csv")

    external_image_probabilities = {
        candidate: external_npz[f"image_{candidate}"]
        for candidate in (
            "convnext_small_full",
            "dinov2_small_moderate",
            "dinov2_base_top4",
            "dinov2_base_multilayer_logistic",
            "dinov2_base_multilayer_rbf",
            "locked_ensemble_collapsed",
            "direct_three_class_probe",
        )
    }
    selective_frame = pd.DataFrame(
        selective_rows(domain="POLAR", labels=test_labels, probabilities=test_probabilities)
        + selective_rows(
            domain="V-COCO",
            labels=external_labels_image,
            probabilities=external_image_probabilities,
        )
    )
    write_csv(selective_frame, result_dir / "polar_exploratory_selective_prediction.csv")

    disagreement_rows = []
    for domain, labels, probabilities, ensemble_name in (
        ("POLAR", test_labels, test_probabilities, "locked_ensemble"),
        (
            "V-COCO",
            external_labels_image,
            external_image_probabilities,
            "locked_ensemble_collapsed",
        ),
    ):
        predictions = np.column_stack(
            [probabilities[candidate].argmax(axis=1) for candidate in COMPONENTS]
        )
        disagreement = np.any(predictions != predictions[:, [0]], axis=1)
        ensemble_error = probabilities[ensemble_name].argmax(axis=1) != labels
        row = {
            "domain": domain,
            "rows": len(labels),
            "component_disagreement_rate": float(disagreement.mean()),
            "ensemble_error_rate_when_unanimous": float(ensemble_error[~disagreement].mean()),
            "ensemble_error_rate_when_disagreeing": float(ensemble_error[disagreement].mean()),
            "disagreement_auroc_for_ensemble_error": float(
                roc_auc_score(ensemble_error.astype(int), disagreement.astype(int))
            ),
            "unanimous_errors": int((ensemble_error & ~disagreement).sum()),
        }
        if domain == "V-COCO":
            ensemble_predictions_domain = probabilities[ensemble_name].argmax(axis=1)
            unanimous_standing_to_locomotion = (
                ~disagreement
                & ensemble_error
                & (labels == CLASS_NAMES_3.index("standing"))
                & (ensemble_predictions_domain == CLASS_NAMES_3.index("walking_running"))
            )
            row["unanimous_standing_to_locomotion_errors"] = int(
                unanimous_standing_to_locomotion.sum()
            )
            row["fraction_unanimous_errors_standing_to_locomotion"] = float(
                unanimous_standing_to_locomotion.sum() / row["unanimous_errors"]
            )
        disagreement_rows.append(row)
    disagreement_frame = pd.DataFrame(disagreement_rows)
    write_csv(
        disagreement_frame,
        result_dir / "polar_exploratory_disagreement_shift.csv",
    )

    development_manifest = pd.read_csv(development_manifest_path)
    geometry_frame, geometry_payload = run_geometry_baseline(
        development_manifest,
        test_manifest,
        test_labels,
        external_manifest,
        external_labels_person,
        resamples=args.bootstrap_resamples,
    )
    write_csv(geometry_frame, result_dir / "polar_exploratory_geometry_baseline.csv")

    clean_manifest = pd.read_csv(clean_manifest_path)
    clean_manifest = clean_manifest[clean_manifest["primary_included"]].copy()
    polar_quartile_thresholds = clean_manifest["bbox_area_fraction"].quantile([0.25, 0.50, 0.75])
    shift_rows = []
    for class_name in CLASS_NAMES_3:
        polar_area = clean_manifest.loc[
            clean_manifest["label_3"].eq(class_name), "bbox_area_fraction"
        ]
        external_area = external_manifest.loc[
            external_manifest["label_3"].eq(class_name), "bbox_area_fraction"
        ]
        statistic = ks_2samp(polar_area, external_area)
        shift_rows.append(
            {
                "class": class_name,
                "polar_rows": len(polar_area),
                "polar_median_bbox_area": polar_area.median(),
                "vcoco_rows": len(external_area),
                "vcoco_median_bbox_area": external_area.median(),
                "vcoco_to_polar_median_ratio": external_area.median() / polar_area.median(),
                "ks_statistic": statistic.statistic,
                "ks_pvalue_descriptive": statistic.pvalue,
            }
        )
    shift_frame = pd.DataFrame(shift_rows)
    write_csv(shift_frame, result_dir / "polar_exploratory_domain_composition.csv")

    external_manifest["source_includes_stand"] = (
        external_manifest["source_actions"]
        .fillna("")
        .str.split("|", regex=False)
        .apply(lambda values: "stand" in values)
    )
    locomotion_people = external_manifest["label_3"].eq("walking_running")
    external_person_ensemble_predictions = external_person_probabilities[
        "locked_ensemble_collapsed"
    ].argmax(axis=1)
    locomotion_image_ids = set(external_npz["image_ids"][external_labels_image == 2].astype(str))
    locomotion_image_rows = external_manifest[
        external_manifest["image_id"].astype(str).isin(locomotion_image_ids)
        & external_manifest["label_3"].eq("walking_running")
    ]
    locomotion_image_semantics = locomotion_image_rows.groupby(
        locomotion_image_rows["image_id"].astype(str)
    ).agg(
        any_stand=("source_includes_stand", "any"),
        all_stand=("source_includes_stand", "all"),
    )
    external_confusion = pd.crosstab(
        pd.Series(external_labels_image, name="true"),
        pd.Series(external_ensemble_predictions, name="predicted"),
    ).reindex(index=range(3), columns=range(3), fill_value=0)
    external_total_errors = int((external_ensemble_predictions != external_labels_image).sum())
    external_semantics = pd.DataFrame(
        [
            {
                "diagnostic": "person_locomotion_labels_with_stand_source_action",
                "numerator": int(
                    external_manifest.loc[locomotion_people, "source_includes_stand"].sum()
                ),
                "denominator": int(locomotion_people.sum()),
            },
            {
                "diagnostic": "locomotion_images_with_any_stand_source_action",
                "numerator": int(locomotion_image_semantics["any_stand"].sum()),
                "denominator": int(len(locomotion_image_semantics)),
            },
            {
                "diagnostic": "locomotion_images_with_all_relevant_people_stand_tagged",
                "numerator": int(locomotion_image_semantics["all_stand"].sum()),
                "denominator": int(len(locomotion_image_semantics)),
            },
            {
                "diagnostic": "ensemble_standing_to_locomotion_errors",
                "numerator": int(external_confusion.loc[1, 2]),
                "denominator": external_total_errors,
            },
            {
                "diagnostic": "ensemble_locomotion_recall_when_stand_cotagged",
                "numerator": int(
                    (external_person_ensemble_predictions[locomotion_people.to_numpy()] == 2)[
                        external_manifest.loc[locomotion_people, "source_includes_stand"].to_numpy()
                    ].sum()
                ),
                "denominator": int(
                    external_manifest.loc[locomotion_people, "source_includes_stand"].sum()
                ),
            },
            {
                "diagnostic": "ensemble_locomotion_recall_without_stand_cotag",
                "numerator": int(
                    (external_person_ensemble_predictions[locomotion_people.to_numpy()] == 2)[
                        ~external_manifest.loc[
                            locomotion_people, "source_includes_stand"
                        ].to_numpy()
                    ].sum()
                ),
                "denominator": int(
                    (~external_manifest.loc[locomotion_people, "source_includes_stand"]).sum()
                ),
            },
        ]
    )
    external_semantics["fraction"] = (
        external_semantics["numerator"] / external_semantics["denominator"]
    )
    write_csv(
        external_semantics,
        result_dir / "polar_exploratory_external_semantics.csv",
    )

    mixed_image_label_counts = external_manifest.groupby("image_id")["label_3"].nunique()
    mixed_image_ids = set(mixed_image_label_counts[mixed_image_label_counts > 1].index)
    mixed_mask = external_manifest["image_id"].isin(mixed_image_ids).to_numpy()
    mixed_labels = external_labels_person[mixed_mask]
    mixed_clusters = external_manifest.loc[mixed_mask, "image_id"].astype(str).to_numpy()
    mixed_rows = []
    for candidate, probabilities in external_person_probabilities.items():
        candidate_predictions = probabilities[mixed_mask].argmax(axis=1)
        varying_images = 0
        for image_id in mixed_image_ids:
            image_mask = mixed_clusters == str(image_id)
            varying_images += int(np.unique(candidate_predictions[image_mask]).size > 1)
        mixed_rows.append(
            {
                "candidate": candidate,
                "images": len(mixed_image_ids),
                "person_rows": int(mixed_mask.sum()),
                "person_accuracy": accuracy_score(mixed_labels, candidate_predictions),
                "person_macro_f1": f1_score(mixed_labels, candidate_predictions, average="macro"),
                "images_with_person_specific_prediction_variation": varying_images,
                "image_variation_rate": varying_images / len(mixed_image_ids),
                "analysis_role": "posthoc_external_mixed_label_diagnostic",
            }
        )
    mixed_frame = pd.DataFrame(mixed_rows)
    write_csv(mixed_frame, result_dir / "polar_exploratory_mixed_scene_persons.csv")
    mixed_interval = cluster_bootstrap_macro_delta(
        mixed_labels,
        external_person_probabilities["dinov2_base_top4"][mixed_mask].argmax(axis=1),
        external_person_probabilities["convnext_small_full"][mixed_mask].argmax(axis=1),
        mixed_clusters,
        resamples=args.bootstrap_resamples,
        seed=ANALYSIS_SEED + 40,
    )

    faithfulness = pd.read_csv(result_dir / "polar_faithfulness_per_image.csv")
    faithfulness_metrics = (
        faithfulness.groupby(["family", "bbox_area_quartile"], as_index=False)
        .agg(
            rows=("image_id", "size"),
            target_accuracy=("target_correct", "mean"),
            person_attribution_mass=("person_attribution_mass", "mean"),
            person_attribution_mass_lift=("person_attribution_mass_lift", "mean"),
            pointing_game=("pointing_game", "mean"),
            deletion_selectivity_gap=("deletion_selectivity_gap", "mean"),
            person_minus_context_occlusion_drop=(
                "person_minus_context_occlusion_drop",
                "mean",
            ),
            full_crop_js_divergence=("full_crop_js_divergence", "mean"),
            target_vs_alternative_spearman=(
                "target_vs_alternative_attribution_spearman",
                "mean",
            ),
        )
        .sort_values(["family", "bbox_area_quartile"])
    )
    write_csv(faithfulness_metrics, result_dir / "polar_exploratory_faithfulness_strata.csv")
    faithfulness_correlations = []
    correlation_pairs = (
        ("bbox_area_fraction_source", "person_attribution_mass"),
        ("bbox_area_fraction_source", "person_attribution_mass_lift"),
        ("bbox_area_fraction_source", "person_minus_context_occlusion_drop"),
        ("locked_target_probability", "deletion_selectivity_gap"),
    )
    for family, family_frame in faithfulness.groupby("family"):
        for left, right in correlation_pairs:
            selected = family_frame[[left, right]].dropna()
            correlation = spearmanr(selected[left], selected[right])
            faithfulness_correlations.append(
                {
                    "family": family,
                    "left": left,
                    "right": right,
                    "rows": len(selected),
                    "spearman_rho": correlation.statistic,
                    "pvalue_descriptive_unadjusted": correlation.pvalue,
                    "analysis_role": "posthoc_exploratory",
                }
            )
    faithfulness_correlation_frame = pd.DataFrame(faithfulness_correlations)
    write_csv(
        faithfulness_correlation_frame,
        result_dir / "polar_exploratory_faithfulness_correlations.csv",
    )
    faithfulness_error_detection_rows = []
    for family_index, (family, family_frame) in enumerate(faithfulness.groupby("family")):
        errors = (~family_frame["target_correct"].astype(bool)).to_numpy(dtype=int)
        for signal_index, (signal, values) in enumerate(
            (
                (
                    "full_crop_js_divergence",
                    family_frame["full_crop_js_divergence"].to_numpy(dtype=float),
                ),
                (
                    "one_minus_locked_confidence",
                    1.0 - family_frame["locked_target_probability"].to_numpy(dtype=float),
                ),
            )
        ):
            faithfulness_error_detection_rows.append(
                {
                    "family": family,
                    "signal": signal,
                    "rows": len(errors),
                    "errors": int(errors.sum()),
                    **bootstrap_auroc_interval(
                        errors,
                        values,
                        resamples=args.bootstrap_resamples,
                        seed=ANALYSIS_SEED + 50 + family_index * 2 + signal_index,
                    ),
                    "analysis_role": "posthoc_small_cohort_hypothesis",
                }
            )
    faithfulness_error_detection_frame = pd.DataFrame(faithfulness_error_detection_rows)
    write_csv(
        faithfulness_error_detection_frame,
        result_dir / "polar_exploratory_faithfulness_error_detection.csv",
    )

    fault_npz = np.load(fault_npz_path, allow_pickle=True)
    fault_rows = []
    for family in ("convnext_small_full", "dinov2_base_top4"):
        clean_probabilities = fault_npz[f"{family}__clean_float"]
        fault_probabilities = fault_npz[f"{family}__input_rate_0.001"]
        clean_predictions = clean_probabilities.argmax(axis=1)
        fault_predictions = fault_probabilities.argmax(axis=1)
        drift = np.abs(clean_probabilities - fault_probabilities).mean(axis=1)
        ordered = np.sort(clean_probabilities, axis=1)
        margin = ordered[:, -1] - ordered[:, -2]
        changed = clean_predictions != fault_predictions
        labels = fault_npz["labels"]
        for class_index, class_name in enumerate(CLASS_NAMES_4):
            mask = labels == class_index
            fault_rows.append(
                {
                    "family": family,
                    "class": class_name,
                    "rows": int(mask.sum()),
                    "mean_absolute_probability_drift": float(drift[mask].mean()),
                    "median_absolute_probability_drift": float(np.median(drift[mask])),
                    "prediction_flips": int(changed[mask].sum()),
                    "clean_accuracy": float((clean_predictions[mask] == labels[mask]).mean()),
                    "fault_accuracy": float((fault_predictions[mask] == labels[mask]).mean()),
                }
            )
        fault_rows.append(
            {
                "family": family,
                "class": "all",
                "rows": len(labels),
                "mean_absolute_probability_drift": float(drift.mean()),
                "median_absolute_probability_drift": float(np.median(drift)),
                "prediction_flips": int(changed.sum()),
                "clean_accuracy": float((clean_predictions == labels).mean()),
                "fault_accuracy": float((fault_predictions == labels).mean()),
                "changed_prediction_margin_median": (
                    float(np.median(margin[changed])) if changed.any() else np.nan
                ),
                "stable_prediction_margin_median": float(np.median(margin[~changed])),
            }
        )
    fault_frame = pd.DataFrame(fault_rows)
    write_csv(fault_frame, result_dir / "polar_exploratory_fault_by_class.csv")

    training = pd.read_csv(result_dir / "polar_training_runs.csv")
    regularization = regularization_summary(training)
    write_csv(regularization, result_dir / "polar_exploratory_regularization.csv")
    scale_frame, scale_payload = scale_summary(pd.read_csv(result_dir / "polar_scale_runs.csv"))
    write_csv(scale_frame, result_dir / "polar_exploratory_scale_integrity.csv")
    scale_class_rows = []
    for training_rows, prediction_path in scale_prediction_paths.items():
        predictions = np.load(prediction_path, allow_pickle=True)
        labels = predictions["labels"]
        predicted = predictions["probabilities"].argmax(axis=1)
        class_f1 = f1_score(labels, predicted, average=None, labels=range(4))
        confusion = pd.crosstab(
            pd.Series(labels, name="true"),
            pd.Series(predicted, name="predicted"),
        ).reindex(index=range(4), columns=range(4), fill_value=0)
        total_errors_at_size = int((predicted != labels).sum())
        adjacent_errors = int(
            sum(
                confusion.loc[left, right]
                for left in range(4)
                for right in range(4)
                if abs(left - right) == 1
            )
        )
        for class_index, class_name in enumerate(CLASS_NAMES_4):
            scale_class_rows.append(
                {
                    "training_rows": training_rows,
                    "class": class_name,
                    "class_f1": class_f1[class_index],
                    "total_errors": total_errors_at_size,
                    "adjacent_state_errors": adjacent_errors,
                    "adjacent_state_error_fraction": adjacent_errors / total_errors_at_size,
                    "analysis_role": "posthoc_fixed_curve_decomposition",
                }
            )
    scale_class_frame = pd.DataFrame(scale_class_rows)
    write_csv(
        scale_class_frame,
        result_dir / "polar_exploratory_scale_per_class.csv",
    )
    first_scale = scale_class_frame[scale_class_frame["training_rows"].eq(242)].set_index("class")
    last_scale = scale_class_frame[scale_class_frame["training_rows"].eq(9_958)].set_index("class")
    scale_payload["per_class_f1_gain_242_to_9958"] = {
        class_name: float(
            last_scale.loc[class_name, "class_f1"] - first_scale.loc[class_name, "class_f1"]
        )
        for class_name in CLASS_NAMES_4
    }

    plot_bbox_effects(bbox_frame, asset_dir)
    plot_selective_shift(selective_frame, asset_dir)
    plot_faithfulness_scale(faithfulness_metrics, asset_dir)
    plot_error_structure(ensemble_frame, asset_dir)
    plot_regularization_tradeoff(regularization, asset_dir)
    plot_scale_class_curves(scale_class_frame, asset_dir)
    plot_shift_diagnostics(disagreement_frame, mixed_frame, asset_dir)

    polar_global_median = clean_manifest["bbox_area_fraction"].median()
    external_global_median = external_manifest["bbox_area_fraction"].median()
    rbf_logistic = pairwise_frame[
        pairwise_frame["left"].eq("dinov2_base_multilayer_rbf")
        & pairwise_frame["right"].eq("dinov2_base_multilayer_logistic")
    ].iloc[0]
    dino_external = pairwise_frame[
        pairwise_frame["domain"].eq("V-COCO")
        & pairwise_frame["left"].eq("dinov2_base_top4")
        & pairwise_frame["right"].eq("locked_ensemble_collapsed")
    ].iloc[0]
    q1_delta = pairwise_frame[pairwise_frame["stratum"].eq("bbox_Q1")].iloc[0]
    ensemble_selective = selective_frame[
        selective_frame["domain"].eq("POLAR")
        & selective_frame["candidate"].eq("locked_ensemble")
        & selective_frame["coverage"].eq(0.90)
    ].iloc[0]
    external_ensemble_selective = selective_frame[
        selective_frame["domain"].eq("V-COCO")
        & selective_frame["candidate"].eq("locked_ensemble_collapsed")
        & selective_frame["coverage"].eq(0.90)
    ].iloc[0]
    disagreement_payload = {}
    for record in disagreement_frame.to_dict(orient="records"):
        domain = record.pop("domain")
        disagreement_payload[domain] = {
            key: value for key, value in record.items() if pd.notna(value)
        }
    summary = {
        "status": "POSTHOC_EXPLORATORY_ANALYSIS_COMPLETE",
        "analysis_role": (
            "Hypothesis-generating only; no result changes the locked primary selection or "
            "confirmatory interpretation."
        ),
        "analysis_seed": ANALYSIS_SEED,
        "bootstrap_resamples": args.bootstrap_resamples,
        "source_hashes": {
            path.relative_to(ROOT).as_posix(): file_sha256(path) for path in required
        },
        "findings": {
            "small_person_ensemble_gain": {
                "ensemble_minus_rbf_macro_f1_q1": q1_delta["point_estimate"],
                "ci_95": [q1_delta["ci_95_low"], q1_delta["ci_95_high"]],
                "interpretation": (
                    "The ensemble gain is largest in the smallest-person quartile; this was not "
                    "a predeclared interaction."
                ),
            },
            "component_oracle": {
                "oracle_accuracy": float(any_component_correct.mean()),
                "locked_ensemble_accuracy": float((ensemble_predictions == test_labels).mean()),
                "oracle_headroom": float(
                    any_component_correct.mean() - (ensemble_predictions == test_labels).mean()
                ),
                "interpretation": (
                    "The oracle is an unattainable label-informed upper bound demonstrating "
                    "remaining error diversity, not a deployable score."
                ),
            },
            "error_concentration": {
                "standing_walking_errors": error_families["standing <-> walking"],
                "walking_running_errors": error_families["walking <-> running"],
                "adjacent_state_errors": int(
                    error_families["standing <-> walking"]
                    + error_families["walking <-> running"]
                    + error_families["sitting <-> standing"]
                ),
                "adjacent_state_error_fraction": float(
                    (
                        error_families["standing <-> walking"]
                        + error_families["walking <-> running"]
                        + error_families["sitting <-> standing"]
                    )
                    / total_errors
                ),
                "total_errors": total_errors,
            },
            "rbf_vs_logistic": {
                "macro_f1_delta": rbf_logistic["point_estimate"],
                "paired_ci_95": [
                    rbf_logistic["ci_95_low"],
                    rbf_logistic["ci_95_high"],
                ],
                "interpretation": (
                    "The standalone point-estimate advantage is not statistically resolved by "
                    "this exploratory paired interval."
                ),
            },
            "external_top4_vs_ensemble": {
                "macro_f1_delta": dino_external["point_estimate"],
                "paired_ci_95": [
                    dino_external["ci_95_low"],
                    dino_external["ci_95_high"],
                ],
                "interpretation": (
                    "The external point-estimate ordering is not statistically resolved, while "
                    "the ensemble retains better tracked log loss."
                ),
            },
            "confidence_transfer": {
                "polar_90_percent_coverage_accuracy": ensemble_selective["selective_accuracy"],
                "vcoco_90_percent_coverage_accuracy": external_ensemble_selective[
                    "selective_accuracy"
                ],
                "polar_correctness_auroc": ensemble_selective["correctness_auroc"],
                "vcoco_correctness_auroc": external_ensemble_selective["correctness_auroc"],
            },
            "box_geometry_baseline": geometry_payload,
            "domain_composition": {
                "polar_median_person_box_area": polar_global_median,
                "vcoco_median_person_box_area": external_global_median,
                "vcoco_to_polar_median_ratio": external_global_median / polar_global_median,
                "vcoco_fraction_below_polar_q1": float(
                    (
                        external_manifest["bbox_area_fraction"]
                        <= polar_quartile_thresholds.loc[0.25]
                    ).mean()
                ),
            },
            "external_label_semantics": {
                row.diagnostic: {
                    "numerator": int(row.numerator),
                    "denominator": int(row.denominator),
                    "fraction": float(row.fraction),
                }
                for row in external_semantics.itertuples(index=False)
            },
            "disagreement_shift": disagreement_payload,
            "mixed_scene_person_crops": {
                "images": int(len(mixed_image_ids)),
                "person_rows": int(mixed_mask.sum()),
                "dinov2_base_minus_convnext_macro_f1": mixed_interval,
                "interpretation": (
                    "Person-conditioned crops can produce different predictions for different "
                    "people in the same image; a single full-frame prediction cannot."
                ),
            },
            "faithfulness_error_detection": {
                f"{row.family}__{row.signal}": {
                    "rows": int(row.rows),
                    "errors": int(row.errors),
                    "auroc": float(row.auroc),
                    "ci_95": [float(row.ci_95_low), float(row.ci_95_high)],
                }
                for row in faithfulness_error_detection_frame.itertuples(index=False)
            },
            "scale_integrity": scale_payload,
        },
    }
    write_json(summary, result_dir / "polar_exploratory_summary.json")
    print(f"Wrote exploratory analysis outputs under {result_dir} and {asset_dir}")


if __name__ == "__main__":
    main()
