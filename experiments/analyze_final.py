"""Select downstream heads on OOF evidence and report locked test results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FAMILIES = ("convnext_small", "dinov2_small")
DISPLAY_NAMES = {
    "convnext_small": "ConvNeXt-Small",
    "dinov2_small": "DINOv2-Small",
    "convnext_small_svm": "ConvNeXt + RBF/linear SVM",
    "dinov2_small_svm": "DINOv2 + RBF/linear SVM",
    "probability_blend": "OOF-weighted probability blend",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    one_hot = np.eye(probs.shape[1], dtype=float)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:], strict=False):
        include_right = right == 1.0
        mask = (confidence >= left) & (
            (confidence <= right) if include_right else (confidence < right)
        )
        if mask.any():
            value += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return value


def metric_bundle(y_true: np.ndarray, probs: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, preds, average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "log_loss": float(log_loss(y_true, probs, labels=np.arange(probs.shape[1]))),
        "brier_score": multiclass_brier(y_true, probs),
        "ece": expected_calibration_error(y_true, probs),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.astype(int).tolist(),
    }


def l2_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


def load_ids(path: Path) -> list[str]:
    return pd.read_csv(path)["image_id"].astype(str).tolist()


def discover_seeds(artifact_root: Path, family: str) -> list[int]:
    base = artifact_root / "predictions" / family
    seeds = []
    for path in base.glob("seed_*"):
        if (path / "final" / "seed_result.json").is_file():
            seeds.append(int(path.name.split("_", 1)[1]))
    seeds = sorted(seeds)
    if len(seeds) < 3:
        raise RuntimeError(f"Expected at least three complete seeds for {family}; found {seeds}")
    return seeds


def assert_same(reference, candidate, label: str) -> None:
    if isinstance(reference, np.ndarray):
        equal = np.array_equal(reference, candidate)
    else:
        equal = reference == candidate
    if not equal:
        raise RuntimeError(f"Alignment mismatch: {label}")


def load_oof_family(artifact_root: Path, family: str, seeds: list[int]) -> dict:
    center_probs, flip_tta_probs, features = [], [], []
    labels_ref = ids_ref = None
    for seed in seeds:
        base = artifact_root / "predictions" / family / f"seed_{seed}" / "cv"
        labels = np.load(base / "oof_calibrated_labels.npy")
        ids = load_ids(base / "oof_image_ids.csv")
        if labels_ref is None:
            labels_ref, ids_ref = labels, ids
        else:
            assert_same(labels_ref, labels, f"{family} OOF labels seed {seed}")
            assert_same(ids_ref, ids, f"{family} OOF IDs seed {seed}")
        center_probs.append(np.load(base / "oof_calibrated_probs.npy"))
        flip_tta_probs.append(np.load(base / "oof_tta_calibrated_probs.npy"))
        features.append(l2_rows(np.load(base / "oof_raw_features.npy")))
    return {
        "labels": labels_ref,
        "ids": ids_ref,
        "center_probs": np.mean(center_probs, axis=0),
        "flip_tta_probs": np.mean(flip_tta_probs, axis=0),
        "features": l2_rows(np.mean(features, axis=0)),
    }


def load_full_pool_family(artifact_root: Path, family: str, seeds: list[int]) -> dict:
    features = []
    labels_ref = ids_ref = None
    for seed in seeds:
        base = artifact_root / "predictions" / family / f"seed_{seed}" / "final"
        labels = np.load(base / "full_pool_labels.npy")
        ids = load_ids(base / "full_pool_image_ids.csv")
        if labels_ref is None:
            labels_ref, ids_ref = labels, ids
        else:
            assert_same(labels_ref, labels, f"{family} pool labels seed {seed}")
            assert_same(ids_ref, ids, f"{family} pool IDs seed {seed}")
        features.append(l2_rows(np.load(base / "full_pool_features.npy")))
    return {
        "labels": labels_ref,
        "ids": ids_ref,
        "features": l2_rows(np.mean(features, axis=0)),
    }


def svm_grid() -> list[dict]:
    grid = [
        {"kernel": "linear", "C": value, "gamma": "scale"}
        for value in (0.01, 0.1, 1.0, 10.0, 100.0)
    ]
    grid.extend(
        {"kernel": "rbf", "C": c_value, "gamma": gamma}
        for c_value in (0.1, 1.0, 10.0, 100.0)
        for gamma in ("scale", 0.001, 0.01, 0.1)
    )
    return grid


def build_svm(params: dict) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel=params["kernel"],
                    C=float(params["C"]),
                    gamma=params["gamma"],
                    probability=True,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def select_svm(features: np.ndarray, labels: np.ndarray) -> tuple[dict, np.ndarray, pd.DataFrame]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    predictions: dict[str, np.ndarray] = {}
    for params in svm_grid():
        key = json.dumps(params, sort_keys=True)
        probs = np.zeros((len(labels), len(np.unique(labels))), dtype=float)
        for train_idx, val_idx in folds.split(features, labels):
            model = build_svm(params)
            model.fit(features[train_idx], labels[train_idx])
            if not np.array_equal(model.classes_, np.arange(probs.shape[1])):
                raise RuntimeError("Unexpected SVM class ordering.")
            probs[val_idx] = model.predict_proba(features[val_idx])
        metrics = metric_bundle(labels, probs)
        rows.append(
            {**params, **{k: v for k, v in metrics.items() if not k.startswith("per_class")}}
        )
        predictions[key] = probs
    grid_df = (
        pd.DataFrame(rows)
        .sort_values(["macro_f1", "log_loss", "C"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    winner = grid_df.iloc[0]
    params = {
        "kernel": str(winner["kernel"]),
        "C": float(winner["C"]),
        "gamma": str(winner["gamma"])
        if isinstance(winner["gamma"], str)
        else float(winner["gamma"]),
    }
    return params, predictions[json.dumps(params, sort_keys=True)], grid_df


def select_blend(
    conv_probs: np.ndarray, dino_probs: np.ndarray, labels: np.ndarray
) -> tuple[float, np.ndarray, pd.DataFrame]:
    rows = []
    candidates = {}
    for conv_weight in np.linspace(0.0, 1.0, 41):
        probs = conv_weight * conv_probs + (1.0 - conv_weight) * dino_probs
        metrics = metric_bundle(labels, probs)
        rows.append(
            {
                "convnext_weight": float(conv_weight),
                "dinov2_weight": float(1.0 - conv_weight),
                **{k: v for k, v in metrics.items() if not k.startswith("per_class")},
            }
        )
        candidates[float(conv_weight)] = probs
    table = pd.DataFrame(rows)
    table["distance_from_equal_weight"] = (table["convnext_weight"] - 0.5).abs()
    table = table.sort_values(
        ["macro_f1", "log_loss", "distance_from_equal_weight", "convnext_weight"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    winner = float(table.iloc[0]["convnext_weight"])
    return winner, candidates[winner], table


def select_downstream(
    artifact_root: Path, output_dir: Path, seeds_by_family: dict[str, list[int]]
) -> tuple[dict, dict, dict]:
    oof = {
        family: load_oof_family(artifact_root, family, seeds_by_family[family])
        for family in FAMILIES
    }
    assert_same(oof[FAMILIES[0]]["labels"], oof[FAMILIES[1]]["labels"], "cross-family OOF labels")
    assert_same(oof[FAMILIES[0]]["ids"], oof[FAMILIES[1]]["ids"], "cross-family OOF IDs")
    labels = oof[FAMILIES[0]]["labels"]

    evaluation_policy = {}
    method_probs = {}
    evaluation_rows = []
    for family in FAMILIES:
        policy_candidates = {
            "center_crop": oof[family]["center_probs"],
            "center_plus_horizontal_flip": oof[family]["flip_tta_probs"],
        }
        policy_scores = []
        for policy_name, probs in policy_candidates.items():
            metrics = metric_bundle(labels, probs)
            policy_scores.append(
                {
                    "family": family,
                    "policy": policy_name,
                    "macro_f1": metrics["macro_f1"],
                    "log_loss": metrics["log_loss"],
                    "prefer_simpler": 0 if policy_name == "center_crop" else 1,
                }
            )
        policy_scores = sorted(
            policy_scores,
            key=lambda row: (-row["macro_f1"], row["log_loss"], row["prefer_simpler"]),
        )
        winner = policy_scores[0]["policy"]
        evaluation_policy[family] = winner
        method_probs[family] = policy_candidates[winner]
        for row in policy_scores:
            row["selected"] = row["policy"] == winner
            evaluation_rows.append(row)
    pd.DataFrame(evaluation_rows).to_csv(
        output_dir / "evaluation_policy_oof_selection.csv", index=False
    )
    svm_selections = {}
    for family in FAMILIES:
        params, probs, grid_df = select_svm(oof[family]["features"], labels)
        method_name = f"{family}_svm"
        method_probs[method_name] = probs
        svm_selections[family] = params
        grid_df.to_csv(output_dir / f"{family}_svm_oof_search.csv", index=False)

    blend_weight, blend_probs, blend_table = select_blend(
        method_probs["convnext_small"], method_probs["dinov2_small"], labels
    )
    method_probs["probability_blend"] = blend_probs
    blend_table.to_csv(output_dir / "probability_blend_oof_search.csv", index=False)

    ranking_rows = []
    for method, probs in method_probs.items():
        metrics = metric_bundle(labels, probs)
        ranking_rows.append(
            {
                "method": method,
                **{key: value for key, value in metrics.items() if not key.startswith("per_class")},
            }
        )
    ranking = (
        pd.DataFrame(ranking_rows)
        .sort_values(["macro_f1", "log_loss", "method"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    ranking["oof_rank"] = np.arange(1, len(ranking) + 1)
    ranking["selected_champion"] = ranking["oof_rank"].eq(1)
    ranking.to_csv(output_dir / "downstream_oof_ranking.csv", index=False)

    lock = {
        "status": "LOCKED_FROM_OOF_BEFORE_DOWNSTREAM_TEST_EVALUATION",
        "analysis_runner": Path(__file__).name,
        "analysis_runner_sha256": sha256_file(Path(__file__).resolve()),
        "test_used_for_selection": False,
        "seed_ensemble": {family: seeds_by_family[family] for family in FAMILIES},
        "evaluation_policy": evaluation_policy,
        "svm_selection": svm_selections,
        "probability_blend": {
            "convnext_weight": blend_weight,
            "dinov2_weight": 1.0 - blend_weight,
            "grid_step": 0.025,
        },
        "champion_method": str(ranking.iloc[0]["method"]),
        "selection_metric": "pooled_oof_macro_f1",
        "tie_breakers": ["lower_oof_log_loss", "method_name"],
    }
    lock_path = output_dir / "downstream_selection_lock.json"
    with lock_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(lock), handle, indent=2)
    lock["lock_sha256"] = sha256_file(lock_path)
    return lock, oof, method_probs


def load_test_family(artifact_root: Path, family: str, seeds: list[int]) -> dict:
    center_probs, flip_tta_probs, features = [], [], []
    labels_ref = ids_ref = None
    for seed in seeds:
        base = artifact_root / "predictions" / family / f"seed_{seed}" / "final"
        labels = np.load(base / "test_calibrated_labels.npy")
        ids = load_ids(base / "test_image_ids.csv")
        if labels_ref is None:
            labels_ref, ids_ref = labels, ids
        else:
            assert_same(labels_ref, labels, f"{family} test labels seed {seed}")
            assert_same(ids_ref, ids, f"{family} test IDs seed {seed}")
        center_probs.append(np.load(base / "test_calibrated_probs.npy"))
        flip_tta_probs.append(np.load(base / "test_tta_calibrated_probs.npy"))
        features.append(l2_rows(np.load(base / "test_raw_features.npy")))
    return {
        "labels": labels_ref,
        "ids": ids_ref,
        "center_probs": np.mean(center_probs, axis=0),
        "flip_tta_probs": np.mean(flip_tta_probs, axis=0),
        "features": l2_rows(np.mean(features, axis=0)),
    }


def evaluate_locked_methods(
    artifact_root: Path,
    output_dir: Path,
    lock: dict,
    seeds_by_family: dict[str, list[int]],
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    pool = {
        family: load_full_pool_family(artifact_root, family, seeds_by_family[family])
        for family in FAMILIES
    }
    test = {
        family: load_test_family(artifact_root, family, seeds_by_family[family])
        for family in FAMILIES
    }
    assert_same(
        test[FAMILIES[0]]["labels"], test[FAMILIES[1]]["labels"], "cross-family test labels"
    )
    assert_same(test[FAMILIES[0]]["ids"], test[FAMILIES[1]]["ids"], "cross-family test IDs")
    labels = test[FAMILIES[0]]["labels"]

    method_probs = {}
    for family in FAMILIES:
        policy = lock["evaluation_policy"][family]
        if policy == "center_crop":
            method_probs[family] = test[family]["center_probs"]
        elif policy == "center_plus_horizontal_flip":
            method_probs[family] = test[family]["flip_tta_probs"]
        else:
            raise RuntimeError(f"Unknown locked evaluation policy: {policy}")
    for family in FAMILIES:
        assert_same(
            pool[family]["labels"], np.asarray(pool[family]["labels"]), f"{family} pool labels"
        )
        model = build_svm(lock["svm_selection"][family])
        model.fit(pool[family]["features"], pool[family]["labels"])
        method_probs[f"{family}_svm"] = model.predict_proba(test[family]["features"])

    conv_weight = float(lock["probability_blend"]["convnext_weight"])
    method_probs["probability_blend"] = (
        conv_weight * method_probs["convnext_small"]
        + (1.0 - conv_weight) * method_probs["dinov2_small"]
    )

    rows = []
    for method, probs in method_probs.items():
        metrics = metric_bundle(labels, probs)
        rows.append(
            {
                "method": method,
                "display_name": DISPLAY_NAMES[method],
                "selected_champion": method == lock["champion_method"],
                **{key: value for key, value in metrics.items() if not key.startswith("per_class")},
            }
        )
        np.save(output_dir / f"{method}_test_probs.npy", probs)
        pd.DataFrame(
            {
                "image_id": test[FAMILIES[0]]["ids"],
                "y_true": labels,
                "y_pred": probs.argmax(axis=1),
                "confidence": probs.max(axis=1),
            }
        ).to_csv(output_dir / f"{method}_test_predictions.csv", index=False)

    table = (
        pd.DataFrame(rows)
        .sort_values(["selected_champion", "macro_f1"], ascending=[False, False])
        .reset_index(drop=True)
    )
    table.to_csv(output_dir / "locked_test_metrics.csv", index=False)
    return table, method_probs, labels


def stratified_bootstrap_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pieces = []
    for class_id in np.unique(labels):
        members = np.flatnonzero(labels == class_id)
        pieces.append(rng.choice(members, size=len(members), replace=True))
    return np.concatenate(pieces)


def bootstrap_intervals(
    method_probs: dict[str, np.ndarray],
    labels: np.ndarray,
    champion: str,
    base_reference: str,
    n_resamples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260821)
    scores = {method: {"accuracy": [], "macro_f1": []} for method in method_probs}
    differences = {"accuracy": [], "macro_f1": []}
    for _ in range(int(n_resamples)):
        idx = stratified_bootstrap_indices(labels, rng)
        sampled_labels = labels[idx]
        sampled = {}
        for method, probs in method_probs.items():
            preds = probs[idx].argmax(axis=1)
            sampled[method] = {
                "accuracy": accuracy_score(sampled_labels, preds),
                "macro_f1": f1_score(sampled_labels, preds, average="macro"),
            }
            for metric in ("accuracy", "macro_f1"):
                scores[method][metric].append(sampled[method][metric])
        for metric in ("accuracy", "macro_f1"):
            differences[metric].append(sampled[champion][metric] - sampled[base_reference][metric])

    interval_rows = []
    for method, metrics in scores.items():
        for metric, values in metrics.items():
            values_array = np.asarray(values)
            interval_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "bootstrap_mean": float(values_array.mean()),
                    "ci_2_5": float(np.quantile(values_array, 0.025)),
                    "ci_97_5": float(np.quantile(values_array, 0.975)),
                    "resamples": int(n_resamples),
                }
            )
    difference_rows = []
    for metric, values in differences.items():
        values_array = np.asarray(values)
        difference_rows.append(
            {
                "champion": champion,
                "reference": base_reference,
                "metric": metric,
                "mean_difference": float(values_array.mean()),
                "ci_2_5": float(np.quantile(values_array, 0.025)),
                "ci_97_5": float(np.quantile(values_array, 0.975)),
                "probability_difference_gt_zero": float(np.mean(values_array > 0.0)),
                "resamples": int(n_resamples),
            }
        )
    return pd.DataFrame(interval_rows), pd.DataFrame(difference_rows)


def create_plots(
    output_dir: Path,
    metrics: pd.DataFrame,
    method_probs: dict[str, np.ndarray],
    labels: np.ndarray,
    champion: str,
    class_names: list[str],
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plot_df = metrics.sort_values("macro_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#0F766E" if method == champion else "#94A3B8" for method in plot_df["method"]]
    ax.barh(plot_df["display_name"], plot_df["macro_f1"], color=colors)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Macro-F1 on locked test split")
    ax.set_ylabel("")
    ax.set_title("Final method comparison")
    for index, value in enumerate(plot_df["macro_f1"]):
        ax.text(min(value + 0.012, 0.97), index, f"{value:.3f}", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "final_method_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "final_method_comparison.svg", bbox_inches="tight")
    plt.close(fig)

    predictions = method_probs[champion].argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(class_names)))
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        output_dir / "champion_confusion_matrix.csv"
    )
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Champion confusion matrix: {DISPLAY_NAMES[champion]}")
    fig.tight_layout()
    fig.savefig(output_dir / "champion_confusion_matrix.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "champion_confusion_matrix.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds_by_family = {family: discover_seeds(artifact_root, family) for family in FAMILIES}
    if seeds_by_family[FAMILIES[0]] != seeds_by_family[FAMILIES[1]]:
        raise RuntimeError(f"Seed mismatch between families: {seeds_by_family}")

    lock, _, _ = select_downstream(artifact_root, output_dir, seeds_by_family)
    metrics, method_probs, labels = evaluate_locked_methods(
        artifact_root, output_dir, lock, seeds_by_family
    )

    oof_ranking = pd.read_csv(output_dir / "downstream_oof_ranking.csv")
    base_rows = oof_ranking[oof_ranking["method"].isin(FAMILIES)].sort_values("oof_rank")
    base_reference = str(base_rows.iloc[0]["method"])
    intervals, differences = bootstrap_intervals(
        method_probs,
        labels,
        lock["champion_method"],
        base_reference,
        int(args.bootstrap_resamples),
    )
    intervals.to_csv(output_dir / "test_bootstrap_intervals.csv", index=False)
    differences.to_csv(output_dir / "champion_paired_bootstrap_difference.csv", index=False)

    with (artifact_root / "configs" / "label_map.json").open("r", encoding="utf-8") as handle:
        label_map = json.load(handle)
    class_names = [str(label_map[str(index)]) for index in range(len(label_map))]
    create_plots(
        output_dir,
        metrics,
        method_probs,
        labels,
        lock["champion_method"],
        class_names,
    )
    print(metrics.to_string(index=False))
    print(f"[done] analysis: {output_dir}")


if __name__ == "__main__":
    main()
