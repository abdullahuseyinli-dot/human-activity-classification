"""Measure image-grouped V-COCO target-label scaling on a locked frozen representation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_training import normalize_probability_rows

CLASS_NAMES = ("sitting", "standing", "walking_running")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets", default="1,5,10,25,50,100")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_260_823)
    parser.add_argument("--C", dest="c_value", type=float, required=True)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    return parser.parse_args()


def parse_budgets(value: str) -> list[int]:
    budgets = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not budgets or budgets[0] < 1:
        raise ValueError("Few-shot budgets must be positive")
    return budgets


def labels(rows: pd.DataFrame) -> np.ndarray:
    output = rows["label_3"].map({name: index for index, name in enumerate(CLASS_NAMES)})
    if output.isna().any():
        raise RuntimeError("Unknown class label in feature cache")
    return output.to_numpy(dtype=int)


def sample_image_groups(
    train_rows: pd.DataFrame,
    train_labels: np.ndarray,
    budget: int,
    generator: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    groups = train_rows["image_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    order = generator.permutation(unique_groups)
    counts = np.zeros(len(CLASS_NAMES), dtype=int)
    selected_groups = []
    remaining = list(order)
    while (counts < budget).any():
        deficits = counts < budget
        candidates = []
        for group in remaining:
            indices = group_indices[str(group)]
            contribution = int(np.minimum(np.bincount(train_labels[indices], minlength=3), deficits).sum())
            if contribution:
                candidates.append((contribution, generator.random(), str(group)))
        if not candidates:
            raise RuntimeError(f"Could not satisfy {budget}-shot class targets by complete image groups")
        _, _, chosen = max(candidates)
        selected_groups.append(chosen)
        chosen_indices = group_indices[chosen]
        counts += np.bincount(train_labels[chosen_indices], minlength=len(CLASS_NAMES))
        remaining.remove(chosen)
    mask = np.isin(groups, selected_groups)
    selected = np.flatnonzero(mask)
    return selected, {
        "selected_images": len(selected_groups),
        "selected_people": len(selected),
        **{f"selected_{name}": int(counts[index]) for index, name in enumerate(CLASS_NAMES)},
    }


def build_model(c_value: float, class_weight: str, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced" if class_weight == "balanced" else None,
            max_iter=3_000,
            random_state=seed,
            solver="lbfgs",
        ),
    )


def main() -> None:
    args = parse_args()
    budgets = parse_budgets(args.budgets)
    if args.repeats < 2 or args.c_value <= 0.0:
        raise ValueError("At least two repeats and a positive C are required")
    lock_path = args.protocol_lock.resolve()
    protocol_hash = sha256_file(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("Few-shot evaluation requires the locked V-COCO v2 protocol")
    cache = args.cache.resolve()
    provenance = json.loads((cache / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("protocol_lock_sha256") != protocol_hash:
        raise RuntimeError("Feature cache protocol drift")
    if provenance.get("test_rows_read") != 0 or provenance.get("test_predictions_run"):
        raise RuntimeError("Feature cache violated the target-test gate")
    rows_path = cache / "rows.csv"
    features_path = cache / "features.npy"
    if sha256_file(rows_path) != provenance["artifact_sha256"]["rows.csv"]:
        raise RuntimeError("Feature row drift")
    if sha256_file(features_path) != provenance["artifact_sha256"]["features.npy"]:
        raise RuntimeError("Feature array drift")
    rows = pd.read_csv(rows_path, dtype={"person_id": str, "image_id": str})
    features = np.asarray(np.load(features_path, mmap_mode="r"), dtype=np.float32)
    if len(rows) != len(features):
        raise RuntimeError("Feature rows and array do not align")
    train_mask = rows["split"].eq("train").to_numpy()
    val_mask = rows["split"].eq("val").to_numpy()
    train_rows = rows[train_mask].reset_index(drop=True)
    train_features = features[train_mask]
    train_labels = labels(train_rows)
    val_rows = rows[val_mask].reset_index(drop=True)
    val_features = features[val_mask]
    val_labels = labels(val_rows)

    run_rows = []
    class_rows = []
    for budget in budgets:
        for repeat in range(args.repeats):
            run_seed = args.seed + budget * 10_000 + repeat
            generator = np.random.default_rng(run_seed)
            selected, sample_details = sample_image_groups(
                train_rows, train_labels, budget, generator
            )
            model = build_model(args.c_value, args.class_weight, run_seed)
            model.fit(train_features[selected], train_labels[selected])
            probabilities = normalize_probability_rows(model.predict_proba(val_features))
            metrics = classification_metrics(val_labels, probabilities)
            run_rows.append(
                {
                    "budget_people_per_class_minimum": budget,
                    "repeat": repeat,
                    "seed": run_seed,
                    **sample_details,
                    **metrics,
                }
            )
            precision, recall, f1, support = precision_recall_fscore_support(
                val_labels,
                probabilities.argmax(axis=1),
                labels=np.arange(len(CLASS_NAMES)),
                zero_division=0,
            )
            class_rows.extend(
                {
                    "budget_people_per_class_minimum": budget,
                    "repeat": repeat,
                    "class": name,
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
                for index, name in enumerate(CLASS_NAMES)
            )

    runs = pd.DataFrame(run_rows)
    aggregate_rows = []
    for budget, frame in runs.groupby("budget_people_per_class_minimum", sort=True):
        aggregate_rows.append(
            {
                "budget_people_per_class_minimum": int(budget),
                "repeats": len(frame),
                "macro_f1_mean": float(frame["macro_f1"].mean()),
                "macro_f1_std": float(frame["macro_f1"].std(ddof=1)),
                "macro_f1_repeat_interval_low": float(frame["macro_f1"].quantile(0.025)),
                "macro_f1_repeat_interval_high": float(frame["macro_f1"].quantile(0.975)),
                "balanced_accuracy_mean": float(frame["balanced_accuracy"].mean()),
                "accuracy_mean": float(frame["accuracy"].mean()),
                "selected_images_mean": float(frame["selected_images"].mean()),
                "selected_people_mean": float(frame["selected_people"].mean()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / "fewshot_runs.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output_dir / "fewshot_per_class.csv", index=False)
    aggregate.to_csv(output_dir / "fewshot_summary.csv", index=False)
    result = {
        "status": "VCOCO_V2_IMAGE_GROUPED_FEWSHOT_DEVELOPMENT_COMPLETE",
        "model_kind": provenance["model_kind"],
        "view": provenance["view"],
        "preprocess": provenance["preprocess"],
        "C": args.c_value,
        "class_weight": args.class_weight,
        "budgets": budgets,
        "repeats": args.repeats,
        "sampling_unit": "complete_image_group",
        "interval_scope": "empirical_across_repeated_label_budget_draws",
        "protocol_lock_sha256": protocol_hash,
        "test_rows_read": 0,
        "test_predictions_run": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
