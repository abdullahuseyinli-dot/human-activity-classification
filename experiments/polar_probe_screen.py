"""Screen frozen POLAR representations and declared feature fusions on validation."""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hac.metrics import classification_metrics
from hac.polar import sha256_file

LABEL_ORDERS = {
    "label_4": ["sitting", "standing", "walking", "running"],
    "label_3": ["sitting", "standing", "walking_running"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-fusions", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--c-value",
        action="append",
        type=float,
        dest="c_values",
        help="Regularization value; repeat to override the default grid.",
    )
    parser.add_argument("--task", choices=sorted(LABEL_ORDERS), action="append", dest="tasks")
    parser.add_argument(
        "--candidate-contains",
        action="append",
        dest="candidate_filters",
        help="Retain candidates containing at least one supplied substring.",
    )
    return parser.parse_args()


def load_caches(root: Path, manifest_hash: str) -> dict[str, dict]:
    caches = {}
    for provenance_path in sorted(root.rglob("provenance.json")):
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("manifest_sha256") != manifest_hash:
            raise RuntimeError(f"Feature cache manifest drift: {provenance_path}")
        cache_dir = provenance_path.parent
        rows = pd.read_csv(cache_dir / "rows.csv", dtype={"image_id": str})
        features = np.load(cache_dir / "features.npy")
        if len(rows) != len(features):
            raise RuntimeError(f"Feature and metadata rows differ: {cache_dir}")
        name = f"{provenance['model_kind']}__{provenance['view']}"
        caches[name] = {"features": features, "rows": rows, "provenance": provenance}
    if not caches:
        raise FileNotFoundError(f"No feature caches found below {root}")
    return caches


def align_cache(cache: dict, manifest: pd.DataFrame) -> np.ndarray:
    rows = cache["rows"]
    if rows["image_id"].tolist() == manifest["image_id"].tolist():
        return cache["features"]
    index_by_id = {value: index for index, value in enumerate(rows["image_id"].astype(str))}
    try:
        indices = [index_by_id[value] for value in manifest["image_id"].astype(str)]
    except KeyError as error:
        raise RuntimeError(f"Feature cache is missing manifest image {error}") from error
    return cache["features"][indices]


def candidate_features(
    caches: dict[str, dict], manifest: pd.DataFrame, include_fusions: bool
) -> dict[str, np.ndarray]:
    aligned = {name: align_cache(cache, manifest) for name, cache in caches.items()}
    candidates = dict(aligned)
    if not include_fusions:
        return candidates

    names = sorted(aligned)
    for left, right in combinations(names, 2):
        left_model, left_view = left.split("__", maxsplit=1)
        right_model, right_view = right.split("__", maxsplit=1)
        same_model = left_model == right_model
        same_view = left_view == right_view
        if same_model or same_view:
            name = f"fusion__{left}__{right}"
            candidates[name] = np.concatenate([aligned[left], aligned[right]], axis=1)
    return candidates


def fit_candidate(
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    c_value: float,
    class_weight: str | None,
    seed: int,
) -> tuple[dict, np.ndarray]:
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    class_weight=class_weight,
                    max_iter=2000,
                    random_state=int(seed),
                    solver="lbfgs",
                    tol=1e-5,
                ),
            ),
        ]
    )
    started = time.perf_counter()
    pipeline.fit(features[train_mask], labels[train_mask])
    probabilities = pipeline.predict_proba(features[val_mask])
    elapsed = time.perf_counter() - started
    metrics = classification_metrics(labels[val_mask], probabilities)
    metrics["fit_seconds"] = float(elapsed)
    metrics["iterations"] = int(np.max(pipeline.named_steps["classifier"].n_iter_))
    metrics["converged"] = metrics["iterations"] < 2000
    return metrics, probabilities


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = sha256_file(args.manifest)
    manifest = pd.read_csv(args.manifest, dtype={"image_id": str})
    manifest = manifest.sort_values("image_id").reset_index(drop=True)
    if set(manifest["split"].astype(str)) != {"train", "val"}:
        raise ValueError("Probe screen requires train and validation only")
    caches = load_caches(args.feature_root.resolve(), manifest_hash)
    candidates = candidate_features(caches, manifest, args.include_fusions)
    if args.candidate_filters:
        candidates = {
            name: features
            for name, features in candidates.items()
            if any(value in name for value in args.candidate_filters)
        }
    if not candidates:
        raise ValueError("Candidate filters removed every representation")
    train_mask = manifest["split"].astype(str).eq("train").to_numpy()
    val_mask = manifest["split"].astype(str).eq("val").to_numpy()

    rows = []
    best_predictions: dict[str, tuple[tuple, dict]] = {}
    tasks = args.tasks or list(LABEL_ORDERS)
    for task in tasks:
        class_names = LABEL_ORDERS[task]
        label_to_index = {label: index for index, label in enumerate(class_names)}
        labels = manifest[task].map(label_to_index).to_numpy(dtype=int)
        for candidate_name, features in candidates.items():
            for c_value in args.c_values or (0.01, 0.1, 1.0, 10.0, 100.0):
                for class_weight in (None, "balanced"):
                    metrics, probabilities = fit_candidate(
                        features,
                        labels,
                        train_mask,
                        val_mask,
                        c_value=c_value,
                        class_weight=class_weight,
                        seed=args.seed,
                    )
                    row = {
                        "task": task,
                        "candidate": candidate_name,
                        "classifier": "standardized_multinomial_logistic",
                        "C": c_value,
                        "class_weight": class_weight or "none",
                        "feature_dimensions": features.shape[1],
                        "train_rows": int(train_mask.sum()),
                        "validation_rows": int(val_mask.sum()),
                        **metrics,
                    }
                    rows.append(row)
                    rank = (-metrics["macro_f1"], metrics["log_loss"], metrics["ece"])
                    key = f"{task}::{candidate_name}"
                    if key not in best_predictions or rank < best_predictions[key][0]:
                        best_predictions[key] = (
                            rank,
                            {
                                "row": row,
                                "probabilities": probabilities,
                                "labels": labels[val_mask],
                                "image_ids": manifest.loc[val_mask, "image_id"].to_numpy(),
                                "class_names": np.asarray(class_names),
                            },
                        )
                    print(
                        f"[{task}] {candidate_name} C={c_value:g} weight={class_weight or 'none'} "
                        f"macro_f1={metrics['macro_f1']:.4f}",
                        flush=True,
                    )

    results = pd.DataFrame(rows).sort_values(
        ["task", "macro_f1", "log_loss", "candidate"],
        ascending=[True, False, True, True],
        ignore_index=True,
    )
    results.to_csv(output_dir / "probe_screen.csv", index=False)
    selected_rows = []
    for key, (_, payload) in best_predictions.items():
        task, candidate_name = key.split("::", maxsplit=1)
        selected_rows.append(payload["row"])
        np.savez_compressed(
            output_dir / f"validation_{task}_{candidate_name.replace('__', '_')}.npz",
            probabilities=payload["probabilities"],
            labels=payload["labels"],
            image_ids=payload["image_ids"],
            class_names=payload["class_names"],
        )
    selected = pd.DataFrame(selected_rows).sort_values(
        ["task", "macro_f1", "log_loss"], ascending=[True, False, True]
    )
    selected.to_csv(output_dir / "probe_best_per_representation.csv", index=False)
    lock = {
        "status": "DEVELOPMENT_ONLY_PROBE_SCREEN",
        "manifest_sha256": manifest_hash,
        "seed": args.seed,
        "feature_caches": {
            name: cache["provenance"] for name, cache in sorted(caches.items())
        },
        "tasks": LABEL_ORDERS,
        "executed_tasks": tasks,
        "candidate_filters": args.candidate_filters or [],
        "c_values": args.c_values or [0.01, 0.1, 1.0, 10.0, 100.0],
        "test_rows_read": 0,
        "test_used_for_selection": False,
    }
    (output_dir / "probe_provenance.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[done] {output_dir / 'probe_screen.csv'}", flush=True)


if __name__ == "__main__":
    main()
