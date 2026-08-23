"""Run the predeclared POLAR learning curve for a frozen multi-view representation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_features import load_aligned_feature_view
from hac.polar_training import TASK_LABELS, nested_stratified_subset, validate_development_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-kind", required=True)
    parser.add_argument("--view", action="append", required=True, dest="views")
    parser.add_argument("--task", choices=sorted(TASK_LABELS), default="label_4")
    parser.add_argument("--c-value", type=float, default=0.01)
    parser.add_argument(
        "--class-weight", choices=["none", "balanced"], default="none"
    )
    parser.add_argument(
        "--train-size", action="append", default=[], help="Integer or 'all'; repeat as needed."
    )
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--subset-seed", type=int, default=20_260_822)
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
    return value


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_sizes(values: list[str], maximum: int) -> list[tuple[str, int | None]]:
    requested = values or ["242", "500", "1000", "3000", "all"]
    output = []
    for value in requested:
        if value == "all":
            output.append((value, None))
            continue
        size = int(value)
        if size < 1 or size > maximum:
            raise ValueError(f"Invalid train size {size}; clean training rows={maximum}")
        output.append((value, size))
    if len({value for value, _ in output}) != len(output):
        raise ValueError("Train sizes must be unique")
    return output


def main() -> None:
    args = parse_args()
    if len(set(args.views)) != len(args.views):
        raise ValueError("Views must be unique")
    if args.c_value <= 0.0:
        raise ValueError("c-value must be positive")
    seeds = args.seed or [42, 52, 62]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")

    manifest_path = args.manifest.resolve()
    manifest_hash = sha256_file(manifest_path)
    manifest = validate_development_manifest(
        pd.read_csv(manifest_path, dtype={"image_id": str}), args.task
    )
    train_frame = manifest[manifest["split"].eq("train")].copy()
    validation_frame = manifest[manifest["split"].eq("val")].copy()
    sizes = parse_sizes(args.train_size, len(train_frame))
    class_names = list(TASK_LABELS[args.task])
    class_to_index = {name: index for index, name in enumerate(class_names)}
    labels = manifest["label"].map(class_to_index).to_numpy(dtype=int)
    id_to_row = {value: index for index, value in enumerate(manifest["image_id"].astype(str))}
    validation_indices = np.asarray(
        [id_to_row[value] for value in validation_frame["image_id"].astype(str)], dtype=int
    )

    feature_views = []
    cache_provenance = {}
    for view in args.views:
        features, provenance = load_aligned_feature_view(
            args.feature_root.resolve(), args.model_kind, view, manifest, manifest_hash
        )
        feature_views.append(features)
        cache_provenance[view] = provenance
    features = np.concatenate(feature_views, axis=1)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for requested_name, requested_size in sizes:
        subset = nested_stratified_subset(
            train_frame,
            requested_size,
            label_column="label",
            seed=args.subset_seed,
        )
        train_indices = np.asarray(
            [id_to_row[value] for value in subset["image_id"].astype(str)], dtype=int
        )
        subset_hash = hashlib.sha256(
            "\n".join(subset["image_id"].astype(str)).encode("utf-8")
        ).hexdigest()
        for seed in seeds:
            pipeline = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=args.c_value,
                            class_weight=(
                                None if args.class_weight == "none" else args.class_weight
                            ),
                            max_iter=2_000,
                            random_state=seed,
                            solver="lbfgs",
                            tol=1e-5,
                        ),
                    ),
                ]
            )
            started = time.perf_counter()
            pipeline.fit(features[train_indices], labels[train_indices])
            probabilities = pipeline.predict_proba(features[validation_indices])
            fit_seconds = time.perf_counter() - started
            metrics = classification_metrics(labels[validation_indices], probabilities)
            iterations = int(np.max(pipeline.named_steps["classifier"].n_iter_))
            run_id = f"size_{requested_name}_seed_{seed}"
            np.savez_compressed(
                output_dir / f"{run_id}_predictions.npz",
                probabilities=probabilities,
                labels=labels[validation_indices],
                image_ids=validation_frame["image_id"].to_numpy(),
                class_names=np.asarray(class_names),
            )
            records.append(
                {
                    "run_id": run_id,
                    "requested_train_size": requested_name,
                    "actual_train_size": len(subset),
                    "seed": seed,
                    "subset_image_ids_sha256": subset_hash,
                    "feature_dimensions": features.shape[1],
                    "fit_seconds": fit_seconds,
                    "iterations": iterations,
                    "converged": iterations < 2_000,
                    **metrics,
                    "test_rows_read": 0,
                }
            )
            print(
                f"[{run_id}] macro_f1={metrics['macro_f1']:.4f} "
                f"log_loss={metrics['log_loss']:.4f}",
                flush=True,
            )

    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "scale_runs.csv", index=False)
    aggregate = (
        frame.groupby(["requested_train_size", "actual_train_size"], sort=False)
        .agg(
            seeds=("seed", "count"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
            ece_mean=("ece", "mean"),
            fit_seconds_mean=("fit_seconds", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(output_dir / "scale_summary.csv", index=False)
    write_json(
        output_dir / "provenance.json",
        {
            "status": "DEVELOPMENT_ONLY_SCALE_STUDY",
            "architecture": {
                "representation": args.model_kind,
                "views": args.views,
                "classifier": "standardized_multinomial_logistic",
                "C": args.c_value,
                "class_weight": args.class_weight,
            },
            "task": args.task,
            "manifest_sha256": manifest_hash,
            "feature_cache_provenance": cache_provenance,
            "subset_seed": args.subset_seed,
            "training_seeds": seeds,
            "requested_train_sizes": [name for name, _ in sizes],
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "test_rows_read": 0,
            "test_used_for_selection": False,
        },
    )
    print(aggregate.to_string(index=False), flush=True)
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
