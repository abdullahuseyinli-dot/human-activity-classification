"""Analyze validation-only POLAR predictions, uncertainty, and model complementarity."""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_analysis import (
    PredictionArtifact,
    align_prediction_artifacts,
    complementarity_metrics,
    confusion_metrics,
    load_prediction_artifact,
    per_class_metrics,
    select_probability_blend,
    stratified_paired_bootstrap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named validation prediction artifact; repeat for each model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blend-step", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_822)
    return parser.parse_args()


def parse_artifacts(values: list[str]) -> tuple[dict[str, Path], dict[str, PredictionArtifact]]:
    paths: dict[str, Path] = {}
    for value in values:
        name, separator, path_value = value.partition("=")
        if not separator or not re.fullmatch(r"[a-z0-9_]+", name):
            raise ValueError(f"Expected NAME=PATH with a snake-case name, found {value!r}")
        if name in paths:
            raise ValueError(f"Duplicate artifact name: {name}")
        path = Path(path_value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[name] = path
    if len(paths) < 2:
        raise ValueError("Validation analysis requires at least two artifacts")
    return paths, align_prediction_artifacts(
        {name: load_prediction_artifact(path) for name, path in paths.items()}
    )


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


def main() -> None:
    args = parse_args()
    paths, artifacts = parse_artifacts(args.artifact)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = next(iter(artifacts.values()))

    metric_rows = []
    per_class_rows = []
    confusions = {}
    uncertainty = {}
    for name, artifact in artifacts.items():
        metrics = classification_metrics(artifact.labels, artifact.probabilities)
        metric_rows.append({"candidate": name, "candidate_type": "single", **metrics})
        per_class_rows.extend(
            {"candidate": name, **record}
            for record in per_class_metrics(
                artifact.labels, artifact.probabilities, artifact.class_names
            )
        )
        confusions[name] = confusion_metrics(
            artifact.labels, artifact.probabilities, artifact.class_names
        )
        uncertainty[name] = stratified_paired_bootstrap(
            artifact.labels,
            artifact.probabilities,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )

    pair_rows = []
    pair_blends = {}
    for left_name, right_name in combinations(sorted(artifacts), 2):
        left = artifacts[left_name]
        right = artifacts[right_name]
        record = complementarity_metrics(
            reference.labels, left.probabilities, right.probabilities
        )
        pair_rows.append({"left": left_name, "right": right_name, **record})
        weights, metrics, _ = select_probability_blend(
            {left_name: left, right_name: right}, step=args.blend_step
        )
        pair_blends[f"{left_name}__{right_name}"] = {"weights": weights, "metrics": metrics}

    weights, blend_metrics, blend_probabilities = select_probability_blend(
        artifacts, step=args.blend_step
    )
    blend_name = "validation_locked_blend"
    metric_rows.append({"candidate": blend_name, "candidate_type": "blend", **blend_metrics})
    per_class_rows.extend(
        {"candidate": blend_name, **record}
        for record in per_class_metrics(
            reference.labels, blend_probabilities, reference.class_names
        )
    )
    confusions[blend_name] = confusion_metrics(
        reference.labels, blend_probabilities, reference.class_names
    )
    uncertainty[blend_name] = stratified_paired_bootstrap(
        reference.labels,
        blend_probabilities,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    uncertainty[f"{blend_name}_paired_deltas"] = {
        name: stratified_paired_bootstrap(
            reference.labels,
            blend_probabilities,
            artifact.probabilities,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )
        for name, artifact in artifacts.items()
    }

    metric_frame = pd.DataFrame(metric_rows).sort_values(
        ["macro_f1", "log_loss", "candidate"],
        ascending=[False, True, True],
        ignore_index=True,
    )
    metric_frame.to_csv(output_dir / "validation_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).sort_values(
        ["candidate", "class"], ignore_index=True
    ).to_csv(output_dir / "validation_per_class.csv", index=False)
    pd.DataFrame(pair_rows).sort_values(["left", "right"], ignore_index=True).to_csv(
        output_dir / "validation_complementarity.csv", index=False
    )
    write_json(output_dir / "validation_confusions.json", confusions)
    write_json(output_dir / "validation_uncertainty.json", uncertainty)
    write_json(
        output_dir / "validation_blend.json",
        {
            "status": "DEVELOPMENT_ONLY_VALIDATION_BLEND",
            "selection_order": ["macro_f1_desc", "log_loss_asc", "ece_asc"],
            "grid_step": args.blend_step,
            "weights": weights,
            "metrics": blend_metrics,
            "pair_blends": pair_blends,
            "test_rows_read": 0,
            "test_used_for_selection": False,
        },
    )
    np.savez_compressed(
        output_dir / "validation_blend_predictions.npz",
        probabilities=blend_probabilities,
        labels=reference.labels,
        image_ids=reference.image_ids,
        class_names=np.asarray(reference.class_names),
    )
    write_json(
        output_dir / "provenance.json",
        {
            "status": "DEVELOPMENT_ONLY_VALIDATION_ANALYSIS",
            "artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
            "artifact_rows": {name: len(artifact.labels) for name, artifact in artifacts.items()},
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "test_rows_read": 0,
            "test_used_for_selection": False,
        },
    )
    print(metric_frame.to_string(index=False), flush=True)
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
