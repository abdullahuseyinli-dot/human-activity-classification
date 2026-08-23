"""Average aligned POLAR validation probabilities across locked confirmation seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_analysis import align_prediction_artifacts, load_prediction_artifact
from hac.polar_training import normalize_probability_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.artifact) != len(args.seed) or len(args.artifact) < 2:
        raise ValueError("Supply aligned artifact and seed lists with at least two entries")
    if len(set(args.seed)) != len(args.seed):
        raise ValueError("Seed identifiers must be unique")
    paths = [path.resolve() for path in args.artifact]
    aligned = align_prediction_artifacts(
        {
            f"seed_{seed}": load_prediction_artifact(path)
            for seed, path in zip(args.seed, paths, strict=True)
        }
    )
    reference = next(iter(aligned.values()))
    averaged = normalize_probability_rows(
        np.mean([artifact.probabilities for artifact in aligned.values()], axis=0)
    )
    seed_rows = [
        {
            "candidate": args.candidate,
            "seed": seed,
            **classification_metrics(reference.labels, aligned[f"seed_{seed}"].probabilities),
        }
        for seed in args.seed
    ]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        probabilities=averaged,
        labels=reference.labels,
        image_ids=reference.image_ids,
        class_names=np.asarray(reference.class_names),
    )
    pd.DataFrame(seed_rows).sort_values("seed", ignore_index=True).to_csv(
        output_dir / "seed_metrics.csv", index=False
    )
    aggregate_metrics = classification_metrics(reference.labels, averaged)
    provenance = {
        "status": "DEVELOPMENT_ONLY_MULTI_SEED_AGGREGATE",
        "candidate": args.candidate,
        "seeds": args.seed,
        "artifact_sha256": {
            str(seed): sha256_file(path)
            for seed, path in zip(args.seed, paths, strict=True)
        },
        "aggregation": "arithmetic_mean_probabilities",
        "rows": len(reference.labels),
        "metrics": aggregate_metrics,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "test_rows_read": 0,
        "test_used_for_selection": False,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
