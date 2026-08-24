"""Assign recording-grouped temporal development and optional confirmation splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hac.polar import sha256_file
from hac.vcoco_v3_temporal import TEMPORAL_CLASS_NAMES, grouped_recording_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--grid", type=Path, default=Path("experiments/vcoco_v3_temporal_grid.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def held_fold(
    indices: np.ndarray,
    labels: np.ndarray,
    recordings: np.ndarray,
    *,
    folds: int,
    fold: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    splits = grouped_recording_splits(labels[indices], recordings[indices], folds=folds, seed=seed)
    if not 0 <= fold < len(splits):
        raise ValueError("Declared split fold is outside the splitter")
    relative_fit, relative_held = splits[fold]
    return indices[relative_fit], indices[relative_held]


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata.resolve()
    grid_path = args.grid.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"Temporal split output already exists: {output_path}")
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    if grid.get("status") != "DECLARED_BEFORE_TEMPORAL_FITTING":
        raise RuntimeError("The temporal grid is not in its pre-fit state")
    frame = pd.read_csv(
        metadata_path,
        dtype={"sample_id": str, "recording_id": str, "track_id": str},
    )
    required = {
        "sample_id",
        "recording_id",
        "track_id",
        "label",
        "frame_count",
        "center_frame_index",
        "frames_per_second",
        "feature_path",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Temporal metadata is missing columns: {missing}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Temporal sample identifiers must be unique")
    unknown = sorted(set(frame["label"].astype(str)) - set(TEMPORAL_CLASS_NAMES))
    if unknown:
        raise ValueError(f"Temporal metadata contains unknown labels: {unknown}")
    mapping = {name: index for index, name in enumerate(TEMPORAL_CLASS_NAMES)}
    labels = frame["label"].map(mapping).to_numpy(dtype=int)
    recordings = frame["recording_id"].astype(str).to_numpy(dtype=str)
    policy = grid["split_policy"]
    external_confirmation = policy.get("confirmation_partition") == "provider_test"
    if external_confirmation:
        remaining = np.arange(len(frame))
        confirmation = None
    else:
        remaining, confirmation = held_fold(
            np.arange(len(frame)),
            labels,
            recordings,
            folds=int(policy["confirmation_folds"]),
            fold=int(policy["confirmation_fold"]),
            seed=int(policy["confirmation_seed"]),
        )
    remaining, calibration = held_fold(
        remaining,
        labels,
        recordings,
        folds=int(
            policy.get(
                "calibration_folds_within_remainder",
                policy.get("calibration_folds"),
            )
        ),
        fold=int(policy["calibration_fold"]),
        seed=int(policy["calibration_seed"]),
    )
    train, validation = held_fold(
        remaining,
        labels,
        recordings,
        folds=int(policy["validation_folds_within_remainder"]),
        fold=int(policy["validation_fold"]),
        seed=int(policy["validation_seed"]),
    )
    frame["split"] = ""
    assignments = [
        ("train", train),
        ("validation", validation),
        ("calibration", calibration),
    ]
    if confirmation is not None:
        assignments.append(("confirmation", confirmation))
    for name, indices in assignments:
        frame.loc[indices, "split"] = name
        if set(labels[indices]) != set(range(len(TEMPORAL_CLASS_NAMES))):
            raise RuntimeError(f"The recording-grouped {name} split does not contain every class")
    if (frame["split"] == "").any():
        raise RuntimeError("Some temporal rows were not assigned to a split")
    if frame.groupby("recording_id")["split"].nunique().max() != 1:
        raise RuntimeError("A recording crossed the generated temporal split")
    track_key = frame["recording_id"].astype(str) + "::" + frame["track_id"].astype(str)
    if frame.assign(_track=track_key).groupby("_track")["split"].nunique().max() != 1:
        raise RuntimeError("A person track crossed the generated temporal split")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    counts = (
        frame.groupby(["split", "label"], observed=True)
        .size()
        .rename("samples")
        .reset_index()
        .to_dict(orient="records")
    )
    provenance = {
        "status": "VCOCO_V3_TEMPORAL_SPLIT_ASSIGNED_BEFORE_MODEL_OUTCOMES",
        "samples": len(frame),
        "recordings": int(frame["recording_id"].nunique()),
        "tracks": int(track_key.nunique()),
        "counts": counts,
        "split_policy": policy,
        "model_outcomes_read": 0,
        "external_confirmation_partition": (
            str(policy["confirmation_partition"]) if external_confirmation else None
        ),
        "confirmation_rows_read": 0,
        "source_sha256": {
            "provider_metadata": sha256_file(metadata_path),
            "temporal_grid": sha256_file(grid_path),
        },
        "artifact_sha256": {output_path.name: sha256_file(output_path)},
    }
    provenance_path = output_path.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
