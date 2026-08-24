"""Create a prediction-blind V-COCO v3 annotation pilot with hidden sampling evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hac.polar import sha256_file
from hac.vcoco_v3_annotation import create_pilot_tasks, prepare_prediction_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".runs/polar_v2/final_test_evaluation_bound/opened_test_manifest.csv"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(".runs/polar_v2/final_test_evaluation_bound/test_predictions.npz"),
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/protocol/vcoco_v3_lock.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(".runs/vcoco_v3/annotation/pilot"))
    parser.add_argument("--probability-tasks", type=int, default=180)
    parser.add_argument("--error-tasks", type=int, default=100)
    parser.add_argument("--repeat-tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    predictions_path = args.predictions.resolve()
    protocol_lock_path = args.protocol_lock.resolve()
    output_dir = args.output_dir.resolve()
    if min(args.probability_tasks, args.error_tasks, args.repeat_tasks) < 0:
        raise ValueError("Pilot task counts cannot be negative")
    protocol = json.loads(protocol_lock_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "VCOCO_V3_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("The v3 protocol must be locked before pilot sampling")

    manifest = pd.read_csv(
        manifest_path, dtype={"person_id": str, "image_id": str, "annotation_id": str}
    )
    with np.load(predictions_path, allow_pickle=True) as predictions:
        rows = prepare_prediction_frame(manifest, predictions)
    blind, private = create_pilot_tasks(
        rows,
        probability_tasks=args.probability_tasks,
        error_tasks=args.error_tasks,
        repeat_tasks=args.repeat_tasks,
        seed=args.seed,
    )
    if not blind["image_path"].map(lambda value: Path(str(value)).is_file()).all():
        raise FileNotFoundError("One or more pilot images are unavailable")

    output_dir.mkdir(parents=True, exist_ok=True)
    blind_path = output_dir / "blind_tasks.csv"
    private_path = output_dir / "private_sampling_manifest.csv"
    blind.to_csv(blind_path, index=False)
    private.to_csv(private_path, index=False)
    summary = {
        "status": "VCOCO_V3_BLINDED_ANNOTATION_PILOT_READY",
        "guide_version": "v3-pilot-1",
        "tasks": len(blind),
        "unique_people": int(private["person_id"].nunique()),
        "unique_images": int(private["image_id"].nunique()),
        "cohort_counts": {
            str(key): int(value)
            for key, value in private["cohort"].value_counts().sort_index().items()
        },
        "source_labels_or_predictions_in_blind_manifest": False,
        "seed": args.seed,
        "protocol_lock_sha256": sha256_file(protocol_lock_path),
        "source_sha256": {
            "opened_v2_test_manifest": sha256_file(manifest_path),
            "locked_v2_predictions": sha256_file(predictions_path),
        },
        "artifact_sha256": {
            "blind_tasks.csv": sha256_file(blind_path),
            "private_sampling_manifest.csv": sha256_file(private_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
