"""Finalize adjudicated V-COCO v3 labels and quantify the locked pilot mechanisms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file
from hac.vcoco_v3_audit import (
    build_harmonized_annotations,
    discover_annotation_passes,
    harmonization_estimates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agreement-dir",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/agreement"),
    )
    parser.add_argument(
        "--private-manifest",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/pilot/private_sampling_manifest.csv"),
    )
    parser.add_argument(
        "--pilot-annotation-dir",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/pilot/annotations"),
    )
    parser.add_argument(
        "--adjudication-dir",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/adjudication/annotations"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(".runs/vcoco_v3/annotation/final"))
    parser.add_argument("--adjudicator")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples < 1:
        raise ValueError("Bootstrap resamples must be positive")
    agreement_dir = args.agreement_dir.resolve()
    agreement_summary_path = agreement_dir / "summary.json"
    agreement = json.loads(agreement_summary_path.read_text(encoding="utf-8"))
    allowed = {
        "VCOCO_V3_ANNOTATION_READY_FOR_BLINDED_ADJUDICATION",
        "VCOCO_V3_ANNOTATION_AGREEMENT_GATE_PASSED_NO_DISAGREEMENTS",
    }
    if agreement.get("status") not in allowed or not agreement.get("agreement_gate_passed"):
        raise RuntimeError("The independent annotation agreement gate has not passed")

    private_path = args.private_manifest.resolve()
    private = pd.read_csv(
        private_path,
        dtype={"task_id": str, "person_id": str, "image_id": str},
        keep_default_na=False,
    )
    pilot_snapshots, _ = discover_annotation_passes(
        args.pilot_annotation_dir.resolve(), private["task_id"]
    )
    rater_ids = tuple(map(str, agreement["selected_annotators"]))
    if set(rater_ids) - set(pilot_snapshots):
        raise RuntimeError("A selected independent annotation snapshot is missing")

    adjudicator_frame = None
    adjudicator_id = None
    adjudication_manifest_path = agreement_dir / "adjudication_tasks.csv"
    adjudication_tasks = pd.read_csv(adjudication_manifest_path, dtype={"task_id": str})
    if len(adjudication_tasks):
        adjudication_snapshots, adjudication_progress = discover_annotation_passes(
            args.adjudication_dir.resolve(), adjudication_tasks["task_id"]
        )
        complete = (
            adjudication_progress.loc[
                adjudication_progress["complete"].astype(bool), "annotator_id"
            ]
            .astype(str)
            .tolist()
        )
        if args.adjudicator:
            if args.adjudicator not in complete:
                raise RuntimeError("The requested adjudication pass is not complete")
            adjudicator_id = args.adjudicator
        elif len(complete) == 1:
            adjudicator_id = complete[0]
        elif not complete:
            raise RuntimeError("The blinded adjudication pass is incomplete")
        else:
            raise RuntimeError("Choose one predeclared adjudicator with --adjudicator")
        if adjudicator_id in rater_ids:
            raise RuntimeError("The adjudication pass must use a separate annotator ID")
        adjudicator_frame = adjudication_snapshots[adjudicator_id]

    harmonized = build_harmonized_annotations(
        private,
        pilot_snapshots,
        (rater_ids[0], rater_ids[1]),
        adjudicator_frame,
    )
    estimates = harmonization_estimates(
        harmonized,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    harmonized_path = output_dir / "private_harmonized_annotations.csv"
    harmonized.to_csv(harmonized_path, index=False)
    summary = {
        "status": "VCOCO_V3_HUMAN_ANNOTATION_GATE_PASSED",
        "selected_independent_annotators": list(rater_ids),
        "adjudicator": adjudicator_id,
        "unique_people": len(harmonized),
        "adjudicated_people": int(harmonized["adjudicated_axes"].ne("").sum()),
        "bootstrap_unit": "source_image",
        "bootstrap_resamples": args.bootstrap_resamples,
        "probability_cohort_estimates": estimates,
        "interpretation_policy": "descriptive_mechanism_audit_not_model_selection",
        "source_sha256": {
            "agreement_summary": sha256_file(agreement_summary_path),
            "private_sampling_manifest": sha256_file(private_path),
            "adjudication_manifest": sha256_file(adjudication_manifest_path),
        },
        "artifact_sha256": {harmonized_path.name: sha256_file(harmonized_path)},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
