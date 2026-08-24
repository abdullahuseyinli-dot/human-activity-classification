"""Audit two complete blind annotation passes and prepare blind adjudication tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hac.polar import sha256_file
from hac.vcoco_v3 import load_protocol_spec
from hac.vcoco_v3_annotation import BLIND_COLUMNS
from hac.vcoco_v3_audit import (
    agreement_tables,
    discover_annotation_passes,
    make_adjudication_manifest,
    select_complete_raters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("experiments/vcoco_v3_protocol.json"))
    parser.add_argument(
        "--blind-manifest",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/pilot/blind_tasks.csv"),
    )
    parser.add_argument(
        "--private-manifest",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/pilot/private_sampling_manifest.csv"),
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/pilot/annotations"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/agreement"),
    )
    parser.add_argument("--annotator", action="append", default=[])
    parser.add_argument("--adjudication-seed", type=int, default=20260825)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    spec = load_protocol_spec(args.spec.resolve())
    gate = spec["ontology"]["annotation_gate"]
    blind_path = args.blind_manifest.resolve()
    private_path = args.private_manifest.resolve()
    blind = pd.read_csv(blind_path, dtype={"task_id": str})
    private = pd.read_csv(
        private_path,
        dtype={"task_id": str, "person_id": str, "image_id": str},
        keep_default_na=False,
    )
    if tuple(blind.columns) != BLIND_COLUMNS:
        raise RuntimeError("The blind manifest contains unexpected fields")
    if set(blind["task_id"]) != set(private["task_id"]):
        raise RuntimeError("Blind and private task manifests differ")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots, progress = discover_annotation_passes(
        args.annotation_dir.resolve(), blind["task_id"]
    )
    progress_path = output_dir / "annotation_progress.csv"
    progress.to_csv(progress_path, index=False)
    required = int(gate["minimum_independent_annotators"])
    rater_ids = select_complete_raters(snapshots, progress, args.annotator, required=required)
    base = {
        "required_independent_annotators": required,
        "complete_independent_annotators": int(progress["complete"].sum()) if len(progress) else 0,
        "task_count_per_rater": len(blind),
        "source_labels_or_predictions_exposed": False,
        "source_sha256": {
            "protocol_spec": sha256_file(args.spec.resolve()),
            "blind_manifest": sha256_file(blind_path),
            "private_manifest": sha256_file(private_path),
        },
    }
    if rater_ids is None:
        write_json(
            output_dir / "summary.json",
            {**base, "status": "VCOCO_V3_ANNOTATION_WAITING_FOR_COMPLETE_INDEPENDENT_PASSES"},
        )
        print(json.dumps({**base, "status": "WAITING_FOR_ANNOTATORS"}, indent=2))
        return
    if len(rater_ids) != 2:
        raise RuntimeError("This pilot analysis requires exactly two independent raters")

    inter, intra, disagreements = agreement_tables(private, snapshots, (rater_ids[0], rater_ids[1]))
    inter_path = output_dir / "interrater_agreement.csv"
    intra_path = output_dir / "intrarater_repeat_agreement.csv"
    disagreements_path = output_dir / "private_disagreements.csv"
    inter.to_csv(inter_path, index=False)
    intra.to_csv(intra_path, index=False)
    disagreements.to_csv(disagreements_path, index=False)

    minimum_alpha = float(gate["minimum_alpha_each_required_axis"])
    minimum_intra = float(gate["minimum_intrarater_exact_agreement_each_axis"])
    alpha_failed = (
        inter.loc[
            ~np.isfinite(inter["krippendorff_alpha_nominal"])
            | inter["krippendorff_alpha_nominal"].lt(minimum_alpha),
            "axis",
        ]
        .astype(str)
        .tolist()
    )
    intra_failed = intra.loc[
        ~np.isfinite(intra["exact_agreement"]) | intra["exact_agreement"].lt(minimum_intra),
        ["annotator_id", "axis"],
    ]
    gate_passed = not alpha_failed and intra_failed.empty

    adjudication_path = output_dir / "adjudication_tasks.csv"
    if gate_passed and len(disagreements):
        adjudication = make_adjudication_manifest(
            blind,
            disagreements["task_id"],
            seed=args.adjudication_seed,
        )
        adjudication.to_csv(adjudication_path, index=False)
        status = "VCOCO_V3_ANNOTATION_READY_FOR_BLINDED_ADJUDICATION"
    elif gate_passed:
        blind.iloc[:0].to_csv(adjudication_path, index=False)
        status = "VCOCO_V3_ANNOTATION_AGREEMENT_GATE_PASSED_NO_DISAGREEMENTS"
    else:
        status = "VCOCO_V3_ANNOTATION_GUIDE_REVISION_REQUIRED"

    summary = {
        **base,
        "status": status,
        "selected_annotators": list(rater_ids),
        "unique_people_compared": int(private["repeat_of_task_id"].eq("").sum()),
        "hidden_repeat_pairs_per_rater": int(private["repeat_of_task_id"].ne("").sum()),
        "items_requiring_adjudication": len(disagreements),
        "minimum_alpha_each_required_axis": minimum_alpha,
        "minimum_intrarater_exact_agreement_each_axis": minimum_intra,
        "agreement_gate_passed": gate_passed,
        "failed_alpha_axes": alpha_failed,
        "failed_intrarater_cells": intra_failed.to_dict(orient="records"),
        "artifact_sha256": {
            inter_path.name: sha256_file(inter_path),
            intra_path.name: sha256_file(intra_path),
            disagreements_path.name: sha256_file(disagreements_path),
            **(
                {adjudication_path.name: sha256_file(adjudication_path)}
                if adjudication_path.is_file()
                else {}
            ),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
