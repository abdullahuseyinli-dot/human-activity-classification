"""Freeze and audit the fixed 130-presentation V-COCO v3 human pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file
from hac.vcoco_v3 import load_protocol_spec
from hac.vcoco_v3_annotation import BLIND_COLUMNS
from hac.vcoco_v3_audit import load_annotation_snapshot, single_rater_pilot_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, default=Path("experiments/vcoco_v3_protocol.json")
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/pilot/annotations/1.json"),
    )
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
        "--pilot-build-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/pilot/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/final"),
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def fixed_prefix(
    private: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    task_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select a complete task-manifest prefix without using labels or predictions."""

    display_order = pd.to_numeric(private["display_order"], errors="raise").astype(int)
    if display_order.duplicated().any():
        raise RuntimeError("Pilot display order is not unique")
    expected_orders = set(range(1, task_count + 1))
    primary = private.loc[display_order.isin(expected_orders)].copy()
    if len(primary) != task_count or set(primary["display_order"].astype(int)) != expected_orders:
        raise RuntimeError("The requested fixed task prefix is incomplete")

    primary_ids = set(primary["task_id"].astype(str))
    observed_ids = set(ratings["task_id"].astype(str))
    missing = primary_ids - observed_ids
    if missing:
        raise RuntimeError(f"The fixed task prefix has {len(missing)} unanswered items")
    primary_ratings = ratings[ratings["task_id"].astype(str).isin(primary_ids)].copy()
    surplus_ratings = ratings[~ratings["task_id"].astype(str).isin(primary_ids)].copy()
    order_by_task = primary.set_index("task_id")["display_order"].astype(int)
    primary_ratings["display_order"] = primary_ratings["task_id"].map(order_by_task)
    primary_ratings = primary_ratings.sort_values("display_order", kind="stable")
    primary_ratings = primary_ratings.drop(columns="display_order")
    primary = primary.sort_values("display_order", kind="stable")
    return primary, primary_ratings, surplus_ratings


def main() -> None:
    args = parse_args()
    spec_path = args.spec.resolve()
    snapshot_path = args.snapshot.resolve()
    blind_path = args.blind_manifest.resolve()
    private_path = args.private_manifest.resolve()
    build_summary_path = args.pilot_build_summary.resolve()
    spec = load_protocol_spec(spec_path)
    task_count = int(spec["ontology"]["development_pilot"]["primary_task_presentations"])

    blind = pd.read_csv(blind_path, dtype={"task_id": str})
    if tuple(blind.columns) != BLIND_COLUMNS:
        raise RuntimeError("The pilot blind manifest contains unexpected fields")
    private = pd.read_csv(
        private_path,
        dtype={
            "task_id": str,
            "repeat_of_task_id": str,
            "person_id": str,
            "image_id": str,
        },
        keep_default_na=False,
    )
    if set(blind["task_id"]) != set(private["task_id"]):
        raise RuntimeError("Blind and private task manifests differ")
    snapshot, ratings = load_annotation_snapshot(snapshot_path, private["task_id"])
    primary, primary_ratings, surplus = fixed_prefix(
        private,
        ratings,
        task_count=task_count,
    )

    summary, tables = single_rater_pilot_audit(private, primary_ratings)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_ids = set(primary["task_id"].astype(str))
    raw_by_task = {
        str(row["task_id"]): row for row in snapshot.get("annotations", [])
    }
    frozen_annotations = [
        raw_by_task[str(task_id)] for task_id in primary["task_id"].astype(str)
    ]
    frozen_snapshot = {
        "status": "VCOCO_V3_FIXED_PREFIX_ANNOTATION_SNAPSHOT",
        "annotator_id": str(snapshot.get("annotator_id", "")),
        "selection_rule": "first_130_by_blind_manifest_display_order",
        "completed_rows": len(frozen_annotations),
        "task_manifest_rows": len(private),
        "surplus_responses_excluded": len(surplus),
        "source_snapshot_sha256": sha256_file(snapshot_path),
        "annotations": frozen_annotations,
    }
    frozen_snapshot_path = output_dir / "private_primary_annotations.json"
    write_json(frozen_snapshot_path, frozen_snapshot)

    primary_private_path = output_dir / "private_primary_manifest.csv"
    primary_blind_path = output_dir / "blind_primary_tasks.csv"
    primary.to_csv(primary_private_path, index=False)
    blind[blind["task_id"].isin(primary_ids)].sort_values(
        "display_order", kind="stable"
    ).to_csv(primary_blind_path, index=False)

    artifact_hashes = {
        frozen_snapshot_path.name: sha256_file(frozen_snapshot_path),
        primary_private_path.name: sha256_file(primary_private_path),
        primary_blind_path.name: sha256_file(primary_blind_path),
    }
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        artifact_hashes[path.name] = sha256_file(path)

    repeat_table = tables["repeat_summary"]
    repeat_agreement = {
        str(row.axis): {
            "complete_pairs": int(row.complete_pairs),
            "matches": int(row.matches),
            "exact_agreement": (
                float(row.exact_agreement) if pd.notna(row.exact_agreement) else None
            ),
        }
        for row in repeat_table.itertuples(index=False)
    }
    result = {
        **summary,
        "status": "VCOCO_V3_HUMAN_PILOT_AUDIT_COMPLETE",
        "protocol_version": spec["protocol_version"],
        "protocol_amendment_revision": int(spec["amendment"]["revision"]),
        "primary_selection_rule": "first_130_by_blind_manifest_display_order",
        "primary_task_presentations": task_count,
        "annotator_id": str(snapshot.get("annotator_id", "")),
        "guide_versions": sorted(primary_ratings["guide_version"].astype(str).unique()),
        "surplus_responses_preserved_and_excluded": len(surplus),
        "interrater_agreement_available": False,
        "intrarater_repeat_agreement": repeat_agreement,
        "source_tag_development_model_fitting_permitted": True,
        "human_pilot_labels_used_for_candidate_selection": False,
        "human_harmonized_performance_claim_permitted": False,
        "evidence_scope": "fixed_prefix_single_rater_descriptive_mechanism_audit",
        "source_sha256": {
            "protocol_spec": sha256_file(spec_path),
            "annotation_snapshot": sha256_file(snapshot_path),
            "blind_manifest": sha256_file(blind_path),
            "private_manifest": sha256_file(private_path),
            "pilot_build_summary": sha256_file(build_summary_path),
        },
        "artifact_sha256": artifact_hashes,
    }
    result.pop("annotation_gate_passed", None)
    result.pop("model_fitting_permitted_by_annotation_gate", None)
    write_json(output_dir / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
