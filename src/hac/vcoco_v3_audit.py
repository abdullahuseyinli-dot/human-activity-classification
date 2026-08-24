"""Agreement, adjudication, and harmonization for the V-COCO v3 pilot."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from hac.vcoco_v3 import nominal_krippendorff_alpha, validate_annotation
from hac.vcoco_v3_annotation import BLIND_COLUMNS

ANNOTATION_AXES = ("posture", "visible_translation", "gait", "visibility")


def load_annotation_snapshot(
    path: Path, valid_task_ids: Iterable[str]
) -> tuple[dict, pd.DataFrame]:
    """Load and revalidate an annotation snapshot without weakening completion rules."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    annotator_id = str(payload.get("annotator_id", ""))
    rows = []
    for raw in payload.get("annotations", []):
        normalized = validate_annotation(raw, valid_task_ids)
        if normalized["annotator_id"] != annotator_id:
            raise ValueError(f"Mixed annotator identities in {path}")
        rows.append(
            {
                **normalized,
                "saved_at_utc": raw.get("saved_at_utc", ""),
                "revision": int(raw.get("revision", 1)),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) and frame["task_id"].duplicated().any():
        raise ValueError(f"Duplicate task response in {path}")
    if path.stem != annotator_id:
        raise ValueError(f"Snapshot filename and annotator identity differ: {path}")
    return payload, frame


def discover_annotation_passes(
    annotation_dir: Path,
    valid_task_ids: Iterable[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Find valid local annotation passes and report their completion."""

    expected = tuple(map(str, valid_task_ids))
    expected_set = set(expected)
    snapshots = {}
    progress = []
    for path in sorted(annotation_dir.glob("*.json")) if annotation_dir.is_dir() else ():
        payload, frame = load_annotation_snapshot(path, expected)
        observed = set(frame["task_id"].astype(str)) if len(frame) else set()
        if not observed.issubset(expected_set):
            raise ValueError(f"Snapshot contains a task outside the blind manifest: {path}")
        annotator_id = str(payload["annotator_id"])
        snapshots[annotator_id] = frame.set_index("task_id", drop=False)
        progress.append(
            {
                "annotator_id": annotator_id,
                "completed_tasks": len(observed),
                "expected_tasks": len(expected),
                "completion_fraction": len(observed) / len(expected) if expected else 0.0,
                "complete": observed == expected_set,
                "updated_at_utc": str(payload.get("updated_at_utc", "")),
            }
        )
    columns = [
        "annotator_id",
        "completed_tasks",
        "expected_tasks",
        "completion_fraction",
        "complete",
        "updated_at_utc",
    ]
    return snapshots, pd.DataFrame(progress, columns=columns)


def select_complete_raters(
    snapshots: dict[str, pd.DataFrame],
    progress: pd.DataFrame,
    requested: Iterable[str],
    *,
    required: int,
) -> tuple[str, ...] | None:
    """Select predeclared complete raters or infer only an unambiguous complete set."""

    requested = tuple(requested)
    complete = tuple(
        sorted(progress.loc[progress["complete"].astype(bool), "annotator_id"].astype(str))
    )
    if requested:
        if len(requested) != required or len(set(requested)) != required:
            raise ValueError(f"Exactly {required} distinct --annotator values are required")
        missing = set(requested) - set(snapshots)
        incomplete = set(requested) - set(complete)
        if missing:
            raise ValueError(f"Requested annotation snapshots are missing: {sorted(missing)}")
        if incomplete:
            return None
        return requested
    if len(complete) < required:
        return None
    if len(complete) > required:
        raise ValueError("More complete raters exist; choose the predeclared pair with --annotator")
    return complete


def agreement_tables(
    private_manifest: pd.DataFrame,
    snapshots: dict[str, pd.DataFrame],
    rater_ids: tuple[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate inter-rater and hidden-repeat agreement and return disagreements."""

    unique = private_manifest[private_manifest["repeat_of_task_id"].fillna("").eq("")].copy()
    repeats = private_manifest[private_manifest["repeat_of_task_id"].fillna("").ne("")].copy()
    inter_rows = []
    disagreement_rows = []
    left, right = (snapshots[value] for value in rater_ids)
    for axis in ANNOTATION_AXES:
        units = [
            [left.at[task_id, axis], right.at[task_id, axis]]
            for task_id in unique["task_id"].astype(str)
        ]
        matches = np.asarray([values[0] == values[1] for values in units], dtype=bool)
        alpha = nominal_krippendorff_alpha(units)
        inter_rows.append(
            {
                "axis": axis,
                "items": len(units),
                "exact_agreement": float(matches.mean()),
                "krippendorff_alpha_nominal": alpha,
            }
        )

    for row in unique.itertuples(index=False):
        task_id = str(row.task_id)
        axes = [
            axis for axis in ANNOTATION_AXES if left.at[task_id, axis] != right.at[task_id, axis]
        ]
        if axes:
            disagreement_rows.append(
                {
                    "task_id": task_id,
                    "disagreement_axes": "|".join(axes),
                    **{
                        f"{rater_ids[0]}_{axis}": left.at[task_id, axis] for axis in ANNOTATION_AXES
                    },
                    **{
                        f"{rater_ids[1]}_{axis}": right.at[task_id, axis]
                        for axis in ANNOTATION_AXES
                    },
                }
            )

    intra_rows = []
    for rater_id in rater_ids:
        ratings = snapshots[rater_id]
        for axis in ANNOTATION_AXES:
            matches = [
                ratings.at[str(row.task_id), axis] == ratings.at[str(row.repeat_of_task_id), axis]
                for row in repeats.itertuples(index=False)
            ]
            intra_rows.append(
                {
                    "annotator_id": rater_id,
                    "axis": axis,
                    "repeat_pairs": len(matches),
                    "exact_agreement": float(np.mean(matches)) if matches else np.nan,
                }
            )
    return (
        pd.DataFrame(inter_rows),
        pd.DataFrame(intra_rows),
        pd.DataFrame(disagreement_rows),
    )


def make_adjudication_manifest(
    blind_manifest: pd.DataFrame,
    disagreement_task_ids: Iterable[str],
    *,
    seed: int,
) -> pd.DataFrame:
    """Create a newly ordered blind task list for the adjudication pass."""

    task_ids = set(map(str, disagreement_task_ids))
    selected = blind_manifest[blind_manifest["task_id"].astype(str).isin(task_ids)].copy()
    if len(selected) != len(task_ids):
        raise ValueError("A disagreement task is missing from the blind manifest")
    rng = np.random.default_rng(seed)
    selected = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
    selected["display_order"] = np.arange(1, len(selected) + 1)
    return selected[list(BLIND_COLUMNS)]


def harmonized_activity(posture: str, visible_translation: str, gait: str) -> str:
    """Map joint factor labels to the three-class benchmark when identifiable."""

    if visible_translation == "locomoting" and gait in {
        "walking",
        "running",
        "indeterminate",
    }:
        return "walking_running"
    if posture == "seated":
        return "sitting"
    if posture != "upright":
        return "not_resolved"
    if visible_translation == "stationary" and gait == "not_applicable":
        return "standing"
    return "not_resolved"


def single_rater_pilot_audit(
    private_manifest: pd.DataFrame,
    ratings: pd.DataFrame,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Build a descriptive audit without treating one partial pass as a gate result."""

    required_private = {
        "task_id",
        "repeat_of_task_id",
        "person_id",
        "image_id",
        "cohort",
        "label_3",
        "predicted_label",
        "error_type",
    }
    missing_private = required_private - set(private_manifest.columns)
    if missing_private:
        raise ValueError(f"Private manifest fields are missing: {sorted(missing_private)}")
    required_ratings = {"task_id", *ANNOTATION_AXES}
    missing_ratings = required_ratings - set(ratings.columns)
    if missing_ratings:
        raise ValueError(f"Rating fields are missing: {sorted(missing_ratings)}")

    private = private_manifest.copy()
    private["task_id"] = private["task_id"].astype(str)
    private["repeat_of_task_id"] = private["repeat_of_task_id"].fillna("").astype(str)
    observed = ratings.copy()
    observed["task_id"] = observed["task_id"].astype(str)
    if private["task_id"].duplicated().any() or observed["task_id"].duplicated().any():
        raise ValueError("Pilot task identifiers must be unique")
    if not set(observed["task_id"]).issubset(set(private["task_id"])):
        raise ValueError("A rating refers to a task outside the private manifest")

    completed = private.merge(observed, on="task_id", how="inner", validate="one_to_one")
    completed["canonical_task_id"] = np.where(
        completed["repeat_of_task_id"].ne(""),
        completed["repeat_of_task_id"],
        completed["task_id"],
    )
    missing_originals = set(completed["canonical_task_id"]) - set(
        private.loc[private["repeat_of_task_id"].eq(""), "task_id"]
    )
    if missing_originals:
        raise ValueError(f"Repeat originals are missing: {sorted(missing_originals)}")

    selected = (
        completed.assign(is_repeat_observation=completed["repeat_of_task_id"].ne(""))
        .sort_values("is_repeat_observation", kind="stable")
        .drop_duplicates("canonical_task_id", keep="first")
    )
    rating_columns = [value for value in observed.columns if value != "task_id"]
    selection_columns = ["canonical_task_id", "task_id", *rating_columns]
    if "display_order" in selected.columns:
        selection_columns.append("display_order")
    selected_ratings = selected[selection_columns].rename(
        columns={
            "task_id": "observed_task_id",
            "display_order": "observed_display_order",
        }
    )
    canonical = private[private["repeat_of_task_id"].eq("")].copy()
    canonical = canonical.rename(
        columns={"task_id": "canonical_task_id", "display_order": "canonical_display_order"}
    )
    content = canonical.merge(
        selected_ratings,
        on="canonical_task_id",
        how="inner",
        validate="one_to_one",
    )
    content["harmonized_label_3"] = [
        harmonized_activity(posture, translation, gait)
        for posture, translation, gait in zip(
            content["posture"],
            content["visible_translation"],
            content["gait"],
            strict=True,
        )
    ]
    content["resolved_3class"] = content["harmonized_label_3"].ne("not_resolved")

    axis_rows = []
    for axis in ANNOTATION_AXES:
        counts = content[axis].value_counts().sort_index()
        for value, count in counts.items():
            axis_rows.append(
                {
                    "axis": axis,
                    "value": str(value),
                    "count": int(count),
                    "fraction": float(count / len(content)) if len(content) else np.nan,
                }
            )
    axis_counts = pd.DataFrame(axis_rows, columns=["axis", "value", "count", "fraction"])

    joint_counts = (
        content.groupby(["visible_translation", "gait"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    joint_counts["fraction"] = (
        joint_counts["count"] / len(content) if len(content) else np.nan
    )

    cohort_rows = []
    for cohort, frame in content.groupby("cohort", sort=True):
        resolved = frame[frame["resolved_3class"]]
        source_matches = resolved["label_3"].eq(resolved["harmonized_label_3"])
        prediction_matches = resolved["predicted_label"].eq(
            resolved["harmonized_label_3"]
        )
        cohort_rows.append(
            {
                "cohort": str(cohort),
                "rows": len(frame),
                "images": int(frame["image_id"].nunique()),
                "resolved_3class_rows": len(resolved),
                "not_resolved_3class_rows": int((~frame["resolved_3class"]).sum()),
                "source_matches_resolved": int(source_matches.sum()),
                "source_match_fraction_resolved": (
                    float(source_matches.mean()) if len(resolved) else np.nan
                ),
                "prediction_matches_resolved": int(prediction_matches.sum()),
                "prediction_match_fraction_resolved": (
                    float(prediction_matches.mean()) if len(resolved) else np.nan
                ),
            }
        )
    cohort_summary = pd.DataFrame(cohort_rows)

    errors = content[content["cohort"].eq("error_enriched")].copy()
    if len(errors):
        errors["pilot_resolution"] = np.select(
            [
                ~errors["resolved_3class"],
                errors["harmonized_label_3"].eq(errors["label_3"]),
                errors["harmonized_label_3"].eq(errors["predicted_label"]),
            ],
            ["not_resolved", "source_supported", "prediction_supported"],
            default="neither_supported",
        )
        error_resolution = (
            errors.groupby(["error_type", "pilot_resolution"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
    else:
        error_resolution = pd.DataFrame(
            columns=["error_type", "pilot_resolution", "count"]
        )

    indexed = observed.set_index("task_id", drop=False)
    repeat_rows = completed[completed["repeat_of_task_id"].ne("")]
    paired = repeat_rows[repeat_rows["repeat_of_task_id"].isin(indexed.index)]
    repeat_summary_rows = []
    for axis in ANNOTATION_AXES:
        matches = [
            indexed.at[str(row.task_id), axis]
            == indexed.at[str(row.repeat_of_task_id), axis]
            for row in paired.itertuples(index=False)
        ]
        repeat_summary_rows.append(
            {
                "axis": axis,
                "complete_pairs": len(matches),
                "matches": int(sum(matches)),
                "exact_agreement": float(np.mean(matches)) if matches else np.nan,
            }
        )
    repeat_summary = pd.DataFrame(repeat_summary_rows)

    review_parts = []
    review_rules = {
        "non_gait_locomotion": content["visible_translation"].eq("locomoting")
        & content["gait"].eq("not_applicable"),
        "other_posture_with_walk_or_run": content["posture"].eq("other")
        & content["gait"].isin(["walking", "running"]),
        "transition": content["visible_translation"].eq("transition"),
        "annotator_note": content.get("notes", pd.Series("", index=content.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne(""),
    }
    review_columns = [
        "observed_task_id",
        "canonical_task_id",
        "observed_display_order",
        "cohort",
        "posture",
        "visible_translation",
        "gait",
        "visibility",
        "label_3",
        "predicted_label",
        "error_type",
    ]
    available_review_columns = [value for value in review_columns if value in content.columns]
    for reason, mask in review_rules.items():
        selected = content.loc[mask, available_review_columns].copy()
        selected.insert(0, "review_reason", reason)
        review_parts.append(selected)
    review_items = (
        pd.concat(review_parts, ignore_index=True)
        if review_parts
        else pd.DataFrame(columns=["review_reason", *available_review_columns])
    )

    non_gait_locomotion = content["visible_translation"].eq("locomoting") & content[
        "gait"
    ].eq("not_applicable")
    clear_walk_or_run = content["visible_translation"].eq("locomoting") & content[
        "gait"
    ].isin(["walking", "running"])
    notes = observed.get("notes", pd.Series("", index=observed.index)).fillna("").astype(str)
    revisions = pd.to_numeric(
        observed.get("revision", pd.Series(1, index=observed.index)), errors="coerce"
    ).fillna(1)
    summary = {
        "status": "VCOCO_V3_SINGLE_RATER_PILOT_DESCRIPTIVE_AUDIT_COMPLETE",
        "responses": len(observed),
        "unique_content_rows": len(content),
        "repeat_observations": len(repeat_rows),
        "complete_repeat_pairs": len(paired),
        "unique_images": int(content["image_id"].nunique()),
        "resolved_3class_rows": int(content["resolved_3class"].sum()),
        "not_resolved_3class_rows": int((~content["resolved_3class"]).sum()),
        "non_gait_locomotion_rows": int(non_gait_locomotion.sum()),
        "clear_walking_or_running_rows": int(clear_walk_or_run.sum()),
        "nonempty_notes": int(notes.str.strip().ne("").sum()),
        "revised_tasks": int((revisions > 1).sum()),
        "annotation_gate_passed": False,
        "interrater_agreement_available": False,
        "intrarater_agreement_available": bool(len(paired)),
    }
    tables = {
        "axis_counts": axis_counts,
        "joint_translation_gait": joint_counts,
        "cohort_summary": cohort_summary,
        "error_resolution": error_resolution,
        "repeat_summary": repeat_summary,
        "review_items": review_items,
    }
    return summary, tables


def build_harmonized_annotations(
    private_manifest: pd.DataFrame,
    snapshots: dict[str, pd.DataFrame],
    rater_ids: tuple[str, str],
    adjudicator: pd.DataFrame | None,
) -> pd.DataFrame:
    """Resolve each axis by agreement or a separate blinded adjudication pass."""

    unique = private_manifest[private_manifest["repeat_of_task_id"].fillna("").eq("")].copy()
    left, right = (snapshots[value] for value in rater_ids)
    rows = []
    for private_row in unique.itertuples(index=False):
        task_id = str(private_row.task_id)
        output = {"task_id": task_id}
        adjudicated_axes = []
        for axis in ANNOTATION_AXES:
            left_value = str(left.at[task_id, axis])
            right_value = str(right.at[task_id, axis])
            if left_value == right_value:
                output[axis] = left_value
            else:
                if adjudicator is None or task_id not in adjudicator.index:
                    raise ValueError(f"Missing adjudication for {task_id}/{axis}")
                output[axis] = str(adjudicator.at[task_id, axis])
                adjudicated_axes.append(axis)
        output["adjudicated_axes"] = "|".join(adjudicated_axes)
        output["harmonized_label_3"] = harmonized_activity(
            output["posture"], output["visible_translation"], output["gait"]
        )
        rows.append(output)
    harmonized = pd.DataFrame(rows)
    private_columns = [
        "task_id",
        "person_id",
        "image_id",
        "cohort",
        "label_3",
        "predicted_label",
        "confidence",
        "design_weight",
    ]
    available = [value for value in private_columns if value in unique.columns]
    return unique[available].merge(harmonized, on="task_id", validate="one_to_one")


def cluster_weighted_bootstrap_proportion(
    frame: pd.DataFrame,
    values: np.ndarray,
    eligible: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int | None]:
    """Estimate a weighted proportion and recording-cluster bootstrap interval."""

    values = np.asarray(values, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    weights = frame["design_weight"].to_numpy(dtype=float)
    valid = eligible & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return {"estimate": None, "ci_95_low": None, "ci_95_high": None, "rows": 0}
    point = float(np.average(values[valid].astype(float), weights=weights[valid]))
    groups = frame["image_id"].astype(str).to_numpy(dtype=str)
    unique_groups = np.unique(groups[valid])
    index_by_group = {group: np.flatnonzero(valid & (groups == group)) for group in unique_groups}
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    for iteration in range(resamples):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([index_by_group[group] for group in sampled])
        estimates[iteration] = np.average(values[indices].astype(float), weights=weights[indices])
    return {
        "estimate": point,
        "ci_95_low": float(np.quantile(estimates, 0.025)),
        "ci_95_high": float(np.quantile(estimates, 0.975)),
        "rows": int(valid.sum()),
    }


def harmonization_estimates(
    harmonized: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    """Summarize the probability cohort without treating error enrichment as prevalence."""

    frame = harmonized[harmonized["cohort"].eq("probability_sample")].reset_index(drop=True)
    resolved = frame["harmonized_label_3"].ne("not_resolved").to_numpy()
    source_match = frame["harmonized_label_3"].eq(frame["label_3"]).to_numpy()
    prediction_match = frame["harmonized_label_3"].eq(frame["predicted_label"]).to_numpy()
    unidentifiable = ~resolved
    return {
        "static_label_not_resolved": cluster_weighted_bootstrap_proportion(
            frame,
            unidentifiable,
            np.ones(len(frame), dtype=bool),
            resamples=resamples,
            seed=seed,
        ),
        "source_tag_matches_harmonized_when_resolved": cluster_weighted_bootstrap_proportion(
            frame,
            source_match,
            resolved,
            resamples=resamples,
            seed=seed + 1,
        ),
        "locked_prediction_matches_harmonized_when_resolved": cluster_weighted_bootstrap_proportion(
            frame,
            prediction_match,
            resolved,
            resamples=resamples,
            seed=seed + 2,
        ),
    }
