import json

import pandas as pd
import pytest
from PIL import Image

from hac.vcoco_v3_annotation import (
    BLIND_COLUMNS,
    AnnotationApplication,
    AnnotationStore,
    create_pilot_tasks,
)
from hac.vcoco_v3_audit import (
    agreement_tables,
    build_harmonized_annotations,
    harmonization_estimates,
    harmonized_activity,
    make_adjudication_manifest,
    single_rater_pilot_audit,
)
from tools.finalize_vcoco_v3_pilot import fixed_prefix


def candidate_rows(tmp_path):
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 80), (40, 80, 120)).save(image_path)
    rows = []
    for index in range(96):
        label = (index // 4) % 3
        prediction = label
        error_type = "correct"
        if index < 32:
            label, prediction, error_type = 1, 2, "standing_to_locomotion"
        elif index < 64:
            label, prediction, error_type = 2, 1, "locomotion_to_standing"
        rows.append(
            {
                "person_id": f"person-{index:03d}",
                "image_id": f"image-{index // 2:03d}",
                "annotation_id": f"annotation-{index:03d}",
                "image_path": str(image_path),
                "bbox_xmin": 10,
                "bbox_ymin": 8,
                "bbox_xmax": 70,
                "bbox_ymax": 72,
                "image_width": 100,
                "image_height": 80,
                "label_3": ["sitting", "standing", "walking_running"][label],
                "label_index": label,
                "predicted_index": prediction,
                "predicted_label": ["sitting", "standing", "walking_running"][prediction],
                "probability_sitting": 0.1,
                "probability_standing": 0.4,
                "probability_walking_running": 0.5,
                "confidence": 0.5 + index / 1000,
                "error_type": error_type,
                "area_quartile": f"Q{index % 4 + 1}",
                "confidence_quartile": f"Q{index % 4 + 1}",
                "touches_boundary": False,
                "scene_occupancy": "multiple",
            }
        )
    return pd.DataFrame(rows)


def valid_response(task_id, annotator_id="rater-01"):
    return {
        "task_id": task_id,
        "annotator_id": annotator_id,
        "posture": "upright",
        "visible_translation": "locomoting",
        "gait": "walking",
        "visibility": "clear",
        "notes": "",
        "guide_version": "v3-pilot-1",
    }


def test_pilot_manifest_is_blind_and_contains_hidden_repeat_evidence(tmp_path):
    blind, private = create_pilot_tasks(
        candidate_rows(tmp_path),
        probability_tasks=24,
        error_tasks=16,
        repeat_tasks=8,
        seed=42,
    )

    assert tuple(blind.columns) == BLIND_COLUMNS
    assert len(blind) == 48
    assert private["person_id"].nunique() == 40
    assert private["repeat_of_task_id"].ne("").sum() == 8
    assert not {
        "label_3",
        "predicted_label",
        "confidence",
        "error_type",
        "cohort",
        "repeat_of_task_id",
    }.intersection(blind.columns)
    assert blind["task_id"].is_unique


def test_annotation_store_separates_raters_and_preserves_revisions(tmp_path):
    tasks = pd.DataFrame(
        [
            {
                "task_id": "task-1",
                "display_order": 1,
                "image_path": "frame.jpg",
                "bbox_xmin": 0,
                "bbox_ymin": 0,
                "bbox_xmax": 1,
                "bbox_ymax": 1,
                "image_width": 1,
                "image_height": 1,
            }
        ]
    )
    store = AnnotationStore(tmp_path, tasks)

    first = store.save(valid_response("task-1"))
    second = store.save(valid_response("task-1"))
    other = store.save(valid_response("task-1", "rater-02"))

    assert (first["revision"], second["revision"], other["revision"]) == (1, 2, 1)
    assert set(store.load("rater-01")) == {"task-1"}
    assert set(store.load("rater-02")) == {"task-1"}
    events = (tmp_path / "events" / "rater-01.jsonl").read_text().splitlines()
    assert [json.loads(row)["revision"] for row in events] == [1, 2]


def test_annotation_application_exposes_no_private_sampling_fields(tmp_path):
    blind, _ = create_pilot_tasks(
        candidate_rows(tmp_path),
        probability_tasks=8,
        error_tasks=8,
        repeat_tasks=2,
        seed=7,
    )
    manifest_path = tmp_path / "blind.csv"
    blind.to_csv(manifest_path, index=False)
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    application = AnnotationApplication(manifest_path, tmp_path / "output", static_dir)

    state = application.public_state("rater-01")
    assert set(state["tasks"][0]) == {"task_id", "display_order", "annotation"}
    assert state["completed"] == 0
    assert application.render_image(state["tasks"][0]["task_id"], "context")
    assert application.render_image(state["tasks"][0]["task_id"], "person")


def test_store_rejects_motion_gait_conflict(tmp_path):
    tasks = pd.DataFrame(
        [{"task_id": "task-1", "display_order": 1, **dict.fromkeys(BLIND_COLUMNS[2:], 0)}]
    )
    store = AnnotationStore(tmp_path, tasks)
    response = valid_response("task-1")
    response["visible_translation"] = "stationary"

    with pytest.raises(ValueError, match="requires visible locomotion"):
        store.save(response)


def test_harmonized_activity_preserves_not_applicable_motion_distinction():
    assert harmonized_activity("upright", "stationary", "not_applicable") == "standing"
    assert (
        harmonized_activity("upright", "locomoting", "not_applicable")
        == "not_resolved"
    )
    assert harmonized_activity("upright", "locomoting", "walking") == "walking_running"
    assert harmonized_activity("upright", "locomoting", "running") == "walking_running"
    assert harmonized_activity("other", "locomoting", "running") == "walking_running"
    assert (
        harmonized_activity("indeterminate", "locomoting", "walking")
        == "walking_running"
    )
    assert (
        harmonized_activity("upright", "locomoting", "indeterminate")
        == "walking_running"
    )


def audit_fixture():
    private = pd.DataFrame(
        [
            {
                "task_id": "original-1",
                "repeat_of_task_id": "",
                "person_id": "person-1",
                "image_id": "image-1",
                "cohort": "probability_sample",
                "label_3": "standing",
                "predicted_label": "walking_running",
                "error_type": "standing_to_locomotion",
                "confidence": 0.8,
                "design_weight": 2.0,
            },
            {
                "task_id": "original-2",
                "repeat_of_task_id": "",
                "person_id": "person-2",
                "image_id": "image-2",
                "cohort": "probability_sample",
                "label_3": "sitting",
                "predicted_label": "sitting",
                "error_type": "correct",
                "confidence": 0.9,
                "design_weight": 1.0,
            },
            {
                "task_id": "repeat-1",
                "repeat_of_task_id": "original-1",
                "person_id": "person-1",
                "image_id": "image-1",
                "cohort": "intrarater_repeat",
                "label_3": "standing",
                "predicted_label": "walking_running",
                "error_type": "standing_to_locomotion",
                "confidence": 0.8,
                "design_weight": "",
            },
        ]
    )
    base = {
        "posture": "upright",
        "visible_translation": "stationary",
        "gait": "not_applicable",
        "visibility": "clear",
    }
    left = pd.DataFrame(
        [
            {"task_id": "original-1", **base},
            {
                "task_id": "original-2",
                "posture": "seated",
                "visible_translation": "stationary",
                "gait": "not_applicable",
                "visibility": "clear",
            },
            {"task_id": "repeat-1", **base},
        ]
    ).set_index("task_id", drop=False)
    right = left.copy()
    return private, left, right


def test_agreement_audit_uses_unique_items_and_hidden_repeats():
    private, left, right = audit_fixture()

    inter, intra, disagreements = agreement_tables(
        private, {"left": left, "right": right}, ("left", "right")
    )

    assert (inter["items"] == 2).all()
    assert (inter["exact_agreement"] == 1.0).all()
    assert (intra["exact_agreement"] == 1.0).all()
    assert disagreements.empty


def test_single_rater_pilot_audit_excludes_repeats_from_content_counts():
    private, left, _ = audit_fixture()

    summary, tables = single_rater_pilot_audit(private, left.reset_index(drop=True))

    assert summary["responses"] == 3
    assert summary["unique_content_rows"] == 2
    assert summary["repeat_observations"] == 1
    assert summary["complete_repeat_pairs"] == 1
    assert summary["resolved_3class_rows"] == 2
    assert not summary["annotation_gate_passed"]
    assert tables["axis_counts"].query("axis == 'posture'")["count"].sum() == 2
    assert (tables["repeat_summary"]["exact_agreement"] == 1.0).all()


def test_fixed_prefix_uses_display_order_and_preserves_surplus():
    private = pd.DataFrame(
        [
            {"task_id": "task-2", "display_order": 2},
            {"task_id": "task-3", "display_order": 3},
            {"task_id": "task-1", "display_order": 1},
        ]
    )
    ratings = pd.DataFrame(
        [
            {"task_id": "task-3", "posture": "upright"},
            {"task_id": "task-1", "posture": "seated"},
            {"task_id": "task-2", "posture": "other"},
        ]
    )

    primary, primary_ratings, surplus = fixed_prefix(private, ratings, task_count=2)

    assert primary["task_id"].tolist() == ["task-1", "task-2"]
    assert primary_ratings["task_id"].tolist() == ["task-1", "task-2"]
    assert surplus["task_id"].tolist() == ["task-3"]


def test_blind_adjudication_resolves_only_disagreed_axis(tmp_path):
    private, left, right = audit_fixture()
    right.loc["original-1", "visible_translation"] = "not_inferable"
    blind = pd.DataFrame(
        [
            {
                "task_id": task_id,
                "display_order": index + 1,
                "image_path": str(tmp_path / "frame.jpg"),
                "bbox_xmin": 0,
                "bbox_ymin": 0,
                "bbox_xmax": 10,
                "bbox_ymax": 10,
                "image_width": 10,
                "image_height": 10,
            }
            for index, task_id in enumerate(private["task_id"])
        ]
    )
    _, _, disagreements = agreement_tables(
        private, {"left": left, "right": right}, ("left", "right")
    )
    adjudication_tasks = make_adjudication_manifest(blind, disagreements["task_id"], seed=2)
    adjudicator = left.loc[["original-1"]].copy()

    harmonized = build_harmonized_annotations(
        private,
        {"left": left, "right": right},
        ("left", "right"),
        adjudicator,
    )
    estimates = harmonization_estimates(harmonized, resamples=100, seed=4)

    assert adjudication_tasks["task_id"].tolist() == ["original-1"]
    assert harmonized.loc[0, "adjudicated_axes"] == "visible_translation"
    assert harmonized["harmonized_label_3"].tolist() == ["standing", "sitting"]
    assert estimates["source_tag_matches_harmonized_when_resolved"]["estimate"] == 1.0
