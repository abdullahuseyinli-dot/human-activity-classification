import math

import pytest

from hac.vcoco_v3 import (
    nominal_krippendorff_alpha,
    validate_annotation,
    validate_protocol_spec,
)


def valid_protocol():
    return {
        "protocol_version": "3.0.0-dev.2",
        "frozen_reference": {
            "git_tag": "polar-study-v2.0.0",
            "official_test_status": "CONSUMED_EXPLORATORY_ONLY",
        },
        "development_policy": {"current_official_test_may_select_models": False},
        "endpoints": {
            "development_selection": "source_tag_macro_f1_named_explicitly",
            "human_pilot": "descriptive_ontology_and_error_mechanism_audit_only",
            "independent_confirmation": "human_harmonized_macro_f1",
        },
        "ontology": {
            "development_pilot": {
                "primary_task_presentations": 130,
                "selection_rule": "first_130_by_blind_manifest_display_order",
                "annotators": 1,
                "predictions_and_source_tags_blinded": True,
                "labels_used_for_candidate_selection": False,
            },
            "annotation_gate": {
                "minimum_independent_annotators": 2,
                "adjudication_required": True,
                "required_alpha_axes": [
                    "posture",
                    "visible_translation",
                    "gait",
                    "visibility",
                ],
                "minimum_alpha_each_required_axis": 0.8,
                "minimum_intrarater_exact_agreement_each_axis": 0.9,
                "required_completion_fraction": 1.0,
                "predictions_and_source_tags_blinded": True,
            }
        },
        "confirmation": {
            "current_vcoco_test_eligible": False,
            "single_open_after_lock": True,
        },
    }


def test_protocol_rejects_reusing_consumed_test():
    payload = valid_protocol()
    payload["confirmation"]["current_vcoco_test_eligible"] = True

    with pytest.raises(ValueError, match="cannot be reused"):
        validate_protocol_spec(payload)


def test_annotation_requires_consistent_motion_and_gait():
    payload = {
        "task_id": "sample-1",
        "annotator_id": "rater-1",
        "posture": "upright",
        "visible_translation": "stationary",
        "gait": "walking",
        "visibility": "clear",
        "notes": "",
    }

    with pytest.raises(ValueError, match="requires visible locomotion"):
        validate_annotation(payload, {"sample-1"})


def test_nominal_alpha_is_one_for_complete_agreement():
    alpha = nominal_krippendorff_alpha(
        [["upright", "upright"], ["seated", "seated"], ["upright", "upright"]]
    )

    assert alpha == pytest.approx(1.0)


def test_nominal_alpha_is_nan_without_paired_ratings():
    assert math.isnan(nominal_krippendorff_alpha([["upright"], [None]]))
