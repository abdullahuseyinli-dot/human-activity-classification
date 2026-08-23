import pytest

from hac.vcoco import factorize_actions


@pytest.mark.parametrize(
    ("actions", "posture", "motion", "gait", "label_3"),
    [
        ({"sit"}, "seated", "stationary", "not_applicable", "sitting"),
        ({"stand"}, "upright", "stationary", "not_applicable", "standing"),
        ({"walk"}, "upright", "locomoting", "walking", "walking_running"),
        ({"run"}, "upright", "locomoting", "running", "walking_running"),
        ({"stand", "walk"}, "upright", "locomoting", "walking", "walking_running"),
        ({"stand", "run"}, "upright", "locomoting", "running", "walking_running"),
    ],
)
def test_factorize_actions(actions, posture, motion, gait, label_3):
    target = factorize_actions(actions)

    assert target.posture == posture
    assert target.motion == motion
    assert target.gait == gait
    assert target.label_3 == label_3
    assert target.legacy_eligible
    assert target.factorized_clear


def test_sit_stand_is_preserved_as_ambiguous():
    target = factorize_actions({"sit", "stand"})

    assert target.posture == "ambiguous"
    assert target.motion == "ambiguous"
    assert not target.legacy_eligible
    assert not target.factorized_clear
    assert target.label_3 == ""


def test_walk_run_keeps_locomotion_but_masks_gait():
    target = factorize_actions({"stand", "walk", "run"})

    assert target.posture == "upright"
    assert target.motion == "locomoting"
    assert target.gait == "ambiguous"
    assert target.label_3 == "walking_running"
    assert target.label_4 == ""


def test_unknown_action_is_rejected():
    with pytest.raises(ValueError, match="Unexpected V-COCO"):
        factorize_actions({"jump"})
