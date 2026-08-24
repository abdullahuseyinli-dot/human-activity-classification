import numpy as np
import pytest

from hac.vcoco_v3_representations import (
    BOX_PERTURBATIONS,
    aggregate_perturbation_probabilities,
    perturb_person_box,
)


def person_row():
    return {
        "bbox_xmin": 20.0,
        "bbox_ymin": 10.0,
        "bbox_xmax": 60.0,
        "bbox_ymax": 70.0,
        "image_width": 100,
        "image_height": 80,
    }


def test_box_interventions_are_declared_clipped_and_deterministic():
    original = perturb_person_box(person_row(), "none")
    shifted = perturb_person_box(person_row(), "shift_right_05")
    enlarged = perturb_person_box(person_row(), "scale_110")

    assert original["bbox_xmin"] == 20.0
    assert shifted["bbox_xmin"] == pytest.approx(22.0)
    assert enlarged["bbox_xmin"] == pytest.approx(18.0)
    assert 0 <= enlarged["bbox_ymin"] < enlarged["bbox_ymax"] <= 80


def test_robust_probability_aggregation_requires_the_full_suite():
    probabilities = {
        name: np.asarray([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]]) for name in BOX_PERTURBATIONS
    }

    mean = aggregate_perturbation_probabilities(probabilities, method="mean_probability")
    log_mean = aggregate_perturbation_probabilities(probabilities, method="mean_log_probability")

    assert np.allclose(mean.sum(axis=1), 1.0)
    assert np.allclose(mean, log_mean)
    with pytest.raises(ValueError, match="every declared perturbation"):
        aggregate_perturbation_probabilities(
            {"none": probabilities["none"]}, method="mean_probability"
        )
