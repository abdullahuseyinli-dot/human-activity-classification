import numpy as np

from hac.transfer import (
    apply_prior_ratio,
    apply_temperature,
    estimate_label_shift_em,
    image_cluster_paired_bootstrap,
    probability_logits,
    softmax,
)


def test_probability_logits_round_trip():
    probabilities = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])

    assert np.allclose(softmax(probability_logits(probabilities)), probabilities)


def test_temperature_one_is_identity():
    probabilities = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])

    assert np.allclose(apply_temperature(probabilities, 1.0), probabilities)


def test_prior_ratio_moves_probability_mass():
    probabilities = np.asarray([[0.2, 0.4, 0.4], [0.2, 0.4, 0.4]])
    source = np.asarray([0.2, 0.4, 0.4])
    target = np.asarray([0.5, 0.4, 0.1])

    corrected = apply_prior_ratio(probabilities, source, target)

    assert np.allclose(corrected[0], target)
    assert np.allclose(corrected.sum(axis=1), 1.0)


def test_em_recovers_prior_when_probabilities_are_source_posterior():
    probabilities = np.asarray(
        [[0.98, 0.01, 0.01]] * 6 + [[0.01, 0.98, 0.01]] * 3 + [[0.01, 0.01, 0.98]]
    )
    source = np.asarray([1 / 3, 1 / 3, 1 / 3])

    estimated, details = estimate_label_shift_em(probabilities, source)

    assert details["converged"]
    assert np.allclose(estimated, [0.6, 0.3, 0.1], atol=0.03)


def test_cluster_bootstrap_preserves_paired_people():
    labels = np.asarray([0, 0, 1, 2])
    baseline = np.asarray([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.6, 0.2], [0.2, 0.6, 0.2]])
    challenger = np.eye(3)[labels] * 0.8 + 0.2 / 3
    images = np.asarray(["a", "a", "b", "c"])

    interval = image_cluster_paired_bootstrap(
        labels, challenger, baseline, images, resamples=100, seed=7
    )

    assert interval["point_estimate"] > 0
    assert interval["clusters"] == 3
