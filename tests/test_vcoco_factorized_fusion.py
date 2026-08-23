import numpy as np

from experiments.evaluate_vcoco_v2_factorized_fusion import decode_factorized


def test_factorized_decoder_respects_hierarchy():
    probabilities = decode_factorized(
        np.asarray([0.90, 0.05, 0.05]),
        np.asarray([0.90, 0.10, 0.90]),
    )

    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities.argmax(axis=1).tolist() == [0, 1, 2]
    assert probabilities[0, 2] < 0.10
