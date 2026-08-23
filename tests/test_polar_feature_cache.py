import torch
from torch import nn

from hac.polar_features import official_multilayer_features


def test_official_multilayer_features_concatenate_four_cls_and_mean_patch_tokens():
    hidden_states = tuple(
        torch.tensor(
            [
                [
                    [10.0 * layer + 1.0, 10.0 * layer + 2.0],
                    [10.0 * layer + 3.0, 10.0 * layer + 4.0],
                    [10.0 * layer + 5.0, 10.0 * layer + 6.0],
                ]
            ]
        )
        for layer in range(5)
    )
    features = official_multilayer_features(hidden_states, nn.Identity())
    expected = torch.tensor([[11.0, 12.0, 21.0, 22.0, 31.0, 32.0, 41.0, 42.0, 44.0, 45.0]])
    assert torch.equal(features, expected)
