from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from hac.explainability import (
    attribution_spearman,
    calibrated_probabilities,
    curve_auc,
    normalize_attribution,
    patch_keep_mask,
    patch_scores,
)


class DirectionalModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        left = inputs[..., : inputs.shape[-1] // 2].mean(dim=(1, 2, 3))
        right = inputs[..., inputs.shape[-1] // 2 :].mean(dim=(1, 2, 3))
        return torch.stack((left, right, left - right), dim=-1)


def test_center_probability_matches_temperature_scaling() -> None:
    model = DirectionalModel()
    inputs = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    expected = torch.softmax(model(inputs) / 1.7, dim=-1)
    observed = calibrated_probabilities(
        model, inputs, temperature=1.7, policy="center_crop"
    )
    torch.testing.assert_close(observed, expected)


def test_flip_tta_is_invariant_to_input_flip() -> None:
    model = DirectionalModel()
    inputs = torch.randn(2, 3, 8, 8)
    original = calibrated_probabilities(
        model,
        inputs,
        temperature=1.3,
        policy="center_plus_horizontal_flip",
    )
    flipped = calibrated_probabilities(
        model,
        torch.flip(inputs, dims=(-1,)),
        temperature=1.3,
        policy="center_plus_horizontal_flip",
    )
    torch.testing.assert_close(original, flipped)


def test_patch_mask_removes_exact_ranked_fraction() -> None:
    scores = np.arange(16, dtype=np.float32).reshape(4, 4)
    keep = patch_keep_mask(
        scores, 0.25, order="most", output_size=(8, 8)
    ).numpy()
    assert keep.shape == (8, 8)
    assert float((keep == 0.0).mean()) == pytest.approx(0.25)
    assert np.all(keep[6:, 6:] == 0.0)


def test_attribution_normalization_and_patch_pooling() -> None:
    attribution = np.zeros((224, 224), dtype=np.float32)
    attribution[:14, :14] = 2.0
    normalized = normalize_attribution(attribution)
    pooled = patch_scores(normalized, 16)
    assert float(normalized.sum()) == pytest.approx(1.0)
    assert float(pooled.sum()) == pytest.approx(1.0)
    assert np.unravel_index(np.argmax(pooled), pooled.shape) == (0, 0)


def test_similarity_and_curve_auc() -> None:
    values = np.arange(9, dtype=np.float32).reshape(3, 3)
    assert attribution_spearman(values, values) == pytest.approx(1.0)
    assert attribution_spearman(values, np.flip(values)) == pytest.approx(-1.0)
    assert curve_auc(np.array([0.0, 0.5, 1.0]), np.array([1.0, 0.5, 0.0])) == pytest.approx(
        0.5
    )
