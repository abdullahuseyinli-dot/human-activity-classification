import numpy as np
import pandas as pd
import pytest
import torch
from evaluate_polar_faithfulness import randomized_adapted_cascade
from torch import nn

from hac.polar_faithfulness import (
    area_matched_context_mask,
    area_matched_occlusion_masks,
    attribution_localization,
    box_mask,
    flip_uint8_bits_exact,
    projected_person_box,
    quantize_and_flip_parameter_bits,
    select_bbox_stratified_cohort,
)


def test_projected_box_accounts_for_context_crop_and_center_crop():
    row = {
        "actual_width": 400,
        "actual_height": 200,
        "bbox_xmin": 100,
        "bbox_ymin": 50,
        "bbox_xmax": 300,
        "bbox_ymax": 150,
    }
    full = projected_person_box(row, "full_frame")
    context = projected_person_box(row, "person_context_25")
    assert full == (0, 48, 224, 176)
    assert context == (0, 26, 224, 198)


def test_context_mask_matches_area_without_overlap():
    person = box_mask((3, 2, 7, 6), output_size=(10, 10))
    context = area_matched_context_mask(person, seed=17)
    assert int(context.sum()) == int(person.sum())
    assert not torch.logical_and(person, context).any()


def test_matched_occlusion_subsamples_a_majority_person_region():
    person = torch.ones((10, 10), dtype=torch.bool)
    person[:2] = False
    matched_person, context, fraction = area_matched_occlusion_masks(person, seed=17)
    assert int(matched_person.sum()) == int(context.sum()) == 20
    assert not torch.logical_and(matched_person, context).any()
    assert fraction == pytest.approx(0.25)


def test_localization_reports_mass_and_pointing_game():
    person = box_mask((0, 0, 2, 2), output_size=(4, 4))
    attribution = np.ones((4, 4), dtype=float)
    attribution[0, 0] = 5.0
    metrics = attribution_localization(attribution, person)
    assert metrics["pointing_game"] is True
    assert metrics["person_attribution_mass"] == pytest.approx(0.4)
    assert metrics["person_attribution_mass_lift"] == pytest.approx(1.6)


def test_cohort_is_balanced_across_class_and_bbox_quartile():
    frame = pd.DataFrame(
        {
            "image_id": [f"image_{index:03d}" for index in range(64)],
            "label_4": np.repeat(["a", "b", "c", "d"], 16),
            "bbox_area_fraction": np.tile(np.arange(1, 17), 4),
        }
    )
    cohort = select_bbox_stratified_cohort(frame, rows=32, seed=42)
    counts = cohort.groupby(["label_4", "bbox_area_quartile"]).size()
    assert len(cohort) == 32
    assert set(counts) == {2}


def test_uint8_fault_injection_flips_exactly_requested_bits():
    source = np.arange(16, dtype=np.uint8)
    corrupted = flip_uint8_bits_exact(source, bit_flips=19, seed=42)
    changed_bits = np.unpackbits(np.bitwise_xor(source, corrupted)).sum()
    assert int(changed_bits) == 19
    assert np.array_equal(corrupted, flip_uint8_bits_exact(source, bit_flips=19, seed=42))


def test_quantized_parameter_fault_is_deterministic_and_finite():
    source = torch.linspace(-1.0, 1.0, 32).reshape(4, 8)
    corrupted, scale = quantize_and_flip_parameter_bits(source, bit_flips=4, seed=7)
    repeated, repeated_scale = quantize_and_flip_parameter_bits(source, bit_flips=4, seed=7)
    assert scale == repeated_scale
    assert torch.equal(corrupted, repeated)
    assert torch.isfinite(corrupted).all()


class _ConvBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.ModuleList(
            [nn.Sequential(nn.Linear(3, 3)), nn.Sequential(nn.Linear(3, 3))]
        )
        self.classifier = nn.Sequential(nn.Identity(), nn.Identity(), nn.Linear(3, 2))


class _ConvModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _ConvBackbone()


def test_cascade_randomization_changes_only_head_and_adapted_stage():
    torch.manual_seed(3)
    model = _ConvModel()
    early = model.backbone.features[0][0].weight.detach().clone()
    adapted = model.backbone.features[-1][0].weight.detach().clone()
    head = model.backbone.classifier[2].weight.detach().clone()

    randomized_adapted_cascade(model, "convnext_small_full", seed=17)

    assert torch.equal(model.backbone.features[0][0].weight, early)
    assert not torch.equal(model.backbone.features[-1][0].weight, adapted)
    assert not torch.equal(model.backbone.classifier[2].weight, head)
