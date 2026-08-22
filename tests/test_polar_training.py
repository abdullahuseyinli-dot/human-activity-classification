import pandas as pd
import pytest
from torch import nn

from hac.polar_training import (
    inverse_frequency_weights,
    nested_stratified_subset,
    optimizer_parameter_groups,
    validate_development_manifest,
)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": [f"image_{index:02d}" for index in range(12)],
            "image_path": [f"image_{index:02d}.jpg" for index in range(12)],
            "split": ["train"] * 8 + ["val"] * 4,
            "label_4": ["sitting", "standing", "walking", "running"] * 3,
        }
    )


def test_development_manifest_rejects_a_test_row():
    frame = _manifest()
    frame.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="train and val"):
        validate_development_manifest(frame, "label_4")


def test_nested_subsets_are_exact_and_nested():
    frame = pd.DataFrame(
        {
            "image_id": [f"image_{index:03d}" for index in range(40)],
            "label": [label for label in "abcd" for _ in range(10)],
        }
    )
    small = nested_stratified_subset(frame, 12)
    large = nested_stratified_subset(frame, 24)
    assert len(small) == 12
    assert len(large) == 24
    assert set(small["image_id"]) < set(large["image_id"])
    assert small["label"].value_counts().to_dict() == {label: 3 for label in "abcd"}


def test_inverse_frequency_weights_are_mean_normalized():
    weights = inverse_frequency_weights([0, 0, 0, 1, 2, 2], 3)
    assert float(weights.mean()) == pytest.approx(1.0)
    assert weights[1] > weights[2] > weights[0]


def test_optimizer_groups_keep_head_and_backbone_learning_rates_separate():
    model = nn.Module()
    model.backbone = nn.Linear(3, 4)
    model.classifier = nn.Linear(4, 2)
    groups = optimizer_parameter_groups(
        model,
        head_lr=1e-3,
        backbone_lr=1e-5,
        weight_decay=0.01,
    )
    by_name = {group["group_name"]: group for group in groups}
    assert by_name["head_decay"]["lr"] == 1e-3
    assert by_name["backbone_decay"]["lr"] == 1e-5
    assert by_name["head_no_decay"]["weight_decay"] == 0.0
