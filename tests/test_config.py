import pytest

from hac.config import ModelConfig


def test_dino_partial_unfreeze_requires_a_block_count():
    with pytest.raises(ValueError, match="top_n_blocks"):
        ModelConfig(
            model_kind="dinov2_small",
            augmentation_strength="moderate",
            batch_size=4,
            head_lr=1e-3,
            backbone_lr=5e-6,
            weight_decay=1e-4,
            unfreeze_strategy="top_blocks",
        )


def test_convnext_rejects_dino_freeze_strategy():
    with pytest.raises(ValueError, match="invalid"):
        ModelConfig(
            model_kind="convnext_small",
            augmentation_strength="mild",
            batch_size=4,
            head_lr=1e-3,
            backbone_lr=3e-5,
            weight_decay=1e-4,
            unfreeze_strategy="probe_only",
        )


def test_valid_dino_config_round_trips_without_null_fields():
    config = ModelConfig(
        model_kind="dinov2_small",
        augmentation_strength="moderate",
        batch_size=4,
        head_lr=1e-3,
        backbone_lr=5e-6,
        weight_decay=5e-4,
        dropout=0.1,
        unfreeze_strategy="top_blocks",
        top_n_blocks=2,
    )
    assert config.as_dict()["top_n_blocks"] == 2
    assert config.as_dict(include_model_kind=False)["dropout"] == 0.1
