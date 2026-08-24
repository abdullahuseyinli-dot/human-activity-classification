from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from hac.vcoco_v3_neural import (
    LoRALinear,
    PairedPersonSafeTransform,
    SharedMultiviewFactorizedModel,
    configure_parameter_efficient_backbone,
    decode_factorized_logits,
    multiview_factorized_loss,
    parameter_count_summary,
    trainable_parameter_groups,
)


class AttentionBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(self.q_proj(values) + self.v_proj(values))


class DummyBackbone(nn.Module):
    def __init__(self, width: int = 8, blocks: int = 4) -> None:
        super().__init__()
        self.input = nn.Linear(3, width)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([AttentionBlock(width) for _ in range(blocks)])

    def forward(self, pixel_values: torch.Tensor):
        values = self.input(pixel_values.mean(dim=(2, 3)))
        for block in self.encoder.layer:
            values = values + block(values)
        return SimpleNamespace(pooler_output=values)


def test_factorized_decoder_is_a_valid_hierarchy():
    posture = torch.tensor([[5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0]])
    motion = torch.tensor([[0.0, 0.0], [5.0, -5.0], [-5.0, 5.0]])

    probabilities = decode_factorized_logits(posture, motion)

    assert probabilities.shape == (3, 3)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(3))
    assert probabilities.argmax(dim=1).tolist() == [0, 1, 2]


def test_multiview_model_returns_trainable_factorized_outputs():
    backbone = DummyBackbone()
    configure_parameter_efficient_backbone(backbone, strategy="adapter_only")
    model = SharedMultiviewFactorizedModel(
        backbone,
        backbone_dim=8,
        adapter_dim=12,
        geometry_dim=6,
        dropout=0.1,
    )
    tight = torch.rand(6, 3, 16, 16)
    context = torch.rand(6, 3, 16, 16)
    geometry = torch.rand(6, 6)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])

    output = model(tight, context, geometry)
    losses = multiview_factorized_loss(output, labels, auxiliary_view_weight=0.2)
    losses["loss"].backward()

    assert output.probabilities.shape == (6, 3)
    assert torch.allclose(output.probabilities.sum(dim=1), torch.ones(6), atol=1e-6)
    assert torch.allclose(output.gate_weights.sum(dim=1), torch.ones(6), atol=1e-6)
    assert model.adapter.network[1].weight.grad is not None
    assert all(parameter.grad is None for parameter in backbone.parameters())


def test_lora_is_limited_to_declared_top_blocks_and_zero_initialized():
    backbone = DummyBackbone()
    values = torch.rand(3, 3, 8, 8)
    before = backbone(values).pooler_output.detach().clone()

    replacements = configure_parameter_efficient_backbone(
        backbone,
        strategy="lora_top_blocks",
        top_blocks=2,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )
    after = backbone(values).pooler_output.detach()

    assert len(replacements) == 4
    assert all(name.startswith(("block.2.", "block.3.")) for name in replacements)
    assert not isinstance(backbone.encoder.layer[1].q_proj, LoRALinear)
    assert isinstance(backbone.encoder.layer[2].q_proj, LoRALinear)
    assert torch.allclose(before, after)
    assert all(
        parameter.requires_grad
        for name, parameter in backbone.named_parameters()
        if ".lora_a." in name or ".lora_b." in name
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in backbone.named_parameters()
        if ".lora_a." not in name and ".lora_b." not in name
    )


def test_optimizer_groups_are_disjoint_and_report_parameter_efficiency():
    backbone = DummyBackbone()
    configure_parameter_efficient_backbone(
        backbone,
        strategy="lora_top_blocks",
        top_blocks=1,
        lora_rank=2,
    )
    model = SharedMultiviewFactorizedModel(backbone, backbone_dim=8, adapter_dim=8)

    groups = trainable_parameter_groups(
        model,
        head_lr=5e-4,
        adapter_lr=2e-4,
        lora_lr=1e-5,
        weight_decay=1e-2,
    )
    summary = parameter_count_summary(model)
    identifiers = [id(parameter) for group in groups for parameter in group["params"]]

    assert {group["group_name"] for group in groups} == {"head", "adapter", "lora"}
    assert len(identifiers) == len(set(identifiers))
    assert 0 < summary["trainable"] < summary["total"]
    assert 0.0 < summary["trainable_fraction"] < 1.0


def test_paired_eval_transform_is_deterministic_and_preserves_shape():
    tight = Image.new("RGB", (20, 40), color=(255, 20, 10))
    context = Image.new("RGB", (60, 30), color=(10, 20, 255))
    transform = PairedPersonSafeTransform(image_size=64, training=False)

    first = transform(tight, context)
    second = transform(tight, context)

    assert first[0].shape == (3, 64, 64)
    assert first[1].shape == (3, 64, 64)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    with pytest.raises(ValueError, match="at least 32"):
        PairedPersonSafeTransform(image_size=16)
