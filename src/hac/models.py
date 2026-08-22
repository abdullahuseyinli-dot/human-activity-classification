"""Transfer-learning model builders with explicit trainability policies."""

from __future__ import annotations

import re

import torch
from torch import nn
from torchvision import models
from torchvision.models import ConvNeXt_Small_Weights

from .config import ModelConfig


class DropoutLinear(nn.Linear):
    """A linear layer that records and applies input dropout explicitly."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.dropout_p = float(dropout)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(self.dropout(inputs), self.weight, self.bias)


class DinoV2Classifier(nn.Module):
    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-small",
        num_classes: int = 3,
        dropout: float = 0.0,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.backbone_name = backbone_name
        if pretrained:
            self.backbone = AutoModel.from_pretrained(backbone_name)
        else:
            configuration = AutoConfig.from_pretrained(
                backbone_name, local_files_only=True
            )
            self.backbone = AutoModel.from_config(configuration)
        hidden_size = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(hidden_size, int(num_classes))

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        return_features: bool = False,
    ):
        output = self.backbone(pixel_values=pixel_values)
        if getattr(output, "pooler_output", None) is not None:
            features = output.pooler_output
        else:
            features = output.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(features))
        return (logits, features) if return_features else logits


class ConvNeXtFeatureAdapter(nn.Module):
    """Expose ConvNeXt's pooled 768-dimensional representation on request."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    @property
    def classifier(self) -> nn.Module:
        return self.backbone.classifier

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        return_features: bool = False,
    ):
        activations = self.backbone.features(pixel_values)
        activations = self.backbone.avgpool(activations)
        activations = self.backbone.classifier[0](activations)
        features = self.backbone.classifier[1](activations)
        logits = self.backbone.classifier[2](features)
        return (logits, features) if return_features else logits


def _dino_block_names(model: DinoV2Classifier) -> list[str]:
    names = []
    for name, _ in model.backbone.named_parameters():
        match = re.search(r"(encoder\.layer\.\d+)", name)
        if match:
            names.append(match.group(1))
    return sorted(set(names), key=lambda item: int(item.rsplit(".", 1)[1]))


def _matches_block(parameter_name: str, block_name: str) -> bool:
    return (
        parameter_name.startswith(block_name)
        or f".{block_name}." in parameter_name
        or parameter_name == block_name
    )


def configure_convnext(model: nn.Module, strategy: str) -> nn.Module:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if strategy == "head_only":
        return model
    if strategy == "last_stage":
        for name, parameter in model.named_parameters():
            if name.startswith(("features.6", "features.7")):
                parameter.requires_grad = True
        return model
    if strategy == "full_backbone":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return model
    raise ValueError(f"Unsupported ConvNeXt strategy: {strategy}")


def configure_dinov2(
    model: DinoV2Classifier, strategy: str, top_n_blocks: int | None
) -> DinoV2Classifier:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if strategy == "probe_only":
        return model
    if strategy == "top_blocks":
        block_names = _dino_block_names(model)
        selected = set(block_names[-int(top_n_blocks or 0) :])
        for name, parameter in model.backbone.named_parameters():
            if any(_matches_block(name, block) for block in selected):
                parameter.requires_grad = True
            if "layernorm" in name.lower() or "norm" in name.lower():
                parameter.requires_grad = True
        return model
    if strategy == "full_backbone":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return model
    raise ValueError(f"Unsupported DINOv2 strategy: {strategy}")


def build_model(
    config: ModelConfig,
    num_classes: int = 3,
    *,
    pretrained: bool = True,
) -> nn.Module:
    if config.model_kind == "convnext_small":
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_small(weights=weights)
        in_features = int(model.classifier[2].in_features)
        model.classifier[2] = DropoutLinear(
            in_features,
            num_classes,
            dropout=config.dropout,
        )
        model = configure_convnext(model, config.unfreeze_strategy)
        return ConvNeXtFeatureAdapter(model)
    model = DinoV2Classifier(
        num_classes=num_classes,
        dropout=config.dropout,
        pretrained=pretrained,
    )
    return configure_dinov2(model, config.unfreeze_strategy, config.top_n_blocks)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": int(total), "trainable": int(trainable), "frozen": int(total - trainable)}
