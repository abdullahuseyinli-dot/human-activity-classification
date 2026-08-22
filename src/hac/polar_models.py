"""Pinned model builders isolated from the completed legacy experiment."""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .models import build_model, configure_dinov2

DINO_MODEL_SPECS = {
    "dinov2_small": {
        "model_id": "facebook/dinov2-small",
        "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
    },
    "dinov2_base": {
        "model_id": "facebook/dinov2-base",
        "revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
    },
}


class PinnedDinoV2Classifier(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str,
        backbone_revision: str,
        num_classes: int,
        dropout: float,
        pretrained: bool,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.backbone_name = backbone_name
        self.backbone_revision = backbone_revision
        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                backbone_name,
                revision=backbone_revision,
            )
        else:
            configuration = AutoConfig.from_pretrained(
                backbone_name,
                revision=backbone_revision,
                local_files_only=True,
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
        features = (
            output.pooler_output
            if getattr(output, "pooler_output", None) is not None
            else output.last_hidden_state[:, 0]
        )
        logits = self.classifier(self.dropout(features))
        return (logits, features) if return_features else logits


def build_polar_model(
    config: ModelConfig,
    num_classes: int,
    *,
    pretrained: bool = True,
) -> nn.Module:
    if config.model_kind == "convnext_small":
        return build_model(config, num_classes=num_classes, pretrained=pretrained)
    specification = DINO_MODEL_SPECS[config.model_kind]
    model = PinnedDinoV2Classifier(
        backbone_name=specification["model_id"],
        backbone_revision=specification["revision"],
        num_classes=num_classes,
        dropout=config.dropout,
        pretrained=pretrained,
    )
    return configure_dinov2(model, config.unfreeze_strategy, config.top_n_blocks)
