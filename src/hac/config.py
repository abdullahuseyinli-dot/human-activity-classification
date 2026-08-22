"""Validated experiment configuration objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

ModelKind = Literal["convnext_small", "dinov2_small", "dinov2_base"]
UnfreezeStrategy = Literal["head_only", "last_stage", "probe_only", "top_blocks", "full_backbone"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_kind: ModelKind
    augmentation_strength: str
    batch_size: int
    head_lr: float
    backbone_lr: float
    weight_decay: float
    label_smoothing: float = 0.0
    dropout: float = 0.0
    mixup_alpha: float = 0.0
    unfreeze_strategy: UnfreezeStrategy = "full_backbone"
    top_n_blocks: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.head_lr <= 0.0 or self.backbone_lr <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.mixup_alpha < 0.0:
            raise ValueError("mixup_alpha cannot be negative")

        allowed = {
            "convnext_small": {"head_only", "last_stage", "full_backbone"},
            "dinov2_small": {"probe_only", "top_blocks", "full_backbone"},
            "dinov2_base": {"probe_only", "top_blocks", "full_backbone"},
        }
        if self.unfreeze_strategy not in allowed[self.model_kind]:
            raise ValueError(f"{self.unfreeze_strategy!r} is invalid for {self.model_kind}")
        if self.unfreeze_strategy == "top_blocks":
            if self.top_n_blocks is None or self.top_n_blocks < 1:
                raise ValueError("top_blocks requires a positive top_n_blocks value")
        elif self.top_n_blocks is not None:
            raise ValueError("top_n_blocks is only valid with the top_blocks strategy")

    @classmethod
    def from_mapping(cls, model_kind: ModelKind, values: dict) -> ModelConfig:
        return cls(model_kind=model_kind, **values)

    def as_dict(self, *, include_model_kind: bool = True) -> dict:
        values = asdict(self)
        if not include_model_kind:
            values.pop("model_kind")
        if values.get("top_n_blocks") is None:
            values.pop("top_n_blocks", None)
        return values
