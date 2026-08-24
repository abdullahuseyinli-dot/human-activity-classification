"""Paired-view neural components for the gated V-COCO v3 experiments."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from hac.augmentations import IMAGENET_MEAN, IMAGENET_STD, SquarePad
from hac.polar import image_view

NEURAL_CLASS_NAMES = ("sitting", "standing", "walking_running")


@dataclass(frozen=True)
class MultiViewOutput:
    """Outputs needed for supervised training, diagnostics, and calibration."""

    probabilities: torch.Tensor
    posture_logits: torch.Tensor
    motion_logits: torch.Tensor
    tight_posture_logits: torch.Tensor
    tight_motion_logits: torch.Tensor
    context_posture_logits: torch.Tensor
    context_motion_logits: torch.Tensor
    gate_weights: torch.Tensor
    fused_features: torch.Tensor


def extract_backbone_features(output: object) -> torch.Tensor:
    """Extract one embedding per image from common Hugging Face model outputs."""

    if isinstance(output, torch.Tensor):
        if output.ndim == 2:
            return output
        if output.ndim == 3:
            return output[:, 0]
        raise ValueError(f"Unsupported tensor output shape: {tuple(output.shape)}")
    pooler = getattr(output, "pooler_output", None)
    if pooler is not None:
        return pooler
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        return hidden[:, 0]
    if isinstance(output, Mapping):
        if output.get("pooler_output") is not None:
            return output["pooler_output"]
        if output.get("last_hidden_state") is not None:
            return output["last_hidden_state"][:, 0]
    raise TypeError("Backbone output has neither pooler_output nor last_hidden_state")


def decode_factorized_logits(
    posture_logits: torch.Tensor,
    motion_logits: torch.Tensor,
) -> torch.Tensor:
    """Decode seated/upright and stationary/locomoting logits into three classes."""

    if posture_logits.shape != motion_logits.shape or posture_logits.shape[-1] != 2:
        raise ValueError("Posture and motion logits must have matching [batch, 2] shapes")
    posture = posture_logits.softmax(dim=-1)
    motion = motion_logits.softmax(dim=-1)
    probabilities = torch.stack(
        (
            posture[:, 0],
            posture[:, 1] * motion[:, 0],
            posture[:, 1] * motion[:, 1],
        ),
        dim=1,
    )
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)


class ProjectionAdapter(nn.Module):
    """Small residual-free adapter that maps a frozen embedding into fusion space."""

    def __init__(self, input_dim: int, adapter_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(adapter_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(adapter_dim), int(adapter_dim)),
            nn.LayerNorm(int(adapter_dim)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class FactorizedClassifier(nn.Module):
    def __init__(self, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(int(feature_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.posture = nn.Linear(int(feature_dim), 2)
        self.motion = nn.Linear(int(feature_dim), 2)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.dropout(self.normalization(features))
        return self.posture(values), self.motion(values)


class SharedMultiviewFactorizedModel(nn.Module):
    """Share one backbone across tight/context views and learn reliability-aware fusion."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        backbone_dim: int,
        adapter_dim: int = 256,
        geometry_dim: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if backbone_dim < 1 or adapter_dim < 1 or geometry_dim < 1:
            raise ValueError("Model dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.backbone = backbone
        self.adapter = ProjectionAdapter(backbone_dim, adapter_dim, dropout)
        self.geometry_norm = nn.LayerNorm(geometry_dim)
        gate_input_dim = adapter_dim * 4 + geometry_dim
        self.reliability_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, 2),
        )
        self.interaction = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.interaction_scale = nn.Parameter(torch.tensor(0.0))
        self.fusion_norm = nn.LayerNorm(adapter_dim)
        self.classifier = FactorizedClassifier(adapter_dim, dropout)

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        try:
            output = self.backbone(pixel_values=pixel_values)
        except TypeError:
            output = self.backbone(pixel_values)
        return self.adapter(extract_backbone_features(output))

    def forward(
        self,
        tight_pixels: torch.Tensor,
        context_pixels: torch.Tensor,
        geometry: torch.Tensor,
    ) -> MultiViewOutput:
        tight = self.encode(tight_pixels)
        context = self.encode(context_pixels)
        if geometry.ndim != 2 or geometry.shape[0] != tight.shape[0]:
            raise ValueError("Geometry must have shape [batch, geometry_dim]")
        normalized_geometry = self.geometry_norm(geometry)
        gate_input = torch.cat(
            (tight, context, torch.abs(tight - context), tight * context, normalized_geometry),
            dim=1,
        )
        gate_weights = self.reliability_gate(gate_input).softmax(dim=1)
        fused = gate_weights[:, :1] * tight + gate_weights[:, 1:] * context
        interaction = self.interaction(tight * context)
        fused = self.fusion_norm(fused + torch.tanh(self.interaction_scale) * interaction)

        posture_logits, motion_logits = self.classifier(fused)
        tight_posture, tight_motion = self.classifier(tight)
        context_posture, context_motion = self.classifier(context)
        return MultiViewOutput(
            probabilities=decode_factorized_logits(posture_logits, motion_logits),
            posture_logits=posture_logits,
            motion_logits=motion_logits,
            tight_posture_logits=tight_posture,
            tight_motion_logits=tight_motion,
            context_posture_logits=context_posture,
            context_motion_logits=context_motion,
            gate_weights=gate_weights,
            fused_features=fused,
        )


def _factorized_cross_entropy(
    posture_logits: torch.Tensor,
    motion_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float,
    posture_weight: torch.Tensor | None,
    motion_weight: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if labels.ndim != 1 or not torch.all((labels >= 0) & (labels <= 2)):
        raise ValueError("Factorized labels must be a one-dimensional 0/1/2 tensor")
    posture_targets = (labels != 0).long()
    posture_loss = F.cross_entropy(
        posture_logits,
        posture_targets,
        weight=posture_weight,
        label_smoothing=float(label_smoothing),
    )
    upright = labels != 0
    if not torch.any(upright):
        motion_loss = motion_logits.sum() * 0.0
    else:
        motion_targets = (labels[upright] == 2).long()
        motion_loss = F.cross_entropy(
            motion_logits[upright],
            motion_targets,
            weight=motion_weight,
            label_smoothing=float(label_smoothing),
        )
    return posture_loss, motion_loss


def multiview_factorized_loss(
    output: MultiViewOutput,
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    auxiliary_view_weight: float = 0.2,
    posture_weight: torch.Tensor | None = None,
    motion_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Hierarchical likelihood plus light auxiliary supervision of both views."""

    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if not 0.0 <= auxiliary_view_weight <= 1.0:
        raise ValueError("auxiliary_view_weight must be in [0, 1]")
    posture, motion = _factorized_cross_entropy(
        output.posture_logits,
        output.motion_logits,
        labels,
        label_smoothing=label_smoothing,
        posture_weight=posture_weight,
        motion_weight=motion_weight,
    )
    tight_posture, tight_motion = _factorized_cross_entropy(
        output.tight_posture_logits,
        output.tight_motion_logits,
        labels,
        label_smoothing=label_smoothing,
        posture_weight=posture_weight,
        motion_weight=motion_weight,
    )
    context_posture, context_motion = _factorized_cross_entropy(
        output.context_posture_logits,
        output.context_motion_logits,
        labels,
        label_smoothing=label_smoothing,
        posture_weight=posture_weight,
        motion_weight=motion_weight,
    )
    auxiliary = 0.5 * (tight_posture + tight_motion + context_posture + context_motion)
    total = posture + motion + float(auxiliary_view_weight) * auxiliary
    return {
        "loss": total,
        "posture_loss": posture,
        "motion_loss": motion,
        "auxiliary_view_loss": auxiliary,
    }


class LoRALinear(nn.Module):
    """Frozen linear layer with a trainable low-rank residual update."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank < 1 or rank > min(base.in_features, base.out_features):
            raise ValueError("LoRA rank must fit the wrapped linear layer")
        if alpha <= 0.0 or not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA alpha must be positive and dropout must be in [0, 1)")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(base.in_features, int(rank), bias=False)
        self.lora_b = nn.Linear(int(rank), base.out_features, bias=False)
        self.scaling = float(alpha) / int(rank)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        base = self.base(values)
        residual = self.lora_b(self.lora_a(self.lora_dropout(values)))
        return base + self.scaling * residual


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _resolve_path(module: nn.Module, path: str) -> object:
    current: object = module
    for part in path.split("."):
        current = getattr(current, part)
    return current


def transformer_blocks(backbone: nn.Module) -> Sequence[nn.Module]:
    """Return the ordered transformer blocks for supported HF-style backbones."""

    candidates = (
        "encoder.layer",
        "encoder.layers",
        "model.encoder.layer",
        "model.encoder.layers",
        "model.layer",
        "backbone.encoder.layer",
        "backbone.encoder.layers",
        "blocks",
    )
    for path in candidates:
        try:
            value = _resolve_path(backbone, path)
        except AttributeError:
            continue
        if isinstance(value, nn.ModuleList | nn.Sequential) and len(value) > 0:
            return value
    raise ValueError("Could not locate ordered transformer blocks in the backbone")


def _parent_and_leaf(module: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora_top_blocks(
    backbone: nn.Module,
    *,
    top_blocks: int,
    rank: int,
    alpha: float,
    dropout: float,
    target_leaf_names: Iterable[str] = ("query", "value", "q_proj", "v_proj"),
) -> list[str]:
    """Inject LoRA into query/value projections of the selected top blocks."""

    if top_blocks < 1:
        raise ValueError("top_blocks must be positive")
    blocks = transformer_blocks(backbone)
    if top_blocks > len(blocks):
        raise ValueError(f"Requested {top_blocks} top blocks from a {len(blocks)}-block model")
    freeze_module(backbone)
    targets = frozenset(map(str, target_leaf_names))
    replacements: list[str] = []
    first_selected = len(blocks) - int(top_blocks)
    for block_index, block in enumerate(blocks[first_selected:], start=first_selected):
        candidates = [
            name
            for name, child in block.named_modules()
            if name and isinstance(child, nn.Linear) and name.rsplit(".", 1)[-1] in targets
        ]
        for name in candidates:
            parent, leaf = _parent_and_leaf(block, name)
            base = getattr(parent, leaf)
            if isinstance(base, LoRALinear):
                raise RuntimeError(f"LoRA already injected at block {block_index}.{name}")
            setattr(
                parent,
                leaf,
                LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout),
            )
            replacements.append(f"block.{block_index}.{name}")
    if not replacements:
        raise ValueError("No declared query/value projections were found for LoRA injection")
    return replacements


def configure_parameter_efficient_backbone(
    backbone: nn.Module,
    *,
    strategy: str,
    top_blocks: int = 0,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
) -> list[str]:
    """Freeze a backbone and optionally add the preregistered LoRA parameters."""

    freeze_module(backbone)
    if strategy == "adapter_only":
        return []
    if strategy != "lora_top_blocks":
        raise ValueError(f"Unknown parameter-efficient strategy: {strategy}")
    return inject_lora_top_blocks(
        backbone,
        top_blocks=top_blocks,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )


def trainable_parameter_groups(
    model: SharedMultiviewFactorizedModel,
    *,
    head_lr: float,
    adapter_lr: float,
    lora_lr: float,
    weight_decay: float,
) -> list[dict]:
    """Build disjoint optimizer groups and reject accidentally unfrozen weights."""

    if min(head_lr, adapter_lr, lora_lr) <= 0.0 or weight_decay < 0.0:
        raise ValueError("Learning rates must be positive and weight decay nonnegative")
    groups: dict[str, list[nn.Parameter]] = {"head": [], "adapter": [], "lora": []}
    unexpected_backbone = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            if ".lora_a." not in name and ".lora_b." not in name:
                unexpected_backbone.append(name)
            else:
                groups["lora"].append(parameter)
        elif name.startswith("adapter."):
            groups["adapter"].append(parameter)
        else:
            groups["head"].append(parameter)
    if unexpected_backbone:
        raise RuntimeError(f"Unexpected trainable backbone parameters: {unexpected_backbone[:3]}")
    learning_rates = {"head": head_lr, "adapter": adapter_lr, "lora": lora_lr}
    output = []
    for name in ("head", "adapter", "lora"):
        if groups[name]:
            output.append(
                {
                    "params": groups[name],
                    "lr": float(learning_rates[name]),
                    "weight_decay": float(weight_decay),
                    "group_name": name,
                }
            )
    if not output:
        raise RuntimeError("Model has no trainable parameters")
    return output


def parameter_count_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total": int(total),
        "trainable": int(trainable),
        "trainable_fraction": float(trainable / total) if total else 0.0,
    }


class PairedPersonSafeTransform:
    """Apply the same mild geometry and colour perturbation to both person views."""

    def __init__(
        self,
        *,
        image_size: int = 224,
        random_erasing_probability: float = 0.1,
        training: bool = True,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        fill: tuple[int, int, int] = (124, 116, 104),
    ) -> None:
        if image_size < 32:
            raise ValueError("image_size must be at least 32")
        if not 0.0 <= random_erasing_probability <= 1.0:
            raise ValueError("random erasing probability must be in [0, 1]")
        self.image_size = int(image_size)
        self.random_erasing_probability = float(random_erasing_probability)
        self.training = bool(training)
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)
        self.fill = tuple(int(value) for value in fill)
        self.square_pad = SquarePad(fill=self.fill)

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return float(torch.empty(1).uniform_(low, high).item())

    def _prepare(self, image: Image.Image) -> Image.Image:
        return TF.resize(
            self.square_pad(image),
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def _shared_augment(self, images: list[Image.Image]) -> list[Image.Image]:
        if torch.rand(1).item() < 0.5:
            images = [TF.hflip(image) for image in images]
        if torch.rand(1).item() < 0.45:
            angle = self._uniform(-5.0, 5.0)
            max_shift = round(self.image_size * 0.02)
            translations = (
                int(round(self._uniform(-max_shift, max_shift))),
                int(round(self._uniform(-max_shift, max_shift))),
            )
            scale = self._uniform(0.90, 1.0)
            shear = [self._uniform(-2.0, 2.0), 0.0]
            images = [
                TF.affine(
                    image,
                    angle=angle,
                    translate=translations,
                    scale=scale,
                    shear=shear,
                    interpolation=InterpolationMode.BICUBIC,
                    fill=self.fill,
                )
                for image in images
            ]
        brightness = self._uniform(0.82, 1.18)
        contrast = self._uniform(0.82, 1.18)
        saturation = self._uniform(0.88, 1.12)
        hue = self._uniform(-0.025, 0.025)
        output = []
        for image in images:
            image = TF.adjust_brightness(image, brightness)
            image = TF.adjust_contrast(image, contrast)
            image = TF.adjust_saturation(image, saturation)
            output.append(TF.adjust_hue(image, hue))
        return output

    def _shared_erasing(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        if torch.rand(1).item() >= self.random_erasing_probability:
            return tensors
        area = self.image_size * self.image_size
        target_area = self._uniform(0.01, 0.05) * area
        aspect = math.exp(self._uniform(math.log(0.5), math.log(2.0)))
        height = min(self.image_size, max(1, int(round(math.sqrt(target_area * aspect)))))
        width = min(self.image_size, max(1, int(round(math.sqrt(target_area / aspect)))))
        top = int(torch.randint(0, self.image_size - height + 1, (1,)).item())
        left = int(torch.randint(0, self.image_size - width + 1, (1,)).item())
        return [TF.erase(value, top, left, height, width, 0.0, inplace=False) for value in tensors]

    def __call__(
        self, tight: Image.Image, context: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = [self._prepare(tight), self._prepare(context)]
        if self.training:
            images = self._shared_augment(images)
        tensors = [TF.to_tensor(image) for image in images]
        if self.training:
            tensors = self._shared_erasing(tensors)
        tensors = [TF.normalize(value, self.mean, self.std) for value in tensors]
        return tensors[0], tensors[1]


def pinned_backbone_spec(model_kind: str) -> dict:
    if model_kind not in {"dinov2_base", "dinov3_base"}:
        raise ValueError(f"Unsupported multiview backbone: {model_kind}")
    from hac.vcoco_v3_representations import model_provenance

    return model_provenance(model_kind)


def build_pinned_backbone(model_kind: str) -> tuple[nn.Module, int]:
    """Load one exact DINO revision and return its declared embedding width."""

    from transformers import AutoModel

    specification = pinned_backbone_spec(model_kind)
    backbone = AutoModel.from_pretrained(
        specification["model_id"],
        revision=specification["revision"],
    )
    hidden_size = int(backbone.config.hidden_size)
    if hidden_size < 1:
        raise RuntimeError("The pinned backbone has an invalid hidden size")
    return backbone, hidden_size


def row_geometry(row: Mapping) -> torch.Tensor:
    area = max(float(row["bbox_area_fraction"]), 1e-8)
    aspect = max(float(row["bbox_aspect_ratio"]), 1e-6)
    center_x = float(row["bbox_center_x_fraction"])
    center_y = float(row["bbox_center_y_fraction"])
    height = max(float(row["person_pixel_height"]), 1.0)
    edge_distance = min(center_x, 1.0 - center_x, center_y, 1.0 - center_y)
    return torch.tensor(
        [math.log(area), math.log(aspect), center_x, center_y, math.log(height), edge_distance],
        dtype=torch.float32,
    )


class PairedPersonDataset(Dataset):
    """Load aligned tight and context views without exposing test-set rows."""

    def __init__(
        self,
        frame: pd.DataFrame,
        transform: PairedPersonSafeTransform,
        *,
        context_view: str = "person_context_25",
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform
        self.context_view = str(context_view)
        self.class_to_index = {name: index for index, name in enumerate(NEURAL_CLASS_NAMES)}
        if "image_path" in self.frame:
            self.path_column = "image_path"
        elif "resolved_image_path" in self.frame:
            self.path_column = "resolved_image_path"
        else:
            raise ValueError("Paired rows require image_path or resolved_image_path")
        label_column = "label_3" if "label_3" in self.frame else "label"
        if label_column not in self.frame:
            raise ValueError("Paired rows require label_3 or label")
        self.label_column = label_column
        unknown = sorted(set(self.frame[label_column].astype(str)) - set(self.class_to_index))
        if unknown:
            raise ValueError(f"Unknown paired-view labels: {unknown}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        image_path = Path(str(row[self.path_column]))
        with Image.open(image_path) as image:
            source = image.convert("RGB")
            tight = image_view(source, row, "person_tight")
            context = image_view(source, row, self.context_view)
            tight_pixels, context_pixels = self.transform(tight, context)
        return {
            "tight_pixels": tight_pixels,
            "context_pixels": context_pixels,
            "geometry": row_geometry(row),
            "label": self.class_to_index[str(row[self.label_column])],
            "person_id": str(row.get("person_id", index)),
            "image_id": str(row["image_id"]),
        }
