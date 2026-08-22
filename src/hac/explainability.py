"""Attribution and faithfulness primitives for the locked image classifiers.

The functions in this module deliberately target the probability function that
was selected by the experiment protocol.  Temperature scaling and horizontal-
flip TTA therefore remain inside the differentiated graph instead of being
approximated after an attribution has been produced.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F

EvaluationPolicy = Literal["center_crop", "center_plus_horizontal_flip"]


def _autocast_context(inputs: torch.Tensor):
    if inputs.is_cuda:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@dataclass(frozen=True, slots=True)
class ConvAttributions:
    """Class-specific CAM variants derived from one backward pass."""

    gradcam: np.ndarray
    hirescam: np.ndarray
    gradcam_views: tuple[np.ndarray, ...]
    hirescam_views: tuple[np.ndarray, ...]


@dataclass(frozen=True, slots=True)
class DinoAttributions:
    """Raw and target-conditioned transformer attention maps."""

    attention_rollout: np.ndarray
    gradient_attention_rollout: np.ndarray


def calibrated_probabilities(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    temperature: float,
    policy: EvaluationPolicy,
) -> torch.Tensor:
    """Evaluate the exact per-seed probability policy used by the locked run."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    with _autocast_context(inputs):
        center_logits = model(inputs)
        flip_logits = (
            model(torch.flip(inputs, dims=(-1,)))
            if policy == "center_plus_horizontal_flip"
            else None
        )
    center_logits = center_logits.float()
    if policy == "center_crop":
        return torch.softmax(center_logits / float(temperature), dim=-1)
    if policy != "center_plus_horizontal_flip":
        raise ValueError(f"Unsupported evaluation policy: {policy}")

    assert flip_logits is not None
    center = torch.softmax(center_logits, dim=-1)
    flipped = torch.softmax(flip_logits.float(), dim=-1)
    averaged = 0.5 * (center + flipped)
    pseudo_logits = torch.log(averaged.clamp_min(1e-12))
    return torch.softmax(pseudo_logits / float(temperature), dim=-1)


def normalize_attribution(values: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return a finite non-negative attribution map with unit mass."""

    if isinstance(values, torch.Tensor):
        array = values.detach().float().cpu().numpy()
    else:
        array = np.asarray(values, dtype=np.float64)
    array = np.squeeze(array).astype(np.float64, copy=False)
    if array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional map, found shape {array.shape}")
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.maximum(array, 0.0)
    total = float(array.sum())
    if total <= 1e-12:
        return np.full(array.shape, 1.0 / float(array.size), dtype=np.float32)
    return (array / total).astype(np.float32)


def _upsample_map(values: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        values[:, None], size=output_size, mode="bilinear", align_corners=False
    )[:, 0]


def conv_cam_attributions(
    model: nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    *,
    temperature: float,
    policy: EvaluationPolicy,
    target_layer: nn.Module | None = None,
) -> ConvAttributions:
    """Compute Grad-CAM and HiResCAM for one image and the locked score function."""

    if inputs.shape[0] != 1:
        raise ValueError("CAM attribution currently expects a single image")
    if target_layer is None:
        try:
            target_layer = model.backbone.features[-1][-1]
        except AttributeError as exc:  # pragma: no cover - defensive API guard
            raise ValueError("A ConvNeXt target layer must be supplied") from exc

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor | None] = []

    def capture(_module, _arguments, output):
        index = len(activations)
        activations.append(output)
        gradients.append(None)

        def capture_gradient(gradient, *, position=index):
            gradients[position] = gradient

        output.register_hook(capture_gradient)

    handle = target_layer.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        probabilities = calibrated_probabilities(
            model, inputs, temperature=temperature, policy=policy
        )
        probabilities[0, int(target_class)].backward()
    finally:
        handle.remove()

    expected_views = 2 if policy == "center_plus_horizontal_flip" else 1
    if len(activations) != expected_views or any(item is None for item in gradients):
        raise RuntimeError("ConvNeXt CAM hooks did not capture every inference view")

    gradcam_views: list[np.ndarray] = []
    hirescam_views: list[np.ndarray] = []
    gradcam_raw: list[torch.Tensor] = []
    hirescam_raw: list[torch.Tensor] = []
    for index, (activation, gradient) in enumerate(zip(activations, gradients, strict=True)):
        assert gradient is not None
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        gradcam = torch.relu((weights * activation).sum(dim=1))
        hirescam = torch.relu((gradient * activation).sum(dim=1))
        gradcam = _upsample_map(gradcam, tuple(inputs.shape[-2:]))
        hirescam = _upsample_map(hirescam, tuple(inputs.shape[-2:]))
        if index == 1:
            gradcam = torch.flip(gradcam, dims=(-1,))
            hirescam = torch.flip(hirescam, dims=(-1,))
        gradcam_raw.append(gradcam)
        hirescam_raw.append(hirescam)
        gradcam_views.append(normalize_attribution(gradcam[0]))
        hirescam_views.append(normalize_attribution(hirescam[0]))

    return ConvAttributions(
        gradcam=normalize_attribution(torch.stack(gradcam_raw).sum(dim=0)[0]),
        hirescam=normalize_attribution(torch.stack(hirescam_raw).sum(dim=0)[0]),
        gradcam_views=tuple(gradcam_views),
        hirescam_views=tuple(hirescam_views),
    )


def _rollout(attention_layers: list[torch.Tensor]) -> torch.Tensor:
    if not attention_layers:
        raise ValueError("At least one attention layer is required")
    token_count = int(attention_layers[0].shape[-1])
    result = torch.eye(token_count, device=attention_layers[0].device)
    identity = torch.eye(token_count, device=result.device)
    for layer in attention_layers:
        if layer.shape != (token_count, token_count):
            raise ValueError("Attention layers must use a consistent square token matrix")
        adjusted = layer + identity
        adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        result = adjusted @ result
    return result


def _cls_patch_map(rollout: torch.Tensor, output_size: tuple[int, int]) -> np.ndarray:
    patch_values = rollout[0, 1:]
    side = int(round(float(patch_values.numel()) ** 0.5))
    if side * side != patch_values.numel():
        raise RuntimeError("DINOv2 patch tokens do not form a square grid")
    patch_values = patch_values.reshape(1, 1, side, side)
    full = F.interpolate(
        patch_values, size=output_size, mode="bilinear", align_corners=False
    )[0, 0]
    return normalize_attribution(full)


def dino_attention_attributions(
    model: nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    *,
    temperature: float,
) -> DinoAttributions:
    """Compute raw rollout and a class-specific gradient-attention rollout."""

    if inputs.shape[0] != 1:
        raise ValueError("DINOv2 attention attribution currently expects one image")
    original_attention_implementation = getattr(
        model.backbone.config, "_attn_implementation", None
    )
    if hasattr(model.backbone, "set_attn_implementation"):
        model.backbone.set_attn_implementation("eager")

    try:
        model.zero_grad(set_to_none=True)
        with _autocast_context(inputs):
            output = model.backbone(pixel_values=inputs, output_attentions=True)
            attentions = list(output.attentions or ())
            if not attentions:
                raise RuntimeError("DINOv2 did not return attention tensors")
            features = (
                output.pooler_output
                if getattr(output, "pooler_output", None) is not None
                else output.last_hidden_state[:, 0]
            )
            logits = model.classifier(model.dropout(features))
        logits = logits.float()
        score = torch.softmax(logits / float(temperature), dim=-1)[0, int(target_class)]
        gradients = torch.autograd.grad(score, attentions, allow_unused=False)
    finally:
        if (
            original_attention_implementation is not None
            and hasattr(model.backbone, "set_attn_implementation")
        ):
            model.backbone.set_attn_implementation(original_attention_implementation)

    raw_layers = [attention.detach().mean(dim=1)[0] for attention in attentions]
    gradient_layers = [
        torch.relu((gradient * attention).mean(dim=1)[0]).detach()
        for attention, gradient in zip(attentions, gradients, strict=True)
    ]
    raw_rollout = _rollout(raw_layers)
    gradient_rollout = _rollout(gradient_layers)
    output_size = tuple(inputs.shape[-2:])
    return DinoAttributions(
        attention_rollout=_cls_patch_map(raw_rollout, output_size),
        gradient_attention_rollout=_cls_patch_map(gradient_rollout, output_size),
    )


def integrated_gradients_attribution(
    model: nn.Module,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    target_class: int,
    *,
    temperature: float,
    policy: EvaluationPolicy,
    steps: int = 16,
    internal_batch_size: int = 4,
) -> np.ndarray:
    """Integrated gradients for the exact calibrated per-seed probability."""

    if inputs.shape != baseline.shape or inputs.shape[0] != 1:
        raise ValueError("Integrated gradients expects one aligned input/baseline pair")
    if steps < 2 or internal_batch_size < 1:
        raise ValueError("Invalid integrated-gradient sampling configuration")
    alphas = (torch.arange(steps, device=inputs.device, dtype=inputs.dtype) + 0.5) / steps
    accumulated = torch.zeros_like(inputs)
    delta = inputs - baseline
    for start in range(0, steps, internal_batch_size):
        alpha = alphas[start : start + internal_batch_size, None, None, None]
        points = baseline + alpha * delta
        points = points.detach().requires_grad_(True)
        probabilities = calibrated_probabilities(
            model, points, temperature=temperature, policy=policy
        )
        objective = probabilities[:, int(target_class)].sum()
        gradient = torch.autograd.grad(objective, points)[0]
        accumulated += gradient.sum(dim=0, keepdim=True)
    integrated = delta * accumulated / float(steps)
    positive = torch.relu(integrated.sum(dim=1))[0]
    if float(positive.sum()) <= 1e-12:
        positive = integrated.abs().sum(dim=1)[0]
    return normalize_attribution(positive)


def gaussian_baseline(inputs: torch.Tensor, kernel_size: int = 31) -> torch.Tensor:
    """Create a deterministic blurred reference without changing normalization."""

    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer of at least three")
    sigma = max(float(kernel_size) / 6.0, 1.0)
    coordinates = torch.arange(kernel_size, device=inputs.device, dtype=inputs.dtype)
    coordinates = coordinates - (kernel_size - 1) / 2
    kernel = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    horizontal = kernel.view(1, 1, 1, -1).repeat(inputs.shape[1], 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(inputs.shape[1], 1, 1, 1)
    padding = kernel_size // 2
    blurred = F.conv2d(inputs, horizontal, padding=(0, padding), groups=inputs.shape[1])
    return F.conv2d(blurred, vertical, padding=(padding, 0), groups=inputs.shape[1])


def patch_scores(attribution: np.ndarray, grid_size: int = 16) -> np.ndarray:
    """Aggregate an image-space map into a common square perturbation grid."""

    tensor = torch.as_tensor(attribution, dtype=torch.float32)[None, None]
    pooled = F.adaptive_avg_pool2d(tensor, (grid_size, grid_size))[0, 0]
    return normalize_attribution(pooled)


def patch_keep_mask(
    scores: np.ndarray,
    fraction: float,
    *,
    order: Literal["most", "least", "random"],
    random_seed: int = 0,
    output_size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Return a pixel mask after removing an exact fraction of ranked patches."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Patch scores must be a square matrix")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    count = values.size
    remove_count = int(round(float(fraction) * count))
    if order == "most":
        ordering = np.argsort(-values.reshape(-1), kind="stable")
    elif order == "least":
        ordering = np.argsort(values.reshape(-1), kind="stable")
    elif order == "random":
        ordering = np.random.default_rng(int(random_seed)).permutation(count)
    else:
        raise ValueError(f"Unknown patch order: {order}")
    keep = np.ones(count, dtype=np.float32)
    keep[ordering[:remove_count]] = 0.0
    keep_tensor = torch.from_numpy(keep.reshape(values.shape))[None, None]
    return F.interpolate(keep_tensor, size=output_size, mode="nearest")[0, 0]


def attribution_spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman similarity with deterministic handling of constant maps."""

    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    if np.allclose(left_values, left_values[0]) and np.allclose(
        right_values, right_values[0]
    ):
        return 1.0 if np.allclose(left_values, right_values) else 0.0
    result = spearmanr(left_values, right_values).statistic
    return 0.0 if not np.isfinite(result) else float(result)


def curve_auc(fractions: np.ndarray, values: np.ndarray) -> float:
    """Trapezoidal area under a curve on a validated increasing grid."""

    x = np.asarray(fractions, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("AUC inputs must be aligned one-dimensional arrays")
    if len(x) < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("AUC fractions must be strictly increasing")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))
