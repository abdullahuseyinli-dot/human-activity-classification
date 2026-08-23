"""Geometry and cohort helpers for bbox-aware POLAR faithfulness evaluation."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch

from .polar import context_box


def stable_seed(*parts: object) -> int:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**32)


def select_bbox_stratified_cohort(
    frame: pd.DataFrame,
    *,
    rows: int,
    seed: int,
    class_column: str = "label_4",
) -> pd.DataFrame:
    """Select equal counts per class and global bbox-area quartile."""

    if rows < 1:
        raise ValueError("rows must be positive")
    required = {"image_id", class_column, "bbox_area_fraction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cohort frame is missing columns: {sorted(missing)}")
    output = frame.copy()
    output["bbox_area_fraction"] = pd.to_numeric(output["bbox_area_fraction"])
    if output["bbox_area_fraction"].isna().any():
        raise ValueError("bbox_area_fraction contains missing values")
    output["bbox_area_quartile"] = pd.qcut(
        output["bbox_area_fraction"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="raise",
    ).astype(str)
    groups = list(output.groupby([class_column, "bbox_area_quartile"], sort=True))
    if rows % len(groups):
        raise ValueError("rows must divide evenly across class-by-area strata")
    per_group = rows // len(groups)
    selected = []
    for (class_name, quartile), group in groups:
        if len(group) < per_group:
            raise ValueError(
                f"Insufficient rows in stratum {(class_name, quartile)}: "
                f"need {per_group}, found {len(group)}"
            )
        random_state = stable_seed(seed, class_name, quartile)
        selected.append(group.sample(n=per_group, random_state=random_state, replace=False))
    cohort = pd.concat(selected, ignore_index=True)
    return cohort.sort_values("image_id", ignore_index=True)


def projected_person_box(
    row: pd.Series | dict,
    view: str,
    *,
    resize_shorter: int = 256,
    output_size: int = 224,
) -> tuple[int, int, int, int]:
    """Project a source-image bbox through the declared view, resize, and center crop."""

    source_width = int(row["actual_width"])
    source_height = int(row["actual_height"])
    person = tuple(
        float(row[name]) for name in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
    )
    if view == "full_frame":
        left, top, right, bottom = 0, 0, source_width, source_height
    elif view in {"person_context_10", "person_context_25"}:
        context = 0.10 if view == "person_context_10" else 0.25
        left, top, right, bottom = context_box(person, (source_width, source_height), context)
    else:
        raise ValueError(f"Unknown POLAR view: {view}")

    view_width = right - left
    view_height = bottom - top
    scale = float(resize_shorter) / float(min(view_width, view_height))
    resized_width = int(view_width * scale)
    resized_height = int(view_height * scale)
    crop_left = (resized_width - output_size) // 2
    crop_top = (resized_height - output_size) // 2

    coordinates = (
        (person[0] - left) * scale - crop_left,
        (person[1] - top) * scale - crop_top,
        (person[2] - left) * scale - crop_left,
        (person[3] - top) * scale - crop_top,
    )
    xmin = int(np.clip(np.floor(coordinates[0]), 0, output_size - 1))
    ymin = int(np.clip(np.floor(coordinates[1]), 0, output_size - 1))
    xmax = int(np.clip(np.ceil(coordinates[2]), xmin + 1, output_size))
    ymax = int(np.clip(np.ceil(coordinates[3]), ymin + 1, output_size))
    return xmin, ymin, xmax, ymax


def box_mask(
    box: tuple[int, int, int, int], output_size: tuple[int, int] = (224, 224)
) -> torch.Tensor:
    xmin, ymin, xmax, ymax = box
    height, width = output_size
    if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
        raise ValueError(f"Box lies outside the output canvas: {box}")
    mask = torch.zeros(output_size, dtype=torch.bool)
    mask[ymin:ymax, xmin:xmax] = True
    return mask


def area_matched_context_mask(person_mask: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Sample the same number of context pixels outside the person box."""

    if person_mask.ndim != 2 or person_mask.dtype != torch.bool:
        raise ValueError("person_mask must be a two-dimensional boolean tensor")
    person_count = int(person_mask.sum())
    available = torch.nonzero(~person_mask.reshape(-1), as_tuple=False).reshape(-1).numpy()
    if person_count > len(available):
        raise ValueError("The person box is larger than the available matched context")
    selected = np.random.default_rng(int(seed)).choice(available, size=person_count, replace=False)
    output = torch.zeros(person_mask.numel(), dtype=torch.bool)
    output[torch.from_numpy(selected)] = True
    return output.reshape(person_mask.shape)


def area_matched_occlusion_masks(
    person_mask: torch.Tensor, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return equal-area person and context masks, subsampling the larger region."""

    if person_mask.ndim != 2 or person_mask.dtype != torch.bool:
        raise ValueError("person_mask must be a two-dimensional boolean tensor")
    person_indices = torch.nonzero(person_mask.reshape(-1), as_tuple=False).reshape(-1).numpy()
    context_indices = torch.nonzero(~person_mask.reshape(-1), as_tuple=False).reshape(-1).numpy()
    count = min(len(person_indices), len(context_indices))
    if count < 1:
        raise ValueError("Both person and context regions must contain pixels")
    generator = np.random.default_rng(int(seed))
    selected_person = generator.choice(person_indices, size=count, replace=False)
    selected_context = generator.choice(context_indices, size=count, replace=False)
    output_person = torch.zeros(person_mask.numel(), dtype=torch.bool)
    output_context = torch.zeros(person_mask.numel(), dtype=torch.bool)
    output_person[torch.from_numpy(selected_person)] = True
    output_context[torch.from_numpy(selected_context)] = True
    match_fraction = float(count / max(len(person_indices), 1))
    return (
        output_person.reshape(person_mask.shape),
        output_context.reshape(person_mask.shape),
        match_fraction,
    )


def attribution_localization(
    attribution: np.ndarray, person_mask: torch.Tensor
) -> dict[str, float | bool]:
    values = np.asarray(attribution, dtype=np.float64)
    mask = person_mask.cpu().numpy().astype(bool)
    if values.shape != mask.shape:
        raise ValueError("Attribution and bbox mask must have the same shape")
    values = np.maximum(np.nan_to_num(values), 0.0)
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("Attribution map must have positive mass")
    values /= total
    peak = np.unravel_index(int(np.argmax(values)), values.shape)
    box_fraction = float(mask.mean())
    mass = float(values[mask].sum())
    return {
        "person_attribution_mass": mass,
        "person_area_fraction_transformed": box_fraction,
        "person_attribution_mass_lift": mass / max(box_fraction, 1e-12),
        "pointing_game": bool(mask[peak]),
    }


def flip_uint8_bits_exact(values: np.ndarray, *, bit_flips: int, seed: int) -> np.ndarray:
    """Flip an exact number of unique bits in an unsigned 8-bit array."""

    source = np.asarray(values)
    if source.dtype != np.uint8:
        raise ValueError("Bit-flip input must use uint8 storage")
    total_bits = source.size * 8
    if not 0 <= bit_flips <= total_bits:
        raise ValueError("bit_flips lies outside the available bit range")
    output = source.copy()
    if bit_flips == 0:
        return output
    positions = np.random.default_rng(int(seed)).choice(
        total_bits, size=int(bit_flips), replace=False
    )
    byte_indices = positions // 8
    masks = np.left_shift(np.uint8(1), (positions % 8).astype(np.uint8))
    flattened = output.reshape(-1)
    np.bitwise_xor.at(flattened, byte_indices, masks)
    return output


def quantize_and_flip_parameter_bits(
    values: torch.Tensor, *, bit_flips: int, seed: int
) -> tuple[torch.Tensor, float]:
    """Symmetrically quantize a tensor to int8, flip bits, and dequantize it."""

    source = values.detach().float().cpu().numpy()
    maximum = float(np.max(np.abs(source)))
    scale = maximum / 127.0 if maximum > 0.0 else 1.0
    quantized = np.clip(np.rint(source / scale), -127, 127).astype(np.int8)
    encoded = quantized.view(np.uint8)
    corrupted = flip_uint8_bits_exact(encoded, bit_flips=bit_flips, seed=seed)
    restored = corrupted.view(np.int8).astype(np.float32) * scale
    return torch.from_numpy(restored).reshape(values.shape), scale
