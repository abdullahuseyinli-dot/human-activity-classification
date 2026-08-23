"""Feature-cache contracts shared by the frozen POLAR experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


def official_multilayer_features(
    hidden_states: tuple[torch.Tensor, ...], layernorm: nn.Module
) -> torch.Tensor:
    """Match the feature layout used by the official DINOv2 linear classifier."""

    if len(hidden_states) < 4:
        raise ValueError("DINOv2 multi-layer features require at least four hidden states")
    normalized = [layernorm(state) for state in hidden_states[-4:]]
    class_tokens = [state[:, 0] for state in normalized]
    mean_patch_token = normalized[-1][:, 1:].mean(dim=1)
    return torch.cat([*class_tokens, mean_patch_token], dim=1)


class PinnedDinoFeatureModel(nn.Module):
    """Extract a declared representation from a revision-pinned DINOv2 backbone."""

    def __init__(self, model_kind: str, representation: str) -> None:
        super().__init__()
        from transformers import AutoModel

        from .polar_models import DINO_MODEL_SPECS

        if model_kind not in DINO_MODEL_SPECS:
            raise ValueError(f"Frozen DINOv2 features are unavailable for {model_kind!r}")
        if representation not in {"final_cls", "last4_cls_mean_patch"}:
            raise ValueError(f"Unknown DINOv2 representation: {representation!r}")
        specification = DINO_MODEL_SPECS[model_kind]
        self.backbone = AutoModel.from_pretrained(
            specification["model_id"], revision=specification["revision"]
        )
        self.representation = representation

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=self.representation == "last4_cls_mean_patch",
        )
        if self.representation == "last4_cls_mean_patch":
            return official_multilayer_features(output.hidden_states, self.backbone.layernorm)
        if getattr(output, "pooler_output", None) is not None:
            return output.pooler_output
        return output.last_hidden_state[:, 0]


def load_aligned_feature_view(
    root: Path,
    model_kind: str,
    view: str,
    manifest: pd.DataFrame,
    manifest_hash: str,
) -> tuple[np.ndarray, dict]:
    """Load one cache and align its rows to an explicitly ordered manifest."""

    cache_dir = root / model_kind / view
    provenance_path = cache_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("manifest_sha256") != manifest_hash:
        raise RuntimeError(f"Feature cache manifest drift: {provenance_path}")
    test_rows_read = provenance.get("test_rows_read", provenance.get("test_rows"))
    if test_rows_read != 0 or provenance.get("test_labels_read") is True:
        raise RuntimeError(f"Development feature cache violates the test gate: {provenance_path}")
    rows = pd.read_csv(cache_dir / "rows.csv", dtype={"image_id": str})
    features = np.load(cache_dir / "features.npy", mmap_mode="r")
    if len(rows) != len(features):
        raise RuntimeError(f"Feature and metadata rows differ: {cache_dir}")
    if rows["image_id"].astype(str).duplicated().any():
        raise RuntimeError(f"Feature cache identifiers are not unique: {cache_dir}")
    index_by_id = {value: index for index, value in enumerate(rows["image_id"].astype(str))}
    try:
        order = np.asarray(
            [index_by_id[value] for value in manifest["image_id"].astype(str)], dtype=int
        )
    except KeyError as error:
        raise RuntimeError(f"Feature cache is missing manifest image {error}") from error
    aligned = np.asarray(features[order], dtype=np.float32)
    return aligned, provenance
