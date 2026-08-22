"""Dataset and deterministic data-loader helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from .polar import image_view


class ActivityDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_index: dict[str, int],
        transform: Callable,
        view: str = "full_frame",
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.class_to_index = dict(class_to_index)
        self.transform = transform
        self.view = str(view)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        with Image.open(row["resolved_image_path"]) as image:
            source = image.convert("RGB")
            pixels = self.transform(image_view(source, row, self.view))
        return {
            "pixel_values": pixels,
            "label": int(self.class_to_index[str(row["label"])]),
            "image_id": str(row["image_id"]),
            "image_path": str(row["resolved_image_path"]),
        }


def make_loader(
    frame: pd.DataFrame,
    class_to_index: dict[str, int],
    transform: Callable,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int = 0,
    view: str = "full_frame",
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        ActivityDataset(frame, class_to_index, transform, view=view),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator,
    )


def stratified_folds(
    development: pd.DataFrame,
    *,
    splits: int = 5,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=int(splits), shuffle=True, random_state=int(seed))
    dummy = np.zeros(len(development), dtype=np.uint8)
    yield from splitter.split(dummy, development["label"].astype(str).to_numpy())
