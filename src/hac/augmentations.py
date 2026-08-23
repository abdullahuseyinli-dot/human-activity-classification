"""Training and deterministic evaluation transforms used by the study."""

from __future__ import annotations

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AddGaussianNoise:
    def __init__(self, standard_deviation: float = 0.01) -> None:
        if standard_deviation < 0.0:
            raise ValueError("standard_deviation cannot be negative")
        self.standard_deviation = float(standard_deviation)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.standard_deviation
        return torch.clamp(tensor + noise, 0.0, 1.0)


class SquarePad:
    """Pad an image to a square without cropping the person or changing aspect."""

    def __init__(self, fill: tuple[int, int, int] = (124, 116, 104)) -> None:
        self.fill = tuple(int(value) for value in fill)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        horizontal = side - width
        vertical = side - height
        border = (
            horizontal // 2,
            vertical // 2,
            horizontal - horizontal // 2,
            vertical - vertical // 2,
        )
        return ImageOps.expand(image, border=border, fill=self.fill)


def _mild(image_size: int) -> list:
    return [
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.75, 1.0),
            ratio=(0.80, 1.25),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [
                transforms.RandomAffine(
                    degrees=8,
                    translate=(0.04, 0.04),
                    scale=(0.95, 1.05),
                    shear=2,
                    interpolation=InterpolationMode.BICUBIC,
                )
            ],
            p=0.35,
        ),
        transforms.ColorJitter(
            brightness=0.18,
            contrast=0.18,
            saturation=0.12,
            hue=0.03,
        ),
        transforms.ToTensor(),
        transforms.RandomApply([AddGaussianNoise(0.01)], p=0.10),
        transforms.RandomErasing(
            p=0.15,
            scale=(0.02, 0.08),
            ratio=(0.3, 3.0),
            value="random",
        ),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]


def _moderate(image_size: int) -> list:
    return [
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.60, 1.0),
            ratio=(0.75, 1.33),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.06, 0.06),
                    scale=(0.90, 1.10),
                    shear=4,
                    interpolation=InterpolationMode.BICUBIC,
                )
            ],
            p=0.50,
        ),
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.20,
            hue=0.04,
        ),
        transforms.ToTensor(),
        transforms.RandomApply([AddGaussianNoise(0.015)], p=0.12),
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.12),
            ratio=(0.3, 3.3),
            value="random",
        ),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]


def build_train_transform(policy: str, image_size: int = 224) -> transforms.Compose:
    if policy in {"mild", "mild_no_random_erasing", "mild_randaugment_light"}:
        operations = _mild(image_size)
    elif policy == "moderate":
        operations = _moderate(image_size)
    else:
        raise ValueError(f"Unknown augmentation policy: {policy}")

    if policy == "mild_no_random_erasing":
        operations = [item for item in operations if not isinstance(item, transforms.RandomErasing)]
    elif policy == "mild_randaugment_light":
        operations.insert(2, transforms.RandAugment(num_ops=2, magnitude=7))
    return transforms.Compose(operations)


def build_person_train_transform(policy: str, image_size: int = 224) -> transforms.Compose:
    """Augment a person crop while keeping the complete padded view in frame."""

    if policy not in {"person_safe_mild", "person_safe_augmix"}:
        raise ValueError(f"Unknown person-safe augmentation policy: {policy}")
    operations = [
        SquarePad(),
        transforms.Resize(
            (int(image_size), int(image_size)), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.02, 0.02),
                    scale=(0.88, 1.0),
                    shear=2,
                    interpolation=InterpolationMode.BICUBIC,
                    fill=(124, 116, 104),
                )
            ],
            p=0.45,
        ),
        transforms.ColorJitter(
            brightness=0.18,
            contrast=0.18,
            saturation=0.12,
            hue=0.025,
        ),
    ]
    if policy == "person_safe_augmix":
        operations.append(transforms.AugMix(severity=2, mixture_width=3, chain_depth=-1, alpha=1.0))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.RandomApply([AddGaussianNoise(0.01)], p=0.10),
            transforms.RandomErasing(
                p=0.10,
                scale=(0.01, 0.05),
                ratio=(0.5, 2.0),
                value="random",
            ),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_aspect_preserving_eval_transform(
    image_size: int = 224,
    *,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    fill: tuple[int, int, int] = (124, 116, 104),
) -> transforms.Compose:
    """Pad to square and resize so the complete declared view remains visible."""

    return transforms.Compose(
        [
            SquarePad(fill=fill),
            transforms.Resize(
                (int(image_size), int(image_size)), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_flip_tta_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [build_eval_transform(image_size), transforms.RandomHorizontalFlip(p=1.0)]
    )
