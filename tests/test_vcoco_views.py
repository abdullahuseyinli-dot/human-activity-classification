from types import SimpleNamespace

from PIL import Image

from hac.augmentations import (
    SquarePad,
    build_aspect_preserving_eval_transform,
    build_person_train_transform,
)
from hac.polar import image_view


def row():
    return SimpleNamespace(
        bbox_xmin=20,
        bbox_ymin=10,
        bbox_xmax=60,
        bbox_ymax=90,
    ).__dict__


def test_person_views_expand_monotonically():
    image = Image.new("RGB", (100, 100))

    tight = image_view(image, row(), "person_tight")
    context_25 = image_view(image, row(), "person_context_25")
    context_50 = image_view(image, row(), "person_context_50")

    assert tight.size == (40, 80)
    assert context_25.size[0] > tight.size[0]
    assert context_25.size[1] > tight.size[1]
    assert context_50.size[0] >= context_25.size[0]
    assert context_50.size[1] >= context_25.size[1]


def test_square_pad_retains_complete_image():
    image = Image.new("RGB", (40, 80), color=(255, 0, 0))

    padded = SquarePad()(image)
    transformed = build_aspect_preserving_eval_transform(224)(image)

    assert padded.size == (80, 80)
    assert transformed.shape == (3, 224, 224)


def test_aspect_preserving_transform_accepts_model_normalization():
    image = Image.new("RGB", (20, 40), color=(255, 255, 255))

    transformed = build_aspect_preserving_eval_transform(
        32,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        fill=(128, 128, 128),
    )(image)

    assert transformed.shape == (3, 32, 32)
    assert transformed[:, 8:24, :].max().item() == 1.0


def test_background_interventions_preserve_person_pixels():
    image = Image.new("RGB", (100, 100), color=(0, 255, 0))
    image.paste(Image.new("RGB", (40, 80), color=(255, 0, 0)), (20, 10))

    ordinary = image_view(image, row(), "person_context_25")
    blurred = image_view(image, row(), "person_context_25_background_blur")
    masked = image_view(image, row(), "person_context_25_background_mask")

    assert ordinary.size == blurred.size == masked.size
    assert ordinary.getpixel((30, 40)) == blurred.getpixel((30, 40)) == masked.getpixel((30, 40))
    assert masked.getpixel((0, 0)) == (124, 116, 104)
    assert ordinary.getpixel((0, 0)) != masked.getpixel((0, 0))


def test_person_safe_augmentation_returns_model_tensor():
    image = Image.new("RGB", (40, 80), color=(255, 0, 0))

    transformed = build_person_train_transform("person_safe_mild", 64)(image)

    assert transformed.shape == (3, 64, 64)
