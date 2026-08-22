import json
from pathlib import Path

import pandas as pd
from PIL import Image

from hac.polar import (
    apply_quarantine,
    context_box,
    cross_split_embedding_pairs,
    embedding_confirmed_source_pairs,
    image_view,
    near_phash_cross_split_pairs,
    parse_annotation,
    quarantine_components,
    source_related_pairs,
)


def test_parse_annotation_selects_one_target_person(tmp_path: Path):
    path = tmp_path / "p1_00001.json"
    path.write_text(
        json.dumps(
            {
                "filename": "p1_00001.jpg",
                "originalname": "[walk]_source.jpg",
                "width": 100,
                "height": 80,
                "persons": [
                    {
                        "bndbox": {"xmin": 10, "ymin": 20, "xmax": 60, "ymax": 70},
                        "actions": {"walk": 1, "sit": 0, "stand": 0, "run": 0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    annotation = parse_annotation(path)
    assert annotation.image_id == "p1_00001"
    assert annotation.target is not None
    assert annotation.target.label_4 == "walking"
    assert annotation.target.xmax == 60


def test_context_box_expands_and_clips():
    assert context_box((5, 10, 55, 60), (100, 80), 0.10) == (0, 5, 60, 65)


def test_person_view_uses_declared_box():
    image = Image.new("RGB", (100, 80), "white")
    row = {
        "bbox_xmin": 10,
        "bbox_ymin": 20,
        "bbox_xmax": 60,
        "bbox_ymax": 70,
    }
    assert image_view(image, row, "full_frame").size == (100, 80)
    assert image_view(image, row, "person_context_10").size == (60, 60)


def test_near_hash_search_finds_only_cross_split_pairs():
    frame = pd.DataFrame(
        {
            "image_id": ["train_a", "train_b", "test_a", "test_far"],
            "split": ["train", "train", "test", "test"],
            "label_4": ["sitting", "standing", "sitting", "running"],
            "image_path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "phash": [
                "0000000000000000",
                "0000000000000001",
                "0000000000000003",
                "ffffffffffffffff",
            ],
        }
    )
    pairs = near_phash_cross_split_pairs(frame, max_distance=2)
    assert set(zip(pairs["left_image_id"], pairs["right_image_id"], strict=True)) == {
        ("train_a", "test_a"),
        ("train_b", "test_a"),
    }


def test_source_quarantine_uses_connected_components():
    pairs = pd.DataFrame(
        {
            "left_image_id": ["train_a", "test_a", "unrelated"],
            "right_image_id": ["test_a", "val_a", "test_b"],
            "phash_distance": [2, 4, 6],
            "normalized_correlation": [0.99, 0.91, 0.89],
        }
    )
    confirmed = source_related_pairs(pairs, minimum_correlation=0.90)
    quarantine = quarantine_components(confirmed)
    assert set(quarantine["image_id"]) == {"train_a", "test_a", "val_a"}
    assert quarantine["quarantine_group"].nunique() == 1

    frame = pd.DataFrame(
        {
            "image_id": ["train_a", "test_a", "val_a", "clean"],
            "eligible": [True, True, True, True],
            "exclusion_reason": ["", "", "", ""],
        }
    )
    full, clean = apply_quarantine(frame, quarantine)
    assert clean["image_id"].tolist() == ["clean"]
    assert set(full.loc[~full["primary_included"], "exclusion_reason"]) == {
        "cross_split_source_related"
    }


def test_embedding_retrieval_never_returns_same_split_neighbours():
    features = [[1.0, 0.0], [0.999, 0.001], [0.998, 0.002], [0.0, 1.0]]
    frame = pd.DataFrame(
        {
            "image_id": ["train_a", "train_b", "test_a", "test_b"],
            "split": ["train", "train", "test", "test"],
            "label_4": ["sitting"] * 4,
            "image_path": ["a", "b", "c", "d"],
        }
    )
    pairs = cross_split_embedding_pairs(
        features, frame, minimum_cosine=0.99, top_k=2, chunk_size=2
    )
    assert set(pairs["left_image_id"]) | set(pairs["right_image_id"]) == {
        "train_a",
        "train_b",
        "test_a",
    }
    assert all(pairs["left_split"] != pairs["right_split"])


def test_embedding_confirmation_requires_independent_pixel_agreement():
    pairs = pd.DataFrame(
        {
            "left_image_id": ["same_asset", "semantic_only"],
            "right_image_id": ["same_asset_crop", "similar_activity"],
            "embedding_cosine": [0.996, 0.997],
            "normalized_correlation": [0.96, 0.70],
        }
    )
    confirmed = embedding_confirmed_source_pairs(pairs)
    assert confirmed["left_image_id"].tolist() == ["same_asset"]
