import pandas as pd
import pytest
from evaluate_vcoco_external import validate_external_manifest


def external_frame() -> pd.DataFrame:
    labels = [
        ("sitting", "sitting"),
        ("standing", "standing"),
        ("walking_running", "running"),
    ]
    return pd.DataFrame(
        [
            {
                "person_id": f"person-{index}",
                "image_id": f"image-{index}",
                "image_path": f"image-{index}.jpg",
                "label_3": label_3,
                "label_4": label_4,
                "bbox_xmin": 0,
                "bbox_ymin": 0,
                "bbox_xmax": 10,
                "bbox_ymax": 10,
                "image_level_unambiguous": True,
                "eligible_person": True,
            }
            for index, (label_3, label_4) in enumerate(labels)
        ]
    )


def test_external_manifest_preserves_specific_dynamic_label():
    validated = validate_external_manifest(external_frame())

    assert validated.loc[validated["label_3"].eq("walking_running"), "label_4"].item() == (
        "running"
    )


def test_external_manifest_rejects_unknown_four_class_label():
    frame = external_frame()
    frame.loc[0, "label_4"] = "jumping"

    with pytest.raises(ValueError, match="unexpected four-class"):
        validate_external_manifest(frame)
