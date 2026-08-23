import numpy as np
import pandas as pd
import pytest

from experiments.evaluate_vcoco_v2_final_test import (
    collapse_to_three,
    open_test_manifest_once,
)
from hac.polar import sha256_file


def test_official_test_gate_is_idempotent_and_bound_to_selection(tmp_path):
    manifest = tmp_path / "test.csv"
    pd.DataFrame(
        {
            "person_id": ["p1", "p2"],
            "image_id": ["i1", "i2"],
            "label_3": ["sitting", "standing"],
        }
    ).to_csv(manifest, index=False)
    selection = {
        "final_test": {
            "manifest_sha256": sha256_file(manifest),
            "expected_people": 2,
        }
    }
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()

    first, first_gate = open_test_manifest_once(manifest, output_dir, selection, "selection-a")
    second, second_gate = open_test_manifest_once(manifest, output_dir, selection, "selection-a")

    assert first.equals(second)
    assert first_gate == second_gate
    assert second_gate["official_test_label_open_count"] == 1


def test_official_test_gate_rejects_cached_manifest_drift(tmp_path):
    manifest = tmp_path / "test.csv"
    pd.DataFrame(
        {"person_id": ["p1"], "image_id": ["i1"], "label_3": ["sitting"]}
    ).to_csv(manifest, index=False)
    selection = {
        "final_test": {
            "manifest_sha256": sha256_file(manifest),
            "expected_people": 1,
        }
    }
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    open_test_manifest_once(manifest, output_dir, selection, "selection-a")
    (output_dir / "opened_test_manifest.csv").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="gate is invalid"):
        open_test_manifest_once(manifest, output_dir, selection, "selection-a")


def test_four_class_probabilities_collapse_to_locked_three_class_ontology():
    collapsed = collapse_to_three(
        [[0.10, 0.20, 0.30, 0.40], [0.70, 0.10, 0.10, 0.10]]
    )

    assert np.allclose(collapsed, [[0.10, 0.20, 0.70], [0.70, 0.10, 0.20]])
