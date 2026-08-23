import numpy as np
import pandas as pd
import pytest

from hac.polar_analysis import (
    PredictionArtifact,
    align_prediction_artifacts,
    complementarity_metrics,
    select_probability_blend,
    simplex_weights,
    stratified_paired_bootstrap,
)
from hac.polar_features import load_aligned_feature_view


def _artifact(ids, labels, probabilities):
    return PredictionArtifact(
        image_ids=np.asarray(ids),
        labels=np.asarray(labels),
        probabilities=np.asarray(probabilities, dtype=float),
        class_names=("left", "right"),
    )


def test_prediction_alignment_reorders_rows_and_preserves_labels():
    first = _artifact(["b", "a"], [1, 0], [[0.1, 0.9], [0.8, 0.2]])
    second = _artifact(["a", "b"], [0, 1], [[0.6, 0.4], [0.2, 0.8]])
    aligned = align_prediction_artifacts({"first": first, "second": second})
    assert aligned["first"].image_ids.tolist() == ["a", "b"]
    assert aligned["first"].labels.tolist() == [0, 1]
    assert aligned["second"].labels.tolist() == [0, 1]


def test_simplex_grid_is_complete_and_normalized():
    weights = simplex_weights(["a", "b", "c"], step=0.5)
    assert len(weights) == 6
    assert all(sum(item.values()) == pytest.approx(1.0) for item in weights)


def test_blend_selection_can_use_complementary_models():
    labels = [0, 0, 1, 1]
    left = _artifact(
        ["a", "b", "c", "d"],
        labels,
        [[0.9, 0.1], [0.4, 0.6], [0.1, 0.9], [0.6, 0.4]],
    )
    right = _artifact(
        ["a", "b", "c", "d"],
        labels,
        [[0.6, 0.4], [0.9, 0.1], [0.6, 0.4], [0.1, 0.9]],
    )
    weights, metrics, _ = select_probability_blend({"left": left, "right": right}, step=0.5)
    assert weights == {"left": 0.5, "right": 0.5}
    assert metrics["macro_f1"] == 1.0


def test_complementarity_reports_unique_corrections():
    labels = np.asarray([0, 0, 1, 1])
    left = np.asarray([[0.9, 0.1], [0.4, 0.6], [0.1, 0.9], [0.6, 0.4]])
    right = np.asarray([[0.6, 0.4], [0.9, 0.1], [0.6, 0.4], [0.1, 0.9]])
    metrics = complementarity_metrics(labels, left, right)
    assert metrics["left_unique_correct_count"] == 1
    assert metrics["right_unique_correct_count"] == 2
    assert metrics["oracle_accuracy"] == 1.0


def test_stratified_paired_bootstrap_is_reproducible():
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    perfect = np.eye(2)[labels]
    weak = np.asarray(
        [[0.9, 0.1], [0.9, 0.1], [0.4, 0.6], [0.1, 0.9], [0.6, 0.4], [0.1, 0.9]]
    )
    first = stratified_paired_bootstrap(labels, perfect, weak, resamples=100, seed=7)
    second = stratified_paired_bootstrap(labels, perfect, weak, resamples=100, seed=7)
    assert first == second
    assert first["point_estimate"] > 0.0


def test_feature_cache_loader_aligns_manifest_rows(tmp_path):
    cache = tmp_path / "model" / "view"
    cache.mkdir(parents=True)
    pd.DataFrame({"image_id": ["b", "a"]}).to_csv(cache / "rows.csv", index=False)
    np.save(cache / "features.npy", np.asarray([[2.0, 3.0], [0.0, 1.0]], dtype=np.float32))
    (cache / "provenance.json").write_text(
        '{"manifest_sha256": "manifest", "test_rows_read": 0}\n', encoding="utf-8"
    )
    manifest = pd.DataFrame({"image_id": ["a", "b"]})
    features, _ = load_aligned_feature_view(
        tmp_path, "model", "view", manifest, "manifest"
    )
    assert features.tolist() == [[0.0, 1.0], [2.0, 3.0]]
