import numpy as np
import pytest
import torch

from hac.vcoco_v3_models import (
    Candidate,
    CudaPrimalLinearSVM,
    fit_box_augmented_factorized,
    fit_candidate,
    grouped_splits,
    holm_adjust,
    paired_cluster_bootstrap,
    probability_reliability_features,
)


def synthetic_features(seed=5):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(30), 3).astype(str)
    labels = np.tile(np.arange(3), 30)
    centers = np.eye(3)[labels]
    first = np.column_stack([centers, labels / 2]) + rng.normal(0, 0.08, (90, 4))
    second = np.column_stack([centers[:, ::-1], labels / 2]) + rng.normal(0, 0.1, (90, 4))
    geometry = np.column_stack([labels, labels == 2, labels == 0, np.ones((90, 3))]).astype(float)
    return {"first": first, "second": second}, geometry, labels, groups


def test_grouped_splits_keep_source_images_intact():
    _, _, labels, groups = synthetic_features()

    splits = grouped_splits(labels, groups, folds=5, seed=11)

    assert len(splits) == 5
    assert all(not set(groups[train]).intersection(groups[held]) for train, held in splits)
    assert sorted(np.concatenate([held for _, held in splits])) == list(range(len(labels)))


@pytest.mark.parametrize("factorized,reliability", [(False, False), (True, True)])
def test_probability_stack_returns_valid_unseen_probabilities(factorized, reliability):
    features, geometry, labels, groups = synthetic_features()
    train = np.flatnonzero(np.arange(len(labels)) < 63)
    target = np.flatnonzero(np.arange(len(labels)) >= 63)
    family = "factorized" if factorized else "flat"
    candidate = Candidate(
        candidate_id=family,
        family=family,
        component_c=0.1,
        meta_c=0.1,
        svm_c=None,
        class_weight="none",
    )
    declaration = {
        "components": ["first", "second"],
        "factorized": factorized,
        "reliability_features": reliability,
    }

    probabilities = fit_candidate(
        candidate,
        declaration,
        {name: value[train] for name, value in features.items()},
        {name: value[target] for name, value in features.items()},
        geometry[train],
        geometry[target],
        labels[train],
        groups[train],
        stack_folds=3,
        seed=21,
    )

    assert probabilities.shape == (len(target), 3)
    assert np.all(probabilities >= 0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities.argmax(axis=1) == labels[target]).mean() > 0.8


def test_reliability_features_capture_view_disagreement():
    certain_left = np.asarray([[0.98, 0.01, 0.01], [0.98, 0.01, 0.01]])
    views = [certain_left, np.asarray([[0.98, 0.01, 0.01], [0.01, 0.01, 0.98]])]

    reliability = probability_reliability_features(views)

    assert reliability.shape[0] == 2
    assert reliability[1, -1] > reliability[0, -1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_linear_svm_separates_simple_classes():
    feature_blocks, _, labels, _ = synthetic_features()
    features = np.concatenate(list(feature_blocks.values()), axis=1).astype(np.float32)
    model = CudaPrimalLinearSVM(
        c_value=0.1,
        class_weight="none",
        maximum_iterations=100,
        tolerance=1e-5,
        seed=12,
    ).fit(features, labels)

    predictions = model.decision_function(features).argmax(axis=1)

    assert (predictions == labels).mean() > 0.95
    assert np.isfinite(model.optimization_["final_gradient_max"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_linear_svm_integrates_with_calibrated_candidate_pipeline():
    features, geometry, labels, groups = synthetic_features()
    train = np.arange(63)
    target = np.arange(63, 90)
    candidate = Candidate(
        candidate_id="cuda-svm",
        family="dino_siglip_linear_svm_control",
        component_c=None,
        meta_c=None,
        svm_c=0.01,
        class_weight="none",
    )
    declaration = {
        "components": ["first", "second"],
        "maximum_iterations": 100,
        "tolerance": 1e-4,
    }

    probabilities = fit_candidate(
        candidate,
        declaration,
        {name: values[train] for name, values in features.items()},
        {name: values[target] for name, values in features.items()},
        geometry[train],
        geometry[target],
        labels[train],
        groups[train],
        stack_folds=3,
        seed=31,
    )

    assert probabilities.shape == (len(target), len(np.unique(labels)))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_box_augmented_head_returns_each_condition_and_aggregate():
    features, geometry, labels, _ = synthetic_features()
    train = np.arange(63)
    target = np.arange(63, 90)
    contexts_train = {
        "nominal": features["second"][train],
        "shifted": features["second"][train] + 0.02,
    }
    contexts_target = {
        "nominal": features["second"][target],
        "shifted": features["second"][target] + 0.02,
    }

    aggregate, conditions = fit_box_augmented_factorized(
        features["first"][train],
        contexts_train,
        features["first"][target],
        contexts_target,
        geometry[train],
        geometry[target],
        labels[train],
        c_value=0.1,
        class_weight="none",
        seed=8,
    )

    assert set(conditions) == {"nominal", "shifted"}
    assert aggregate.shape == (len(target), 3)
    assert np.allclose(aggregate.sum(axis=1), 1.0)


def test_cluster_bootstrap_and_holm_report_paired_improvement():
    _, _, labels, groups = synthetic_features()
    challenger = np.eye(3)[labels] * 0.9 + 0.1 / 3
    baseline_labels = np.roll(labels, 1)
    baseline = np.eye(3)[baseline_labels] * 0.9 + 0.1 / 3

    result = paired_cluster_bootstrap(labels, challenger, baseline, groups, resamples=200, seed=9)
    adjusted = holm_adjust({"strong": 0.001, "weak": 0.04})

    assert result["macro_f1"]["point_estimate"] == pytest.approx(1.0)
    assert result["macro_f1"]["ci_95_low"] > 0
    assert adjusted == {"strong": pytest.approx(0.002), "weak": pytest.approx(0.04)}
