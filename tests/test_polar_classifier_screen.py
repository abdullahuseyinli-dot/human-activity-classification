import pytest
from screen_polar_embedding_classifiers import (
    Candidate,
    build_estimator,
    calibrated_final_estimator,
    declared_candidates,
)
from sklearn.calibration import CalibratedClassifierCV


def test_classifier_grid_matches_the_locked_protocol():
    candidates = declared_candidates()
    family_counts = {}
    for candidate in candidates:
        family_counts[candidate.family] = family_counts.get(candidate.family, 0) + 1
    assert family_counts == {
        "multinomial_logistic": 8,
        "linear_svm": 10,
        "rbf_svm": 18,
        "shrinkage_lda": 4,
    }


def test_rbf_gamma_is_relative_to_the_fused_feature_count():
    candidate = Candidate(
        "rbf_svm",
        {"C": 10.0, "gamma_multiplier": 1.0, "class_weight": None},
    )
    estimator = build_estimator(candidate, feature_count=1536, seed=42)
    assert estimator.named_steps["classifier"].gamma == pytest.approx(1.0 / 1536.0)


def test_svm_final_fit_uses_training_only_probability_calibration():
    candidate = Candidate("linear_svm", {"C": 0.01, "class_weight": None})
    estimator = calibrated_final_estimator(
        candidate,
        feature_count=1536,
        seed=42,
        calibration_folds=5,
    )
    assert isinstance(estimator, CalibratedClassifierCV)
    assert estimator.method == "sigmoid"
    assert estimator.cv.n_splits == 5
