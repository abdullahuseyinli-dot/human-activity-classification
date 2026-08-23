from argparse import Namespace

import numpy as np
from fit_polar_final_probe import build_estimator, configuration


def _arguments(classifier: str) -> Namespace:
    return Namespace(
        model_kind="dinov2_base",
        representation="last4_cls_mean_patch",
        views=["full_frame", "person_context_10"],
        task="label_4",
        classifier=classifier,
        c_value=10.0 if classifier == "calibrated_rbf_svm" else 0.001,
        class_weight="none" if classifier == "calibrated_rbf_svm" else "balanced",
        gamma_multiplier=1.0,
        calibration_folds=2,
        seed=42,
    )


def test_final_logistic_configuration_records_representation():
    config = configuration(_arguments("standardized_multinomial_logistic"))
    assert config["representation"] == "last4_cls_mean_patch"
    assert config["solver"] == "lbfgs"
    assert "gamma_multiplier" not in config


def test_final_rbf_estimator_exposes_calibrated_class_order():
    arguments = _arguments("calibrated_rbf_svm")
    config = configuration(arguments)
    assert config["calibration"] == "sigmoid"
    assert config["gamma_multiplier"] == 1.0

    features = np.asarray(
        [[class_id * 3.0 + offset, offset] for class_id in range(3) for offset in range(4)],
        dtype=np.float32,
    )
    labels = np.repeat(np.arange(3), 4)
    estimator = build_estimator(arguments, features.shape[1])
    estimator.fit(features, labels)
    assert np.array_equal(estimator.classes_, np.arange(3))
    assert estimator.predict_proba(features).shape == (12, 3)
