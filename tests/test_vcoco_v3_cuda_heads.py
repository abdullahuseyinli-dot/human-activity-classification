import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hac.vcoco_v3_cuda_heads import (
    CudaStandardizedLogisticRegression,
    cuda_logistic_fit_audit,
    fit_probability_head_cuda,
    predict_probability_head_cuda,
    reset_cuda_logistic_fit_audit,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def test_cuda_multinomial_logistic_matches_sklearn_lbfgs() -> None:
    rng = np.random.default_rng(17)
    features = rng.normal(size=(900, 32)).astype(np.float32)
    labels = rng.integers(0, 3, size=len(features))
    reference = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=3_000, random_state=17, solver="lbfgs"),
    ).fit(features, labels)
    model = CudaStandardizedLogisticRegression(
        c_value=0.1,
        class_weight="none",
        maximum_iterations=400,
        tolerance=1e-6,
        seed=17,
    ).fit(features, labels)

    reference_probabilities = reference.predict_proba(features)
    cuda_probabilities = model.predict_proba(features)
    assert np.mean(np.abs(reference_probabilities - cuda_probabilities)) < 2e-4
    assert np.mean(
        reference_probabilities.argmax(axis=1) == cuda_probabilities.argmax(axis=1)
    ) > 0.995
    assert not model.optimization_["iteration_limit_reached"]


def test_cuda_factorized_head_is_normalized_and_audited() -> None:
    rng = np.random.default_rng(23)
    features = rng.normal(size=(300, 24)).astype(np.float32)
    labels = np.tile(np.arange(3), 100)
    reset_cuda_logistic_fit_audit()
    head = fit_probability_head_cuda(
        features,
        labels,
        factorized=True,
        c_value=1.0,
        class_weight="balanced",
        seed=23,
        maximum_iterations=400,
        tolerance=1e-6,
    )
    probabilities = predict_probability_head_cuda(head, features[:19])

    assert probabilities.shape == (19, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    records = cuda_logistic_fit_audit()
    assert len(records) == 2
    assert all(record["classes"] == 2 for record in records)
    assert all(not record["iteration_limit_reached"] for record in records)
