import json

import numpy as np
import pandas as pd
import pytest
import torch

from hac.vcoco_v3_pose_control import (
    build_pose_svm,
    extract_pose_control_features,
    fit_pose_score_calibrator,
    pose_decision_scores,
    predict_pose_probabilities,
)
from hac.vcoco_v3_temporal import (
    StaticIdentifiabilityStudent,
    TemporalFactorizedTeacher,
    TemporalFeatureDataset,
    aps_nonconformity_scores,
    aps_prediction_sets,
    evaluate_routing_curve,
    fit_aps_threshold,
    grouped_recording_splits,
    pose_velocity_summary,
    route_by_budget,
    static_student_distillation_loss,
    teacher_advantage_targets,
    temporal_teacher_loss,
    uniform_clip_indices,
    validate_temporal_manifest,
)


def test_temporal_teacher_masks_padding_and_trains_hierarchically():
    torch.manual_seed(3)
    model = TemporalFactorizedTeacher(
        12,
        model_dim=16,
        layers=1,
        attention_heads=4,
        feedforward_dim=32,
        dropout=0.0,
        maximum_length=8,
    )
    features = torch.randn(4, 6, 12)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    labels = torch.tensor([0, 1, 2, 2])

    output = model(features, mask)
    loss = temporal_teacher_loss(output, labels, label_smoothing=0.02)
    loss.backward()

    assert output.probabilities.shape == (4, 3)
    assert torch.allclose(output.probabilities.sum(dim=1), torch.ones(4), atol=1e-6)
    assert torch.all(output.attention_weights[~mask] == 0)
    assert model.input_projection[1].weight.grad is not None


def test_static_student_combines_supervision_distillation_and_routing_target():
    torch.manual_seed(4)
    model = StaticIdentifiabilityStudent(10, hidden_dim=16, geometry_dim=2, dropout=0.0)
    output = model(torch.randn(6, 10), torch.randn(6, 2))
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    teacher = torch.eye(3)[labels] * 0.9 + 0.1 / 3
    targets = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.float32)

    losses = static_student_distillation_loss(
        output,
        labels,
        teacher,
        identifiability_targets=targets,
        supervised_weight=0.6,
        distillation_weight=0.3,
        identifiability_weight=0.1,
    )
    losses["loss"].backward()

    assert output.probabilities.shape == (6, 3)
    assert output.identifiability_logit.shape == (6,)
    assert losses["loss"].item() > 0
    assert model.identifiability[-1].weight.grad is not None


def test_teacher_advantage_requires_correct_cross_fitted_improvement():
    labels = np.asarray([0, 1, 2, 2])
    static = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.4, 0.4],
            [0.1, 0.6, 0.3],
            [0.1, 0.2, 0.7],
        ]
    )
    teacher = np.asarray(
        [
            [0.85, 0.1, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.25, 0.65],
        ]
    )

    targets = teacher_advantage_targets(
        labels,
        static,
        teacher,
        minimum_log_likelihood_gain=0.2,
    )

    assert targets.tolist() == [0, 1, 1, 0]


def test_split_conformal_sets_and_budgeted_routing_are_deterministic():
    calibration_labels = np.asarray([0, 1, 2, 0, 1, 2])
    calibration_probabilities = np.eye(3)[calibration_labels] * 0.8 + 0.2 / 3
    scores = aps_nonconformity_scores(calibration_labels, calibration_probabilities)
    threshold = fit_aps_threshold(scores, miscoverage=0.2)
    sets = aps_prediction_sets(calibration_probabilities, threshold)

    routing_scores = np.asarray([0.1, 0.9, 0.4, 0.8, 0.2, 0.3])
    routed = route_by_budget(routing_scores, clip_fraction=1 / 3)
    temporal = np.eye(3)[calibration_labels] * 0.9 + 0.1 / 3
    static = np.roll(temporal, 1, axis=1)
    curve = evaluate_routing_curve(
        calibration_labels,
        static,
        temporal,
        routing_scores,
        clip_fractions=[0.0, 1 / 3, 1.0],
    )

    assert np.all(sets[np.arange(6), calibration_labels])
    assert routed.tolist() == [False, True, False, True, False, False]
    assert curve[0]["macro_f1"] < curve[-1]["macro_f1"]
    assert curve[-1]["macro_f1"] == pytest.approx(1.0)


def test_temporal_manifest_and_grouped_splits_protect_recordings():
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(18)],
            "recording_id": np.repeat([f"r{index}" for index in range(6)], 3),
            "track_id": np.tile(["a", "b", "c"], 6),
            "label": np.tile(["sitting", "standing", "walking_running"], 6),
            "split": ["train"] * 18,
            "frame_count": [20] * 18,
            "center_frame_index": [10] * 18,
            "frames_per_second": [25.0] * 18,
            "feature_path": [f"features/{index}.npz" for index in range(18)],
        }
    )
    validated = validate_temporal_manifest(frame)
    labels = np.tile(np.arange(3), 6)
    groups = validated["recording_id"].to_numpy()
    splits = grouped_recording_splits(labels, groups, folds=3, seed=8)

    assert len(validated) == 18
    assert all(not set(groups[fit]).intersection(groups[held]) for fit, held in splits)
    assert uniform_clip_indices(5, center_index=0, samples=4, span_frames=7).tolist() == [
        0,
        0,
        1,
        3,
    ]

    drifted = frame.copy()
    drifted.loc[1, "split"] = "validation"
    with pytest.raises(ValueError, match="recording crosses"):
        validate_temporal_manifest(drifted)


def test_temporal_feature_dataset_and_pose_control(tmp_path):
    feature_path = tmp_path / "clip.npz"
    frames = 9
    tight = np.arange(frames * 4, dtype=np.float32).reshape(frames, 4)
    context = tight + 100
    geometry = np.ones((frames, 6), dtype=np.float32)
    pose = np.zeros((frames, 3, 3), dtype=np.float32)
    pose[..., 2] = 1.0
    pose[:, :, 0] = np.arange(frames)[:, None] * 0.1
    np.savez_compressed(
        feature_path,
        tight=tight,
        context=context,
        geometry=geometry,
        pose=pose,
    )
    frame = pd.DataFrame(
        {
            "sample_id": ["sample"],
            "recording_id": ["recording"],
            "track_id": ["track"],
            "label": ["walking_running"],
            "split": ["train"],
            "frame_count": [frames],
            "center_frame_index": [4],
            "frames_per_second": [10.0],
            "feature_path": [feature_path.name],
        }
    )

    item = TemporalFeatureDataset(
        frame,
        uniform_samples=5,
        window_seconds=0.5,
        manifest_directory=tmp_path,
    )[0]
    motion = pose_velocity_summary(pose, frames_per_second=10.0)
    sampled_motion = pose_velocity_summary(
        pose[[0, 2, 4, 6, 8]],
        frames_per_second=10.0,
        frame_indices=np.asarray([0, 2, 4, 6, 8]),
    )
    pose_features = extract_pose_control_features(
        frame,
        candidate={"uniform_samples": 5, "window_seconds": 0.5},
        manifest_directory=tmp_path,
    )

    assert item["clip_features"].shape == (5, 14)
    assert item["static_features"].shape == (14,)
    assert item["label"] == 2
    assert motion.shape == (12,)
    assert motion[2] > 0
    assert motion[-1] == pytest.approx(0.0)
    assert sampled_motion[2] == pytest.approx(motion[2])
    assert pose_features.values.shape == (1, 12)
    assert pose_features.labels.tolist() == [2]


def test_temporal_feature_dataset_reads_packed_memmap_store(tmp_path):
    rows, frames, dimensions = 2, 7, 5
    tight = np.arange(rows * frames * dimensions, dtype=np.float32).reshape(
        rows, frames, dimensions
    )
    context = tight + 1_000
    geometry = np.ones((rows, frames, 6), dtype=np.float32)
    np.save(tmp_path / "tight.npy", tight)
    np.save(tmp_path / "context.npy", context)
    np.save(tmp_path / "geometry.npy", geometry)
    store_path = tmp_path / "store.json"
    store_path.write_text(
        json.dumps(
            {
                "status": "VCOCO_V3_PACKED_TEMPORAL_FEATURE_STORE_COMPLETE",
                "arrays": {
                    "tight": {"path": "tight.npy"},
                    "context": {"path": "context.npy"},
                    "geometry": {"path": "geometry.npy"},
                },
            }
        ),
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "sample_id": ["sample"],
            "recording_id": ["scenario"],
            "track_id": ["video::track"],
            "label": ["standing"],
            "split": ["train"],
            "frame_count": [frames],
            "center_frame_index": [3],
            "frames_per_second": [6.0],
            "feature_path": [store_path.name],
            "feature_index": [1],
        }
    )

    item = TemporalFeatureDataset(
        frame,
        uniform_samples=4,
        window_seconds=1.0,
        manifest_directory=tmp_path,
    )[0]

    assert item["clip_features"].shape == (4, 2 * dimensions + 6)
    assert np.array_equal(item["static_features"].numpy()[:dimensions], tight[1, 3])
    assert item["recording_id"] == "scenario"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pose-control fitting requires CUDA")
def test_pose_svm_bundle_returns_calibrated_three_class_probabilities():
    rng = np.random.default_rng(12)
    labels = np.repeat(np.arange(3), 20)
    values = rng.normal(0.0, 0.15, size=(len(labels), 12))
    values[:, 0] += labels * 2.0
    values[:, 1] += (labels == 2) * 2.0
    model = build_pose_svm(c_value=0.1, class_weight="balanced", seed=4)
    model.fit(values, labels)
    scores = pose_decision_scores(model, values)
    calibrator = fit_pose_score_calibrator(scores, labels, seed=4)

    probabilities = predict_pose_probabilities(
        {"svm": model, "score_calibrator": calibrator}, values
    )

    assert probabilities.shape == (60, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities.argmax(axis=1) == labels).mean() > 0.9
