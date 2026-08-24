import numpy as np
import torch

from experiments.cache_okutama_cptr_parts import anatomical_region_weights
from hac.cptr import CenterQueryEncoder, PartTrajectoryResidualNetwork, cptr_loss
from hac.cptr_features import (
    BODY_REGION_NAMES,
    QUALITY_DIM,
    TRAJECTORY_SEQUENCE_DIM,
    TRAJECTORY_SUMMARY_DIM,
    build_trajectory_features,
    jittered_camera_kwargs,
    motion_null_kwargs,
    reversed_kwargs,
    sample_indices_with_centre,
)
from hac.vcoco_v3_temporal import StaticIdentifiabilityStudent, TemporalFactorizedTeacher


def test_centre_preserving_sampler_handles_even_clip_length():
    indices, centre = sample_indices_with_centre(
        17,
        centre_index=8,
        samples=8,
        span_frames=8,
    )
    assert len(indices) == 8
    assert indices[centre] == 8
    assert np.all(np.diff(indices) >= 0)


def test_trajectory_features_separate_camera_translation_from_person_translation():
    frames = np.arange(17) * 2
    boxes = np.column_stack(
        (
            200 + np.arange(17) * 3,
            np.full(17, 100),
            240 + np.arange(17) * 3,
            np.full(17, 180),
        )
    )
    raw, raw_summary, raw_quality = build_trajectory_features(
        boxes,
        frames,
        np.ones(17, dtype=bool),
        centre_index=8,
    )
    transforms = np.repeat(np.eye(3)[None], 17, axis=0)
    transforms[:, 0, 2] = -(np.arange(17) - 8) * 3
    compensated, compensated_summary, quality = build_trajectory_features(
        boxes,
        frames,
        np.ones(17, dtype=bool),
        centre_index=8,
        to_centre_homographies=transforms,
        camera_quality=np.ones(17),
    )
    assert raw.shape == compensated.shape == (17, TRAJECTORY_SEQUENCE_DIM)
    assert raw_summary.shape == compensated_summary.shape == (TRAJECTORY_SUMMARY_DIM,)
    assert raw_quality.shape == quality.shape == (17,)
    assert raw[:, 6].mean() > 0.0
    assert abs(compensated[:, 6].mean()) < abs(raw[:, 6].mean()) * 0.05


def _dummy_model(**overrides) -> PartTrajectoryResidualNetwork:
    static = StaticIdentifiabilityStudent(14, hidden_dim=16, dropout=0.0)
    teacher = TemporalFactorizedTeacher(
        14,
        model_dim=16,
        layers=1,
        attention_heads=4,
        feedforward_dim=32,
        dropout=0.0,
        maximum_length=17,
    )
    options = {
        "use_short": True,
        "use_long": True,
        "use_trajectory": True,
        "use_parts": True,
        "use_pose": True,
        "use_siglip": True,
    }
    options.update(overrides)
    return PartTrajectoryResidualNetwork(
        static,
        teacher,
        frame_input_dim=14,
        quality_dim=QUALITY_DIM,
        trajectory_sequence_dim=TRAJECTORY_SEQUENCE_DIM,
        trajectory_summary_dim=TRAJECTORY_SUMMARY_DIM,
        model_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=32,
        dropout=0.0,
        use_short=options["use_short"],
        use_long=options["use_long"],
        use_trajectory=options["use_trajectory"],
        use_parts=options["use_parts"],
        part_input_dim=12,
        part_count=len(BODY_REGION_NAMES),
        use_pose=options["use_pose"],
        pose_joint_count=5,
        use_siglip=options["use_siglip"],
        siglip_dim=10,
        expert_roles=overrides.get("expert_roles"),
    )


def _dummy_kwargs(batch: int = 3) -> dict:
    return {
        "static_features": torch.randn(batch, 14),
        "short_features": torch.randn(batch, 8, 14),
        "short_valid_mask": torch.ones(batch, 8, dtype=torch.bool),
        "short_centre_index": 4,
        "quality_features": torch.randn(batch, QUALITY_DIM),
        "long_features": torch.randn(batch, 8, 14),
        "long_valid_mask": torch.ones(batch, 8, dtype=torch.bool),
        "long_centre_index": 4,
        "trajectory_sequence": torch.randn(batch, 17, TRAJECTORY_SEQUENCE_DIM),
        "trajectory_summary": torch.randn(batch, TRAJECTORY_SUMMARY_DIM),
        "trajectory_valid_mask": torch.ones(batch, 17, dtype=torch.bool),
        "camera_quality": torch.ones(batch, 17),
        "part_tokens": torch.randn(batch, 8, len(BODY_REGION_NAMES), 12),
        "part_confidence": torch.ones(batch, 8, len(BODY_REGION_NAMES)),
        "part_valid_mask": torch.ones(batch, 8, dtype=torch.bool),
        "part_centre_index": 4,
        "pose": torch.randn(batch, 8, 5, 3),
        "pose_valid_mask": torch.ones(batch, 8, dtype=torch.bool),
        "pose_centre_index": 4,
        "siglip_features": torch.randn(batch, 10),
    }


def test_full_cptr_forward_and_counterfactual_losses_are_finite():
    model = _dummy_model()
    kwargs = _dummy_kwargs()
    output = model(**kwargs)
    null = model(**motion_null_kwargs(kwargs))
    reversed_output = model(**reversed_kwargs(kwargs))
    jittered = model(**jittered_camera_kwargs(kwargs))
    assert output.probabilities.shape == (3, 3)
    assert torch.allclose(output.probabilities.sum(dim=1), torch.ones(3), atol=1e-5)
    assert output.posture_gates.shape[1] == len(output.expert_names)
    losses = cptr_loss(
        output,
        torch.tensor([0, 1, 2]),
        transition_targets=torch.tensor([0, 1, 0]),
        gait_targets=torch.tensor([-1, -1, 0]),
        occlusion_targets=torch.tensor([0.0, 1.0, 0.0]),
        null_output=null,
        reversed_output=reversed_output,
        jittered_output=jittered,
    )
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    assert model.posture_gate.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.static_fallback.parameters())


def test_part_confidence_can_be_fully_missing_without_nan():
    model = _dummy_model().eval()
    kwargs = _dummy_kwargs(batch=2)
    kwargs["part_confidence"].zero_()
    kwargs["pose"][..., 2].zero_()
    with torch.inference_mode():
        output = model(**kwargs)
    assert torch.isfinite(output.probabilities).all()
    part_index = output.expert_names.index("parts")
    assert torch.equal(output.expert_reliability[:, part_index], torch.zeros(2))


def test_anatomical_region_queries_normalize_for_tall_and_wide_crops():
    weights = anatomical_region_weights(
        torch.tensor([20.0, 100.0]),
        torch.tensor([100.0, 20.0]),
        grid_size=16,
        device=torch.device("cpu"),
    )
    assert weights.shape == (2, len(BODY_REGION_NAMES), 256)
    assert torch.allclose(weights.sum(dim=2), torch.ones(2, len(BODY_REGION_NAMES)))
    assert torch.isfinite(weights).all()


def test_expert_roles_hard_mask_inapplicable_residual_paths():
    model = _dummy_model(
        expert_roles={
            "centre_short": "posture",
            "centre_long": "motion",
        }
    ).eval()
    with torch.inference_mode():
        output = model(**_dummy_kwargs(batch=2))
    short_index = output.expert_names.index("centre_short")
    long_index = output.expert_names.index("centre_long")
    assert torch.all(output.motion_gates[:, short_index] == 0)
    assert torch.all(output.posture_gates[:, long_index] == 0)


def test_centre_encoder_exposes_masked_pretraining_tokens():
    encoder = CenterQueryEncoder(
        14,
        model_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=32,
        dropout=0.0,
        maximum_length=8,
    )
    values = torch.randn(2, 8, 14)
    mask = torch.ones(2, 8, dtype=torch.bool)
    encoded, repaired = encoder.encode_tokens(values, mask, centre_index=4)
    pooled = encoder.pool_tokens(encoded, repaired, centre_index=4)
    direct = encoder(values, mask, centre_index=4)
    assert encoded.shape == (2, 8, 16)
    assert pooled.shape == (2, 16)
    assert torch.allclose(pooled, direct.features)
