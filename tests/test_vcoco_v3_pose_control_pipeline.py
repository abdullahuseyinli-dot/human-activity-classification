import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from hac.polar import sha256_file


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pose-control fitting requires CUDA")
def test_pose_control_runner_scores_only_development_and_calibration(tmp_path):
    root = Path(__file__).resolve().parents[1]
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    rows = []
    development_evidence = {}
    split_counts = {"train": 30, "validation": 15, "calibration": 9, "confirmation": 3}
    sample_index = 0
    for split, count in split_counts.items():
        for within_split in range(count):
            label_index = within_split % 3
            label = ("sitting", "standing", "walking_running")[label_index]
            sample_id = f"sample-{sample_index:03d}"
            relative_path = Path("features") / f"{sample_id}.npz"
            rows.append(
                {
                    "sample_id": sample_id,
                    "recording_id": f"recording-{sample_index:03d}",
                    "track_id": "track-0",
                    "label": label,
                    "split": split,
                    "frame_count": 9,
                    "center_frame_index": 4,
                    "frames_per_second": 10.0,
                    "feature_path": relative_path.as_posix(),
                }
            )
            if split != "confirmation":
                pose = np.zeros((9, 5, 3), dtype=np.float32)
                pose[..., 2] = 1.0
                if label_index == 0:
                    pose[:, :, 1] = np.arange(9)[:, None] * 0.015
                    pose[:, 0, 0] = np.sin(np.arange(9)) * 0.05
                elif label_index == 2:
                    pose[:, :, 0] = np.arange(9)[:, None] * 0.12
                path = tmp_path / relative_path
                np.savez_compressed(path, pose=pose)
                development_evidence[sample_id] = {"pose_available": True}
            sample_index += 1
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    candidate = {
        "candidate_id": "temporal_8f_050s",
        "uniform_samples": 8,
        "window_seconds": 0.5,
        "frame_backbone": "dinov3_base",
    }
    temporal_lock_path = tmp_path / "temporal_lock.json"
    write_json(
        temporal_lock_path,
        {
            "status": "VCOCO_V3_TEMPORAL_GRID_LOCKED_BEFORE_FIT",
            "teacher_candidates": [candidate],
        },
    )
    manifest_lock_path = tmp_path / "manifest_lock.json"
    write_json(
        manifest_lock_path,
        {
            "status": "VCOCO_V3_TEMPORAL_MANIFEST_LOCKED",
            "confirmation_feature_arrays_opened": 0,
            "source_sha256": {"manifest": sha256_file(manifest_path)},
            "development_feature_sha256": development_evidence,
        },
    )
    development_path = tmp_path / "development.json"
    write_json(
        development_path,
        {
            "status": "VCOCO_V3_TEMPORAL_DEVELOPMENT_COMPLETE",
            "selected_teacher": candidate,
        },
    )
    output_dir = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "experiments/evaluate_vcoco_v3_pose_velocity_control.py"),
            "--grid",
            str(root / "experiments/vcoco_v3_temporal_grid.json"),
            "--temporal-lock",
            str(temporal_lock_path),
            "--manifest-lock",
            str(manifest_lock_path),
            "--development-summary",
            str(development_path),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    validation = pd.read_csv(output_dir / "validation_grid.csv")
    with np.load(output_dir / "calibration_predictions.npz", allow_pickle=False) as payload:
        probabilities = payload["probabilities"]
        calibration_samples = payload["sample_ids"]

    assert result.returncode == 0
    assert summary["status"] == "VCOCO_V3_POSE_CONTROL_COMPLETE"
    assert summary["calibration_samples_used_for_model_selection"] == 0
    assert summary["confirmation_samples_read"] == 0
    assert len(validation) == 8
    assert probabilities.shape == (split_counts["calibration"], 3)
    assert len(calibration_samples) == split_counts["calibration"]
    assert not any(
        (tmp_path / "features" / f"sample-{index:03d}.npz").exists() for index in range(54, 57)
    )
