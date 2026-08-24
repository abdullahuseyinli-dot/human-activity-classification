import argparse
import importlib.util
import json
from pathlib import Path

from hac.polar import sha256_file


def load_script(name: str):
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_feature_queue_derives_only_missing_v3_cache_jobs():
    queue = load_script("run_vcoco_v3_feature_queue.py")
    root = Path(__file__).resolve().parents[1]
    spatial = json.loads((root / "experiments/vcoco_v3_spatial_grid.json").read_text())
    representations = json.loads(
        (root / "experiments/vcoco_v3_representation_grid.json").read_text()
    )

    spatial_jobs = queue.declared_jobs(spatial, "spatial")
    representation_jobs = queue.declared_jobs(representations, "representation")

    assert len(spatial_jobs) == 10
    assert len(representation_jobs) == 2
    assert {job["model_kind"] for job in spatial_jobs} == {"dinov2_base"}
    assert {job["model_kind"] for job in representation_jobs} == {"dinov3_base"}
    assert {job["image_size"] for job in spatial_jobs} == {224, 336, 448}


def test_temporal_queue_declares_every_seed_fold_and_selected_final_model(tmp_path):
    queue = load_script("run_vcoco_v3_temporal_queue.py")
    lock_path = tmp_path / "lock.json"
    lock = {
        "seeds": [42, 43, 44, 45, 46],
        "teacher_candidates": [
            {"candidate_id": name}
            for name in ("temporal_8f_050s", "temporal_8f_100s", "temporal_16f_100s")
        ],
        "student_candidates": [
            {"candidate_id": "distilled_static"},
            {"candidate_id": "identifiability_conditioned_static"},
        ],
    }
    write_json(lock_path, lock)
    selection_path = tmp_path / "selection.json"
    write_json(
        selection_path,
        {
            "status": "VCOCO_V3_TEMPORAL_TEACHER_SELECTED",
            "source_sha256": {"temporal_grid_lock": sha256_file(lock_path)},
        },
    )
    targets_path = tmp_path / "targets.json"
    write_json(targets_path, {"status": "VCOCO_V3_TEMPORAL_STUDENT_TARGETS_LOCKED"})
    development_path = tmp_path / "development.json"
    write_json(
        development_path,
        {
            "status": "VCOCO_V3_TEMPORAL_DEVELOPMENT_COMPLETE",
            "classification_student": "distilled_static",
            "routing_student": "identifiability_conditioned_static",
        },
    )
    base = {
        "manifest": tmp_path / "manifest.csv",
        "grid": tmp_path / "grid.json",
        "temporal_lock": lock_path,
        "manifest_lock": tmp_path / "manifest-lock.json",
        "teacher_selection": selection_path,
        "student_target_summary": targets_path,
        "development_summary": development_path,
        "temporal_root": tmp_path / "runs",
        "workers": 0,
    }
    grid = {"training": {"teacher_crossfit_folds": 5}}

    counts = {}
    for phase in ("development", "crossfit", "students", "final"):
        arguments = argparse.Namespace(**base, phase=phase)
        counts[phase] = len(queue.build_jobs(arguments, lock, grid, tmp_path))

    assert counts == {"development": 20, "crossfit": 50, "students": 10, "final": 20}
