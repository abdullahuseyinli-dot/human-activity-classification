"""Plan or execute locked temporal training jobs sequentially on one GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from hac.polar import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["development", "crossfit", "students", "final"], required=True
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--grid", type=Path, default=Path("experiments/vcoco_v3_temporal_grid.json")
    )
    parser.add_argument(
        "--temporal-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/temporal_grid_lock.json"),
    )
    parser.add_argument(
        "--manifest-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/temporal_manifest_lock.json"),
    )
    parser.add_argument(
        "--teacher-selection",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/teacher_selection_lock.json"),
    )
    parser.add_argument(
        "--student-target-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/student_targets/summary.json"),
    )
    parser.add_argument(
        "--development-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/development_final/summary.json"),
    )
    parser.add_argument("--temporal-root", type=Path, default=Path(".runs/vcoco_v3/temporal"))
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_terminal(path: Path, status: str, label: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != status:
        raise RuntimeError(f"{label} is incomplete")
    return payload


def job_complete(job: dict) -> bool:
    summary_path = Path(job["output_dir"]) / "summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return all(summary.get(field) == value for field, value in job["expected"].items())


def build_jobs(args: argparse.Namespace, lock: dict, grid: dict, root: Path) -> list[dict]:
    temporal_root = args.temporal_root.resolve()
    seeds = list(map(int, lock["seeds"]))
    jobs = []
    if args.phase == "development":
        for seed in seeds:
            jobs.append(
                {
                    "model_role": "static",
                    "candidate_id": None,
                    "seed": seed,
                    "output_dir": temporal_root / "development/static" / f"seed-{seed}",
                    "expected": {
                        "status": "VCOCO_V3_TEMPORAL_DEVELOPMENT_RUN_COMPLETE",
                        "model_role": "static",
                        "candidate_id": "static_center_frame",
                        "seed": seed,
                    },
                }
            )
        for candidate in lock["teacher_candidates"]:
            for seed in seeds:
                candidate_id = str(candidate["candidate_id"])
                jobs.append(
                    {
                        "model_role": "teacher",
                        "candidate_id": candidate_id,
                        "seed": seed,
                        "output_dir": (
                            temporal_root / "development/teacher" / candidate_id / f"seed-{seed}"
                        ),
                        "expected": {
                            "status": "VCOCO_V3_TEMPORAL_DEVELOPMENT_RUN_COMPLETE",
                            "model_role": "teacher",
                            "candidate_id": candidate_id,
                            "seed": seed,
                        },
                    }
                )
        return jobs

    selection = load_terminal(
        args.teacher_selection.resolve(),
        "VCOCO_V3_TEMPORAL_TEACHER_SELECTED",
        "Temporal teacher selection",
    )
    if selection["source_sha256"].get("temporal_grid_lock") != sha256_file(
        args.temporal_lock.resolve()
    ):
        raise RuntimeError("Temporal teacher selection belongs to a different grid")
    if args.phase == "crossfit":
        folds = int(grid["training"]["teacher_crossfit_folds"])
        for role in ("static", "teacher"):
            for fold in range(folds):
                for seed in seeds:
                    jobs.append(
                        {
                            "model_role": role,
                            "fold": fold,
                            "seed": seed,
                            "output_dir": (
                                temporal_root / "crossfit" / role / f"fold-{fold}" / f"seed-{seed}"
                            ),
                            "expected": {
                                "status": "VCOCO_V3_TEMPORAL_CROSSFIT_RUN_COMPLETE",
                                "model_role": role,
                                "fold": fold,
                                "seed": seed,
                            },
                        }
                    )
        return jobs

    if args.phase == "students":
        load_terminal(
            args.student_target_summary.resolve(),
            "VCOCO_V3_TEMPORAL_STUDENT_TARGETS_LOCKED",
            "Temporal student targets",
        )
        for candidate in lock["student_candidates"]:
            candidate_id = str(candidate["candidate_id"])
            for seed in seeds:
                jobs.append(
                    {
                        "candidate_id": candidate_id,
                        "seed": seed,
                        "output_dir": (temporal_root / "students" / candidate_id / f"seed-{seed}"),
                        "expected": {
                            "status": "VCOCO_V3_TEMPORAL_STUDENT_RUN_COMPLETE",
                            "candidate_id": candidate_id,
                            "seed": seed,
                        },
                    }
                )
        return jobs

    development = load_terminal(
        args.development_summary.resolve(),
        "VCOCO_V3_TEMPORAL_DEVELOPMENT_COMPLETE",
        "Temporal development",
    )
    roles = [("static", None), ("teacher", None)]
    roles.extend(
        ("student", candidate)
        for candidate in sorted(
            {str(development["classification_student"]), str(development["routing_student"])}
        )
    )
    for role, candidate in roles:
        for seed in seeds:
            if role == "student":
                output = temporal_root / "pipeline_models/student" / str(candidate) / f"seed-{seed}"
            else:
                output = temporal_root / "pipeline_models" / role / f"seed-{seed}"
            jobs.append(
                {
                    "model_role": role,
                    "student_candidate": candidate,
                    "seed": seed,
                    "output_dir": output,
                    "expected": {
                        "status": "VCOCO_V3_TEMPORAL_FINAL_MODEL_COMPLETE",
                        "model_role": role,
                        "student_candidate": candidate,
                        "seed": seed,
                    },
                }
            )
    return jobs


def command_for_job(
    args: argparse.Namespace,
    job: dict,
    *,
    root: Path,
) -> list[str]:
    common = [
        "--grid",
        str(args.grid.resolve()),
        "--temporal-lock",
        str(args.temporal_lock.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--seed",
        str(job["seed"]),
        "--output-dir",
        str(job["output_dir"]),
        "--workers",
        str(args.workers),
    ]
    if args.phase == "development":
        command = [
            sys.executable,
            str(root / "experiments/train_vcoco_v3_temporal_candidate.py"),
            *common,
            "--manifest-lock",
            str(args.manifest_lock.resolve()),
            "--model-role",
            job["model_role"],
        ]
        if job["candidate_id"] is not None:
            command.extend(["--candidate-id", job["candidate_id"]])
        return command
    if args.phase == "crossfit":
        return [
            sys.executable,
            str(root / "experiments/crossfit_vcoco_v3_temporal.py"),
            *common,
            "--teacher-selection",
            str(args.teacher_selection.resolve()),
            "--model-role",
            job["model_role"],
            "--fold",
            str(job["fold"]),
        ]
    if args.phase == "students":
        return [
            sys.executable,
            str(root / "experiments/train_vcoco_v3_temporal_student.py"),
            *common,
            "--teacher-selection",
            str(args.teacher_selection.resolve()),
            "--student-target-summary",
            str(args.student_target_summary.resolve()),
            "--candidate-id",
            job["candidate_id"],
        ]
    command = [
        sys.executable,
        str(root / "experiments/fit_vcoco_v3_temporal_pipeline_model.py"),
        *common,
        "--development-summary",
        str(args.development_summary.resolve()),
        "--student-target-summary",
        str(args.student_target_summary.resolve()),
        "--model-role",
        job["model_role"],
    ]
    if job["student_candidate"] is not None:
        command.extend(["--student-candidate", job["student_candidate"]])
    return command


def main() -> None:
    args = parse_args()
    if args.max_runs is not None and args.max_runs < 1:
        raise ValueError("max-runs must be positive")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    root = Path.cwd().resolve()
    lock_path = args.temporal_lock.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    grid = json.loads(args.grid.resolve().read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_TEMPORAL_GRID_LOCKED_BEFORE_FIT":
        raise RuntimeError("The temporal grid is not locked")
    if lock["source_sha256"].get("temporal_grid") != sha256_file(args.grid.resolve()):
        raise RuntimeError("The temporal grid changed after locking")
    jobs = build_jobs(args, lock, grid, root)
    pending = [job for job in jobs if not job_complete(job)]
    selected = pending[: args.max_runs] if args.max_runs is not None else pending
    status_path = args.temporal_root.resolve() / "queues" / f"{args.phase}.json"
    status = {
        "status": "PLANNED" if not args.execute else "RUNNING",
        "phase": args.phase,
        "declared_jobs": len(jobs),
        "complete_before_start": len(jobs) - len(pending),
        "pending_before_start": len(pending),
        "selected_this_invocation": len(selected),
        "temporal_grid_lock_sha256": sha256_file(lock_path),
        "jobs": [
            {
                **{key: value for key, value in job.items() if key != "expected"},
                "output_dir": str(job["output_dir"]),
                "complete": job_complete(job),
            }
            for job in jobs
        ],
    }
    write_json(status_path, status)
    print(json.dumps(status, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return
    started = time.perf_counter()
    for index, job in enumerate(selected, start=1):
        result = subprocess.run(command_for_job(args, job, root=root), cwd=root, check=False)
        if result.returncode:
            status.update(
                {
                    "status": "FAILED",
                    "completed_this_invocation": index - 1,
                    "failed_job": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in job.items()
                        if key != "expected"
                    },
                    "return_code": result.returncode,
                    "runtime_seconds": time.perf_counter() - started,
                }
            )
            write_json(status_path, status)
            raise RuntimeError(f"Temporal queue stopped at {job['output_dir']}")
        status.update(
            {
                "status": "RUNNING",
                "completed_this_invocation": index,
                "runtime_seconds": time.perf_counter() - started,
            }
        )
        write_json(status_path, status)
    status.update(
        {
            "status": "INVOCATION_COMPLETE",
            "runtime_seconds": time.perf_counter() - started,
        }
    )
    write_json(status_path, status)
    print(json.dumps(status, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
