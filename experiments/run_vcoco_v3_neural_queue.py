"""Plan or execute the locked V-COCO v3 neural jobs sequentially on one GPU."""

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
    parser.add_argument("--stage", choices=["inner", "outer"], required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--grid", type=Path, default=Path("experiments/vcoco_v3_neural_grid.json"))
    parser.add_argument(
        "--neural-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/neural/neural_grid_lock.json"),
    )
    parser.add_argument(
        "--selection-lock",
        type=Path,
        default=Path(".runs/vcoco_v3/neural/inner_selection_lock.json"),
    )
    parser.add_argument("--run-root", type=Path, default=Path(".runs/vcoco_v3/neural"))
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def output_directory(
    root: Path,
    *,
    stage: str,
    outer: int,
    inner: int | None,
    candidate: str,
    seed: int,
) -> Path:
    if stage == "inner":
        return root / "inner" / f"outer-{outer}" / f"inner-{inner}" / candidate / f"seed-{seed}"
    return root / "outer" / f"outer-{outer}" / candidate / f"seed-{seed}"


def complete(path: Path, expected: dict) -> bool:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return all(summary.get(field) == value for field, value in expected.items())


def main() -> None:
    args = parse_args()
    if args.max_runs is not None and args.max_runs < 1:
        raise ValueError("max-runs must be positive")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    root = Path.cwd().resolve()
    grid_path = args.grid.resolve()
    lock_path = args.neural_lock.resolve()
    run_root = args.run_root.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V3_NEURAL_GRID_LOCKED_BEFORE_FIT":
        raise RuntimeError("The neural grid is not eligible and locked")
    if sha256_file(grid_path) != lock["source_sha256"]["neural_grid"]:
        raise RuntimeError("The neural grid changed after locking")
    cross_validation = grid["cross_validation"]
    jobs = []
    if args.stage == "inner":
        candidates = [candidate["candidate_id"] for candidate in lock["candidates"]]
        for outer in range(int(cross_validation["outer_folds"])):
            for inner in range(int(cross_validation["inner_folds"])):
                for candidate in candidates:
                    for seed in cross_validation["screening_seeds"]:
                        jobs.append((outer, inner, candidate, int(seed)))
    else:
        selection_path = args.selection_lock.resolve()
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("status") != "VCOCO_V3_NEURAL_INNER_SELECTION_LOCKED":
            raise RuntimeError("Outer queue requires the completed inner selection lock")
        if selection.get("source_sha256", {}).get("neural_grid_lock") != sha256_file(lock_path):
            raise RuntimeError("The selection lock belongs to a different neural grid")
        for outer in range(int(cross_validation["outer_folds"])):
            candidate = selection["selected_by_outer_fold"][str(outer)]["candidate_id"]
            for seed in cross_validation["outer_fit_seeds"]:
                jobs.append((outer, None, candidate, int(seed)))

    pending = []
    completed_count = 0
    for outer, inner, candidate, seed in jobs:
        path = output_directory(
            run_root,
            stage=args.stage,
            outer=outer,
            inner=inner,
            candidate=candidate,
            seed=seed,
        )
        expected = {
            "status": "VCOCO_V3_NEURAL_RUN_COMPLETE",
            "role": "inner_screen" if args.stage == "inner" else "outer_fit",
            "outer_fold": outer,
            "inner_fold": inner,
            "candidate_id": candidate,
            "seed": seed,
        }
        if complete(path, expected):
            completed_count += 1
        else:
            pending.append((outer, inner, candidate, seed, path))
    selected_pending = pending[: args.max_runs] if args.max_runs is not None else pending
    status_path = run_root / f"{args.stage}_queue_status.json"
    status = {
        "status": "PLANNED" if not args.execute else "RUNNING",
        "stage": args.stage,
        "declared_jobs": len(jobs),
        "complete_before_start": completed_count,
        "pending_before_start": len(pending),
        "selected_this_invocation": len(selected_pending),
        "neural_grid_lock_sha256": sha256_file(lock_path),
    }
    write_json(status_path, status)
    print(json.dumps(status, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        for outer, inner, candidate, seed, path in selected_pending[:20]:
            print(
                json.dumps(
                    {
                        "outer_fold": outer,
                        "inner_fold": inner,
                        "candidate_id": candidate,
                        "seed": seed,
                        "output_dir": str(path),
                    },
                    sort_keys=True,
                )
            )
        return

    started = time.perf_counter()
    executed = 0
    runner = root / "experiments" / "train_vcoco_v3_neural.py"
    for outer, inner, candidate, seed, path in selected_pending:
        command = [
            sys.executable,
            str(runner),
            "--grid",
            str(grid_path),
            "--neural-lock",
            str(lock_path),
            "--role",
            "inner_screen" if args.stage == "inner" else "outer_fit",
            "--candidate-id",
            candidate,
            "--outer-fold",
            str(outer),
            "--seed",
            str(seed),
            "--output-dir",
            str(path),
            "--workers",
            str(args.workers),
        ]
        if inner is not None:
            command.extend(["--inner-fold", str(inner)])
        if args.stage == "outer":
            command.extend(["--selection-lock", str(args.selection_lock.resolve())])
        result = subprocess.run(command, cwd=root, check=False)
        if result.returncode:
            status.update(
                {
                    "status": "FAILED",
                    "executed": executed,
                    "failed_job": {
                        "outer_fold": outer,
                        "inner_fold": inner,
                        "candidate_id": candidate,
                        "seed": seed,
                        "return_code": result.returncode,
                    },
                    "runtime_seconds": time.perf_counter() - started,
                }
            )
            write_json(status_path, status)
            raise RuntimeError(f"Neural queue stopped after failed job: {path}")
        executed += 1
        status.update(
            {
                "status": "RUNNING",
                "executed": executed,
                "runtime_seconds": time.perf_counter() - started,
            }
        )
        write_json(status_path, status)
    status.update(
        {
            "status": "INVOCATION_COMPLETE",
            "executed": executed,
            "runtime_seconds": time.perf_counter() - started,
        }
    )
    write_json(status_path, status)
    print(json.dumps(status, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
