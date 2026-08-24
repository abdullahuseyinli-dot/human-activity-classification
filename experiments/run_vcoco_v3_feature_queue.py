"""Plan or execute the declared v3 feature caches sequentially on one GPU."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from hac.polar import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["spatial", "representation"], required=True)
    parser.add_argument("--grid", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--status-dir", type=Path, default=Path(".runs/vcoco_v3/feature_queues"))
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def spatial_declaration(name: str, value: str) -> dict:
    path = Path(value)
    parts = list(path.parts)
    try:
        feature_index = parts.index("features")
    except ValueError as error:
        raise ValueError(f"Spatial cache path lacks the features directory: {name}") from error
    fields = parts[feature_index + 1 :]
    if len(fields) not in {3, 4}:
        raise ValueError(f"Spatial cache path has an unexpected layout: {name}")
    match = re.fullmatch(r"aspect_pad_(224|336|448)", fields[2])
    if not match:
        raise ValueError(f"Spatial cache preprocessing is not declared: {name}")
    return {
        "name": name,
        "path": value,
        "model_kind": fields[0],
        "view": fields[1],
        "preprocess": "aspect_preserving_pad",
        "image_size": int(match.group(1)),
        "box_perturbation": fields[3] if len(fields) == 4 else "none",
    }


def declared_jobs(grid: dict, stage: str) -> list[dict]:
    jobs = []
    if stage == "spatial":
        for name, value in grid["feature_caches"].items():
            if Path(value).as_posix().startswith(".runs/vcoco_v3/features/"):
                jobs.append(spatial_declaration(str(name), str(value)))
    else:
        for name, declaration in grid["feature_caches"].items():
            if Path(declaration["path"]).as_posix().startswith(".runs/vcoco_v3/features/"):
                jobs.append(
                    {
                        "name": str(name),
                        **declaration,
                        "box_perturbation": "none",
                    }
                )
    if not jobs:
        raise RuntimeError(f"The {stage} grid contains no v3 feature jobs")
    return jobs


def cache_state(root: Path, job: dict, stage: str) -> str:
    output = (root / job["path"]).resolve()
    provenance_path = output / "provenance.json"
    if not provenance_path.is_file():
        return "partial" if output.is_dir() and any(output.iterdir()) else "pending"
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = {
        "status": "VCOCO_V3_GATED_DEVELOPMENT_FEATURE_CACHE_COMPLETE",
        "stage": stage,
        "model_kind": job["model_kind"],
        "view": job["view"],
        "preprocess": job["preprocess"],
        "image_size": int(job["image_size"]),
        "box_perturbation": job["box_perturbation"],
        "official_v2_test_rows_read": 0,
        "official_v2_test_predictions_run": False,
    }
    if any(payload.get(field) != value for field, value in expected.items()):
        return "drifted"
    for artifact in ("rows.csv", "features.npy"):
        path = output / artifact
        if not path.is_file() or sha256_file(path) != payload["artifact_sha256"].get(artifact):
            return "drifted"
    return "complete"


def main() -> None:
    args = parse_args()
    if args.max_runs is not None and args.max_runs < 1:
        raise ValueError("max-runs must be positive")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers cannot be negative")
    root = Path.cwd().resolve()
    default_grid = root / f"experiments/vcoco_v3_{args.stage}_grid.json"
    grid_path = args.grid.resolve() if args.grid is not None else default_grid
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    expected_grid_status = f"DECLARED_BEFORE_{args.stage.upper()}_FITTING"
    if grid.get("status") != expected_grid_status:
        raise RuntimeError(f"The {args.stage} grid is not in its declared pre-fit state")
    jobs = declared_jobs(grid, args.stage)
    for job in jobs:
        scale = (224.0 / int(job["image_size"])) ** 2
        job["batch_size"] = max(1, int(args.batch_size * scale))
    states = {job["name"]: cache_state(root, job, args.stage) for job in jobs}
    blocked = [name for name, state in states.items() if state in {"partial", "drifted"}]
    if blocked:
        raise RuntimeError(
            "Feature queue found partial or drifted outputs; preserve and inspect them: "
            + ", ".join(blocked)
        )
    pending = [job for job in jobs if states[job["name"]] == "pending"]
    selected = pending[: args.max_runs] if args.max_runs is not None else pending
    status_path = args.status_dir.resolve() / f"{args.stage}.json"
    status = {
        "status": "PLANNED" if not args.execute else "RUNNING",
        "stage": args.stage,
        "declared_jobs": len(jobs),
        "complete_before_start": len(jobs) - len(pending),
        "pending_before_start": len(pending),
        "selected_this_invocation": len(selected),
        "grid_sha256": sha256_file(grid_path),
        "jobs": [
            {
                **job,
                "state": states[job["name"]],
                "path": str((root / job["path"]).resolve()),
            }
            for job in jobs
        ],
    }
    write_json(status_path, status)
    print(json.dumps(status, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return

    runner = root / "experiments/cache_vcoco_v3_features.py"
    started = time.perf_counter()
    for index, job in enumerate(selected, start=1):
        command = [
            sys.executable,
            str(runner),
            "--stage",
            args.stage,
            "--output-dir",
            str((root / job["path"]).resolve()),
            "--model-kind",
            job["model_kind"],
            "--view",
            job["view"],
            "--preprocess",
            job["preprocess"],
            "--image-size",
            str(job["image_size"]),
            "--box-perturbation",
            job["box_perturbation"],
            "--batch-size",
            str(job["batch_size"]),
            "--workers",
            str(args.workers),
        ]
        result = subprocess.run(command, cwd=root, check=False)
        if result.returncode:
            status.update(
                {
                    "status": "FAILED",
                    "completed_this_invocation": index - 1,
                    "failed_job": job["name"],
                    "return_code": result.returncode,
                    "runtime_seconds": time.perf_counter() - started,
                }
            )
            write_json(status_path, status)
            raise RuntimeError(f"Feature queue stopped at {job['name']}")
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
