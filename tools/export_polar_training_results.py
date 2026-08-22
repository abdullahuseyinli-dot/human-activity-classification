"""Export path-free development training summaries and preserved failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hac.polar import sha256_file


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def flattened_record(root: Path, summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("test_rows_read") != 0 or summary.get("test_used_for_selection"):
        raise RuntimeError(f"Training summary violates the test gate: {summary_path}")
    request_path = summary_path.parent / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    configuration = summary["configuration"]
    metrics = summary["best_validation_metrics"]
    counts = summary["parameter_counts"]
    return {
        "run_id": summary_path.parent.relative_to(root).as_posix(),
        **configuration,
        "train_rows": summary["train_rows"],
        "validation_rows": summary["validation_rows"],
        "best_epoch": summary["best_epoch"],
        "epochs_completed": summary["epochs_completed"],
        "stopped_early": summary["stopped_early"],
        **metrics,
        "parameters_total": counts["total"],
        "parameters_trainable": counts["trainable"],
        "runtime_seconds": summary["runtime_seconds_this_invocation"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "request_sha256": summary["request_sha256"],
        "git_revision_at_start": request.get("git_revision_at_start", ""),
        "implementation_hash_count": len(request.get("implementation_sha256", {})),
        "summary_sha256": sha256_file(summary_path),
        "test_rows_read": 0,
    }


def failure_records(root: Path) -> list[dict]:
    records = []
    failure_paths = sorted(root.rglob("failure.json")) + sorted(
        root.rglob("failure_observed.json")
    )
    for path in failure_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("test_rows_read") != 0:
            raise RuntimeError(f"Failure record violates the test gate: {path}")
        records.append(
            {
                "run_id": path.parent.relative_to(root).as_posix(),
                "error_type": payload.get("error_type", ""),
                "error": payload.get("error", ""),
                "disposition": payload.get("disposition", "preserved; excluded from selection"),
                "record_sha256": sha256_file(path),
                "test_rows_read": 0,
            }
        )
    return records


def main() -> None:
    args = parse_args()
    root = args.training_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [flattened_record(root, path) for path in sorted(root.rglob("summary.json"))]
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame = frame.sort_values(
            ["run_role", "task", "macro_f1", "log_loss", "run_id"],
            ascending=[True, True, False, True, True],
            ignore_index=True,
        )
    frame.to_csv(output_dir / "polar_training_runs.csv", index=False)
    failures = failure_records(root)
    (output_dir / "polar_training_failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selection = (
        frame[frame["run_role"].ne("engineering_smoke")].copy()
        if not frame.empty
        else frame.copy()
    )
    top = []
    if not selection.empty:
        top = json_safe(
            selection.sort_values(
                ["task", "macro_f1", "log_loss"],
                ascending=[True, False, True],
            )
            .groupby("task", sort=True)
            .head(10)
            .to_dict("records")
        )
    summary = {
        "status": "DEVELOPMENT_ONLY_TRAINING_EVIDENCE",
        "completed_runs": len(frame),
        "selection_eligible_runs": len(selection),
        "preserved_failures": len(failures),
        "top_selection_runs": top,
        "test_rows_read": 0,
        "test_used_for_selection": False,
    }
    (output_dir / "polar_training_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
