"""Export compact, path-free POLAR probe evidence for version control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Named probe output directory; may be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_screen(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Screens must use NAME=DIR")
    name, path = value.split("=", maxsplit=1)
    if not name or not path:
        raise ValueError("Screens must use non-empty NAME=DIR")
    return name, Path(path).resolve()


def best_records(frame: pd.DataFrame, count: int = 10) -> list[dict]:
    columns = [
        "task",
        "candidate",
        "classifier",
        "C",
        "class_weight",
        "feature_dimensions",
        "train_rows",
        "validation_rows",
        "macro_f1",
        "accuracy",
        "balanced_accuracy",
        "log_loss",
        "brier_score",
        "ece",
        "iterations",
        "converged",
        "screen",
    ]
    ranked = frame.sort_values(
        ["task", "macro_f1", "log_loss", "candidate"],
        ascending=[True, False, True, True],
    )
    return ranked.groupby("task", sort=True).head(int(count))[columns].to_dict("records")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    sources = []
    seen_names = set()
    manifest_hashes = set()
    for value in args.screen:
        name, directory = parse_screen(value)
        if name in seen_names:
            raise ValueError(f"Duplicate screen name: {name}")
        seen_names.add(name)
        results_path = directory / "probe_screen.csv"
        provenance_path = directory / "probe_provenance.json"
        frame = pd.read_csv(results_path)
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("test_rows_read") != 0 or provenance.get("test_used_for_selection"):
            raise RuntimeError(f"Probe provenance violates the test gate: {directory}")
        manifest_hashes.add(str(provenance["manifest_sha256"]))
        frame["screen"] = name
        if "converged" not in frame:
            frame["converged"] = frame["iterations"].astype(int) < 2000
        frames.append(frame)
        sources.append(
            {
                "screen": name,
                "result_rows": len(frame),
                "results_sha256": sha256_file(results_path),
                "provenance_sha256": sha256_file(provenance_path),
                "executed_tasks": provenance.get("executed_tasks", list(provenance["tasks"])),
                "candidate_filters": provenance.get("candidate_filters", []),
            }
        )
    if len(manifest_hashes) != 1:
        raise RuntimeError("Probe screens use different development manifests")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["task", "macro_f1", "log_loss", "candidate", "screen"],
        ascending=[True, False, True, True, True],
        ignore_index=True,
    )
    combined.to_csv(output_dir / "polar_probe_screen.csv", index=False)
    summary = {
        "status": "DEVELOPMENT_ONLY_PROBE_EVIDENCE",
        "development_manifest_sha256": next(iter(manifest_hashes)),
        "screens": sources,
        "result_rows": len(combined),
        "top_results_per_task": best_records(combined),
        "test_rows_read": 0,
        "test_used_for_selection": False,
    }
    (output_dir / "polar_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
