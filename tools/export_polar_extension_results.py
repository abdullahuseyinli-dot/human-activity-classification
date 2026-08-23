"""Publish path-free validation and scale evidence from the local POLAR study."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--scale-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_gate_checked_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test_rows_read") != 0 or payload.get("test_used_for_selection") is True:
        raise RuntimeError(f"Development evidence violates the test gate: {path}")
    return payload


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copyfile(source, destination)


def main() -> None:
    args = parse_args()
    validation_dir = args.validation_dir.resolve()
    scale_dir = args.scale_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_provenance = load_gate_checked_json(validation_dir / "provenance.json")
    validation_blend = load_gate_checked_json(validation_dir / "validation_blend.json")
    scale_provenance = load_gate_checked_json(scale_dir / "provenance.json")

    mappings = {
        validation_dir / "validation_metrics.csv": output_dir / "polar_validation_metrics.csv",
        validation_dir
        / "validation_per_class.csv": output_dir
        / "polar_validation_per_class.csv",
        validation_dir
        / "validation_complementarity.csv": output_dir
        / "polar_validation_complementarity.csv",
        validation_dir
        / "validation_confusions.json": output_dir
        / "polar_validation_confusions.json",
        validation_dir
        / "validation_uncertainty.json": output_dir
        / "polar_validation_uncertainty.json",
        validation_dir / "validation_blend.json": output_dir / "polar_validation_blend.json",
        validation_dir / "provenance.json": output_dir / "polar_validation_provenance.json",
        scale_dir / "scale_runs.csv": output_dir / "polar_scale_runs.csv",
        scale_dir / "scale_summary.csv": output_dir / "polar_scale_summary.csv",
        scale_dir / "provenance.json": output_dir / "polar_scale_provenance.json",
    }
    for source, destination in mappings.items():
        copy_file(source, destination)

    validation_metrics = pd.read_csv(validation_dir / "validation_metrics.csv")
    scale_summary = pd.read_csv(scale_dir / "scale_summary.csv")
    if validation_metrics.empty or scale_summary.empty:
        raise RuntimeError("POLAR extension evidence is unexpectedly empty")
    summary = {
        "status": "DEVELOPMENT_ONLY_POLAR_EXTENSION_EVIDENCE",
        "best_validation_candidate": validation_metrics.iloc[0].to_dict(),
        "validation_blend_weights": validation_blend["weights"],
        "scale_architecture": scale_provenance["architecture"],
        "scale_curve": scale_summary.to_dict("records"),
        "source_sha256": {
            "validation_provenance": sha256_file(validation_dir / "provenance.json"),
            "scale_provenance": sha256_file(scale_dir / "provenance.json"),
        },
        "artifact_sha256": validation_provenance["artifact_sha256"],
        "test_rows_read": 0,
        "test_used_for_selection": False,
    }
    (output_dir / "polar_extension_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
