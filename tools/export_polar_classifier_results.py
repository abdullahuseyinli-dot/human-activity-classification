"""Export compact, path-free evidence from the POLAR classifier experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-screen", type=Path, required=True)
    parser.add_argument("--transferred-rbf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def gated_payload(path: Path, expected_status: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != expected_status:
        raise RuntimeError(f"Unexpected evidence status in {path}")
    if payload.get("test_rows_read") != 0 or payload.get("test_used_for_selection"):
        raise RuntimeError(f"Classifier evidence violates the test gate: {path}")
    return payload


def main() -> None:
    args = parse_args()
    screen_dir = args.classifier_screen.resolve()
    rbf_dir = args.transferred_rbf.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance_path = screen_dir / "classifier_screen_provenance.json"
    rbf_summary_path = rbf_dir / "transferred_rbf_summary.json"
    provenance = gated_payload(provenance_path, "DEVELOPMENT_ONLY_CLASSIFIER_SCREEN")
    rbf_summary = gated_payload(rbf_summary_path, "DEVELOPMENT_ONLY_TRANSFERRED_RBF")
    inner_path = screen_dir / "classifier_inner_cv.csv"
    validation_path = screen_dir / "classifier_validation.csv"
    inner = pd.read_csv(inner_path)
    validation = pd.read_csv(validation_path)
    if len(inner) != 40 or len(validation) != 4:
        raise RuntimeError("Classifier screen does not contain the predeclared 40/4 rows")

    shutil.copyfile(inner_path, output_dir / "polar_classifier_inner_cv.csv")
    shutil.copyfile(validation_path, output_dir / "polar_classifier_validation.csv")
    path_free_provenance = {
        key: value
        for key, value in provenance.items()
        if key not in {"manifest_path", "protocol_path"}
    }
    (output_dir / "polar_classifier_provenance.json").write_text(
        json.dumps(path_free_provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "polar_multilayer_rbf_summary.json").write_text(
        json.dumps(rbf_summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = json_safe({
        "status": "DEVELOPMENT_ONLY_CLASSIFIER_EVIDENCE",
        "screen_configurations": len(inner),
        "validation_family_evaluations": len(validation),
        "best_screen_family": validation.iloc[0].to_dict(),
        "transferred_multilayer_rbf": rbf_summary,
        "source_sha256": {
            "inner_cv": sha256_file(inner_path),
            "validation": sha256_file(validation_path),
            "screen_provenance": sha256_file(provenance_path),
            "transferred_rbf_summary": sha256_file(rbf_summary_path),
        },
        "test_rows_read": 0,
        "test_used_for_selection": False,
    })
    (output_dir / "polar_classifier_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
