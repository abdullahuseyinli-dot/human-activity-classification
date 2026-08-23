"""Validate development evidence and write the immutable pre-test POLAR selection lock."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

from hac.polar import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def implementation_hashes(root: Path, relative_paths: list[str]) -> dict[str, str]:
    output = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        output[relative] = sha256_file(path)
    return output


def validate_development_summary(path: Path) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE":
        raise RuntimeError(f"Selection evidence is incomplete: {path}")
    if summary.get("test_rows_read") != 0 or summary.get("test_used_for_selection"):
        raise RuntimeError(f"Selection evidence violates the test gate: {path}")
    return summary


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    if git_output(repository_root, "status", "--porcelain"):
        raise RuntimeError("Pre-test selection must be locked from a clean worktree")
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("status") != "DEVELOPMENT_SELECTION_SPEC":
        raise RuntimeError("Expected a DEVELOPMENT_SELECTION_SPEC")
    if spec.get("test_rows_read") != 0 or spec.get("test_used_for_selection"):
        raise RuntimeError("Selection spec violates the test gate")

    development_manifest_path = args.development_manifest.resolve()
    development_hash = sha256_file(development_manifest_path)
    if development_hash != spec["data"]["development_manifest_sha256"]:
        raise RuntimeError("Development manifest hash differs from the selection spec")
    manifest = pd.read_csv(development_manifest_path, dtype={"image_id": str})
    if set(manifest["split"].astype(str)) != {"train", "val"}:
        raise RuntimeError("Development lock input contains a non-development split")
    if len(manifest) != spec["data"]["development_rows"]:
        raise RuntimeError("Development row count differs from the selection spec")
    quarantine_hash = sha256_file(args.quarantine)
    if quarantine_hash != spec["data"]["quarantine_sha256"]:
        raise RuntimeError("Quarantine hash differs from the selection spec")

    protocol_path = args.protocol_json.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol_version"] != spec["protocol_version"]:
        raise RuntimeError("Protocol version differs from the selection spec")

    validation_dir = args.validation_dir.resolve()
    validation_provenance = json.loads(
        (validation_dir / "provenance.json").read_text(encoding="utf-8")
    )
    validation_blend = json.loads(
        (validation_dir / "validation_blend.json").read_text(encoding="utf-8")
    )
    for payload in (validation_provenance, validation_blend):
        if payload.get("test_rows_read") != 0 or payload.get("test_used_for_selection"):
            raise RuntimeError("Validation analysis violates the test gate")
    if validation_blend["weights"] != spec["ensemble"]["weights"]:
        raise RuntimeError("Locked ensemble weights differ from validation selection")

    training_root = args.training_root.resolve()
    training_evidence = {}
    for run_id in spec["confirmation_run_ids"]:
        summary_path = training_root / run_id / "summary.json"
        summary = validate_development_summary(summary_path)
        training_evidence[run_id] = {
            "summary_sha256": sha256_file(summary_path),
            "request_sha256": summary["request_sha256"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "configuration": summary["configuration"],
            "best_epoch": summary["best_epoch"],
            "validation_metrics": summary["best_validation_metrics"],
        }

    weight_total = float(sum(spec["ensemble"]["weights"].values()))
    if not np.isclose(weight_total, 1.0, atol=1e-12):
        raise RuntimeError("Ensemble weights do not sum to one")
    if set(spec["ensemble"]["weights"]) != set(spec["ensemble"]["components"]):
        raise RuntimeError("Ensemble component metadata and weights differ")

    lock = {
        "status": "FINAL_SELECTION_LOCKED_PRE_TEST",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_version": spec["protocol_version"],
        "protocol_sha256": sha256_file(protocol_path),
        "selection_spec_sha256": sha256_file(spec_path),
        "git_revision": git_output(repository_root, "rev-parse", "HEAD"),
        "data": {
            **spec["data"],
            "development_manifest_sha256": development_hash,
            "quarantine_sha256": quarantine_hash,
            "development_split_counts": manifest["split"].value_counts().sort_index().to_dict(),
        },
        "development_selection": {
            "validation_provenance_sha256": sha256_file(validation_dir / "provenance.json"),
            "validation_metrics_sha256": sha256_file(validation_dir / "validation_metrics.csv"),
            "validation_blend_sha256": sha256_file(validation_dir / "validation_blend.json"),
            "training_evidence": training_evidence,
            "scale_evidence": spec["scale_evidence"],
        },
        "final_neural_fits": spec["final_neural_fits"],
        "final_probe_fits": spec["final_probe_fits"],
        "ensemble": spec["ensemble"],
        "calibration": spec["calibration"],
        "evaluation": spec["evaluation"],
        "faithfulness": spec["faithfulness"],
        "fault_robustness": spec["fault_robustness"],
        "external_validation": spec["external_validation"],
        "test_gate": {
            **spec["test_gate"],
            "required_state": "ALL_LOCKED_FINAL_FITS_COMPLETE",
            "official_test_manifest_open_count": 0,
        },
        "implementation_sha256": implementation_hashes(
            repository_root, spec["implementation_files"]
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "test_rows_read": 0,
        "test_used_for_selection": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(lock, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(lock, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
