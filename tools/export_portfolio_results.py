"""Export compact, path-sanitized evidence from local experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import pandas as pd

TENSOR_ID = re.compile(r"^tensor\(([^)]+)\)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def sanitize_selection_lock(lock: dict) -> dict:
    selected = {}
    for family, row in lock["selected"].items():
        selected[family] = {
            key: value for key, value in row.items() if key not in {"evidence_path"}
        }
    return {
        key: value for key, value in lock.items() if key not in {"protocol_path", "selected"}
    } | {"selected": selected}


def sanitize_data_protocol(protocol: dict) -> dict:
    """Normalize legacy experiment labels without altering measured fields."""

    sanitized = dict(protocol)
    if str(sanitized.get("protocol", "")).endswith(
        "fixed_test_plus_internal_stratified_cv"
    ):
        sanitized["protocol"] = "fixed_test_plus_internal_stratified_cv"
    return sanitized


def export_parameter_summary(final_root: Path, results_dir: Path) -> None:
    rows = []
    for family in ("convnext_small", "dinov2_small"):
        result_paths = sorted(
            (final_root / "predictions" / family).glob("seed_*/final/seed_result.json")
        )
        if not result_paths:
            raise RuntimeError(f"No final seed result for {family}")
        seed_result = load_json(result_paths[0])
        checkpoint = Path(seed_result["checkpoint_path"])
        run_summary = load_json(checkpoint.parent / "run_summary.json")
        rows.append(
            {
                "model_kind": family,
                "candidate_id": seed_result["candidate_id"],
                "total_params": int(run_summary["total_params"]),
                "trainable_params": int(run_summary["trainable_params"]),
                "trainable_percent": 100.0
                * float(run_summary["trainable_params"])
                / float(run_summary["total_params"]),
            }
        )
    pd.DataFrame(rows).to_csv(results_dir / "model_parameter_summary.csv", index=False)


def dataset_content_fingerprint(manifest_path: Path) -> str:
    frame = pd.read_csv(manifest_path, dtype={"image_id": str, "sha256": str})
    rows = frame[["image_id", "label", "split", "sha256"]].astype(str)
    rows = rows.sort_values("image_id")
    payload = "\n".join("|".join(row) for row in rows.to_numpy())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_image_id(value: object) -> str:
    """Convert IDs emitted by a collated numeric manifest back to portable text."""
    text = str(value).strip()
    match = TENSOR_ID.fullmatch(text)
    return match.group(1).strip() if match else text


def normalize_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    selection_root = args.selection_root.resolve()
    final_root = args.final_root.resolve()
    analysis_dir = args.analysis_dir.resolve()
    repository = args.repository.resolve()
    results_dir = repository / "results"
    assets_dir = repository / "assets"
    results_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        selection_root / "candidate_selection_summary.csv",
        results_dir / "candidate_selection_summary.csv",
    )
    lock_path = args.selection_lock.resolve()
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    selection_lock = load_json(lock_path)
    write_json(results_dir / "selection_lock.json", sanitize_selection_lock(selection_lock))
    shutil.copy2(lock_path.with_suffix(".csv"), results_dir / "confirmation_ranking.csv")
    shutil.copy2(final_root / "final_seed_metrics.csv", results_dir / "final_seed_metrics.csv")

    analysis_files = [
        "downstream_oof_ranking.csv",
        "evaluation_policy_oof_selection.csv",
        "probability_blend_oof_search.csv",
        "convnext_small_svm_oof_search.csv",
        "dinov2_small_svm_oof_search.csv",
        "locked_test_metrics.csv",
        "test_bootstrap_intervals.csv",
        "champion_paired_bootstrap_difference.csv",
        "champion_confusion_matrix.csv",
        "downstream_selection_lock.json",
    ]
    for name in analysis_files:
        shutil.copy2(analysis_dir / name, results_dir / name)

    downstream_lock = load_json(analysis_dir / "downstream_selection_lock.json")
    champion = downstream_lock["champion_method"]
    champion_predictions = pd.read_csv(
        analysis_dir / f"{champion}_test_predictions.csv", dtype={"image_id": str}
    )
    champion_predictions["image_id"] = champion_predictions["image_id"].map(
        normalize_image_id
    )
    public_manifest = pd.read_csv(
        repository / "data" / "manifest.csv", dtype={"image_id": str}
    )
    expected_test_ids = set(
        public_manifest.loc[public_manifest["split"].eq("test"), "image_id"]
    )
    if (
        champion_predictions["image_id"].duplicated().any()
        or set(champion_predictions["image_id"]) != expected_test_ids
    ):
        raise RuntimeError("Champion predictions do not align with the fixed test IDs.")
    champion_predictions.to_csv(
        results_dir / "champion_test_predictions.csv", index=False, lineterminator="\n"
    )
    for name in (
        "final_method_comparison.png",
        "final_method_comparison.svg",
        "champion_confusion_matrix.png",
        "champion_confusion_matrix.svg",
    ):
        destination = assets_dir / name
        shutil.copy2(analysis_dir / name, destination)
        if destination.suffix == ".svg":
            normalize_svg(destination)

    runtime = load_json(final_root / "configs" / "runtime_provenance.json")
    protocol = load_json(final_root / "configs" / "selection_protocol.json")
    execution = load_json(final_root / "configs" / "final_execution_plan.json")
    execution.pop("selection_lock", None)
    provenance = {
        "runtime": runtime,
        "data_protocol": sanitize_data_protocol(protocol),
        "final_execution": execution,
        "selection_protocol_sha256": selection_lock.get("protocol_sha256"),
        "downstream_lock": downstream_lock,
        "portable_dataset_content_sha256": dataset_content_fingerprint(
            repository / "data" / "manifest.csv"
        ),
    }
    write_json(results_dir / "run_provenance.json", provenance)
    export_parameter_summary(final_root, results_dir)
    print(f"Exported portfolio evidence to {results_dir}")


if __name__ == "__main__":
    main()
