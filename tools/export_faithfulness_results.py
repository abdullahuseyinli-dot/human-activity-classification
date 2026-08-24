"""Promote validated attribution evidence into the portable repository release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath

import pandas as pd

RESULT_FILES = (
    "faithfulness_method_selection.csv",
    "faithfulness_selection_lock.json",
    "oof_method_selection_per_image.csv",
    "oof_replay_validation.csv",
    "faithfulness_test_summary.csv",
    "faithfulness_test_per_image.csv",
    "faithfulness_test_curves.csv",
    "faithfulness_replay_validation.csv",
    "faithfulness_sanity_checks.csv",
    "faithfulness_sanity_summary.csv",
    "faithfulness_stability_checks.csv",
    "faithfulness_stability_summary.csv",
    "faithfulness_checkpoint_manifest.csv",
)
PORTABLE_ASSET_FILES = (
    "faithfulness_method_selection.png",
    "faithfulness_method_selection.svg",
    "faithfulness_perturbation_curves.png",
    "faithfulness_perturbation_curves.svg",
)
LOCAL_QUALITATIVE_ASSET_FILES = (
    "convnext_small_faithfulness_gallery.jpg",
    "dinov2_small_faithfulness_gallery.jpg",
    "probability_blend_faithfulness_gallery.jpg",
)
EXCLUDED_MEDIA_REASON = (
    "Qualitative composites contain COCO/Flickr source photographs and remain part "
    "of validated local run evidence, but are not redistributed in the public release."
)
EXPECTED_MODELS = {"convnext_small", "dinov2_small", "probability_blend"}
EXPECTED_FAMILIES = {"convnext_small", "dinov2_small"}
EXPECTED_CANDIDATES = {
    "convnext_small": {"gradcam", "hirescam", "integrated_gradients"},
    "dinov2_small": {
        "attention_rollout",
        "gradient_attention_rollout",
        "integrated_gradients",
    },
}
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(line.rstrip() for line in lines) + "\n")


def require_files(source_dir: Path) -> None:
    required = (
        *RESULT_FILES,
        *PORTABLE_ASSET_FILES,
        *LOCAL_QUALITATIVE_ASSET_FILES,
        "faithfulness_provenance.json",
        "oof_selection_cohort.csv",
    )
    missing = [name for name in required if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Faithfulness evidence is incomplete: {missing}")


def validate_evidence(source_dir: Path, repository: Path) -> None:
    lock_path = source_dir / "faithfulness_selection_lock.json"
    lock = read_json(lock_path)
    if lock.get("status") != "LOCKED_FROM_OOF_BEFORE_FAITHFULNESS_TEST_EVALUATION":
        raise RuntimeError("Invalid attribution-selection lock status")
    if lock.get("test_used_for_selection") is not False:
        raise RuntimeError("Attribution selection is not test-independent")
    if set(lock.get("selected", {})) != EXPECTED_FAMILIES:
        raise RuntimeError("Attribution lock does not cover both model families")

    provenance = read_json(source_dir / "faithfulness_provenance.json")
    if provenance.get("status") != "LOCKED_TEST_EVALUATED_AFTER_OOF_ATTRIBUTION_SELECTION":
        raise RuntimeError("Faithfulness run did not reach its completed status")
    runner_path = repository / "experiments" / "evaluate_faithfulness.py"
    if provenance.get("runtime", {}).get("runner_sha256") != sha256_file(runner_path):
        raise RuntimeError("Faithfulness runner changed after the recorded execution")
    recorded_lock_hash = provenance.get("inputs", {}).get("faithfulness_selection_lock_sha256")
    if recorded_lock_hash != sha256_file(lock_path):
        raise RuntimeError("Faithfulness lock fingerprint does not match provenance")
    recorded_evidence = provenance.get("evidence", {})
    for name in (
        *RESULT_FILES,
        *PORTABLE_ASSET_FILES,
        *LOCAL_QUALITATIVE_ASSET_FILES,
        "oof_selection_cohort.csv",
    ):
        if recorded_evidence.get(name) != sha256_file(source_dir / name):
            raise RuntimeError(f"Evidence fingerprint mismatch: {name}")

    cohort = pd.read_csv(source_dir / "oof_selection_cohort.csv", dtype={"image_id": str})
    if (
        len(cohort) != 36
        or cohort["image_id"].duplicated().any()
        or not cohort.groupby("label").size().eq(12).all()
    ):
        raise RuntimeError("OOF attribution-selection cohort is not 12 rows per class")
    selection_detail = pd.read_csv(
        source_dir / "oof_method_selection_per_image.csv", dtype={"image_id": str}
    )
    for family, candidates in EXPECTED_CANDIDATES.items():
        family_rows = selection_detail[selection_detail["family"].eq(family)]
        if set(family_rows["method"]) != candidates:
            raise RuntimeError(f"OOF attribution candidates are incomplete for {family}")
        if not family_rows.groupby("method").size().eq(36).all():
            raise RuntimeError(f"OOF attribution evidence is incomplete for {family}")

    manifest = pd.read_csv(repository / "data" / "manifest.csv", dtype={"image_id": str})
    expected_ids = set(manifest.loc[manifest["split"].eq("test"), "image_id"])
    per_image = pd.read_csv(source_dir / "faithfulness_test_per_image.csv", dtype={"image_id": str})
    if set(per_image["model"]) != EXPECTED_MODELS:
        raise RuntimeError("Faithfulness test evidence has an unexpected model set")
    for model, rows in per_image.groupby("model"):
        if len(rows) != 43 or rows["image_id"].duplicated().any():
            raise RuntimeError(f"Invalid locked-test evidence cardinality for {model}")
        if set(rows["image_id"]) != expected_ids:
            raise RuntimeError(f"Locked-test IDs do not match the manifest for {model}")

    summary = pd.read_csv(source_dir / "faithfulness_test_summary.csv")
    if set(summary["model"]) != EXPECTED_MODELS or not summary["test_rows"].eq(43).all():
        raise RuntimeError("Faithfulness summary does not cover all 43 locked-test rows")
    for name in ("oof_replay_validation.csv", "faithfulness_replay_validation.csv"):
        replay = pd.read_csv(source_dir / name)
        passed = replay["passed"].astype(str).str.casefold().eq("true")
        if replay.empty or not passed.all():
            raise RuntimeError(f"Probability replay validation failed in {name}")

    checkpoint_manifest = pd.read_csv(source_dir / "faithfulness_checkpoint_manifest.csv")
    if len(checkpoint_manifest) != 36 or set(checkpoint_manifest["family"]) != EXPECTED_FAMILIES:
        raise RuntimeError("Checkpoint manifest must contain 36 OOF/final checkpoints")
    for value in checkpoint_manifest["artifact_relative_path"].astype(str):
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts or WINDOWS_ABSOLUTE.match(value):
            raise RuntimeError(f"Non-portable checkpoint reference: {value}")


def export_selection_cohort(source_dir: Path, results_dir: Path) -> None:
    cohort = pd.read_csv(source_dir / "oof_selection_cohort.csv", dtype={"image_id": str})
    columns = ["image_id", "label", "cv_row_id", "fold_id", "selection_key"]
    missing = [column for column in columns if column not in cohort]
    if missing:
        raise RuntimeError(f"OOF cohort is missing columns: {missing}")
    cohort[columns].to_csv(
        results_dir / "faithfulness_oof_selection_cohort.csv",
        index=False,
        lineterminator="\n",
    )


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    repository = args.repository.resolve()
    require_files(source_dir)
    validate_evidence(source_dir, repository)

    results_dir = repository / "results"
    assets_dir = repository / "assets"
    results_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    export_selection_cohort(source_dir, results_dir)

    for name in RESULT_FILES:
        shutil.copy2(source_dir / name, results_dir / name)
    for name in PORTABLE_ASSET_FILES:
        destination = assets_dir / name
        shutil.copy2(source_dir / name, destination)
        if destination.suffix == ".svg":
            normalize_svg(destination)

    source_provenance_path = source_dir / "faithfulness_provenance.json"
    provenance = read_json(source_provenance_path)
    promoted_paths = [
        *(results_dir / name for name in RESULT_FILES),
        results_dir / "faithfulness_oof_selection_cohort.csv",
        *(assets_dir / name for name in PORTABLE_ASSET_FILES),
    ]
    provenance["release_export"] = {
        "status": "VALIDATED_PATH_SANITIZED_EVIDENCE_PROMOTED",
        "source_provenance_sha256": sha256_file(source_provenance_path),
        "implementation_fingerprints": {
            name: sha256_file(repository / name)
            for name in (
                "experiments/evaluate_faithfulness.py",
                "src/hac/explainability.py",
                "src/hac/models.py",
                "tools/export_faithfulness_results.py",
                "tools/render_faithfulness_figures.py",
            )
        },
        "tracked_evidence": {
            path.relative_to(repository).as_posix(): sha256_file(path)
            for path in sorted(promoted_paths)
        },
        "excluded_third_party_media": {
            "reason": EXCLUDED_MEDIA_REASON,
            "artifacts": {
                name: sha256_file(source_dir / name) for name in LOCAL_QUALITATIVE_ASSET_FILES
            },
        },
    }
    provenance.pop("evidence", None)
    write_json(results_dir / "faithfulness_provenance.json", provenance)
    print(f"Exported validated faithfulness evidence to {results_dir}")


if __name__ == "__main__":
    main()
