"""Fail fast on portability, notebook, evidence, and repository-size regressions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

import pandas as pd

TEXT_SUFFIXES = {".csv", ".ipynb", ".json", ".md", ".py", ".toml", ".yaml", ".yml"}
LEGACY_FINGERPRINT_TEXT_SUFFIXES = {".csv", ".json", ".py", ".svg"}
EXCLUDED_PARTS = {".git", ".runs", "__pycache__", ".pytest_cache", ".ruff_cache"}
REQUIRED = {
    ".github/workflows/ci.yml",
    ".gitignore",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "data/manifest.csv",
    "docs/EXPERIMENT_PROTOCOL.md",
    "experiments/evaluate_faithfulness.py",
    "human_activity_classification.ipynb",
    "pyproject.toml",
    "src/hac/explainability.py",
    "assets/faithfulness_method_selection.png",
    "assets/faithfulness_perturbation_curves.png",
    "assets/convnext_small_faithfulness_gallery.jpg",
    "assets/dinov2_small_faithfulness_gallery.jpg",
    "results/locked_test_metrics.csv",
    "results/faithfulness_selection_lock.json",
    "results/faithfulness_test_summary.csv",
    "results/faithfulness_test_per_image.csv",
    "results/faithfulness_replay_validation.csv",
    "results/faithfulness_sanity_summary.csv",
    "results/faithfulness_stability_summary.csv",
    "results/faithfulness_checkpoint_manifest.csv",
    "results/faithfulness_oof_selection_cohort.csv",
    "results/faithfulness_provenance.json",
    "results/oof_replay_validation.csv",
    "results/run_provenance.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    return parser.parse_args()


def included_files(repository: Path):
    for path in repository.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repository)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if relative.parts[:2] == ("data", "images"):
            continue
        yield path, relative


def validate_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if int(notebook.get("nbformat", 0)) != 4:
        raise RuntimeError(f"Unsupported notebook version: {path}")
    kernelspec = notebook.get("metadata", {}).get("kernelspec", {})
    if kernelspec.get("name") != "python3":
        raise RuntimeError(f"Non-portable notebook kernel: {path}")
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                raise RuntimeError(f"Notebook contains an error output: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def legacy_release_hashes(path: Path) -> set[str]:
    """Return byte-equivalent hashes across historical Git line-ending checkouts."""

    encoded = path.read_bytes()
    candidates = {hashlib.sha256(encoded).hexdigest()}
    if path.suffix.lower() in LEGACY_FINGERPRINT_TEXT_SUFFIXES:
        linefeed = encoded.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        candidates.add(hashlib.sha256(linefeed).hexdigest())
        candidates.add(hashlib.sha256(linefeed.replace(b"\n", b"\r\n")).hexdigest())
    return candidates


def validate_faithfulness(repository: Path, expected_test_ids: set[str]) -> None:
    results = repository / "results"
    with (results / "faithfulness_selection_lock.json").open(encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("status") != "LOCKED_FROM_OOF_BEFORE_FAITHFULNESS_TEST_EVALUATION":
        raise RuntimeError("Invalid attribution-selection lock status")
    if selection.get("test_used_for_selection") is not False:
        raise RuntimeError("Attribution selection is not independent of the test split")
    selected = selection.get("selected", {})
    if set(selected) != {"convnext_small", "dinov2_small"}:
        raise RuntimeError("Attribution selection does not cover both model families")

    cohort = pd.read_csv(
        results / "faithfulness_oof_selection_cohort.csv", dtype={"image_id": str}
    )
    if (
        len(cohort) != 36
        or cohort["image_id"].duplicated().any()
        or not cohort.groupby("label").size().eq(12).all()
    ):
        raise RuntimeError("Invalid OOF attribution-selection cohort")

    per_image = pd.read_csv(
        results / "faithfulness_test_per_image.csv", dtype={"image_id": str}
    )
    expected_models = {"convnext_small", "dinov2_small", "probability_blend"}
    if set(per_image["model"]) != expected_models:
        raise RuntimeError("Unexpected model set in faithfulness test evidence")
    for model, rows in per_image.groupby("model"):
        if len(rows) != 43 or rows["image_id"].duplicated().any():
            raise RuntimeError(f"Invalid faithfulness row count for {model}")
        if set(rows["image_id"]) != expected_test_ids:
            raise RuntimeError(f"Faithfulness IDs do not match the test split for {model}")

    summary = pd.read_csv(results / "faithfulness_test_summary.csv")
    if set(summary["model"]) != expected_models or not summary["test_rows"].eq(43).all():
        raise RuntimeError("Faithfulness summary does not cover the fixed test split")
    for name in ("oof_replay_validation.csv", "faithfulness_replay_validation.csv"):
        replay = pd.read_csv(results / name)
        if replay.empty or not replay["passed"].astype(str).str.casefold().eq("true").all():
            raise RuntimeError(f"Probability replay validation failed in {name}")

    checkpoints = pd.read_csv(results / "faithfulness_checkpoint_manifest.csv")
    if len(checkpoints) != 36:
        raise RuntimeError("Faithfulness checkpoint manifest must contain 36 checkpoints")
    for value in checkpoints["artifact_relative_path"].astype(str):
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Invalid relative checkpoint reference: {value}")

    provenance_path = results / "faithfulness_provenance.json"
    with provenance_path.open(encoding="utf-8") as handle:
        provenance = json.load(handle)
    if provenance.get("status") != "LOCKED_TEST_EVALUATED_AFTER_OOF_ATTRIBUTION_SELECTION":
        raise RuntimeError("Faithfulness provenance is incomplete")
    release = provenance.get("release_export", {})
    if release.get("status") != "VALIDATED_PATH_SANITIZED_EVIDENCE_PROMOTED":
        raise RuntimeError("Faithfulness release evidence was not validated before export")
    for relative, expected_hash in release.get("tracked_evidence", {}).items():
        path = repository / PurePosixPath(relative)
        if not path.is_file() or expected_hash not in legacy_release_hashes(path):
            raise RuntimeError(f"Faithfulness release fingerprint mismatch: {relative}")
    for relative, expected_hash in release.get("implementation_fingerprints", {}).items():
        path = repository / PurePosixPath(relative)
        if not path.is_file() or expected_hash not in legacy_release_hashes(path):
            raise RuntimeError(f"Faithfulness implementation fingerprint mismatch: {relative}")


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    missing = sorted(name for name in REQUIRED if not (repository / name).is_file())
    if missing:
        raise RuntimeError(f"Missing release files: {missing}")

    license_text = (repository / "LICENSE").read_text(encoding="utf-8")
    for marker in (
        "MIT License",
        "Copyright (c) 2026 Abdulla Huseyinli",
        "Permission is hereby granted, free of charge",
        'THE SOFTWARE IS PROVIDED "AS IS"',
    ):
        if marker not in license_text:
            raise RuntimeError(f"LICENSE is missing: {marker}")

    notices = (repository / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for marker in (
        "cocodataset.org/#termsofuse",
        "facebookresearch/dinov2",
        "pytorch/vision",
        "jacobgil/pytorch-grad-cam",
    ):
        if marker not in notices:
            raise RuntimeError(f"Third-party notice is missing: {marker}")

    windows_user_path = re.compile(
        r"[A-Za-z]:\\" + "Users" + r"\\", flags=re.IGNORECASE
    )
    local_file_scheme = "file" + "://"
    oversized = []
    for path, relative in included_files(repository):
        if path.stat().st_size > 10 * 1024 * 1024:
            oversized.append((str(relative), path.stat().st_size))
        if path.name in {".gitignore", ".gitattributes"} or path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            if windows_user_path.search(text) or local_file_scheme in text.lower():
                raise RuntimeError(f"Local absolute path leaked into {relative}")
        if path.suffix.lower() == ".ipynb":
            validate_notebook(path)
    if oversized:
        raise RuntimeError(f"Files exceed the 10 MiB portfolio limit: {oversized}")

    with (repository / "results" / "selection_lock.json").open(encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("status") != "LOCKED_BEFORE_FINAL_TEST":
        raise RuntimeError("Invalid model-selection lock status")
    with (repository / "results" / "downstream_selection_lock.json").open(
        encoding="utf-8"
    ) as handle:
        downstream = json.load(handle)
    if downstream.get("status") != "LOCKED_FROM_OOF_BEFORE_DOWNSTREAM_TEST_EVALUATION":
        raise RuntimeError("Invalid downstream-selection lock status")

    sys.path.insert(0, str(repository / "src"))
    from hac.protocol import load_and_validate_manifest

    manifest, protocol = load_and_validate_manifest(
        repository / "data" / "manifest.csv", require_images=False
    )
    if protocol.development_rows != 242 or protocol.test_rows != 43:
        raise RuntimeError("Tracked manifest no longer matches the fixed protocol")
    expected_test_ids = set(
        manifest.loc[manifest["split"].eq("test"), "image_id"].astype(str)
    )
    validate_faithfulness(repository, expected_test_ids)
    print("Repository validation passed")


if __name__ == "__main__":
    main()
