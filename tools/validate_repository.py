"""Fail fast on portability, notebook, evidence, and repository-size regressions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TEXT_SUFFIXES = {".csv", ".ipynb", ".json", ".md", ".py", ".toml", ".yaml", ".yml"}
EXCLUDED_PARTS = {".git", ".runs", "__pycache__", ".pytest_cache", ".ruff_cache"}
REQUIRED = {
    ".github/workflows/ci.yml",
    ".gitignore",
    "README.md",
    "data/manifest.csv",
    "docs/EXPERIMENT_PROTOCOL.md",
    "human_activity_classification.ipynb",
    "pyproject.toml",
    "results/locked_test_metrics.csv",
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


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    missing = sorted(name for name in REQUIRED if not (repository / name).is_file())
    if missing:
        raise RuntimeError(f"Missing release files: {missing}")

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

    _, protocol = load_and_validate_manifest(
        repository / "data" / "manifest.csv", require_images=False
    )
    if protocol.development_rows != 242 or protocol.test_rows != 43:
        raise RuntimeError("Tracked manifest no longer matches the fixed protocol")
    print("Repository validation passed")


if __name__ == "__main__":
    main()
