"""Build or verify the POLAR Study Report v1.0.0 release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from pathlib import Path, PurePosixPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ID = "polar-study-v1.0.0"
REPORT_VERSION = "1.0.0"
SOFTWARE_VERSION = "2.0.0"
DEFAULT_OUTPUT = PurePosixPath("results/polar_study_v1.0.0_manifest.json")
DEFAULT_CHECKSUMS = PurePosixPath("release/POLAR_STUDY_V1.0.0_SHA256SUMS.txt")

CORE_ARTIFACTS = (
    ".zenodo.json",
    "CHANGELOG.md",
    "CITATION.cff",
    "README.md",
    "docs/POLAR_PUBLIC_REPORT.md",
    "docs/releases/POLAR_STUDY_V1.0.0.md",
    "experiments/analyze_polar_exploratory.py",
    "output/pdf/README.md",
    "output/pdf/polar_public_report_v1.0.0.pdf",
    "results/README.md",
    "results/polar_final_evidence_manifest.json",
    "results/polar_training_failures.json",
    "tools/build_study_papers.py",
    "tools/build_study_release_manifest.py",
    "tools/export_polar_training_results.py",
)
ARTIFACT_GLOBS = (
    "assets/polar_exploratory_*.png",
    "assets/polar_exploratory_*.svg",
    "results/polar_exploratory_*.csv",
    "results/polar_exploratory_*.json",
)
EXPECTED_GLOB_COUNTS = {
    "assets/polar_exploratory_*.png": 7,
    "assets/polar_exploratory_*.svg": 7,
    "results/polar_exploratory_*.csv": 17,
    "results/polar_exploratory_*.json": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_paths(repository: Path) -> list[Path]:
    paths = {repository / PurePosixPath(relative) for relative in CORE_ARTIFACTS}
    for pattern in ARTIFACT_GLOBS:
        matches = set(repository.glob(pattern))
        expected_count = EXPECTED_GLOB_COUNTS[pattern]
        if len(matches) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} release artifacts for {pattern}, found {len(matches)}"
            )
        paths.update(matches)
    missing = sorted(path for path in paths if not path.is_file())
    if missing:
        formatted = ", ".join(path.relative_to(repository).as_posix() for path in missing)
        raise RuntimeError(f"Release artifacts are missing: {formatted}")
    return sorted(paths, key=lambda path: path.relative_to(repository).as_posix())


def validate_versions(repository: Path) -> None:
    with (repository / "pyproject.toml").open("rb") as handle:
        package_version = tomllib.load(handle)["project"]["version"]
    if package_version != SOFTWARE_VERSION:
        raise RuntimeError(
            f"Expected software version {SOFTWARE_VERSION}, found {package_version}"
        )

    report = (repository / "docs/POLAR_PUBLIC_REPORT.md").read_text(encoding="utf-8")
    match = re.search(r"(?m)^version: ([^\s]+)$", report)
    if match is None or match.group(1) != REPORT_VERSION:
        found = match.group(1) if match else "missing"
        raise RuntimeError(f"Expected report version {REPORT_VERSION}, found {found}")


def build_manifest(repository: Path) -> dict:
    repository = repository.resolve()
    validate_versions(repository)
    artifacts = {}
    for path in artifact_paths(repository):
        relative = path.relative_to(repository).as_posix()
        artifacts[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "report_version": REPORT_VERSION,
        "software_version": SOFTWARE_VERSION,
        "release_date": "2026-08-23",
        "artifact_scope": "Public report, post-lock analysis supplement, and release metadata",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def encoded_manifest(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def checksum_text(repository: Path, manifest_path: Path) -> str:
    pdf_path = repository / "output/pdf/polar_public_report_v1.0.0.pdf"
    targets = (pdf_path, manifest_path)
    return "".join(
        f"{sha256_file(path)}  {path.relative_to(repository).as_posix()}\n" for path in targets
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output", type=PurePosixPath, default=DEFAULT_OUTPUT)
    parser.add_argument("--checksums", type=PurePosixPath, default=DEFAULT_CHECKSUMS)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    output = repository / args.output
    checksums = repository / args.checksums
    expected = encoded_manifest(build_manifest(repository))

    if args.check:
        if not output.is_file() or output.read_bytes() != expected:
            raise RuntimeError(f"Release manifest is stale: {output}")
        expected_checksums = checksum_text(repository, output)
        if not checksums.is_file() or checksums.read_text(encoding="utf-8") != expected_checksums:
            raise RuntimeError(f"Release checksums are stale: {checksums}")
        print(f"Release manifest verified: {output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(expected)
    checksums.parent.mkdir(parents=True, exist_ok=True)
    checksums.write_text(checksum_text(repository, output), encoding="utf-8", newline="\n")
    print(f"Release manifest written: {output}")
    print(f"Release checksums written: {checksums}")


if __name__ == "__main__":
    main()
