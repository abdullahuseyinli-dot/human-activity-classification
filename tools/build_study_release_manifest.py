"""Build or verify the Human Activity Classification Study v2.0.0 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path, PurePosixPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ID = "polar-study-v2.0.0"
REPORT_VERSION = "2.0.0"
SOFTWARE_VERSION = "2.0.0"
RELEASE_DATE = "2026-08-24"
DEFAULT_OUTPUT = PurePosixPath("results/polar_study_v2.0.0_manifest.json")
DEFAULT_CHECKSUMS = PurePosixPath("release/POLAR_STUDY_V2.0.0_SHA256SUMS.txt")

CORE_ARTIFACTS = (
    ".zenodo.json",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/VCOCO_V2_EXTERNAL_TRANSFER.md",
    "docs/releases/POLAR_STUDY_V2.0.0.md",
    "experiments/README.md",
    "experiments/train_polar_candidate.py",
    "human_activity_classification.ipynb",
    "output/pdf/README.md",
    "output/pdf/vcoco_v2_external_transfer_v2.0.0.pdf",
    "pyproject.toml",
    "requirements-lock.txt",
    "results/README.md",
    "src/hac/augmentations.py",
    "src/hac/metrics.py",
    "src/hac/polar.py",
    "src/hac/transfer.py",
    "src/hac/vcoco.py",
    "tests/test_metrics.py",
    "tests/test_transfer.py",
    "tools/build_portfolio_notebook.py",
    "tools/build_readme.py",
    "tools/build_study_papers.py",
    "tools/build_study_release_manifest.py",
    "tools/validate_repository.py",
)
ARTIFACT_GLOBS = (
    "assets/vcoco_v2_*.png",
    "assets/vcoco_v2_*.svg",
    "experiments/*vcoco_v2*.py",
    "results/vcoco_v2/*",
    "tests/test_vcoco_*.py",
    "tools/*vcoco_v2*.py",
)
EXPECTED_GLOB_COUNTS = {
    "assets/vcoco_v2_*.png": 5,
    "assets/vcoco_v2_*.svg": 5,
    "experiments/*vcoco_v2*.py": 13,
    "results/vcoco_v2/*": 20,
    "tests/test_vcoco_*.py": 6,
    "tools/*vcoco_v2*.py": 6,
}
TEXT_ARTIFACT_SUFFIXES = frozenset(
    {".cff", ".csv", ".ipynb", ".json", ".md", ".py", ".svg", ".toml", ".txt", ".yaml", ".yml"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_paths(repository: Path) -> list[Path]:
    paths = {repository / PurePosixPath(relative) for relative in CORE_ARTIFACTS}
    for pattern in ARTIFACT_GLOBS:
        matches = {path for path in repository.glob(pattern) if path.is_file()}
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
        raise RuntimeError(f"Expected software version {SOFTWARE_VERSION}, found {package_version}")

    report = (repository / "docs/VCOCO_V2_EXTERNAL_TRANSFER.md").read_text(encoding="utf-8")
    match = re.search(r"(?m)^version: ([^\s]+)$", report)
    if match is None or match.group(1) != REPORT_VERSION:
        found = match.group(1) if match else "missing"
        raise RuntimeError(f"Expected report version {REPORT_VERSION}, found {found}")

    citation = (repository / "CITATION.cff").read_text(encoding="utf-8")
    if (
        f"version: {SOFTWARE_VERSION}" not in citation
        or f"  version: {REPORT_VERSION}" not in citation
        or f"date-released: {RELEASE_DATE}" not in citation
    ):
        raise RuntimeError("Citation metadata does not match the v2 release")


def build_manifest(repository: Path) -> dict:
    repository = repository.resolve()
    validate_versions(repository)
    artifacts = {}
    for path in artifact_paths(repository):
        relative = path.relative_to(repository).as_posix()
        if path.suffix.lower() in TEXT_ARTIFACT_SUFFIXES and b"\r" in path.read_bytes():
            raise RuntimeError(f"Release text artifact must use LF line endings: {relative}")
        artifacts[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "report_version": REPORT_VERSION,
        "software_version": SOFTWARE_VERSION,
        "release_date": RELEASE_DATE,
        "artifact_scope": (
            "V-COCO v2 technical report, portable evidence, figures, metadata, "
            "and implementation entry points"
        ),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def encoded_manifest(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def checksum_text(repository: Path, manifest_path: Path) -> str:
    pdf_path = repository / "output/pdf/vcoco_v2_external_transfer_v2.0.0.pdf"
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


def verify_frozen_release(repository: Path, manifest_path: Path, checksums_path: Path) -> None:
    if not manifest_path.is_file():
        raise RuntimeError(f"Release manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("release_id") != RELEASE_ID
        or manifest.get("report_version") != REPORT_VERSION
        or manifest.get("software_version") != SOFTWARE_VERSION
    ):
        raise RuntimeError("Release manifest identity changed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) != manifest.get("artifact_count"):
        raise RuntimeError("Release manifest artifact inventory is invalid")
    for relative, evidence in artifacts.items():
        blob = subprocess.run(
            ["git", "show", f"{RELEASE_ID}:{relative}"],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != evidence.get("sha256"):
            raise RuntimeError(f"Tagged release artifact hash differs: {relative}")
        if len(blob) != int(evidence.get("size_bytes", -1)):
            raise RuntimeError(f"Tagged release artifact size differs: {relative}")
    expected_checksums = checksum_text(repository, manifest_path)
    if (
        not checksums_path.is_file()
        or checksums_path.read_text(encoding="utf-8") != expected_checksums
    ):
        raise RuntimeError(f"Release checksums are stale: {checksums_path}")


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    output = repository / args.output
    checksums = repository / args.checksums

    if args.check:
        verify_frozen_release(repository, output, checksums)
        print(f"Release manifest verified: {output}")
        return

    expected = encoded_manifest(build_manifest(repository))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(expected)
    checksums.parent.mkdir(parents=True, exist_ok=True)
    checksums.write_text(checksum_text(repository, output), encoding="utf-8", newline="\n")
    print(f"Release manifest written: {output}")
    print(f"Release checksums written: {checksums}")


if __name__ == "__main__":
    main()
