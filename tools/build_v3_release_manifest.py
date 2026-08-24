"""Build or verify the Human Activity Classification Study v3.0.0 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path, PurePosixPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ID = "human-activity-study-v3.0.0"
REPORT_VERSION = "3.0.0"
SOFTWARE_VERSION = "3.0.0"
RELEASE_DATE = "2026-08-24"
DEFAULT_OUTPUT = PurePosixPath("results/human_activity_study_v3.0.0_manifest.json")
DEFAULT_CHECKSUMS = PurePosixPath("release/HUMAN_ACTIVITY_STUDY_V3.0.0_SHA256SUMS.txt")

REQUIRED_ARTIFACTS = frozenset(
    {
        PurePosixPath(".zenodo.json"),
        PurePosixPath("CHANGELOG.md"),
        PurePosixPath("CITATION.cff"),
        PurePosixPath("README.md"),
        PurePosixPath("docs/OKUTAMA_CPTR_DEVELOPMENT.md"),
        PurePosixPath("docs/VCOCO_V3_MOTION_IDENTIFIABILITY.md"),
        PurePosixPath("docs/releases/HUMAN_ACTIVITY_STUDY_V3.0.0.md"),
        PurePosixPath("human_activity_classification.ipynb"),
        PurePosixPath("output/pdf/okutama_cptr_development_v3.0.0.pdf"),
        PurePosixPath("output/pdf/vcoco_v3_motion_identifiability_v3.0.0.pdf"),
        PurePosixPath("pyproject.toml"),
        PurePosixPath("requirements-v3-lock.txt"),
        PurePosixPath("results/okutama_cptr/evidence_manifest.json"),
        PurePosixPath("results/vcoco_v3/evidence_manifest.json"),
        PurePosixPath("tools/build_v3_release_manifest.py"),
    }
)

TEXT_ARTIFACT_SUFFIXES = frozenset(
    {
        ".cff",
        ".css",
        ".csv",
        ".html",
        ".ipynb",
        ".js",
        ".json",
        ".md",
        ".py",
        ".svg",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
TEXT_ARTIFACT_FILENAMES = frozenset(
    {
        ".env.example",
        ".gitattributes",
        ".gitignore",
        "LICENSE",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_bytes(path: Path) -> bytes:
    """Return bytes as Git will store them under the repository attributes."""

    payload = path.read_bytes()
    if path.suffix.lower() in TEXT_ARTIFACT_SUFFIXES or path.name in TEXT_ARTIFACT_FILENAMES:
        payload = payload.replace(b"\r\n", b"\n")
        if b"\r" in payload:
            raise RuntimeError(f"Text artifact contains a lone carriage return: {path}")
    return payload


def artifact_paths(repository: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    relative_paths = {
        PurePosixPath(value.decode("utf-8")) for value in completed.stdout.split(b"\0") if value
    }
    relative_paths.difference_update({DEFAULT_OUTPUT, DEFAULT_CHECKSUMS})
    missing_required = sorted(REQUIRED_ARTIFACTS.difference(relative_paths))
    if missing_required:
        formatted = ", ".join(path.as_posix() for path in missing_required)
        raise RuntimeError(f"Required release artifacts are missing or ignored: {formatted}")
    paths = {repository / relative for relative in relative_paths}
    missing = sorted(path for path in paths if not path.is_file())
    if missing:
        formatted = ", ".join(path.relative_to(repository).as_posix() for path in missing)
        raise RuntimeError(f"Release candidate files are missing: {formatted}")
    if not paths:
        raise RuntimeError("The release candidate inventory is empty")
    return sorted(paths, key=lambda path: path.relative_to(repository).as_posix())


def validate_versions(repository: Path) -> None:
    with (repository / "pyproject.toml").open("rb") as handle:
        package_version = tomllib.load(handle)["project"]["version"]
    if package_version != SOFTWARE_VERSION:
        raise RuntimeError(f"Expected software version {SOFTWARE_VERSION}, found {package_version}")

    for relative in (
        "docs/VCOCO_V3_MOTION_IDENTIFIABILITY.md",
        "docs/OKUTAMA_CPTR_DEVELOPMENT.md",
    ):
        report = (repository / relative).read_text(encoding="utf-8")
        if f"version: {REPORT_VERSION}" not in report:
            raise RuntimeError(f"Report version is missing or stale: {relative}")

    citation = (repository / "CITATION.cff").read_text(encoding="utf-8")
    if (
        f"version: {SOFTWARE_VERSION}" not in citation
        or f"  version: {REPORT_VERSION}" not in citation
        or f"date-released: {RELEASE_DATE}" not in citation
    ):
        raise RuntimeError("Citation metadata does not match the v3 release")

    zenodo = json.loads((repository / ".zenodo.json").read_text(encoding="utf-8"))
    if zenodo.get("version") != REPORT_VERSION or zenodo.get("publication_date") != RELEASE_DATE:
        raise RuntimeError("Zenodo metadata does not match the v3 release")


def build_manifest(repository: Path) -> dict:
    repository = repository.resolve()
    validate_versions(repository)
    artifacts = {}
    for path in artifact_paths(repository):
        relative = path.relative_to(repository).as_posix()
        payload = artifact_bytes(path)
        artifacts[relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    return {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "report_version": REPORT_VERSION,
        "software_version": SOFTWARE_VERSION,
        "release_date": RELEASE_DATE,
        "artifact_scope": (
            "Complete nonignored repository release candidate, including the v3 "
            "motion-identifiability and CPTR reports, portable evidence, figures, "
            "protocols, implementation, tests, and reproducibility entry points"
        ),
        "text_digest_policy": (
            "CRLF is canonicalized to LF for source files and repository metadata "
            "declared as text by .gitattributes"
        ),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def encoded_manifest(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def checksum_text(repository: Path, manifest_path: Path) -> str:
    targets = (
        repository / "output/pdf/vcoco_v3_motion_identifiability_v3.0.0.pdf",
        repository / "output/pdf/okutama_cptr_development_v3.0.0.pdf",
        manifest_path,
    )
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
        if not checksums.is_file() or checksums.read_text(encoding="utf-8") != checksum_text(
            repository, output
        ):
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
