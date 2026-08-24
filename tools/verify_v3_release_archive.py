"""Verify the exact Git archive intended for the v3 archival release."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PurePosixPath("results/human_activity_study_v3.0.0_manifest.json")
CHECKSUM_PATH = PurePosixPath("release/HUMAN_ACTIVITY_STUDY_V3.0.0_SHA256SUMS.txt")
FORBIDDEN_MEDIA = frozenset(
    {
        PurePosixPath("assets/champion_error_gallery.png"),
        PurePosixPath("assets/convnext_small_faithfulness_gallery.jpg"),
        PurePosixPath("assets/dinov2_small_faithfulness_gallery.jpg"),
        PurePosixPath("assets/probability_blend_faithfulness_gallery.jpg"),
    }
)
REQUIRED_ARCHIVE_FILES = frozenset(
    {
        PurePosixPath(".zenodo.json"),
        PurePosixPath("CITATION.cff"),
        PurePosixPath("THIRD_PARTY_NOTICES.md"),
        PurePosixPath("docs/SCIENTIFIC_VALIDATION_PLAN.md"),
        PurePosixPath("human_activity_classification.ipynb"),
        PurePosixPath("output/pdf/okutama_cptr_development_v3.0.0.pdf"),
        PurePosixPath("output/pdf/vcoco_v3_motion_identifiability_v3.0.0.pdf"),
        MANIFEST_PATH,
        CHECKSUM_PATH,
    }
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--treeish", default="HEAD")
    return parser.parse_args()


def build_archive(repository: Path, treeish: str, output: Path) -> None:
    subprocess.run(
        ["git", "archive", "--format=zip", f"--output={output}", treeish],
        cwd=repository,
        check=True,
        capture_output=True,
    )


def validate_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        files = {
            PurePosixPath(name): archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/")
        }

    missing = REQUIRED_ARCHIVE_FILES.difference(files)
    if missing:
        raise RuntimeError(
            "Release archive is missing required files: "
            + ", ".join(item.as_posix() for item in sorted(missing))
        )
    present_forbidden = FORBIDDEN_MEDIA.intersection(files)
    if present_forbidden:
        raise RuntimeError(
            "Release archive contains non-distributable media: "
            + ", ".join(item.as_posix() for item in sorted(present_forbidden))
        )

    manifest = json.loads(files[MANIFEST_PATH])
    artifacts = manifest.get("artifacts", {})
    if int(manifest.get("artifact_count", -1)) != len(artifacts):
        raise RuntimeError("Release manifest artifact count is invalid")
    expected_files = {PurePosixPath(name) for name in artifacts} | {
        MANIFEST_PATH,
        CHECKSUM_PATH,
    }
    if set(files) != expected_files:
        unexpected = sorted(set(files).difference(expected_files))
        absent = sorted(expected_files.difference(files))
        raise RuntimeError(
            "Release archive and manifest inventory differ; "
            f"unexpected={[item.as_posix() for item in unexpected]}, "
            f"missing={[item.as_posix() for item in absent]}"
        )
    for relative, record in artifacts.items():
        payload = files[PurePosixPath(relative)]
        if sha256_bytes(payload) != record.get("sha256") or len(payload) != int(
            record.get("size_bytes", -1)
        ):
            raise RuntimeError(f"Archived artifact differs from manifest: {relative}")

    checksum_lines = files[CHECKSUM_PATH].decode("utf-8").splitlines()
    if len(checksum_lines) != 3:
        raise RuntimeError("Release checksum file must cover two reports and the manifest")
    for line in checksum_lines:
        digest, basename = line.split("  ", 1)
        matches = [payload for relative, payload in files.items() if relative.name == basename]
        if len(matches) != 1 or sha256_bytes(matches[0]) != digest:
            raise RuntimeError(f"Release checksum does not resolve inside the archive: {basename}")


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    with tempfile.TemporaryDirectory(prefix="hac-v3-archive-") as temporary:
        archive_path = Path(temporary) / "human-activity-study-v3.0.0.zip"
        build_archive(repository, args.treeish, archive_path)
        validate_archive(archive_path)
    print(f"Release archive verified from {args.treeish}")


if __name__ == "__main__":
    main()
