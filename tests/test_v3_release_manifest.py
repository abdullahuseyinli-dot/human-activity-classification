from pathlib import Path

import pytest

from tools.build_v3_release_manifest import artifact_bytes


@pytest.mark.parametrize(
    "filename",
    [
        ".env.example",
        ".gitattributes",
        ".gitignore",
        "LICENSE",
        "module.py",
    ],
)
def test_text_artifacts_use_git_normalized_line_endings(tmp_path: Path, filename: str) -> None:
    artifact = tmp_path / filename
    artifact.write_bytes(b"first\r\nsecond\r\n")

    assert artifact_bytes(artifact) == b"first\nsecond\n"


def test_binary_artifacts_retain_original_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "figure.png"
    payload = b"binary\r\npayload\x00"
    artifact.write_bytes(payload)

    assert artifact_bytes(artifact) == payload
