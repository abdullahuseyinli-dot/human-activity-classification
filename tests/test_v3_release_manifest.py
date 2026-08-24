from pathlib import Path

import pytest

from tools.build_v3_release_manifest import (
    EXCLUDED_THIRD_PARTY_MEDIA,
    artifact_bytes,
    checksum_text,
)


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


def test_non_distributed_media_contract_matches_the_release_tree() -> None:
    repository = Path(__file__).resolve().parents[1]
    expected = {
        "assets/champion_error_gallery.png",
        "assets/convnext_small_faithfulness_gallery.jpg",
        "assets/dinov2_small_faithfulness_gallery.jpg",
        "assets/probability_blend_faithfulness_gallery.jpg",
    }

    assert {path.as_posix() for path in EXCLUDED_THIRD_PARTY_MEDIA} == expected
    assert all(len(digest) == 64 for digest in EXCLUDED_THIRD_PARTY_MEDIA.values())
    assert all(not (repository / path).exists() for path in EXCLUDED_THIRD_PARTY_MEDIA)


def test_release_checksums_use_downloadable_asset_names(tmp_path: Path) -> None:
    (tmp_path / "output" / "pdf").mkdir(parents=True)
    (tmp_path / "output" / "pdf" / "vcoco_v3_motion_identifiability_v3.0.0.pdf").write_bytes(
        b"temporal"
    )
    (tmp_path / "output" / "pdf" / "okutama_cptr_development_v3.0.0.pdf").write_bytes(b"cptr")
    manifest = tmp_path / "human_activity_study_v3.0.0_manifest.json"
    manifest.write_bytes(b"manifest")

    lines = checksum_text(tmp_path, manifest).splitlines()

    assert [line.split("  ", 1)[1] for line in lines] == [
        "vcoco_v3_motion_identifiability_v3.0.0.pdf",
        "okutama_cptr_development_v3.0.0.pdf",
        "human_activity_study_v3.0.0_manifest.json",
    ]


def test_scientific_validation_plan_binds_the_remaining_evidence() -> None:
    repository = Path(__file__).resolve().parents[1]
    plan = (repository / "docs" / "SCIENTIFIC_VALIDATION_PLAN.md").read_text(encoding="utf-8")

    for marker in (
        "## Current claim boundary",
        "## Limitation and evidence-gate matrix",
        "## 1. Annotation reliability",
        "## 3. Independent replication on POLIMI-ITW-S",
        "## 4. Grouped inference and prospective precision",
        "## 5. Matched model comparison",
        "## 6. CPTR occlusion study",
        "## 7. Runtime, energy, and memory",
        "## 8. Reproducibility check",
        "## 9. Multiplicity and analysis discipline",
        "## Publication decision rule",
        "The sealed temporal result has five confirmation scenarios",
        "These are evidence-completeness gates, not success gates.",
    ):
        assert marker in plan
