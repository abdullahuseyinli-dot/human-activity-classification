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
POLAR_REQUIRED = {
    ".zenodo.json",
    "CHANGELOG.md",
    "CITATION.cff",
    "docs/LEGACY_COCO_STUDY.md",
    "docs/POLAR_SCALE_STUDY_PROTOCOL.md",
    "docs/POLAR_PUBLIC_REPORT.md",
    "docs/PORTFOLIO_ARTICLE.md",
    "docs/RESULT_LINEAGE.md",
    "docs/releases/POLAR_STUDY_V1.0.0.md",
    "experiments/analyze_polar_exploratory.py",
    "experiments/evaluate_polar_faithfulness.py",
    "experiments/evaluate_polar_fault_robustness.py",
    "experiments/evaluate_polar_final.py",
    "experiments/polar_study_protocol.json",
    "output/pdf/README.md",
    "output/pdf/polar_public_report_v1.0.0.pdf",
    "release/POLAR_STUDY_V1.0.0_SHA256SUMS.txt",
    "results/polar_data_audit.json",
    "results/polar_final_evidence_manifest.json",
    "results/polar_final_fit_manifest.json",
    "results/polar_final_selection_lock.json",
    "results/polar_exploratory_summary.json",
    "results/polar_study_v1.0.0_manifest.json",
    "results/polar_test_access_gate.json",
    "results/polar_test_metrics.csv",
    "results/polar_test_summary.json",
    "results/polar_test_uncertainty.json",
    "results/polar_external_overlap_audit.json",
    "results/polar_external_summary.json",
    "results/polar_faithfulness_summary.json",
    "results/polar_fault_summary.json",
    "assets/polar_attribution_sanity.png",
    "assets/polar_confusion_matrix.png",
    "assets/polar_external_validation.png",
    "assets/polar_faithfulness.png",
    "assets/polar_fault_robustness.png",
    "assets/polar_scale_curve.png",
    "assets/polar_test_comparison.png",
    "tools/build_study_papers.py",
    "tools/build_study_release_manifest.py",
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


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_polar_release(repository: Path) -> None:
    results = repository / "results"
    pdf_path = repository / "output" / "pdf" / "polar_public_report_v1.0.0.pdf"
    with pdf_path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise RuntimeError("POLAR Study Report is not a PDF")
        handle.seek(max(0, pdf_path.stat().st_size - 1024))
        if b"%%EOF" not in handle.read():
            raise RuntimeError("POLAR Study Report PDF is incomplete")
    evidence = read_json(results / "polar_final_evidence_manifest.json")
    if (
        evidence.get("status") != "LOCKED_POLAR_PORTFOLIO_EVIDENCE"
        or evidence.get("test_used_for_selection") is not False
    ):
        raise RuntimeError("Invalid POLAR portable-evidence manifest")
    lock_hash = evidence.get("selection_lock_sha256")
    selection_path = results / "polar_final_selection_lock.json"
    if lock_hash not in legacy_release_hashes(selection_path):
        raise RuntimeError("POLAR selection-lock fingerprint mismatch")

    exported = evidence.get("exported_files", {})
    if len(exported) < 20:
        raise RuntimeError("POLAR portable-evidence export is incomplete")
    for name, expected_hash in exported.items():
        if Path(name).name != name:
            raise RuntimeError(f"Invalid POLAR evidence filename: {name}")
        path = results / name
        if not path.is_file() or expected_hash not in legacy_release_hashes(path):
            raise RuntimeError(f"POLAR evidence fingerprint mismatch: {name}")

    summaries = {
        "polar_test_summary.json": "LOCKED_FINAL_TEST_COMPLETE",
        "polar_external_summary.json": "LOCKED_EXTERNAL_EVALUATION_COMPLETE",
        "polar_faithfulness_summary.json": "LOCKED_POLAR_FAITHFULNESS_COMPLETE",
        "polar_fault_summary.json": "LOCKED_POLAR_FAULT_ROBUSTNESS_COMPLETE",
    }
    loaded = {}
    for name, status in summaries.items():
        payload = read_json(results / name)
        loaded[name] = payload
        if payload.get("status") != status:
            raise RuntimeError(f"Incomplete POLAR result: {name}")
        if payload.get("selection_lock_sha256") != lock_hash:
            raise RuntimeError(f"POLAR result has a different selection lock: {name}")
        if payload.get("test_used_for_selection") not in {None, False}:
            raise RuntimeError(f"POLAR test-selected result cannot be released: {name}")

    audit = read_json(results / "polar_data_audit.json")
    if (
        audit.get("status") != "PRE_SUPERVISED_FIT_DATA_LOCK"
        or audit.get("clean_rows") != 16614
        or audit.get("quarantine_images") != 125
        or audit.get("test_used_for_model_selection") is not False
    ):
        raise RuntimeError("Invalid POLAR pre-fit data audit")

    fits = read_json(results / "polar_final_fit_manifest.json")
    if (
        fits.get("status") != "LOCKED_POLAR_FINAL_FITS_VERIFIED"
        or fits.get("development_rows") != 13285
        or fits.get("test_rows_read") != 0
        or fits.get("test_used_for_selection") is not False
    ):
        raise RuntimeError("Invalid POLAR final-fit gate")
    expected_neural = {"convnext_small_full", "dinov2_base_top4", "dinov2_small_moderate"}
    expected_probes = {
        "dinov2_base_multilayer_logistic",
        "dinov2_base_multilayer_logistic_3class",
        "dinov2_base_multilayer_rbf",
    }
    if set(fits.get("neural", {})) != expected_neural or any(
        set(item.get("seeds", {})) != {"42", "52", "62"}
        for item in fits.get("neural", {}).values()
    ):
        raise RuntimeError("POLAR final neural fits are incomplete")
    if set(fits.get("probes", {})) != expected_probes:
        raise RuntimeError("POLAR final probes are incomplete")

    gate = read_json(results / "polar_test_access_gate.json")
    if (
        gate.get("status") != "POLAR_TEST_GATE_OPEN"
        or gate.get("official_test_manifest_open_count") != 1
        or gate.get("test_rows_read") != 3329
        or gate.get("selection_lock_sha256") != lock_hash
    ):
        raise RuntimeError("Invalid POLAR one-time test-access gate")

    test = loaded["polar_test_summary.json"]
    if (
        test.get("primary_candidate") != "locked_ensemble"
        or test.get("primary_candidate_locked_pre_test") is not True
        or test.get("test_rows_read") != 3329
        or test.get("official_test_manifest_open_count") != 1
        or test.get("test_used_for_selection") is not False
    ):
        raise RuntimeError("Invalid locked POLAR test summary")
    metrics = pd.read_csv(results / "polar_test_metrics.csv")
    expected_candidates = {
        "locked_ensemble",
        "convnext_small_full",
        "dinov2_small_moderate",
        "dinov2_base_top4",
        "dinov2_base_multilayer_logistic",
        "dinov2_base_multilayer_rbf",
    }
    if (
        len(metrics) != len(expected_candidates)
        or set(metrics["candidate"]) != expected_candidates
        or metrics["candidate"].duplicated().any()
    ):
        raise RuntimeError("Invalid POLAR held-out candidate table")
    primary_row = metrics.loc[metrics["candidate"].eq("locked_ensemble")].iloc[0]
    if abs(float(primary_row["macro_f1"]) - float(test["primary_metrics"]["macro_f1"])) > 1e-12:
        raise RuntimeError("POLAR primary metric disagrees with its summary")
    uncertainty = read_json(results / "polar_test_uncertainty.json")
    if uncertainty.get("locked_ensemble", {}).get("resamples") != 10000:
        raise RuntimeError("POLAR uncertainty does not use the locked resample count")
    paired = uncertainty.get("locked_ensemble_paired_deltas", {})
    if set(paired) != expected_candidates - {"locked_ensemble"} or any(
        float(item["ci_95_low"]) <= 0.0 for item in paired.values()
    ):
        raise RuntimeError("POLAR paired component comparisons are incomplete")

    overlap = read_json(results / "polar_external_overlap_audit.json")
    if (
        overlap.get("status") != "POLAR_VCOCO_CROSS_DATASET_OVERLAP_AUDITED"
        or overlap.get("exact_overlap_pairs") != 0
        or overlap.get("perceptual_candidates") != 0
        or overlap.get("confirmed_source_related_pairs") != 0
        or overlap.get("model_predictions_read") != 0
        or overlap.get("test_used_for_selection") is not False
    ):
        raise RuntimeError("Invalid POLAR/V-COCO overlap audit")
    external = loaded["polar_external_summary.json"]
    if (
        external.get("image_level_rows") != 3761
        or external.get("person_rows") != 6640
        or external.get("primary_candidate") != "locked_ensemble_collapsed"
        or external.get("primary_candidate_locked_pre_test") is not True
        or external.get("test_used_for_selection") is not False
    ):
        raise RuntimeError("Invalid locked V-COCO external evaluation")

    faithfulness = loaded["polar_faithfulness_summary.json"]
    if (
        faithfulness.get("cohort_rows") != 256
        or faithfulness.get("selection_role") != "none"
        or faithfulness.get("parameter_randomization_rows_per_family") != 16
        or faithfulness.get("test_used_for_attribution_selection") is not False
        or faithfulness.get("test_used_for_model_selection") is not False
        or float(faithfulness.get("max_probability_parity_absolute_error", 1.0)) > 0.002
    ):
        raise RuntimeError("Invalid locked POLAR faithfulness summary")
    faith_rows = pd.read_csv(
        results / "polar_faithfulness_per_image.csv", dtype={"image_id": str}
    )
    faith_families = {"convnext_small_full", "dinov2_base_top4"}
    if set(faith_rows["family"]) != faith_families:
        raise RuntimeError("Unexpected POLAR faithfulness model family")
    cohort_ids = None
    for family, rows in faith_rows.groupby("family"):
        ids = set(rows["image_id"])
        if len(rows) != 256 or len(ids) != 256:
            raise RuntimeError(f"Invalid POLAR faithfulness rows for {family}")
        cohort_ids = ids if cohort_ids is None else cohort_ids
        if ids != cohort_ids:
            raise RuntimeError("POLAR faithfulness families use different cohorts")

    fault = loaded["polar_fault_summary.json"]
    if (
        fault.get("cohort_rows") != 256
        or fault.get("cohort_sha256") != faithfulness.get("cohort_sha256")
        or fault.get("reported_separately_from_faithfulness") is not True
        or fault.get("selection_role") != "none"
        or fault.get("test_used_for_selection") is not False
    ):
        raise RuntimeError("Invalid locked POLAR fault audit")
    fault_metrics = pd.read_csv(results / "polar_fault_robustness_metrics.csv")
    aggregate_faults = fault_metrics[fault_metrics["fault_seed"].astype(str).eq("aggregate")]
    if set(aggregate_faults["family"]) != faith_families or set(
        aggregate_faults["condition"]
    ) != {"uint8_input_bit_flip_rate", "symmetric_int8_head_weight_bit_flips"}:
        raise RuntimeError("POLAR aggregate fault evidence is incomplete")

def validate_study_report_release(repository: Path) -> None:
    report_source = repository / "docs/POLAR_PUBLIC_REPORT.md"
    report_text = report_source.read_text(encoding="utf-8")
    if (
        "version: 1.0.0" not in report_text
        or "status: Independent technical report" not in report_text
    ):
        raise RuntimeError("POLAR Study Report metadata is not final v1.0.0")
    for stale_marker in ("0.9.0-review", "Pre-release review copy", "Review status:"):
        if stale_marker in report_text:
            raise RuntimeError(f"Stale report-review marker remains: {stale_marker}")
    for relative in ("docs/POLAR_PUBLIC_REPORT.md", "docs/releases/POLAR_STUDY_V1.0.0.md"):
        text = (repository / relative).read_text(encoding="utf-8")
        if any(ord(character) >= 128 for character in text):
            raise RuntimeError(f"Release narrative must remain ASCII-safe: {relative}")

    pdf_path = repository / "output/pdf/polar_public_report_v1.0.0.pdf"
    with pdf_path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise RuntimeError("POLAR Study Report is not a PDF")
        handle.seek(max(0, pdf_path.stat().st_size - 1024))
        if b"%%EOF" not in handle.read():
            raise RuntimeError("POLAR Study Report PDF is incomplete")

    zenodo = read_json(repository / ".zenodo.json")
    if (
        zenodo.get("title")
        != "Source-Overlap-Controlled Transfer Learning for Still-Image Posture Recognition"
        or zenodo.get("version") != "1.0.0"
        or zenodo.get("license") != "MIT"
        or zenodo.get("upload_type") != "software"
        or zenodo.get("creators") != [{"name": "Huseyinli, Abdulla"}]
    ):
        raise RuntimeError("Zenodo metadata does not match the v1.0.0 study release")

    exploratory = read_json(repository / "results/polar_exploratory_summary.json")
    if (
        exploratory.get("analysis_seed") != 20260823
        or exploratory.get("bootstrap_resamples") != 10000
        or not str(exploratory.get("analysis_role", "")).startswith("Hypothesis-generating only")
    ):
        raise RuntimeError("Exploratory evidence role or deterministic settings changed")

    sys.path.insert(0, str(repository / "tools"))
    from build_study_release_manifest import build_manifest, checksum_text, encoded_manifest

    manifest_path = repository / "results/polar_study_v1.0.0_manifest.json"
    if manifest_path.read_bytes() != encoded_manifest(build_manifest(repository)):
        raise RuntimeError("POLAR Study Report release manifest is stale")
    checksums_path = repository / "release/POLAR_STUDY_V1.0.0_SHA256SUMS.txt"
    if checksums_path.read_text(encoding="utf-8") != checksum_text(repository, manifest_path):
        raise RuntimeError("POLAR Study Report release checksums are stale")


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    missing = sorted(
        name for name in REQUIRED | POLAR_REQUIRED if not (repository / name).is_file()
    )
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
            invalid_control = next(
                (
                    character
                    for character in text
                    if ord(character) < 32 and character not in "\n\r\t"
                ),
                None,
            )
            if invalid_control is not None:
                raise RuntimeError(
                    f"Control character U+{ord(invalid_control):04X} leaked into {relative}"
                )
            if windows_user_path.search(text) or local_file_scheme in text.lower():
                raise RuntimeError(f"Local absolute path leaked into {relative}")
            if "\ufffd" in text:
                raise RuntimeError(f"Unicode replacement character leaked into {relative}")
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
    validate_polar_release(repository)
    validate_study_report_release(repository)
    print("Repository validation passed")


if __name__ == "__main__":
    main()
