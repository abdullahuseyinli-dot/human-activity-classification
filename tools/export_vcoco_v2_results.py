"""Export compact, path-sanitized V-COCO v2 evidence for the repository."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import pandas as pd

from hac.polar import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--final-fit-summary", type=Path, required=True)
    parser.add_argument("--feature-summary", type=Path, required=True)
    parser.add_argument("--test-evaluation", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/vcoco_v2"))
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def repository_relative(value: str, repository: Path) -> str:
    path = Path(value)
    try:
        return path.resolve().relative_to(repository).as_posix()
    except ValueError:
        return path.name


def sanitized_selection(lock: dict, repository: Path, source_hash: str) -> dict:
    result = copy.deepcopy(lock)
    result["source_lock_sha256"] = source_hash
    result["final_fit"]["path"] = repository_relative(result["final_fit"]["path"], repository)
    for record in result["historical_baseline"]["checkpoints"]:
        record["path"] = repository_relative(record["path"], repository)
    result["evidence_sha256"] = {
        repository_relative(path, repository): digest
        for path, digest in result["evidence_sha256"].items()
    }
    result["decision_notes"] = {
        "augmix_lpft": "Statistically tied with the selected stack; paired interval includes zero.",
        "factorized_head": "Supported over the flat matched-input control.",
        "label_definition": "Deterministic mapping of V-COCO action tags.",
    }
    return result


def sanitized_protocol(lock: dict, source_hash: str) -> dict:
    result = copy.deepcopy(lock)
    result["source_lock_sha256"] = source_hash
    ontology = result.get("ontology", {})
    ontology.pop("manual_annotation_status", None)
    ontology["label_definition"] = "Deterministic mapping of V-COCO action tags."
    ontology["current_labels"] = "factorized V-COCO source-action labels"
    return result


def sanitized_test_summary(summary: dict, repository: Path, source_hash: str) -> dict:
    result = copy.deepcopy(summary)
    result["source_summary_sha256"] = source_hash
    result["label_definition"] = "deterministic mapping of V-COCO action tags"
    result.pop("ontology_scope", None)
    for record in result["historical_baseline_checkpoints"]:
        record["path"] = repository_relative(record["path"], repository)
    return result


def candidate_row(
    method: str,
    family: str,
    training_scope: str,
    metrics: dict,
    *,
    view: str,
    role: str,
) -> dict:
    return {
        "method": method,
        "family": family,
        "training_scope": training_scope,
        "view": view,
        "role": role,
        "accuracy": metrics.get("accuracy", metrics.get("val_accuracy")),
        "macro_f1": metrics.get("macro_f1", metrics.get("val_macro_f1")),
        "balanced_accuracy": metrics.get(
            "balanced_accuracy", metrics.get("val_balanced_accuracy")
        ),
        "log_loss": metrics.get("log_loss", metrics.get("val_log_loss")),
        "brier_score": metrics.get("brier_score", metrics.get("val_brier_score")),
        "ece": metrics.get("ece", metrics.get("val_ece")),
    }


def development_candidates(runs_root: Path, protocol_hash: str) -> pd.DataFrame:
    factor_dir = runs_root / "factorized_fusion" / "dinov2_base"
    factor = pd.read_csv(factor_dir / "factorized_fusion_summary.csv").set_index("method")
    sources = {
        "dino": runs_root / "feature_screen" / "summary.json",
        "siglip": runs_root / "siglip2_feature_screen" / "summary.json",
        "convnext": runs_root / "convnext_feature_screen" / "summary.json",
        "pose": runs_root / "pose_oracle_rbf" / "summary.json",
        "source": runs_root / "source_model_views" / "dinov2_base_top4" / "summary.json",
        "mild": runs_root
        / "neural_evaluation"
        / "dinov2_base_lpft_top1_tight_mild_seed42"
        / "summary.json",
        "augmix": runs_root
        / "neural_evaluation"
        / "dinov2_base_lpft_top1_tight_augmix_seed42"
        / "summary.json",
    }
    summaries = {name: read_json(path) for name, path in sources.items()}
    if any(summary.get("protocol_lock_sha256") != protocol_hash for summary in summaries.values()):
        raise RuntimeError("Development summaries do not share the locked protocol")
    if any(summary.get("test_predictions_run") for summary in summaries.values()):
        raise RuntimeError("A development summary reports test prediction access")

    rows = [
        candidate_row(
            "historical_source_dino",
            "DINOv2-B",
            "POLAR source only",
            factor.loc["locked_v1_dinov2"].to_dict(),
            view="person_context_25, legacy center crop",
            role="historical baseline",
        ),
        candidate_row(
            "source_dino_tight_pad",
            "DINOv2-B",
            "POLAR source weights; V-COCO validation selected preprocessing",
            summaries["source"]["best_validation_result"],
            view="person_tight, aspect-preserving pad",
            role="source-model view control",
        ),
        candidate_row(
            "pose_geometry_rbf_oracle",
            "COCO ground-truth pose",
            "V-COCO train",
            summaries["pose"]["best_validation_result"],
            view="normalized person keypoints plus geometry",
            role="diagnostic oracle",
        ),
        candidate_row(
            "convnext_frozen_probe",
            "ConvNeXt-S",
            "V-COCO train",
            summaries["convnext"]["best_validation_result"],
            view="person_tight, aspect-preserving pad",
            role="controlled representation screen",
        ),
        candidate_row(
            "siglip2_frozen_probe",
            "SigLIP2-B",
            "V-COCO train",
            summaries["siglip"]["best_validation_result"],
            view="person_context_25, aspect-preserving pad",
            role="controlled representation screen",
        ),
        candidate_row(
            "dino_frozen_single_view",
            "DINOv2-B",
            "V-COCO train",
            summaries["dino"]["best_validation_result"],
            view="person_tight, aspect-preserving pad",
            role="best single-view control",
        ),
        candidate_row(
            "dino_flat_same_features",
            "DINOv2-B",
            "V-COCO train",
            factor.loc["flat_tight_context_geometry"].to_dict(),
            view="tight plus context plus geometry",
            role="factorization control",
        ),
        candidate_row(
            "dino_factorized_head",
            "DINOv2-B",
            "V-COCO train",
            factor.loc["factorized_tight_context_geometry"].to_dict(),
            view="tight plus context plus geometry",
            role="factorized classifier",
        ),
        candidate_row(
            "dino_lpft_mild",
            "DINOv2-B",
            "V-COCO train",
            summaries["mild"]["metrics"],
            view="person_tight, aspect-preserving pad",
            role="linear-probe then top-block fine-tune",
        ),
        candidate_row(
            "dino_lpft_augmix",
            "DINOv2-B",
            "V-COCO train",
            summaries["augmix"]["metrics"],
            view="person_tight, aspect-preserving pad",
            role="linear-probe then top-block fine-tune",
        ),
        candidate_row(
            "dino_scale_conditioned_stack",
            "DINOv2-B",
            "V-COCO train",
            factor.loc["scale_conditioned_stacking"].to_dict(),
            view="tight and context probabilities plus geometry",
            role="selected development champion",
        ),
    ]
    return pd.DataFrame(rows).sort_values(
        ["macro_f1", "log_loss", "method"], ascending=[False, True, True], ignore_index=True
    )


def copy_evidence(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() in {".csv", ".json", ".md"}:
        text = source.read_text(encoding="utf-8")
        destination.write_text(
            "\n".join(text.splitlines()) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    else:
        shutil.copyfile(source, destination)


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    protocol_path = args.protocol_lock.resolve()
    selection_path = args.selection_lock.resolve()
    final_fit_path = args.final_fit_summary.resolve()
    feature_path = args.feature_summary.resolve()
    evaluation = args.test_evaluation.resolve()
    runs_root = args.runs_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    protocol = read_json(protocol_path)
    selection = read_json(selection_path)
    final_fit = read_json(final_fit_path)
    features = read_json(feature_path)
    test = read_json(evaluation / "summary.json")
    gate = read_json(evaluation / "test_access_gate.json")
    protocol_hash = sha256_file(protocol_path)
    selection_hash = sha256_file(selection_path)
    if protocol.get("status") != "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("Invalid V-COCO v2 protocol lock")
    if selection.get("status") != "VCOCO_V2_FINAL_SELECTION_LOCKED_PRE_TEST":
        raise RuntimeError("Invalid V-COCO v2 final selection lock")
    if final_fit.get("status") != "VCOCO_V2_FINAL_DEVELOPMENT_STACK_FIT_COMPLETE":
        raise RuntimeError("Invalid final development refit")
    if features.get("status") != "VCOCO_V2_LOCKED_TEST_FEATURES_COMPLETE":
        raise RuntimeError("Invalid locked test feature cache")
    if test.get("status") != "VCOCO_V2_OFFICIAL_TEST_EVALUATION_COMPLETE":
        raise RuntimeError("Official test evaluation is incomplete")
    if gate.get("official_test_label_open_count") != 1 or test.get("test_used_for_selection"):
        raise RuntimeError("Official test access contract was not satisfied")
    if features.get("test_label_columns_read") != 0:
        raise RuntimeError("Feature extraction read official test labels")
    if {
        selection.get("protocol_lock_sha256"),
        final_fit.get("protocol_lock_sha256"),
        features.get("protocol_lock_sha256"),
        test.get("protocol_lock_sha256"),
    } != {protocol_hash}:
        raise RuntimeError("Protocol hashes do not align")
    if {features.get("selection_lock_sha256"), test.get("selection_lock_sha256")} != {
        selection_hash
    }:
        raise RuntimeError("Selection hashes do not align")
    if sha256_file(evaluation / "test_predictions.npz") != test["predictions_sha256"]:
        raise RuntimeError("Official test prediction artifact drift")

    write_json(output / "protocol_lock.json", sanitized_protocol(protocol, protocol_hash))
    write_json(
        output / "final_selection_lock.json",
        sanitized_selection(selection, repository, selection_hash),
    )
    write_json(
        output / "official_test_summary.json",
        sanitized_test_summary(test, repository, sha256_file(evaluation / "summary.json")),
    )
    copy_evidence(evaluation / "test_access_gate.json", output / "test_access_gate.json")
    aggregate_files = {
        evaluation / "test_metrics.csv": output / "official_test_metrics.csv",
        evaluation / "test_per_class.csv": output / "official_test_per_class.csv",
        evaluation / "test_confusions.json": output / "official_test_confusions.json",
        evaluation / "test_selective_metrics.json": output / "official_test_selective_metrics.json",
        evaluation / "test_strata.csv": output / "official_test_strata.csv",
        evaluation / "test_paired_uncertainty.json": output / "official_test_uncertainty.json",
        runs_root / "fewshot" / "dinov2_base_tight" / "fewshot_summary.csv": output
        / "fewshot_curve.csv",
        runs_root / "factorized_fusion" / "dinov2_base" / "factorized_fusion_summary.csv": output
        / "factorized_fusion.csv",
        runs_root / "factorized_fusion" / "dinov2_base" / "factorized_fusion_per_class.csv": output
        / "factorized_fusion_per_class.csv",
        runs_root / "factorized_fusion" / "dinov2_base" / "factorized_fusion_uncertainty.json": output
        / "factorized_fusion_uncertainty.json",
        runs_root / "mechanism_analysis" / "mechanism_correlations.csv": output
        / "mechanism_correlations.csv",
        runs_root / "mechanism_analysis" / "mechanism_error_transitions.csv": output
        / "mechanism_error_transitions.csv",
        runs_root / "mechanism_analysis" / "mechanism_strata.csv": output
        / "mechanism_strata.csv",
    }
    for source, destination in aggregate_files.items():
        copy_evidence(source, destination)
    development_candidates(runs_root, protocol_hash).to_csv(
        output / "development_candidates.csv", index=False, lineterminator="\n"
    )

    artifacts = {}
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "evidence_manifest.json":
            artifacts[path.name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    manifest = {
        "status": "VCOCO_V2_PORTABLE_EVIDENCE_COMPLETE",
        "protocol_lock_sha256": protocol_hash,
        "selection_lock_sha256": selection_hash,
        "repository_revision_at_selection_lock": selection["repository_revision_at_lock"],
        "official_test_label_open_count": gate["official_test_label_open_count"],
        "official_test_people": protocol["split_summary"]["test"]["people"],
        "official_test_images": protocol["split_summary"]["test"]["images"],
        "test_used_for_selection": False,
        "exporter_sha256": sha256_file(Path(__file__).resolve()),
        "artifacts": artifacts,
        "local_artifacts_retained": [
            "dense probability arrays",
            "feature tensors",
            "model checkpoints",
            "image manifests with machine-local paths",
        ],
    }
    write_json(output / "evidence_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
