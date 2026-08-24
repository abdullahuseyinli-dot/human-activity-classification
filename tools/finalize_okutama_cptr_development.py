"""Aggregate the locked CPTR development and grouped cross-fit evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hac.cptr_training import classification_summary
from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.vcoco_v3_models import paired_cluster_bootstrap

CLASS_NAMES = ("sitting", "standing", "walking_running")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiments/okutama_cptr_protocol.json")
    )
    parser.add_argument(
        "--protocol-lock", type=Path, default=Path(".runs/cptr/protocol_lock.json")
    )
    parser.add_argument(
        "--plan", type=Path, default=Path("experiments/okutama_cptr_crossfit_plan.json")
    )
    parser.add_argument(
        "--plan-lock", type=Path, default=Path(".runs/cptr/crossfit_plan_lock.json")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/development_manifest.csv"),
    )
    parser.add_argument(
        "--crossfit-root",
        type=Path,
        default=Path(".runs/cptr/crossfit/centre_short_parts"),
    )
    parser.add_argument(
        "--promotion-root",
        type=Path,
        default=Path(".runs/cptr/promotion/centre_short_parts"),
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path(".runs/cptr/baseline/summary.json"),
    )
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=Path(".runs/cptr/baseline/baseline_and_diagnostic_predictions.npz"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".runs/cptr/development_final")
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260919)
    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def full_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    normalized = np.asarray(probabilities, dtype=float)
    normalized = normalized / normalized.sum(axis=1, keepdims=True).clip(min=1e-12)
    values = classification_metrics(labels, normalized)
    values.update(classification_summary(labels, normalized))
    return values


def subgroup_metrics(
    labels: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    transitions: np.ndarray,
    occlusions: np.ndarray,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    masks = {
        "transition": transitions.astype(bool),
        "non_transition": ~transitions.astype(bool),
        "window_occluded": occlusions.astype(bool),
        "window_clear": ~occlusions.astype(bool),
    }
    for subgroup, mask in masks.items():
        if int(mask.sum()) < 3:
            continue
        candidate_metrics = classification_summary(labels[mask], candidate[mask])
        baseline_metrics = classification_summary(labels[mask], baseline[mask])
        rows.append(
            {
                "subgroup": subgroup,
                "samples": int(mask.sum()),
                **{f"candidate_{key}": value for key, value in candidate_metrics.items()},
                **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
                "macro_f1_delta": candidate_metrics["macro_f1"]
                - baseline_metrics["macro_f1"],
            }
        )
    return rows


def exact_group_swap_test(
    labels: np.ndarray,
    candidate_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | int | str]:
    """Exact paired randomisation test by swapping whole recording predictions."""

    labels = np.asarray(labels, dtype=int)
    candidate = np.asarray(candidate_probabilities).argmax(axis=1)
    baseline = np.asarray(baseline_probabilities).argmax(axis=1)
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    if len(unique) > 20:
        raise ValueError("Exact group swapping is limited to at most 20 groups")
    candidate_confusion = np.zeros((len(unique), 3, 3), dtype=np.int64)
    baseline_confusion = np.zeros_like(candidate_confusion)
    positions = {value: index for index, value in enumerate(unique)}
    encoded = np.asarray([positions[value] for value in groups], dtype=int)
    np.add.at(candidate_confusion, (encoded, labels, candidate), 1)
    np.add.at(baseline_confusion, (encoded, labels, baseline), 1)

    def macro_f1(confusion: np.ndarray) -> float:
        true_positive = np.diag(confusion).astype(float)
        denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
        per_class = np.divide(
            2.0 * true_positive,
            denominator,
            out=np.zeros(3, dtype=float),
            where=denominator != 0,
        )
        return float(per_class.mean())

    observed = macro_f1(candidate_confusion.sum(axis=0)) - macro_f1(
        baseline_confusion.sum(axis=0)
    )
    permutations = 1 << len(unique)
    deltas = np.empty(permutations, dtype=float)
    for permutation in range(permutations):
        select_candidate = np.asarray(
            [(permutation >> index) & 1 for index in range(len(unique))], dtype=bool
        )
        first = np.where(
            select_candidate[:, None, None], candidate_confusion, baseline_confusion
        ).sum(axis=0)
        second = np.where(
            select_candidate[:, None, None], baseline_confusion, candidate_confusion
        ).sum(axis=0)
        deltas[permutation] = macro_f1(first) - macro_f1(second)
    tolerance = 1e-12
    p_value = float(np.mean(np.abs(deltas) >= abs(observed) - tolerance))
    return {
        "method": "exact_paired_recording_prediction_swap",
        "point_estimate": observed,
        "two_sided_p": p_value,
        "recordings": int(len(unique)),
        "permutations": int(permutations),
    }


def validate_summary(summary: dict, expected: dict, path: Path) -> None:
    for field, value in expected.items():
        if summary.get(field) != value:
            raise RuntimeError(f"Run contract drift at {path}: {field}")
    if summary.get("calibration_samples_read") != 0:
        raise RuntimeError(f"Calibration data entered development evidence: {path}")
    if summary.get("confirmation_samples_read") != 0:
        raise RuntimeError(f"Confirmation data entered development evidence: {path}")


def load_crossfit_run(path: Path, *, fold: int, seed: int) -> dict:
    summary_path = path / "summary.json"
    predictions_path = path / "held_predictions.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_summary(
        summary,
        {
            "status": "OKUTAMA_CPTR_CROSSFIT_RUN_COMPLETE",
            "candidate_id": "centre_short_parts",
            "fold": fold,
            "seed": seed,
            "validation_samples_read": 0,
        },
        path,
    )
    if sha256_file(predictions_path) != summary["artifact_sha256"][predictions_path.name]:
        raise RuntimeError(f"Cross-fit predictions changed: {path}")
    payload = np.load(predictions_path, allow_pickle=False)
    return {
        "summary": summary,
        "summary_path": summary_path,
        "predictions_path": predictions_path,
        **{name: payload[name] for name in payload.files},
    }


def load_validation_run(path: Path, *, seed: int) -> dict:
    summary_path = path / "summary.json"
    predictions_path = path / "validation_predictions.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_summary(
        summary,
        {
            "status": "OKUTAMA_CPTR_CANDIDATE_COMPLETE",
            "candidate_id": "centre_short_parts",
            "seed": seed,
        },
        path,
    )
    if sha256_file(predictions_path) != summary["artifact_sha256"][predictions_path.name]:
        raise RuntimeError(f"Validation predictions changed: {path}")
    payload = np.load(predictions_path, allow_pickle=False)
    return {
        "summary": summary,
        "summary_path": summary_path,
        "predictions_path": predictions_path,
        **{name: payload[name] for name in payload.files},
    }


def assert_aligned(runs: list[dict], fields: tuple[str, ...], context: str) -> None:
    reference = runs[0]
    for run in runs[1:]:
        for field in fields:
            if not np.array_equal(run[field], reference[field]):
                raise RuntimeError(f"{context} rows differ for {field}")


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples < 1:
        raise ValueError("bootstrap-resamples must be positive")
    protocol_path = args.protocol.resolve()
    protocol_lock_path = args.protocol_lock.resolve()
    plan_path = args.plan.resolve()
    plan_lock_path = args.plan_lock.resolve()
    manifest_path = args.manifest.resolve()
    crossfit_root = args.crossfit_root.resolve()
    promotion_root = args.promotion_root.resolve()
    baseline_summary_path = args.baseline_summary.resolve()
    baseline_predictions_path = args.baseline_predictions.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_lock = json.loads(protocol_lock_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_lock = json.loads(plan_lock_path.read_text(encoding="utf-8"))
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    if protocol_lock.get("status") != "OKUTAMA_CPTR_PROTOCOL_LOCKED_BEFORE_FIT":
        raise RuntimeError("The CPTR protocol lock is invalid")
    if plan_lock.get("status") != "OKUTAMA_CPTR_CROSSFIT_PLAN_LOCKED_BEFORE_FIT":
        raise RuntimeError("The CPTR cross-fit plan lock is invalid")
    if plan_lock["source_sha256"]["plan"] != sha256_file(plan_path):
        raise RuntimeError("The CPTR cross-fit plan changed after locking")
    if baseline_summary.get("status") != "OKUTAMA_CPTR_BASELINE_AND_DIAGNOSTICS_COMPLETE":
        raise RuntimeError("The CPTR baseline evidence is incomplete")
    if (
        sha256_file(baseline_predictions_path)
        != baseline_summary["artifact_sha256"][baseline_predictions_path.name]
    ):
        raise RuntimeError("The baseline prediction artifact changed")

    folds = int(plan["folds"])
    seeds = [int(item["seed"]) for item in plan["seeds"]]
    run_evidence: dict[str, dict[str, str]] = {}
    fold_blocks = []
    fold_seed_rows = []
    seed_blocks: dict[int, list[dict]] = {seed: [] for seed in seeds}
    for fold in range(folds):
        runs = []
        for seed in seeds:
            path = crossfit_root / f"fold-{fold}" / f"seed-{seed}"
            run = load_crossfit_run(path, fold=fold, seed=seed)
            runs.append(run)
            run_evidence[path.relative_to(crossfit_root).as_posix()] = {
                "summary": sha256_file(run["summary_path"]),
                "predictions": sha256_file(run["predictions_path"]),
            }
            seed_blocks[seed].append(run)
            seed_metrics = classification_summary(run["labels"], run["probabilities"])
            seed_baseline_metrics = classification_summary(
                run["labels"], run["baseline_probabilities"]
            )
            fold_seed_rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "fixed_epochs": int(run["summary"]["fixed_epochs"]),
                    "samples": len(run["labels"]),
                    "candidate_macro_f1": seed_metrics["macro_f1"],
                    "baseline_macro_f1": seed_baseline_metrics["macro_f1"],
                    "macro_f1_delta": seed_metrics["macro_f1"]
                    - seed_baseline_metrics["macro_f1"],
                }
            )
        assert_aligned(
            runs,
            (
                "sample_ids",
                "recording_ids",
                "track_ids",
                "labels",
                "transition_targets",
                "occlusion_targets",
            ),
            f"Cross-fit fold {fold}",
        )
        reference = runs[0]
        candidate = np.stack([run["probabilities"] for run in runs]).mean(axis=0)
        baseline = np.stack([run["baseline_probabilities"] for run in runs]).mean(axis=0)
        candidate_metrics = classification_summary(reference["labels"], candidate)
        baseline_metrics = classification_summary(reference["labels"], baseline)
        fold_seed_rows.append(
            {
                "fold": fold,
                "seed": "ensemble",
                "fixed_epochs": "development_locked",
                "samples": len(reference["labels"]),
                "candidate_macro_f1": candidate_metrics["macro_f1"],
                "baseline_macro_f1": baseline_metrics["macro_f1"],
                "macro_f1_delta": candidate_metrics["macro_f1"]
                - baseline_metrics["macro_f1"],
            }
        )
        fold_blocks.append(
            {
                **{
                    field: reference[field]
                    for field in (
                        "sample_ids",
                        "recording_ids",
                        "track_ids",
                        "labels",
                        "transition_targets",
                        "occlusion_targets",
                    )
                },
                "probabilities": candidate,
                "baseline_probabilities": baseline,
            }
        )

    oof = {
        field: np.concatenate([block[field] for block in fold_blocks])
        for field in fold_blocks[0]
    }
    if len(set(oof["sample_ids"].astype(str))) != len(oof["sample_ids"]):
        raise RuntimeError("CPTR cross-fit folds contain duplicate samples")
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
    expected_train = set(manifest.loc[manifest["split"].eq("train"), "sample_id"])
    if set(oof["sample_ids"].astype(str)) != expected_train:
        raise RuntimeError("CPTR cross-fit does not cover the locked training split")

    per_seed_oof = []
    for seed, runs in seed_blocks.items():
        candidate = np.concatenate([run["probabilities"] for run in runs])
        baseline = np.concatenate([run["baseline_probabilities"] for run in runs])
        labels = np.concatenate([run["labels"] for run in runs])
        candidate_metrics = classification_summary(labels, candidate)
        baseline_metrics = classification_summary(labels, baseline)
        per_seed_oof.append(
            {
                "fold": "all_oof",
                "seed": seed,
                "fixed_epochs": next(
                    int(item["fixed_epochs"]) for item in plan["seeds"] if item["seed"] == seed
                ),
                "samples": len(labels),
                "candidate_macro_f1": candidate_metrics["macro_f1"],
                "baseline_macro_f1": baseline_metrics["macro_f1"],
                "macro_f1_delta": candidate_metrics["macro_f1"]
                - baseline_metrics["macro_f1"],
            }
        )

    validation_runs = [
        load_validation_run(promotion_root / f"seed-{seed}", seed=seed) for seed in seeds
    ]
    assert_aligned(
        validation_runs,
        (
            "sample_ids",
            "recording_ids",
            "track_ids",
            "labels",
            "transition_targets",
            "occlusion_targets",
        ),
        "Validation seed",
    )
    validation_reference = validation_runs[0]
    validation_candidate = np.stack(
        [run["probabilities"] for run in validation_runs]
    ).mean(axis=0)
    baseline_payload = np.load(baseline_predictions_path, allow_pickle=False)
    baseline_positions = {
        sample_id: index
        for index, sample_id in enumerate(baseline_payload["sample_ids"].astype(str))
    }
    validation_order = np.asarray(
        [baseline_positions[value] for value in validation_reference["sample_ids"].astype(str)]
    )
    validation_baseline = baseline_payload["teacher__real_all_frames"][validation_order]
    if not np.array_equal(
        baseline_payload["labels"][validation_order], validation_reference["labels"]
    ):
        raise RuntimeError("Validation baseline and CPTR labels differ")

    oof_candidate_metrics = full_metrics(oof["labels"], oof["probabilities"])
    oof_baseline_metrics = full_metrics(oof["labels"], oof["baseline_probabilities"])
    validation_candidate_metrics = full_metrics(
        validation_reference["labels"], validation_candidate
    )
    validation_baseline_metrics = full_metrics(
        validation_reference["labels"], validation_baseline
    )
    oof_uncertainty = paired_cluster_bootstrap(
        oof["labels"],
        oof["probabilities"],
        oof["baseline_probabilities"],
        oof["recording_ids"],
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    validation_uncertainty = paired_cluster_bootstrap(
        validation_reference["labels"],
        validation_candidate,
        validation_baseline,
        validation_reference["recording_ids"],
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed + 1,
    )
    uncertainty = {
        "crossfit_oof_cluster_bootstrap": oof_uncertainty,
        "crossfit_oof_exact_group_swap": exact_group_swap_test(
            oof["labels"],
            oof["probabilities"],
            oof["baseline_probabilities"],
            oof["recording_ids"],
        ),
        "validation_cluster_bootstrap": validation_uncertainty,
        "validation_exact_group_swap": exact_group_swap_test(
            validation_reference["labels"],
            validation_candidate,
            validation_baseline,
            validation_reference["recording_ids"],
        ),
    }

    subgroup_rows = []
    for scope, labels, candidate, baseline, transitions, occlusions in (
        (
            "crossfit_oof",
            oof["labels"],
            oof["probabilities"],
            oof["baseline_probabilities"],
            oof["transition_targets"],
            oof["occlusion_targets"],
        ),
        (
            "validation",
            validation_reference["labels"],
            validation_candidate,
            validation_baseline,
            validation_reference["transition_targets"],
            validation_reference["occlusion_targets"],
        ),
    ):
        subgroup_rows.extend(
            {"scope": scope, **row}
            for row in subgroup_metrics(
                labels, candidate, baseline, transitions, occlusions
            )
        )

    recording_rows = []
    for scope, labels, candidate, baseline, groups in (
        (
            "crossfit_oof",
            oof["labels"],
            oof["probabilities"],
            oof["baseline_probabilities"],
            oof["recording_ids"],
        ),
        (
            "validation",
            validation_reference["labels"],
            validation_candidate,
            validation_baseline,
            validation_reference["recording_ids"],
        ),
    ):
        for recording_id in sorted(set(groups.astype(str))):
            mask = groups.astype(str) == recording_id
            candidate_metrics = classification_summary(labels[mask], candidate[mask])
            baseline_metrics = classification_summary(labels[mask], baseline[mask])
            recording_rows.append(
                {
                    "scope": scope,
                    "recording_id": recording_id,
                    "samples": int(mask.sum()),
                    "candidate_macro_f1": candidate_metrics["macro_f1"],
                    "baseline_macro_f1": baseline_metrics["macro_f1"],
                    "macro_f1_delta": candidate_metrics["macro_f1"]
                    - baseline_metrics["macro_f1"],
                }
            )

    threshold = float(protocol["promotion"]["final_minimum_macro_f1_gain"])
    regression_limit = float(protocol["promotion"]["maximum_allowed_per_class_f1_regression"])
    validation_gain = (
        validation_candidate_metrics["macro_f1"] - validation_baseline_metrics["macro_f1"]
    )
    oof_gain = oof_candidate_metrics["macro_f1"] - oof_baseline_metrics["macro_f1"]
    per_class_deltas = {
        name: validation_candidate_metrics[f"{name}_f1"]
        - validation_baseline_metrics[f"{name}_f1"]
        for name in CLASS_NAMES
    }
    worst_recording_delta = min(row["macro_f1_delta"] for row in recording_rows)
    promotion_checks = {
        "validation_macro_f1_gain_at_least_0_01": validation_gain >= threshold,
        "validation_cluster_interval_lower_bound_above_zero": validation_uncertainty[
            "macro_f1"
        ]["ci_95_low"]
        > 0.0,
        "validation_per_class_regression_within_0_01": min(per_class_deltas.values())
        >= -regression_limit,
        "crossfit_oof_macro_f1_nonnegative": oof_gain >= 0.0,
        "worst_recording_delta_at_least_minus_0_01": worst_recording_delta
        >= -regression_limit,
    }
    promoted = all(promotion_checks.values())

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = output_dir / "crossfit_oof_predictions.npz"
    validation_path = output_dir / "validation_predictions.npz"
    fold_seed_path = output_dir / "fold_seed_metrics.csv"
    subgroup_path = output_dir / "subgroup_metrics.csv"
    recording_path = output_dir / "recording_metrics.csv"
    uncertainty_path = output_dir / "uncertainty.json"
    np.savez_compressed(oof_path, **oof)
    np.savez_compressed(
        validation_path,
        sample_ids=validation_reference["sample_ids"],
        recording_ids=validation_reference["recording_ids"],
        track_ids=validation_reference["track_ids"],
        labels=validation_reference["labels"],
        probabilities=validation_candidate,
        baseline_probabilities=validation_baseline,
        transition_targets=validation_reference["transition_targets"],
        occlusion_targets=validation_reference["occlusion_targets"],
    )
    pd.DataFrame(fold_seed_rows + per_seed_oof).to_csv(fold_seed_path, index=False)
    pd.DataFrame(subgroup_rows).to_csv(subgroup_path, index=False)
    pd.DataFrame(recording_rows).to_csv(recording_path, index=False)
    write_json(uncertainty_path, uncertainty)
    summary = {
        "status": (
            "OKUTAMA_CPTR_DEVELOPMENT_PROMOTION_PASSED"
            if promoted
            else "OKUTAMA_CPTR_DEVELOPMENT_COMPLETE_NO_PROMOTION"
        ),
        "candidate_id": plan["candidate_id"],
        "decision": "promote" if promoted else "retain_as_exploratory_component",
        "development_validation": {
            "samples": int(len(validation_reference["labels"])),
            "recordings": int(len(set(validation_reference["recording_ids"].astype(str)))),
            "candidate_metrics": validation_candidate_metrics,
            "baseline_metrics": validation_baseline_metrics,
            "macro_f1_delta": validation_gain,
            "per_class_f1_delta": per_class_deltas,
        },
        "grouped_crossfit_oof": {
            "samples": int(len(oof["labels"])),
            "recordings": int(len(set(oof["recording_ids"].astype(str)))),
            "candidate_metrics": oof_candidate_metrics,
            "baseline_metrics": oof_baseline_metrics,
            "macro_f1_delta": oof_gain,
        },
        "promotion_checks": promotion_checks,
        "promotion_passed": promoted,
        "worst_recording_macro_f1_delta": worst_recording_delta,
        "fixed_epochs": {
            str(item["seed"]): int(item["fixed_epochs"]) for item in plan["seeds"]
        },
        "seeds": seeds,
        "folds": folds,
        "validation_samples_read": int(len(validation_reference["labels"])),
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "protocol": sha256_file(protocol_path),
            "protocol_lock": sha256_file(protocol_lock_path),
            "crossfit_plan": sha256_file(plan_path),
            "crossfit_plan_lock": sha256_file(plan_lock_path),
            "manifest": sha256_file(manifest_path),
            "baseline_summary": sha256_file(baseline_summary_path),
            "baseline_predictions": sha256_file(baseline_predictions_path),
            "runner": sha256_file(Path(__file__).resolve()),
        },
        "run_artifact_sha256": run_evidence,
        "artifact_sha256": {
            oof_path.name: sha256_file(oof_path),
            validation_path.name: sha256_file(validation_path),
            fold_seed_path.name: sha256_file(fold_seed_path),
            subgroup_path.name: sha256_file(subgroup_path),
            recording_path.name: sha256_file(recording_path),
            uncertainty_path.name: sha256_file(uncertainty_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
