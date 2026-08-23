"""Rank test-blind POLAR confirmations with the predeclared practical-tie rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hac.polar import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--aggregate-root", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def sample_std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def main() -> None:
    args = parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "DEVELOPMENT_CONFIRMATION_PLAN":
        raise RuntimeError("Expected a DEVELOPMENT_CONFIRMATION_PLAN")
    if plan.get("test_rows_read") != 0 or plan.get("test_used_for_selection"):
        raise RuntimeError("Confirmation plan violates the test gate")

    protocol_path = args.protocol_json.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if plan["protocol_version"] != protocol["protocol_version"]:
        raise RuntimeError("Confirmation plan and protocol versions differ")
    selection = protocol["selection"]
    practical_tie = float(selection["practical_tie_absolute"])
    expected_seeds = [int(seed) for seed in selection["confirmation_seeds"]]
    training_root = args.training_root.resolve()
    aggregate_root = args.aggregate_root.resolve()

    records = []
    detailed = {}
    for candidate in plan["candidates"]:
        candidate_id = candidate["candidate_id"]
        seed_rows = []
        summary_hashes = {}
        for seed in expected_seeds:
            run_id = candidate["confirmation_run_ids"][str(seed)]
            summary_path = training_root / run_id / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") != "COMPLETE":
                raise RuntimeError(f"Confirmation is incomplete: {summary_path}")
            if summary.get("test_rows_read") != 0 or summary.get("test_used_for_selection"):
                raise RuntimeError(f"Confirmation violates the test gate: {summary_path}")
            if int(summary["configuration"]["seed"]) != seed:
                raise RuntimeError(f"Confirmation seed differs from its plan: {summary_path}")
            summary_hashes[str(seed)] = sha256_file(summary_path)
            seed_rows.append(
                {
                    "seed": seed,
                    "run_id": run_id,
                    "best_epoch": int(summary["best_epoch"]),
                    "runtime_seconds": float(summary["runtime_seconds_this_invocation"]),
                    **summary["best_validation_metrics"],
                }
            )

        aggregate_path = aggregate_root / candidate_id / "provenance.json"
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if aggregate.get("status") != "DEVELOPMENT_ONLY_MULTI_SEED_AGGREGATE":
            raise RuntimeError(f"Unexpected aggregate status: {aggregate_path}")
        if aggregate.get("test_rows_read") != 0 or aggregate.get("test_used_for_selection"):
            raise RuntimeError(f"Aggregate violates the test gate: {aggregate_path}")

        macro_values = [row["macro_f1"] for row in seed_rows]
        log_loss_values = [row["log_loss"] for row in seed_rows]
        ece_values = [row["ece"] for row in seed_rows]
        runtime_values = [row["runtime_seconds"] for row in seed_rows]
        best_epochs = [row["best_epoch"] for row in seed_rows]
        record = {
            "candidate": candidate_id,
            "seed_count": len(seed_rows),
            "seed_mean_macro_f1": mean(macro_values),
            "seed_std_macro_f1": sample_std(macro_values),
            "seed_mean_log_loss": mean(log_loss_values),
            "seed_mean_ece": mean(ece_values),
            "seed_mean_runtime_seconds": mean(runtime_values),
            "median_best_epoch": int(np.median(best_epochs)),
            **{f"aggregate_{key}": value for key, value in aggregate["metrics"].items()},
        }
        records.append(record)
        detailed[candidate_id] = {
            "configuration": candidate["configuration"],
            "seed_metrics": seed_rows,
            "training_summary_sha256": summary_hashes,
            "aggregate_provenance_sha256": sha256_file(aggregate_path),
            "aggregate_prediction_sha256": sha256_file(
                aggregate_root / candidate_id / "validation_predictions.npz"
            ),
        }

    leader_mean = max(record["seed_mean_macro_f1"] for record in records)
    for record in records:
        record["difference_from_seed_mean_leader"] = (
            leader_mean - record["seed_mean_macro_f1"]
        )
        record["inside_practical_tie"] = (
            record["difference_from_seed_mean_leader"] < practical_tie
        )

    tied = sorted(
        (record for record in records if record["inside_practical_tie"]),
        key=lambda record: (
            record["seed_std_macro_f1"],
            record["seed_mean_log_loss"],
            record["seed_mean_ece"],
            record["seed_mean_runtime_seconds"],
            record["candidate"],
        ),
    )
    outside = sorted(
        (record for record in records if not record["inside_practical_tie"]),
        key=lambda record: (
            -record["seed_mean_macro_f1"],
            record["seed_std_macro_f1"],
            record["seed_mean_log_loss"],
            record["candidate"],
        ),
    )
    ranked = [*tied, *outside]
    for rank, record in enumerate(ranked, start=1):
        record["rank"] = rank
        record["selected"] = rank == 1
    winner = ranked[0]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(ranked).sort_values("rank", ignore_index=True)
    ordered = ["rank", "selected", "candidate", *[c for c in frame if c not in {"rank", "selected", "candidate"}]]
    frame[ordered].to_csv(output_dir / "polar_confirmation_ranking.csv", index=False)
    summary = {
        "status": "DEVELOPMENT_ONLY_CONFIRMATION_RANKING",
        "selected_candidate": winner["candidate"],
        "selection_rule": {
            "primary_metric": selection["primary_metric"],
            "practical_tie_absolute": practical_tie,
            "tie_breakers": selection["tie_breakers"],
            "seed_mean_leader": leader_mean,
            "selected_because": "lowest seed macro-F1 standard deviation inside the practical-tie band",
        },
        "selected_record": winner,
        "candidates": detailed,
        "source_sha256": {
            "confirmation_plan": sha256_file(plan_path),
            "protocol": sha256_file(protocol_path),
        },
        "test_rows_read": 0,
        "test_used_for_selection": False,
    }
    (output_dir / "polar_confirmation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(frame[ordered].to_string(index=False), flush=True)
    print(f"[selected] {winner['candidate']}", flush=True)


if __name__ == "__main__":
    main()
