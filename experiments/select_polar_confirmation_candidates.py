"""Derive the test-blind POLAR confirmation queue from completed seed-42 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hac.polar import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_eligible_runs(training_root: Path) -> list[dict]:
    records = []
    for path in sorted(training_root.rglob("summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        configuration = summary.get("configuration", {})
        if summary.get("status") != "COMPLETE":
            continue
        if summary.get("test_rows_read") != 0 or summary.get("test_used_for_selection"):
            raise RuntimeError(f"Training evidence violates the test gate: {path}")
        if configuration.get("model_kind") != "dinov2_small":
            continue
        if configuration.get("task") != "label_4" or configuration.get("train_size") != "all":
            continue
        if configuration.get("seed") != 42:
            continue
        if configuration.get("run_role") not in {"adaptation_screen", "regularization_screen"}:
            continue
        records.append(
            {
                "run_id": path.parent.relative_to(training_root).as_posix(),
                "summary_path": path,
                "summary_sha256": sha256_file(path),
                "configuration": configuration,
                "metrics": summary["best_validation_metrics"],
                "best_epoch": summary["best_epoch"],
            }
        )
    if not records:
        raise RuntimeError("No eligible seed-42 DINOv2-S runs were found")
    return records


def confirmation_plan(training_root: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    selection = protocol["selection"]
    seeds = [int(seed) for seed in selection["confirmation_seeds"]]
    if seeds[0] != 42 or len(set(seeds)) != len(seeds):
        raise RuntimeError("Confirmation seeds must be unique and start with screen seed 42")

    records = load_eligible_runs(training_root)
    leader = max(records, key=lambda record: float(record["metrics"]["macro_f1"]))
    leader_f1 = float(leader["metrics"]["macro_f1"])
    band = float(selection["confirmation_band_absolute"])
    candidates = [
        record
        for record in records
        if leader_f1 - float(record["metrics"]["macro_f1"]) <= band + 1e-12
    ]
    candidates.sort(
        key=lambda record: (
            -float(record["metrics"]["macro_f1"]),
            float(record["metrics"]["log_loss"]),
            record["run_id"],
        )
    )

    candidate_rows = []
    additional_runs = []
    for record in candidates:
        configuration = dict(record["configuration"])
        source_name = Path(record["run_id"]).name
        if not source_name.endswith("_seed42"):
            raise RuntimeError(f"Seed-42 run id has an unexpected form: {record['run_id']}")
        confirmation_run_ids = {"42": record["run_id"]}
        for seed in seeds[1:]:
            run_name = f"{source_name[:-len('_seed42')]}_seed{seed}"
            run_id = f"confirmation/{run_name}"
            run_configuration = {
                **configuration,
                "run_role": "confirmation",
                "seed": seed,
            }
            confirmation_run_ids[str(seed)] = run_id
            additional_runs.append(
                {
                    "source_run_id": record["run_id"],
                    "run_id": run_id,
                    "seed": seed,
                    "configuration": run_configuration,
                }
            )
        candidate_rows.append(
            {
                "candidate_id": source_name.removesuffix("_seed42"),
                "screen_run_id": record["run_id"],
                "screen_summary_sha256": record["summary_sha256"],
                "screen_macro_f1": record["metrics"]["macro_f1"],
                "screen_log_loss": record["metrics"]["log_loss"],
                "screen_ece": record["metrics"]["ece"],
                "best_epoch": record["best_epoch"],
                "configuration": configuration,
                "confirmation_run_ids": confirmation_run_ids,
            }
        )

    return {
        "status": "DEVELOPMENT_CONFIRMATION_PLAN",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256_file(protocol_path),
        "selector": {
            "primary_metric": selection["primary_metric"],
            "leader_macro_f1": leader_f1,
            "confirmation_band_absolute": band,
            "confirmation_seeds": seeds,
            "eligible_family": "dinov2_small",
            "eligible_tasks": ["label_4"],
            "eligible_roles": ["adaptation_screen", "regularization_screen"],
        },
        "leader_screen_run_id": leader["run_id"],
        "candidates": candidate_rows,
        "additional_runs": additional_runs,
        "test_rows_read": 0,
        "test_used_for_selection": False,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    plan = confirmation_plan(args.training_root.resolve(), args.protocol_json.resolve())
    output.write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
