"""Build a test-blind multi-seed plan for neural POLAR ensemble contributors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hac.polar import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument(
        "--component",
        action="append",
        required=True,
        metavar="NAME=RUN_ID",
        help="Positive-weight neural component and its completed seed-42 run.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_component(value: str) -> tuple[str, str]:
    name, separator, run_id = value.partition("=")
    if not separator or not name or not run_id:
        raise ValueError(f"Expected NAME=RUN_ID, found {value!r}")
    return name, run_id


def main() -> None:
    args = parse_args()
    training_root = args.training_root.resolve()
    protocol_path = args.protocol_json.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in protocol["selection"]["confirmation_seeds"]]
    if seeds[0] != 42 or len(set(seeds)) != len(seeds):
        raise RuntimeError("Confirmation seeds must be unique and start with seed 42")

    validation_dir = args.validation_dir.resolve()
    blend_path = validation_dir / "validation_blend.json"
    provenance_path = validation_dir / "provenance.json"
    blend = json.loads(blend_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for payload in (blend, provenance):
        if payload.get("test_rows_read") != 0 or payload.get("test_used_for_selection"):
            raise RuntimeError("Validation evidence violates the test gate")

    candidates = []
    additional_runs = []
    seen = set()
    for value in args.component:
        name, run_id = parse_component(value)
        if name in seen:
            raise ValueError(f"Duplicate component name: {name}")
        seen.add(name)
        if float(blend["weights"].get(name, 0.0)) <= 0.0:
            raise RuntimeError(f"Ensemble component does not have positive weight: {name}")
        summary_path = training_root / run_id / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        configuration = summary["configuration"]
        if summary.get("status") != "COMPLETE" or configuration.get("seed") != 42:
            raise RuntimeError(f"Expected a completed seed-42 source run: {summary_path}")
        if summary.get("test_rows_read") != 0 or summary.get("test_used_for_selection"):
            raise RuntimeError(f"Source run violates the test gate: {summary_path}")

        source_name = Path(run_id).name
        if not source_name.endswith("_seed42"):
            raise RuntimeError(f"Seed-42 run id has an unexpected form: {run_id}")
        run_ids = {"42": run_id}
        for seed in seeds[1:]:
            run_name = f"{source_name[:-len('_seed42')]}_seed{seed}"
            confirmation_run_id = f"ensemble_confirmation/{run_name}"
            run_configuration = {
                **configuration,
                "run_role": "confirmation",
                "seed": seed,
            }
            run_ids[str(seed)] = confirmation_run_id
            additional_runs.append(
                {
                    "source_run_id": run_id,
                    "component": name,
                    "run_id": confirmation_run_id,
                    "seed": seed,
                    "configuration": run_configuration,
                }
            )
        candidates.append(
            {
                "candidate_id": name,
                "screen_run_id": run_id,
                "screen_summary_sha256": sha256_file(summary_path),
                "screen_metrics": summary["best_validation_metrics"],
                "configuration": configuration,
                "confirmation_run_ids": run_ids,
            }
        )

    plan = {
        "status": "DEVELOPMENT_CONFIRMATION_PLAN",
        "plan_role": "positive_weight_neural_ensemble_component_confirmation",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256_file(protocol_path),
        "confirmation_seeds": seeds,
        "validation_blend_sha256": sha256_file(blend_path),
        "validation_provenance_sha256": sha256_file(provenance_path),
        "validation_weights": blend["weights"],
        "candidates": candidates,
        "additional_runs": additional_runs,
        "test_rows_read": 0,
        "test_used_for_selection": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
