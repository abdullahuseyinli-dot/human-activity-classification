"""Lock CPTR grouped cross-fit epochs and evidence before fold fitting."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from hac.polar import sha256_file
from hac.retained_lock import validate_retained_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("experiments/okutama_cptr_crossfit_plan.json")
    )
    parser.add_argument("--protocol-lock", type=Path, default=Path(".runs/cptr/protocol_lock.json"))
    parser.add_argument("--runner", type=Path, default=Path("experiments/crossfit_okutama_cptr.py"))
    parser.add_argument("--training-module", type=Path, default=Path("src/hac/cptr_training.py"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/development_manifest.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path(".runs/cptr/crossfit_plan_lock.json"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate an existing retained lock without modifying it",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    plan_path = args.plan.resolve()
    protocol_lock_path = args.protocol_lock.resolve()
    runner_path = args.runner.resolve()
    training_module_path = args.training_module.resolve()
    manifest_path = args.manifest.resolve()
    plan = load_json(plan_path)
    protocol_lock = load_json(protocol_lock_path)
    if plan.get("status") != "DECLARED_AFTER_FIVE_SEED_DEVELOPMENT_BEFORE_CPTR_CROSSFIT":
        raise RuntimeError("The CPTR cross-fit plan is not in its declared pre-fit state")
    if protocol_lock.get("status") != "OKUTAMA_CPTR_PROTOCOL_LOCKED_BEFORE_FIT":
        raise RuntimeError("The CPTR protocol lock is invalid")
    candidate_grid_path = (root / plan["candidate_grid"]).resolve()
    candidate_grid_lock_path = (root / plan["candidate_grid_lock"]).resolve()
    candidate_grid = load_json(candidate_grid_path)
    candidate_grid_lock = load_json(candidate_grid_lock_path)
    if candidate_grid_lock.get("status") != "OKUTAMA_CPTR_ADAPTIVE_GRID_LOCKED_BEFORE_FIT":
        raise RuntimeError("The adaptive CPTR grid lock is invalid")
    candidate_grid_hash = sha256_file(candidate_grid_path)
    if candidate_grid_lock["source_sha256"]["adaptive_grid"] != candidate_grid_hash:
        raise RuntimeError("The candidate grid changed after adaptive locking")
    if candidate_grid_lock["source_sha256"]["protocol_lock"] != sha256_file(protocol_lock_path):
        raise RuntimeError("The candidate grid and protocol locks do not match")
    candidate_ids = [item["candidate_id"] for item in candidate_grid.get("candidates", [])]
    if plan["candidate_id"] not in candidate_ids or plan["candidate_id"] not in set(
        candidate_grid_lock.get("candidate_order", [])
    ):
        raise RuntimeError("The cross-fit candidate is not present in the adaptive grid lock")
    seeds = [int(item["seed"]) for item in plan.get("seeds", [])]
    if seeds != [42, 43, 44, 45, 46] or int(plan.get("folds", 0)) != 5:
        raise RuntimeError("The CPTR cross-fit seed/fold contract changed")
    evidence = {}
    for item in plan["seeds"]:
        path = (root / item["development_summary"]).resolve()
        digest = sha256_file(path)
        if digest != item["development_summary_sha256"]:
            raise RuntimeError(f"Development selection evidence changed: {path}")
        summary = load_json(path)
        expected_epochs = max(0, int(summary["best_epoch"]) + 1)
        if expected_epochs != int(item["fixed_epochs"]):
            raise RuntimeError("A cross-fit fixed epoch count differs from development selection")
        if summary.get("calibration_samples_read") != 0:
            raise RuntimeError("Calibration data entered cross-fit epoch selection")
        if summary.get("confirmation_samples_read") != 0:
            raise RuntimeError("Confirmation data entered cross-fit epoch selection")
        evidence[str(item["seed"])] = digest
    payload = {
        "status": "OKUTAMA_CPTR_CROSSFIT_PLAN_LOCKED_BEFORE_FIT",
        "plan_version": plan["plan_version"],
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "candidate_id": plan["candidate_id"],
        "folds": int(plan["folds"]),
        "seeds": seeds,
        "validation_samples_read": 0,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "plan": sha256_file(plan_path),
            "protocol_lock": sha256_file(protocol_lock_path),
            "manifest": sha256_file(manifest_path),
            "runner": sha256_file(runner_path),
            "training_module": sha256_file(training_module_path),
            "candidate_grid": candidate_grid_hash,
            "candidate_grid_lock": sha256_file(candidate_grid_lock_path),
            "locker": sha256_file(Path(__file__).resolve()),
            "development_summaries": evidence,
        },
    }
    output = args.output.resolve()
    if args.check:
        if not output.is_file():
            raise FileNotFoundError(f"CPTR cross-fit plan lock does not exist: {output}")
        retained_text = output.read_text(encoding="utf-8")
        retained = json.loads(retained_text)
        new_bindings = (
            "source_sha256.candidate_grid",
            "source_sha256.candidate_grid_lock",
            "source_sha256.training_module",
        )
        retained_sources = retained.get("source_sha256", {})
        binding_presence = (
            [path.rsplit(".", maxsplit=1)[1] in retained_sources for path in new_bindings]
            if isinstance(retained_sources, dict)
            else []
        )
        if binding_presence and any(binding_presence) and not all(binding_presence):
            raise RuntimeError("The cross-fit lock has an incomplete source-binding schema")
        historical_schema = bool(binding_presence) and not any(binding_presence)
        check = validate_retained_lock(
            retained,
            payload,
            allowed_legacy_omissions=new_bindings if historical_schema else (),
            allowed_historical_mismatches=("source_sha256.runner",) if historical_schema else (),
        )
        if check.legacy_omissions:
            print(
                "Retained cross-fit lock uses the historical schema; later source "
                f"bindings are absent: {', '.join(check.legacy_omissions)}",
                file=sys.stderr,
            )
        if check.historical_mismatches:
            print(
                "Retained cross-fit lock preserves an earlier source revision: "
                f"{', '.join(check.historical_mismatches)}",
                file=sys.stderr,
            )
        print(retained_text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output.is_file() and output.read_text(encoding="utf-8") != serialized:
        raise RuntimeError("An incompatible CPTR cross-fit plan lock already exists")
    if not output.is_file():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(output)
    print(serialized, end="")


if __name__ == "__main__":
    main()
