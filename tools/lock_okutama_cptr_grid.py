"""Bind the declared CPTR candidate grid to the pre-fit protocol lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hac.polar import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", type=Path, default=Path("experiments/okutama_cptr_grid.json")
    )
    parser.add_argument(
        "--protocol-lock", type=Path, default=Path(".runs/cptr/protocol_lock.json")
    )
    parser.add_argument("--output", type=Path, default=Path(".runs/cptr/grid_lock.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid_path = args.grid.resolve()
    lock_path = args.protocol_lock.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if grid.get("status") != "DECLARED_BEFORE_CPTR_CANDIDATE_FITTING":
        raise RuntimeError("The CPTR candidate grid is not in its pre-fit state")
    if lock.get("status") != "OKUTAMA_CPTR_PROTOCOL_LOCKED_BEFORE_FIT":
        raise RuntimeError("The CPTR protocol is not locked")
    identifiers = [item["candidate_id"] for item in grid.get("candidates", [])]
    if len(identifiers) != len(set(identifiers)) or len(identifiers) < 5:
        raise RuntimeError("The CPTR candidate identifiers are empty or duplicated")
    required = {
        "trajectory_raw",
        "trajectory_compensated",
        "centre_short",
        "dual_clock",
        "dual_clock_trajectory_parts",
        "counterfactual_full",
        "masked_adapted_full",
        "group_robust_full",
        "siglip_specialist_full",
    }
    if not required.issubset(identifiers):
        raise RuntimeError("The CPTR grid omits a required architecture stage")
    payload = {
        "status": "OKUTAMA_CPTR_GRID_LOCKED_BEFORE_FIT",
        "grid_version": grid["grid_version"],
        "candidate_order": identifiers,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "grid": sha256_file(grid_path),
            "protocol_lock": sha256_file(lock_path),
            "locker": sha256_file(Path(__file__).resolve()),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output.is_file():
        if output.read_text(encoding="utf-8") != serialized:
            raise RuntimeError("An incompatible CPTR grid lock already exists")
    else:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(output)
    print(serialized, end="")


if __name__ == "__main__":
    main()
