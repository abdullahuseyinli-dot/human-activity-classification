"""Lock the second CPTR adaptive stage against completed component screens."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from hac.polar import sha256_file
from hac.retained_lock import validate_retained_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", type=Path, default=Path("experiments/okutama_cptr_stage2_grid.json")
    )
    parser.add_argument(
        "--adaptive-lock", type=Path, default=Path(".runs/cptr/adaptive_grid_lock.json")
    )
    parser.add_argument("--protocol-lock", type=Path, default=Path(".runs/cptr/protocol_lock.json"))
    parser.add_argument("--output", type=Path, default=Path(".runs/cptr/stage2_grid_lock.json"))
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
    grid_path = args.grid.resolve()
    adaptive_lock_path = args.adaptive_lock.resolve()
    protocol_lock_path = args.protocol_lock.resolve()
    grid = load_json(grid_path)
    adaptive_lock = load_json(adaptive_lock_path)
    protocol_lock = load_json(protocol_lock_path)
    if grid.get("status") != "DECLARED_AFTER_INTEGRATION_SCREEN_BEFORE_STAGE2_FITTING":
        raise RuntimeError("The stage-two grid is not in its declared pre-fit state")
    if adaptive_lock.get("status") != "OKUTAMA_CPTR_ADAPTIVE_GRID_LOCKED_BEFORE_FIT":
        raise RuntimeError("The adaptive CPTR grid lock is invalid")
    if protocol_lock.get("status") != "OKUTAMA_CPTR_PROTOCOL_LOCKED_BEFORE_FIT":
        raise RuntimeError("The CPTR protocol lock is invalid")
    identifiers = [item["candidate_id"] for item in grid.get("candidates", [])]
    if len(identifiers) != len(set(identifiers)) or len(identifiers) < 4:
        raise RuntimeError("Stage-two candidate identifiers are empty or duplicated")

    evidence_hashes: dict[str, str] = {}
    for item in grid.get("screen_evidence", []):
        path = (root / item["summary"]).resolve()
        digest = sha256_file(path)
        if digest != item["summary_sha256"]:
            raise RuntimeError(f"Screen evidence changed: {path}")
        summary = load_json(path)
        if summary.get("status") != "OKUTAMA_CPTR_CANDIDATE_COMPLETE":
            raise RuntimeError(f"Screen evidence is incomplete: {path}")
        if summary.get("calibration_samples_read") != 0:
            raise RuntimeError("Calibration data entered stage-two selection")
        if summary.get("confirmation_samples_read") != 0:
            raise RuntimeError("Confirmation data entered stage-two selection")
        evidence_hashes[item["candidate_id"]] = digest

    payload = {
        "status": "OKUTAMA_CPTR_STAGE2_GRID_LOCKED_BEFORE_FIT",
        "grid_version": grid["grid_version"],
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "candidate_order": identifiers,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "stage2_grid": sha256_file(grid_path),
            "adaptive_grid_lock": sha256_file(adaptive_lock_path),
            "protocol_lock": sha256_file(protocol_lock_path),
            "locker": sha256_file(Path(__file__).resolve()),
            "screen_evidence": evidence_hashes,
        },
    }
    output = args.output.resolve()
    if args.check:
        if not output.is_file():
            raise FileNotFoundError(f"Stage-two CPTR grid lock does not exist: {output}")
        retained_text = output.read_text(encoding="utf-8")
        validate_retained_lock(json.loads(retained_text), payload)
        print(retained_text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output.is_file() and output.read_text(encoding="utf-8") != serialized:
        raise RuntimeError("An incompatible stage-two CPTR grid lock already exists")
    if not output.is_file():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(output)
    print(serialized, end="")


if __name__ == "__main__":
    main()
