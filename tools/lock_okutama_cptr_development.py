"""Lock the completed CPTR development decision and its evidence graph."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from hac.polar import sha256_file
from hac.retained_lock import validate_retained_lock, validate_single_field_normalization

LOCKED_PROTOCOL_SNAPSHOT = Path(
    ".runs/cptr/source_snapshots/"
    "okutama_cptr_protocol_locked_"
    "0bcb58909c04708447607fabccf7efbc8617c09e2d9fdfb7f3545e38e15954c8.json"
)
PORTABLE_ARCHIVE_PATH = "data/external/OkutamaAction/TrainSetFrames.zip"

COMPONENT_SUMMARIES = (
    ".runs/cptr/baseline/summary.json",
    ".runs/cptr/candidates/trajectory_raw/seed-42-v3/summary.json",
    ".runs/cptr/candidates/trajectory_compensated/seed-42-v2/summary.json",
    ".runs/cptr/candidates/centre_short/seed-42-v1/summary.json",
    ".runs/cptr/candidates/dual_clock/seed-42-v1/summary.json",
    ".runs/cptr/adaptive/dual_clock_specialized/seed-42/summary.json",
    ".runs/cptr/adaptive/centre_short_trajectory/seed-42/summary.json",
    ".runs/cptr/adaptive/centre_short_parts/seed-42/summary.json",
    ".runs/cptr/adaptive/cptr_integrated/seed-42/summary.json",
    ".runs/cptr/stage2/centre_short_parts_counterfactual/seed-42/summary.json",
    ".runs/cptr/stage3/centre_short_parts_counterfactual_refined/seed-42/summary.json",
    ".runs/cptr/stage3/centre_short_parts_masked_only/seed-42/summary.json",
    ".runs/cptr/stage4/centre_short_parts_siglip/seed-42/summary.json",
    ".runs/cptr/stage4/centre_short_parts_group_dro/seed-42/summary.json",
    ".runs/cptr/router/summary.json",
    ".runs/cptr/masked_pretraining/seed-42/summary.json",
    ".runs/cptr/lora_specialist-v4/summary.json",
)

FEATURE_STORES = (
    ".runs/cptr/motion_features/store.json",
    ".runs/cptr/part_features/store.json",
    ".runs/cptr/siglip_features/store.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiments/okutama_cptr_protocol.json")
    )
    parser.add_argument("--protocol-lock", type=Path, default=Path(".runs/cptr/protocol_lock.json"))
    parser.add_argument("--protocol-snapshot", type=Path, default=LOCKED_PROTOCOL_SNAPSHOT)
    parser.add_argument(
        "--crossfit-plan-lock",
        type=Path,
        default=Path(".runs/cptr/crossfit_plan_lock.json"),
    )
    parser.add_argument(
        "--development-summary",
        type=Path,
        default=Path(".runs/cptr/development_final/summary.json"),
    )
    parser.add_argument(
        "--faithfulness-summary",
        type=Path,
        default=Path(".runs/cptr/development_final/faithfulness/summary.json"),
    )
    parser.add_argument("--output", type=Path, default=Path(".runs/cptr/development_lock.json"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate an existing retained lock without modifying it",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_closed_evaluation(path: Path, payload: dict) -> None:
    if payload.get("calibration_samples_read", 0) != 0:
        raise RuntimeError(f"Calibration data entered CPTR development: {path}")
    if payload.get("confirmation_samples_read", 0) != 0:
        raise RuntimeError(f"Confirmation data entered CPTR development: {path}")


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    protocol_path = args.protocol.resolve()
    protocol_lock_path = args.protocol_lock.resolve()
    crossfit_lock_path = args.crossfit_plan_lock.resolve()
    development_path = args.development_summary.resolve()
    faithfulness_path = args.faithfulness_summary.resolve()
    protocol = load_json(protocol_path)
    protocol_lock = load_json(protocol_lock_path)
    crossfit_lock = load_json(crossfit_lock_path)
    development = load_json(development_path)
    faithfulness = load_json(faithfulness_path)
    if protocol_lock.get("status") != "OKUTAMA_CPTR_PROTOCOL_LOCKED_BEFORE_FIT":
        raise RuntimeError("The CPTR protocol lock is invalid")
    locked_protocol_digest = protocol_lock["source_sha256"]["protocol"]
    public_protocol_digest = sha256_file(protocol_path)
    protocol_digest = public_protocol_digest
    portable_normalization = False
    if locked_protocol_digest != public_protocol_digest:
        if not args.check:
            raise RuntimeError("The CPTR protocol changed after locking")
        snapshot_path = args.protocol_snapshot.resolve()
        if sha256_file(snapshot_path) != locked_protocol_digest:
            raise RuntimeError("The retained CPTR protocol snapshot does not match its lock")
        snapshot = load_json(snapshot_path)
        _, normalized_path = validate_single_field_normalization(
            snapshot,
            protocol,
            field="input.raw_development_archive",
        )
        if normalized_path != PORTABLE_ARCHIVE_PATH:
            raise RuntimeError("The public CPTR archive path is not the declared portable path")
        protocol_digest = locked_protocol_digest
        portable_normalization = True
    if crossfit_lock.get("status") != "OKUTAMA_CPTR_CROSSFIT_PLAN_LOCKED_BEFORE_FIT":
        raise RuntimeError("The CPTR cross-fit plan lock is invalid")
    if development.get("status") != "OKUTAMA_CPTR_DEVELOPMENT_COMPLETE_NO_PROMOTION":
        raise RuntimeError("The CPTR development decision is not a completed no-promotion result")
    if development.get("promotion_passed") is not False:
        raise RuntimeError("The recorded promotion decision is inconsistent")
    if faithfulness.get("status") != "OKUTAMA_CPTR_FAITHFULNESS_COMPLETE":
        raise RuntimeError("The CPTR faithfulness evaluation is incomplete")
    for path, payload in (
        (development_path, development),
        (faithfulness_path, faithfulness),
    ):
        require_closed_evaluation(path, payload)

    evidence: dict[str, str] = {}
    for relative in (*COMPONENT_SUMMARIES, *FEATURE_STORES):
        path = (root / relative).resolve()
        payload = load_json(path)
        require_closed_evaluation(path, payload)
        evidence[Path(relative).as_posix()] = sha256_file(path)
    for seed in map(int, protocol["training"]["promotion_seeds"]):
        relative = f".runs/cptr/promotion/centre_short_parts/seed-{seed}/summary.json"
        path = (root / relative).resolve()
        payload = load_json(path)
        if payload.get("status") != "OKUTAMA_CPTR_CANDIDATE_COMPLETE":
            raise RuntimeError(f"Promotion seed is incomplete: {path}")
        require_closed_evaluation(path, payload)
        evidence[Path(relative).as_posix()] = sha256_file(path)

    payload = {
        "status": "OKUTAMA_CPTR_DEVELOPMENT_LOCKED_NO_PROMOTION",
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "candidate_id": development["candidate_id"],
        "decision": {
            "default_model": "v3_temporal_8f_050s_five_seed_ensemble",
            "research_component": "centre_short_parts_five_seed_ensemble",
            "research_component_status": "exploratory_not_promoted",
            "calibration_opened": False,
            "reason": "the grouped cross-fit and preregistered aggregate gain gates did not pass",
        },
        "development_validation_macro_f1": development["development_validation"][
            "candidate_metrics"
        ]["macro_f1"],
        "development_validation_macro_f1_delta": development["development_validation"][
            "macro_f1_delta"
        ],
        "grouped_crossfit_oof_macro_f1": development["grouped_crossfit_oof"]["candidate_metrics"][
            "macro_f1"
        ],
        "grouped_crossfit_oof_macro_f1_delta": development["grouped_crossfit_oof"][
            "macro_f1_delta"
        ],
        "promotion_checks": development["promotion_checks"],
        "validation_samples_read": int(development["validation_samples_read"]),
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "source_sha256": {
            "protocol": protocol_digest,
            "protocol_lock": sha256_file(protocol_lock_path),
            "crossfit_plan_lock": sha256_file(crossfit_lock_path),
            "development_summary": sha256_file(development_path),
            "faithfulness_summary": sha256_file(faithfulness_path),
            "locker": sha256_file(Path(__file__).resolve()),
        },
        "component_evidence_sha256": evidence,
    }
    output = args.output.resolve()
    if args.check:
        if not output.is_file():
            raise FileNotFoundError(f"CPTR development lock does not exist: {output}")
        retained_text = output.read_text(encoding="utf-8")
        validate_retained_lock(json.loads(retained_text), payload)
        if portable_normalization:
            print(
                "Retained CPTR protocol verified from its locked snapshot; the public "
                "protocol differs only at input.raw_development_archive, normalized to "
                f"{PORTABLE_ARCHIVE_PATH}",
                file=sys.stderr,
            )
        print(retained_text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output.is_file() and output.read_text(encoding="utf-8") != serialized:
        raise RuntimeError("An incompatible CPTR development lock already exists")
    if not output.is_file():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(output)
    print(serialized, end="")


if __name__ == "__main__":
    main()
