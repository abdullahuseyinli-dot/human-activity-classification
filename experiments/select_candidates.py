"""Lock final model configurations from test-blind confirmation evidence.

The script accepts only completed five-fold confirmation runs. It ranks each
model family by pooled out-of-fold macro-F1, then uses fold stability and OOF
log-loss as deterministic tie-breakers. The resulting lock file is the only
configuration input accepted by the final training runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_FAMILIES = ("convnext_small", "dinov2_small")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_confirmation_records(selection_root: Path) -> list[dict]:
    result_dir = selection_root / "candidate_results"
    records: list[dict] = []
    for path in sorted(result_dir.glob("confirm_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        record["_evidence_path"] = str(path.resolve())
        record["_evidence_sha256"] = sha256_file(path)
        records.append(record)
    if not records:
        raise RuntimeError(f"No confirmation records found under {result_dir}")
    return records


def validate_record(record: dict) -> None:
    if record.get("stage") != "confirm":
        raise RuntimeError(f"Non-confirmation record supplied: {record.get('candidate_id')}")
    if int(record.get("cv_n_splits", 0)) != 5:
        raise RuntimeError(f"Expected five folds for {record.get('candidate_id')}")
    if bool(record.get("test_evaluated", True)):
        raise RuntimeError(f"Test-tainted selection evidence: {record.get('candidate_id')}")
    fold_scores = record.get("fold_macro_f1", [])
    if len(fold_scores) != 5 or not np.isfinite(np.asarray(fold_scores, dtype=float)).all():
        raise RuntimeError(f"Invalid fold evidence for {record.get('candidate_id')}")


def rank_key(record: dict) -> tuple[float, float, float, str]:
    fold_std = float(np.std(record["fold_macro_f1"], ddof=1))
    return (
        -float(record["oof_macro_f1"]),
        fold_std,
        float(record["oof_log_loss"]),
        str(record["candidate_id"]),
    )


def main() -> None:
    args = parse_args()
    selection_root = args.selection_root.resolve()
    records = load_confirmation_records(selection_root)
    for record in records:
        validate_record(record)

    selected: dict[str, dict] = {}
    ranking_rows: list[dict] = []
    for family in MODEL_FAMILIES:
        family_records = [row for row in records if row.get("model_kind") == family]
        if not family_records:
            raise RuntimeError(f"No confirmation evidence for {family}")
        ranked = sorted(family_records, key=rank_key)
        winner = ranked[0]
        selected[family] = {
            "candidate_id": winner["candidate_id"],
            "config": winner["config"],
            "confirmation_seed": int(winner["seed"]),
            "confirmation_oof_macro_f1": float(winner["oof_macro_f1"]),
            "confirmation_oof_log_loss": float(winner["oof_log_loss"]),
            "confirmation_fold_macro_f1_std": float(np.std(winner["fold_macro_f1"], ddof=1)),
            "confirmation_derived_final_epochs": int(winner["derived_final_epochs"]),
            "evidence_path": winner["_evidence_path"],
            "evidence_sha256": winner["_evidence_sha256"],
        }
        for rank, row in enumerate(ranked, start=1):
            ranking_rows.append(
                {
                    "model_kind": family,
                    "rank": rank,
                    "candidate_id": row["candidate_id"],
                    "oof_macro_f1": float(row["oof_macro_f1"]),
                    "fold_macro_f1_std": float(np.std(row["fold_macro_f1"], ddof=1)),
                    "oof_log_loss": float(row["oof_log_loss"]),
                    "selected": rank == 1,
                }
            )

    protocol_path = selection_root / "configs" / "selection_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)

    lock = {
        "status": "LOCKED_BEFORE_FINAL_TEST",
        "selection_runner": Path(__file__).name,
        "selection_runner_sha256": sha256_file(Path(__file__).resolve()),
        "selection_metric": "pooled_oof_macro_f1",
        "tie_breakers": ["lower_fold_macro_f1_std", "lower_oof_log_loss", "candidate_id"],
        "selection_stage": "five_fold_confirmation",
        "test_used_for_selection": False,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "selected": selected,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2)
    pd.DataFrame(ranking_rows).to_csv(args.output.with_suffix(".csv"), index=False)
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
