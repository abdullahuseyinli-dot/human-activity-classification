from __future__ import annotations

from copy import deepcopy

import pytest

from hac.retained_lock import validate_retained_lock, validate_single_field_normalization


def lock_payload() -> dict:
    return {
        "status": "LOCKED",
        "locked_at_utc": "2026-08-24T16:20:56.230728+00:00",
        "source_sha256": {
            "input": "1" * 64,
            "runner": "2" * 64,
            "locker": "3" * 64,
        },
        "candidate_order": ["baseline", "candidate"],
    }


def test_retained_lock_ignores_only_creation_metadata() -> None:
    retained = lock_payload()
    current = deepcopy(retained)
    current["locked_at_utc"] = "2027-01-01T00:00:00+00:00"
    current["source_sha256"]["locker"] = "4" * 64

    check = validate_retained_lock(retained, current)

    assert check.legacy_omissions == ()
    assert check.historical_mismatches == ()


def test_retained_lock_reports_explicit_schema_compatibility() -> None:
    retained = lock_payload()
    current = deepcopy(retained)
    current["source_sha256"].update(
        {
            "candidate_grid": "5" * 64,
            "candidate_grid_lock": "6" * 64,
            "training_module": "7" * 64,
        }
    )
    current["source_sha256"]["runner"] = "8" * 64

    check = validate_retained_lock(
        retained,
        current,
        allowed_legacy_omissions=(
            "source_sha256.candidate_grid",
            "source_sha256.candidate_grid_lock",
            "source_sha256.training_module",
        ),
        allowed_historical_mismatches=("source_sha256.runner",),
    )

    assert check.legacy_omissions == (
        "source_sha256.candidate_grid",
        "source_sha256.candidate_grid_lock",
        "source_sha256.training_module",
    )
    assert check.historical_mismatches == ("source_sha256.runner",)


def test_retained_lock_rejects_unlisted_input_drift() -> None:
    retained = lock_payload()
    current = deepcopy(retained)
    current["source_sha256"]["input"] = "9" * 64

    with pytest.raises(RuntimeError, match="source_sha256.input"):
        validate_retained_lock(retained, current)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("locked_at_utc", "2026-08-24T16:20:56+01:00", "not UTC"),
        ("locked_at_utc", "not-a-time", "invalid creation timestamp"),
        ("source_sha256.locker", "not-a-digest", "invalid locker source digest"),
    ),
)
def test_retained_lock_rejects_invalid_creation_metadata(
    field: str, value: str, message: str
) -> None:
    retained = lock_payload()
    if field == "locked_at_utc":
        retained[field] = value
    else:
        retained["source_sha256"]["locker"] = value

    with pytest.raises(RuntimeError, match=message):
        validate_retained_lock(retained, lock_payload())


def test_single_field_normalization_rejects_any_additional_drift() -> None:
    retained = {
        "input": {"archive": "C:/local/archive.zip"},
        "training": {"seeds": [42, 43]},
    }
    current = {
        "input": {"archive": "data/external/archive.zip"},
        "training": {"seeds": [42, 43]},
    }

    original, normalized = validate_single_field_normalization(
        retained, current, field="input.archive"
    )

    assert original == "C:/local/archive.zip"
    assert normalized == "data/external/archive.zip"
    current["training"]["seeds"] = [42, 44]
    with pytest.raises(RuntimeError, match=r"training\.seeds\[1\]"):
        validate_single_field_normalization(retained, current, field="input.archive")
