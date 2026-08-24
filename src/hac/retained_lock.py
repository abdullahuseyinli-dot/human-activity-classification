"""Validation helpers for immutable experiment lock artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CREATION_TIMESTAMP = "locked_at_utc"
_LOCKER_DIGEST = "source_sha256.locker"


@dataclass(frozen=True)
class RetainedLockCheck:
    """Compatibility details from a successful retained-lock check."""

    legacy_omissions: tuple[str, ...]
    historical_mismatches: tuple[str, ...]


def _parts(path: str) -> tuple[str, ...]:
    parts = tuple(path.split("."))
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid retained-lock field path: {path!r}")
    return parts


def _contains(payload: Mapping[str, Any], path: str) -> bool:
    current: Any = payload
    for part in _parts(path):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def _get(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in _parts(path):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def _delete(payload: dict[str, Any], path: str) -> None:
    parts = _parts(path)
    current = payload
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise KeyError(path)
        current = child
    del current[parts[-1]]


def _set(payload: dict[str, Any], path: str, value: Any) -> None:
    parts = _parts(path)
    current = payload
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise KeyError(path)
        current = child
    current[parts[-1]] = value


def _difference_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(_difference_paths(left[key], right[key], path))
        return differences
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes))
        and isinstance(right, Sequence)
        and not isinstance(right, (str, bytes))
    ):
        if len(left) != len(right):
            return [prefix]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            path = f"{prefix}[{index}]"
            differences.extend(_difference_paths(left_item, right_item, path))
        return differences
    return [] if left == right else [prefix]


def _validate_creation_metadata(payload: Mapping[str, Any]) -> None:
    timestamp = payload.get(_CREATION_TIMESTAMP)
    if not isinstance(timestamp, str):
        raise RuntimeError("Retained lock has no creation timestamp")
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise RuntimeError("Retained lock has an invalid creation timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise RuntimeError("Retained lock creation timestamp is not UTC")

    try:
        locker = _get(payload, _LOCKER_DIGEST)
    except KeyError as error:
        raise RuntimeError("Retained lock has no locker source digest") from error
    if not isinstance(locker, str) or _SHA256_PATTERN.fullmatch(locker) is None:
        raise RuntimeError("Retained lock has an invalid locker source digest")


def validate_retained_lock(
    retained: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    allowed_legacy_omissions: Sequence[str] = (),
    allowed_historical_mismatches: Sequence[str] = (),
) -> RetainedLockCheck:
    """Validate a retained lock without rewriting creation-time metadata.

    Creation timestamp and locker digest identify the original lock operation, so a
    later checker validates their form but does not replace them with current values.
    Explicit compatibility allowances are returned to the caller for reporting.
    """

    _validate_creation_metadata(retained)
    retained_copy = deepcopy(dict(retained))
    current_copy = deepcopy(dict(current))
    for path in (_CREATION_TIMESTAMP, _LOCKER_DIGEST):
        if not _contains(current_copy, path):
            raise ValueError(f"Current lock payload has no {path}")
        _delete(retained_copy, path)
        _delete(current_copy, path)

    omissions: list[str] = []
    for path in allowed_legacy_omissions:
        if not _contains(current_copy, path):
            raise ValueError(f"Current lock payload has no compatibility field {path}")
        if not _contains(retained_copy, path):
            _delete(current_copy, path)
            omissions.append(path)

    historical_mismatches: list[str] = []
    for path in allowed_historical_mismatches:
        if not _contains(current_copy, path):
            raise ValueError(f"Current lock payload has no historical field {path}")
        if _contains(retained_copy, path) and _get(retained_copy, path) != _get(current_copy, path):
            _set(current_copy, path, _get(retained_copy, path))
            historical_mismatches.append(path)

    differences = _difference_paths(retained_copy, current_copy)
    if differences:
        joined = ", ".join(differences[:8])
        suffix = " ..." if len(differences) > 8 else ""
        raise RuntimeError(f"Retained lock differs from current inputs at: {joined}{suffix}")
    return RetainedLockCheck(tuple(omissions), tuple(historical_mismatches))


def validate_single_field_normalization(
    retained: Mapping[str, Any], current: Mapping[str, Any], *, field: str
) -> tuple[Any, Any]:
    """Require two JSON objects to differ at exactly one declared field."""

    if not _contains(retained, field) or not _contains(current, field):
        raise RuntimeError(f"Normalization field is missing: {field}")
    retained_value = _get(retained, field)
    current_value = _get(current, field)
    if retained_value == current_value:
        raise RuntimeError(f"Normalization field did not change: {field}")
    normalized = deepcopy(dict(retained))
    _set(normalized, field, current_value)
    differences = _difference_paths(normalized, current)
    if differences:
        joined = ", ".join(differences[:8])
        suffix = " ..." if len(differences) > 8 else ""
        raise RuntimeError(
            f"Retained and public payloads differ outside {field} at: {joined}{suffix}"
        )
    return retained_value, current_value
