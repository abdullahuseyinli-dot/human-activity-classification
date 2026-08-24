"""Protocol and annotation primitives for the V-COCO v3 research line."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

POSTURE_LABELS = frozenset({"seated", "upright", "other", "indeterminate"})
TRANSLATION_LABELS = frozenset({"stationary", "locomoting", "transition", "not_inferable"})
GAIT_LABELS = frozenset({"walking", "running", "not_applicable", "indeterminate"})
VISIBILITY_LABELS = frozenset({"clear", "occluded", "truncated", "too_small"})
ANNOTATION_GUIDE_VERSION = "v3-pilot-1"

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def load_protocol_spec(path: str | Path) -> dict:
    """Load and validate the human-readable v3 protocol specification."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_protocol_spec(payload)
    return payload


def validate_protocol_spec(payload: Mapping) -> None:
    """Reject protocol drift that would weaken the declared research gates."""

    if payload.get("protocol_version") != "3.0.0-dev.2":
        raise ValueError("Expected the amended v3 development protocol")
    reference = payload.get("frozen_reference", {})
    if reference.get("official_test_status") != "CONSUMED_EXPLORATORY_ONLY":
        raise ValueError("The V-COCO v2 test must remain consumed and exploratory")
    if reference.get("git_tag") != "polar-study-v2.0.0":
        raise ValueError("The v2 release tag changed")
    if payload.get("development_policy", {}).get("current_official_test_may_select_models"):
        raise ValueError("The consumed official test cannot select v3 models")

    ontology = payload.get("ontology", {})
    pilot = ontology.get("development_pilot", {})
    if int(pilot.get("primary_task_presentations", 0)) != 130:
        raise ValueError("The development pilot must use the fixed 130-task prefix")
    if pilot.get("selection_rule") != "first_130_by_blind_manifest_display_order":
        raise ValueError("The development pilot selection rule changed")
    if int(pilot.get("annotators", 0)) != 1:
        raise ValueError("The descriptive development pilot uses one blinded annotator")
    if not pilot.get("predictions_and_source_tags_blinded"):
        raise ValueError("The development pilot must remain prediction- and source-tag-blind")
    if pilot.get("labels_used_for_candidate_selection"):
        raise ValueError("Development pilot labels cannot select model candidates")

    gate = ontology.get("annotation_gate", {})
    if int(gate.get("minimum_independent_annotators", 0)) < 2:
        raise ValueError("At least two independent annotators are required")
    if not gate.get("adjudication_required"):
        raise ValueError("Annotation disagreements must be adjudicated")
    required_axes = set(gate.get("required_alpha_axes", ()))
    expected_axes = {"posture", "visible_translation", "gait", "visibility"}
    if required_axes != expected_axes:
        raise ValueError("All four annotation axes must pass the agreement gate")
    if float(gate.get("minimum_alpha_each_required_axis", 0.0)) < 0.8:
        raise ValueError("The per-axis agreement gate cannot be below 0.80")
    if float(gate.get("minimum_intrarater_exact_agreement_each_axis", 0.0)) < 0.9:
        raise ValueError("The intra-rater agreement gate cannot be below 0.90")
    if float(gate.get("required_completion_fraction", 0.0)) != 1.0:
        raise ValueError("Each independent annotation pass must be complete")
    if not gate.get("predictions_and_source_tags_blinded"):
        raise ValueError("Annotation must remain prediction- and source-tag-blind")

    endpoints = payload.get("endpoints", {})
    if endpoints.get("development_selection") != "source_tag_macro_f1_named_explicitly":
        raise ValueError("Development selection must name its source-tag endpoint")
    if endpoints.get("human_pilot") != "descriptive_ontology_and_error_mechanism_audit_only":
        raise ValueError("The consumed-test human pilot cannot select candidates")
    if endpoints.get("independent_confirmation") != "human_harmonized_macro_f1":
        raise ValueError("Independent confirmation must use harmonized labels")

    confirmation = payload.get("confirmation", {})
    if confirmation.get("current_vcoco_test_eligible"):
        raise ValueError("V-COCO v2 test cannot be reused for confirmation")
    if not confirmation.get("single_open_after_lock"):
        raise ValueError("The new confirmation set must use a one-open gate")


def validate_annotator_id(value: str) -> str:
    """Return a safe local annotator identifier."""

    normalized = str(value).strip()
    if not _SAFE_IDENTIFIER.fullmatch(normalized):
        raise ValueError(
            "Annotator ID must start with a letter or number and contain only "
            "letters, numbers, dot, underscore, or hyphen"
        )
    return normalized


def validate_annotation(payload: Mapping, valid_task_ids: Iterable[str]) -> dict:
    """Validate and normalize one blinded annotation response."""

    valid_tasks = set(map(str, valid_task_ids))
    task_id = str(payload.get("task_id", "")).strip()
    if task_id not in valid_tasks:
        raise ValueError("Unknown annotation task")

    annotator_id = validate_annotator_id(str(payload.get("annotator_id", "")))
    normalized = {
        "task_id": task_id,
        "annotator_id": annotator_id,
        "posture": str(payload.get("posture", "")).strip(),
        "visible_translation": str(payload.get("visible_translation", "")).strip(),
        "gait": str(payload.get("gait", "")).strip(),
        "visibility": str(payload.get("visibility", "")).strip(),
        "notes": str(payload.get("notes", "")).strip(),
        "guide_version": str(payload.get("guide_version", ANNOTATION_GUIDE_VERSION)).strip(),
    }
    allowed = {
        "posture": POSTURE_LABELS,
        "visible_translation": TRANSLATION_LABELS,
        "gait": GAIT_LABELS,
        "visibility": VISIBILITY_LABELS,
    }
    for field, choices in allowed.items():
        if normalized[field] not in choices:
            raise ValueError(f"Invalid {field}: {normalized[field]!r}")
    if len(normalized["notes"]) > 1000:
        raise ValueError("Notes must not exceed 1000 characters")
    if normalized["guide_version"] != ANNOTATION_GUIDE_VERSION:
        raise ValueError("Unknown annotation guide version")
    if normalized["visible_translation"] != "locomoting" and normalized["gait"] in {
        "walking",
        "running",
    }:
        raise ValueError("Walking or running gait requires visible locomotion")
    return normalized


def nominal_krippendorff_alpha(units: Iterable[Iterable[str | None]]) -> float:
    """Compute Krippendorff's alpha for nominal labels.

    Each element in ``units`` contains the available ratings for one item. Missing
    values may be represented by ``None`` or an empty string. Units with fewer than
    two ratings do not contribute.
    """

    coincidence: Counter[tuple[str, str]] = Counter()
    for raw_labels in units:
        labels = [str(value) for value in raw_labels if value not in {None, ""}]
        count = len(labels)
        if count < 2:
            continue
        frequencies = Counter(labels)
        for left, left_count in frequencies.items():
            for right, right_count in frequencies.items():
                pairs = left_count * (right_count - (1 if left == right else 0))
                coincidence[(left, right)] += pairs / (count - 1)

    categories = sorted({value for pair in coincidence for value in pair})
    total = float(sum(coincidence.values()))
    if total <= 1.0 or not categories:
        return math.nan

    observed_disagreement = 1.0 - (
        sum(coincidence[(category, category)] for category in categories) / total
    )
    marginals = {
        category: sum(coincidence[(category, other)] for other in categories)
        for category in categories
    }
    expected_disagreement = (total * total - sum(value * value for value in marginals.values())) / (
        total * (total - 1.0)
    )
    if expected_disagreement <= 0.0:
        return 1.0 if observed_disagreement <= 0.0 else math.nan
    return float(1.0 - observed_disagreement / expected_disagreement)
