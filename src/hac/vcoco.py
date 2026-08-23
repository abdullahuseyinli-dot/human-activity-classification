"""V-COCO ontology helpers for the external-transfer study.

The legacy benchmark requires mutually exclusive activity labels. V-COCO does
not: a person can be annotated as both ``stand`` and ``walk`` or ``run``. This
module preserves those source annotations and exposes a factorized view without
changing the completed v1 mapping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

TARGET_ACTIONS = frozenset({"sit", "stand", "walk", "run"})


@dataclass(frozen=True, slots=True)
class FactorizedActivity:
    """Source-derived posture, motion and gait targets for one V-COCO person."""

    posture: str
    motion: str
    gait: str
    label_3: str
    label_4: str
    legacy_eligible: bool
    factorized_clear: bool

    def as_dict(self) -> dict:
        return asdict(self)


def factorize_actions(actions: set[str] | frozenset[str]) -> FactorizedActivity:
    """Translate V-COCO target verbs while retaining compatible co-labels.

    The factor labels are derived from V-COCO tags, not from a new human visual
    audit. Simultaneous ``sit`` and ``stand`` is therefore kept as ambiguous
    rather than silently removed or adjudicated.
    """

    observed = frozenset(str(value) for value in actions)
    if not observed or not observed.issubset(TARGET_ACTIONS):
        raise ValueError(f"Unexpected V-COCO target actions: {sorted(observed)}")

    has_sit = "sit" in observed
    has_stand = "stand" in observed
    dynamic = observed & {"walk", "run"}

    if has_sit and has_stand:
        posture = "ambiguous"
    elif has_sit:
        posture = "seated"
    else:
        # Walking and running imply an upright posture even when V-COCO does
        # not redundantly attach the stand action.
        posture = "upright"

    if dynamic:
        motion = "locomoting"
    elif has_sit or has_stand:
        motion = "ambiguous" if has_sit and has_stand else "stationary"
    else:  # pragma: no cover - guarded by the target-action validation above
        motion = "ambiguous"

    if dynamic == {"walk"}:
        gait = "walking"
    elif dynamic == {"run"}:
        gait = "running"
    elif dynamic == {"walk", "run"}:
        gait = "ambiguous"
    else:
        gait = "not_applicable"

    legacy_eligible = not (has_sit and has_stand)
    if dynamic:
        label_3 = "walking_running"
        label_4 = gait if gait in {"walking", "running"} else ""
    elif has_sit and not has_stand:
        label_3 = "sitting"
        label_4 = "sitting"
    elif has_stand and not has_sit:
        label_3 = "standing"
        label_4 = "standing"
    else:
        label_3 = ""
        label_4 = ""

    return FactorizedActivity(
        posture=posture,
        motion=motion,
        gait=gait,
        label_3=label_3,
        label_4=label_4,
        legacy_eligible=legacy_eligible,
        factorized_clear=posture != "ambiguous" and motion != "ambiguous",
    )
