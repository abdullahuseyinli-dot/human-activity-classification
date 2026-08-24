"""Audit the sealed Okutama development archive and select temporal centres."""

from __future__ import annotations

import argparse
import json
import shlex
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from hac.polar import sha256_file

BASE_ACTIONS = {"Sitting", "Standing", "Walking", "Running"}
TARGET_LABEL = {
    "Sitting": "sitting",
    "Standing": "standing",
    "Walking": "walking_running",
    "Running": "walking_running",
}
WINDOW_OFFSETS = tuple(np.rint(np.arange(-8, 9) * 30.0 / 16.0).astype(int))


@dataclass(frozen=True)
class Annotation:
    track_id: int
    bbox: tuple[int, int, int, int]
    frame: int
    lost: bool
    occluded: bool
    generated: bool
    actions: tuple[str, ...]

    @property
    def base_actions(self) -> tuple[str, ...]:
        return tuple(action for action in self.actions if action in BASE_ACTIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiments/okutama_action_protocol.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".runs/vcoco_v3/okutama/development_audit")
    )
    return parser.parse_args()


def parse_annotation(line: str) -> Annotation:
    fields = shlex.split(line)
    if len(fields) < 10 or fields[9] != "Person":
        raise ValueError(f"Malformed Okutama annotation: {line[:120]}")
    return Annotation(
        track_id=int(fields[0]),
        bbox=tuple(map(int, fields[1:5])),
        frame=int(fields[5]),
        lost=bool(int(fields[6])),
        occluded=bool(int(fields[7])),
        generated=bool(int(fields[8])),
        actions=tuple(fields[10:]),
    )


def annotation_rows(archive: zipfile.ZipFile, path: str) -> list[Annotation]:
    text = archive.read(path).decode("utf-8")
    return [parse_annotation(line) for line in text.splitlines() if line.strip()]


def recording_metadata(recording_id: str) -> tuple[str, str, str]:
    drone, part_of_day, scenario = recording_id.split(".")
    period = {"1": "morning", "2": "noon"}.get(part_of_day)
    if period is None or drone not in {"1", "2"}:
        raise ValueError(f"Unexpected Okutama recording identifier: {recording_id}")
    return f"{part_of_day}.{scenario}", f"drone_{drone}", period


def expected_frame_path(recording_id: str, frame: int) -> str:
    drone, part_of_day, _ = recording_id.split(".")
    period = {"1": "Morning", "2": "Noon"}[part_of_day]
    return (
        f"Drone{drone}/{period}/Extracted-Frames-1280x720/"
        f"{recording_id}/{frame}.jpg"
    )


def box_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    xmin = max(left[0], right[0])
    ymin = max(left[1], right[1])
    xmax = min(left[2], right[2])
    ymax = min(left[3], right[3])
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return float(intersection / union) if union else 0.0


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    archive_path = args.archive.resolve()
    protocol_path = args.protocol.resolve()
    output_dir = args.output_dir.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != (
        "DECLARED_BEFORE_OKUTAMA_AGGREGATE_LABEL_AUDIT_OR_MODEL_FITTING"
    ):
        raise RuntimeError("The Okutama protocol is not in its pre-audit state")
    declaration = protocol["archives"]["development"]
    if archive_path.name != declaration["file_name"]:
        raise RuntimeError("The development archive name differs from the declaration")
    if archive_path.stat().st_size != int(declaration["expected_bytes"]):
        raise RuntimeError("The development archive byte count differs from the declaration")
    archive_hash = sha256_file(archive_path)
    if archive_hash != declaration["sha256"]:
        raise RuntimeError("The development archive hash differs from the declaration")

    selected_rows = []
    aggregate_actions: Counter[str] = Counter()
    base_action_rows: Counter[str] = Counter()
    join_mismatches = 0
    lost_rows_without_tracking = 0
    unmatched_tracking_rows = 0
    boundary_overlap_rows = 0
    boundary_base_conflicts = 0
    ambiguous_base_rows = 0
    missing_base_rows = 0
    missing_frame_members = 0
    recording_evidence = []
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("The Okutama development ZIP failed its CRC audit")
        members = set(archive.namelist())
        multi_paths = sorted(
            name
            for name in members
            if name.startswith("Labels/MultiActionLabels/3840x2160/")
            and PurePosixPath(name).suffix == ".txt"
        )
        tracking_paths = {
            PurePosixPath(name).name: name
            for name in members
            if name.startswith("Labels/SingleActionTrackingLabels/3840x2160/")
            and PurePosixPath(name).suffix == ".txt"
        }
        if not multi_paths or len(multi_paths) != len(tracking_paths):
            raise RuntimeError("Okutama multi-action and tracking label sets do not align")

        for multi_path in multi_paths:
            file_name = PurePosixPath(multi_path).name
            recording_id = PurePosixPath(multi_path).stem
            tracking_path = tracking_paths.get(file_name)
            if tracking_path is None:
                raise RuntimeError(f"Missing tracking labels for {recording_id}")
            multi = annotation_rows(archive, multi_path)
            tracking = annotation_rows(archive, tracking_path)
            tracking_by_key: dict[tuple, Annotation] = {}
            tracking_by_frame: dict[int, list[Annotation]] = {}
            for row in tracking:
                key = (row.frame, row.bbox, row.occluded, row.generated)
                if key in tracking_by_key:
                    raise RuntimeError(f"Duplicate tracking join key in {recording_id}")
                tracking_by_key[key] = row
                tracking_by_frame.setdefault(row.frame, []).append(row)

            by_track_frame: dict[tuple[int, int], Annotation] = {}
            recording_action_rows: Counter[str] = Counter()
            matched_tracking_keys = set()
            for source in multi:
                if source.lost:
                    lost_rows_without_tracking += 1
                    continue
                join_key = (source.frame, source.bbox, source.occluded, source.generated)
                identity = tracking_by_key.get(join_key)
                if identity is None:
                    nearest = max(
                        tracking_by_frame.get(source.frame, ()),
                        key=lambda row: box_iou(source.bbox, row.bbox),
                        default=None,
                    )
                    overlap = box_iou(source.bbox, nearest.bbox) if nearest is not None else 0.0
                    if source.frame % 180 == 0 and overlap >= 0.2:
                        boundary_overlap_rows += 1
                        source_base = set(source.base_actions)
                        tracking_base = set(nearest.base_actions)
                        if source_base and tracking_base and source_base != tracking_base:
                            boundary_base_conflicts += 1
                    else:
                        join_mismatches += 1
                    continue
                matched_tracking_keys.add(join_key)
                base = source.base_actions
                aggregate_actions.update(source.actions)
                recording_action_rows.update(source.actions)
                if len(base) == 0:
                    missing_base_rows += 1
                elif len(base) > 1:
                    ambiguous_base_rows += 1
                else:
                    base_action_rows[base[0]] += 1
                key = (identity.track_id, source.frame)
                if key in by_track_frame:
                    raise RuntimeError(f"Duplicate stable track/frame row in {recording_id}")
                by_track_frame[key] = Annotation(
                    track_id=identity.track_id,
                    bbox=source.bbox,
                    frame=source.frame,
                    lost=source.lost,
                    occluded=source.occluded,
                    generated=source.generated,
                    actions=source.actions,
                )
            unmatched_tracking_rows += len(set(tracking_by_key) - matched_tracking_keys)

            scenario_id, drone_view, part_of_day = recording_metadata(recording_id)
            recording_selected = 0
            for (track_id, frame), center in sorted(by_track_frame.items()):
                if (
                    frame % 30
                    or frame % 180 == 0
                    or center.generated
                    or center.lost
                    or len(center.base_actions) != 1
                ):
                    continue
                window = [by_track_frame.get((track_id, frame + offset)) for offset in WINDOW_OFFSETS]
                if any(row is None or row.lost for row in window):
                    continue
                frame_paths = [
                    expected_frame_path(recording_id, frame + offset)
                    for offset in WINDOW_OFFSETS
                ]
                if any(path not in members for path in frame_paths):
                    missing_frame_members += 1
                    continue
                base_action = center.base_actions[0]
                window_base = {
                    action
                    for row in window
                    for action in row.base_actions
                }
                xmin, ymin, xmax, ymax = center.bbox
                selected_rows.append(
                    {
                        "sample_id": (
                            f"train__{recording_id}__track-{track_id}__frame-{frame:06d}"
                        ),
                        "provider_partition": "train",
                        "recording_id": recording_id,
                        "scenario_id": scenario_id,
                        "drone_view": drone_view,
                        "part_of_day": part_of_day,
                        "track_id": str(track_id),
                        "center_frame": frame,
                        "provider_base_action": base_action,
                        "label": TARGET_LABEL[base_action],
                        "walking_running_subtype": (
                            base_action.lower()
                            if base_action in {"Walking", "Running"}
                            else "not_applicable"
                        ),
                        "transition_window": len(window_base) > 1,
                        "center_occluded": center.occluded,
                        "window_any_occluded": any(row.occluded for row in window),
                        "bbox_xmin": xmin / 3840.0,
                        "bbox_ymin": ymin / 2160.0,
                        "bbox_xmax": xmax / 3840.0,
                        "bbox_ymax": ymax / 2160.0,
                        "bbox_area_fraction": ((xmax - xmin) * (ymax - ymin)) / (3840 * 2160),
                        "window_frames": ";".join(str(frame + value) for value in WINDOW_OFFSETS),
                        "window_boxes_1280x720": ";".join(
                            ",".join(f"{coordinate / 3.0:.6f}" for coordinate in row.bbox)
                            for row in window
                        ),
                        "window_occluded": ";".join(
                            "1" if row.occluded else "0" for row in window
                        ),
                    }
                )
                recording_selected += 1
            recording_evidence.append(
                {
                    "recording_id": recording_id,
                    "scenario_id": scenario_id,
                    "annotation_rows": len(multi),
                    "selected_centres": recording_selected,
                    "action_rows": dict(sorted(recording_action_rows.items())),
                }
            )

    if join_mismatches:
        raise RuntimeError(f"Tracking/multi-action join mismatches: {join_mismatches}")
    if unmatched_tracking_rows:
        raise RuntimeError(f"Tracking rows without a multi-action match: {unmatched_tracking_rows}")
    if missing_frame_members:
        raise RuntimeError(f"Selected windows reference missing frames: {missing_frame_members}")
    centres = pd.DataFrame(selected_rows).sort_values(
        ["scenario_id", "recording_id", "track_id", "center_frame"], ignore_index=True
    )
    if centres.empty or centres["sample_id"].duplicated().any():
        raise RuntimeError("Okutama centre selection is empty or non-unique")
    if set(centres["label"]) != set(TARGET_LABEL.values()):
        raise RuntimeError("Selected Okutama centres do not contain every target class")

    output_dir.mkdir(parents=True, exist_ok=True)
    centres_path = output_dir / "development_centres.csv"
    centres.to_csv(centres_path, index=False)
    counts = (
        centres.groupby(["label", "provider_base_action"], observed=True)
        .size()
        .rename("samples")
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "status": "OKUTAMA_DEVELOPMENT_ARCHIVE_AND_CENTRES_AUDITED",
        "confirmation_archive_opened": False,
        "development_archive": {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": archive_hash,
        },
        "recordings": len(recording_evidence),
        "scenarios": int(centres["scenario_id"].nunique()),
        "stable_tracks": int(
            (centres["recording_id"] + "::" + centres["track_id"]).nunique()
        ),
        "selected_centres": len(centres),
        "counts": counts,
        "transition_centres": int(centres["transition_window"].sum()),
        "occluded_centres": int(centres["center_occluded"].sum()),
        "windows_with_occlusion": int(centres["window_any_occluded"].sum()),
        "all_action_rows": dict(sorted(aggregate_actions.items())),
        "base_action_rows": dict(sorted(base_action_rows.items())),
        "ambiguous_base_action_rows_quarantined": ambiguous_base_rows,
        "rows_without_base_action": missing_base_rows,
        "lost_rows_without_tracking": lost_rows_without_tracking,
        "boundary_overlap_rows_quarantined": boundary_overlap_rows,
        "boundary_overlap_base_conflicts": boundary_base_conflicts,
        "tracking_join_mismatches": join_mismatches,
        "unmatched_tracking_rows": unmatched_tracking_rows,
        "recording_evidence": recording_evidence,
        "runtime_seconds": time.perf_counter() - started,
        "source_sha256": {
            "okutama_protocol": sha256_file(protocol_path),
            "auditor": sha256_file(Path(__file__).resolve()),
        },
        "artifact_sha256": {centres_path.name: sha256_file(centres_path)},
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
