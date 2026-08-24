"""Blinded sampling and local annotation service for the V-COCO v3 pilot."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import mimetypes
import os
import threading
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from hac.vcoco_v3 import (
    ANNOTATION_GUIDE_VERSION,
    validate_annotation,
    validate_annotator_id,
)

CLASS_NAMES = ("sitting", "standing", "walking_running")
BLIND_COLUMNS = (
    "task_id",
    "display_order",
    "image_path",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "image_width",
    "image_height",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _allocate_stratified_counts(frame: pd.DataFrame, strata: list[str], total: int) -> dict:
    groups = list(frame.groupby(strata, dropna=False, sort=True).groups)
    if not groups:
        raise ValueError("No sampling cells are available")
    base, remainder = divmod(total, len(groups))
    return {key: base + (index < remainder) for index, key in enumerate(groups)}


def stratified_sample(
    frame: pd.DataFrame,
    *,
    total: int,
    strata: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw a deterministic row-level stratified sample and record design weights."""

    if total < 1 or total > len(frame):
        raise ValueError("Requested sample size is outside the candidate cohort")
    allocations = _allocate_stratified_counts(frame, strata, total)
    selected_parts = []
    selected_indices: set[int] = set()
    grouped = frame.groupby(strata, dropna=False, sort=True)
    for key, group in grouped:
        requested = min(int(allocations[key]), len(group))
        if requested == 0:
            continue
        indices = rng.choice(group.index.to_numpy(), size=requested, replace=False)
        selected_indices.update(map(int, indices))
        part = frame.loc[indices].copy()
        part["sampling_cell"] = json.dumps(key if isinstance(key, tuple) else [key])
        part["sampling_probability"] = requested / len(group)
        part["design_weight"] = len(group) / requested
        selected_parts.append(part)

    shortfall = total - len(selected_indices)
    if shortfall:
        remaining = frame.loc[~frame.index.isin(selected_indices)]
        extra_indices = rng.choice(remaining.index.to_numpy(), size=shortfall, replace=False)
        extra = remaining.loc[extra_indices].copy()
        extra["sampling_cell"] = "FILL"
        extra["sampling_probability"] = shortfall / len(remaining)
        extra["design_weight"] = len(remaining) / shortfall
        selected_parts.append(extra)
    return pd.concat(selected_parts, ignore_index=True)


class MappingLike(Protocol):
    """Subset shared by numpy ``NpzFile`` and mapping-based test fixtures."""

    def __getitem__(self, key: str): ...


def prepare_prediction_frame(manifest: pd.DataFrame, predictions: MappingLike) -> pd.DataFrame:
    """Align a manifest with locked predictions and derive sampling strata."""

    person_ids = np.asarray(predictions["person_ids"]).astype(str)
    labels = np.asarray(predictions["labels"], dtype=int)
    probabilities = np.asarray(predictions["scale_conditioned_stacking"], dtype=float)
    if probabilities.shape != (len(person_ids), len(CLASS_NAMES)):
        raise ValueError("Unexpected prediction tensor shape")
    index = {person_id: row for row, person_id in enumerate(person_ids)}
    rows = manifest.copy()
    rows["person_id"] = rows["person_id"].astype(str)
    if set(rows["person_id"]) != set(person_ids):
        raise ValueError("Prediction and manifest person IDs differ")
    order = np.asarray([index[value] for value in rows["person_id"]], dtype=int)
    labels = labels[order]
    probabilities = probabilities[order]
    predictions_index = probabilities.argmax(axis=1)
    rows["label_index"] = labels
    rows["predicted_index"] = predictions_index
    rows["predicted_label"] = [CLASS_NAMES[value] for value in predictions_index]
    rows["confidence"] = probabilities.max(axis=1)
    for class_index, class_name in enumerate(CLASS_NAMES):
        rows[f"probability_{class_name}"] = probabilities[:, class_index]
    rows["error_type"] = "correct"
    rows.loc[(labels == 1) & (predictions_index == 2), "error_type"] = "standing_to_locomotion"
    rows.loc[(labels == 2) & (predictions_index == 1), "error_type"] = "locomotion_to_standing"
    rows.loc[(labels != predictions_index) & rows["error_type"].eq("correct"), "error_type"] = (
        "other_error"
    )
    rows["area_quartile"] = pd.qcut(
        rows["bbox_area_fraction"], 4, labels=["Q1", "Q2", "Q3", "Q4"]
    ).astype(str)
    rows["confidence_quartile"] = pd.qcut(
        rows["confidence"], 4, labels=["Q1", "Q2", "Q3", "Q4"]
    ).astype(str)
    rows["touches_boundary"] = (
        rows["bbox_xmin"].le(0.5)
        | rows["bbox_ymin"].le(0.5)
        | rows["bbox_xmax"].ge(rows["image_width"] - 0.5)
        | rows["bbox_ymax"].ge(rows["image_height"] - 0.5)
    )
    occupancy = rows.groupby("image_id")["person_id"].transform("size")
    rows["scene_occupancy"] = np.where(occupancy.eq(1), "single", "multiple")
    return rows


def create_pilot_tasks(
    rows: pd.DataFrame,
    *,
    probability_tasks: int,
    error_tasks: int,
    repeat_tasks: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build blind and private task manifests for the annotation pilot."""

    if min(probability_tasks, error_tasks, repeat_tasks) < 0:
        raise ValueError("Task counts cannot be negative")
    rng = np.random.default_rng(seed)
    probability = stratified_sample(
        rows,
        total=probability_tasks,
        strata=["label_3", "area_quartile"],
        rng=rng,
    )
    probability["cohort"] = "probability_sample"

    error_pool = rows[
        rows["error_type"].isin({"standing_to_locomotion", "locomotion_to_standing"})
        & ~rows["person_id"].isin(set(probability["person_id"]))
    ].copy()
    errors = stratified_sample(
        error_pool,
        total=error_tasks,
        strata=["error_type", "confidence_quartile"],
        rng=rng,
    )
    errors["cohort"] = "error_enriched"
    errors["sampling_probability"] = np.nan
    errors["design_weight"] = np.nan

    unique = pd.concat([probability, errors], ignore_index=True)
    unique["repeat_of_task_id"] = ""
    unique["task_id"] = [
        "v3p-" + hashlib.sha256(f"{seed}:{row.person_id}:{row.cohort}".encode()).hexdigest()[:12]
        for row in unique.itertuples()
    ]
    if repeat_tasks > len(unique):
        raise ValueError("Repeat task count exceeds unique task count")
    repeats = unique.loc[
        rng.choice(unique.index.to_numpy(), size=repeat_tasks, replace=False)
    ].copy()
    repeats["repeat_of_task_id"] = repeats["task_id"]
    repeats["cohort"] = "intrarater_repeat"
    repeats["sampling_probability"] = np.nan
    repeats["design_weight"] = np.nan
    repeats["task_id"] = [
        "v3r-" + hashlib.sha256(f"{seed}:{value}:repeat".encode()).hexdigest()[:12]
        for value in repeats["repeat_of_task_id"]
    ]
    tasks = pd.concat([unique, repeats], ignore_index=True)

    # Keep repeated items apart so they are not obvious to the annotator.
    for _ in range(200):
        order = rng.permutation(len(tasks))
        candidate = tasks.iloc[order].reset_index(drop=True)
        positions = {task_id: index for index, task_id in enumerate(candidate["task_id"])}
        separation = [
            abs(index - positions[repeat_of])
            for index, repeat_of in enumerate(candidate["repeat_of_task_id"])
            if repeat_of
        ]
        if not separation or min(separation) >= min(25, len(tasks) // 5):
            tasks = candidate
            break
    tasks["display_order"] = np.arange(1, len(tasks) + 1)

    missing = set(BLIND_COLUMNS) - set(tasks)
    if missing:
        raise ValueError(f"Annotation tasks lack display columns: {sorted(missing)}")
    blind = tasks[list(BLIND_COLUMNS)].copy()
    private = tasks.copy()
    return blind, private


class AnnotationStore:
    """Atomic, resumable storage for local independent annotation sessions."""

    def __init__(self, output_dir: Path, tasks: pd.DataFrame):
        self.output_dir = output_dir.resolve()
        self.annotation_dir = self.output_dir / "annotations"
        self.event_dir = self.output_dir / "events"
        self.annotation_dir.mkdir(parents=True, exist_ok=True)
        self.event_dir.mkdir(parents=True, exist_ok=True)
        self.tasks = tasks.sort_values("display_order", ignore_index=True)
        self.task_ids = tuple(self.tasks["task_id"].astype(str))
        self._lock = threading.Lock()

    def _snapshot_path(self, annotator_id: str) -> Path:
        return self.annotation_dir / f"{validate_annotator_id(annotator_id)}.json"

    def load(self, annotator_id: str) -> dict[str, dict]:
        path = self._snapshot_path(annotator_id)
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("annotator_id") != validate_annotator_id(annotator_id):
            raise RuntimeError("Annotation snapshot identity mismatch")
        return {str(row["task_id"]): row for row in payload.get("annotations", [])}

    def save(self, payload: dict) -> dict:
        annotation = validate_annotation(payload, self.task_ids)
        annotation["saved_at_utc"] = utc_now()
        annotator_id = annotation["annotator_id"]
        with self._lock:
            current = self.load(annotator_id)
            revision = int(current.get(annotation["task_id"], {}).get("revision", 0)) + 1
            annotation["revision"] = revision
            current[annotation["task_id"]] = annotation
            ordered = [current[task] for task in self.task_ids if task in current]
            snapshot = {
                "status": "VCOCO_V3_BLINDED_ANNOTATION_IN_PROGRESS",
                "annotator_id": annotator_id,
                "task_manifest_rows": len(self.task_ids),
                "completed_rows": len(ordered),
                "updated_at_utc": annotation["saved_at_utc"],
                "annotations": ordered,
            }
            path = self._snapshot_path(annotator_id)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
            with (self.event_dir / f"{annotator_id}.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(annotation, sort_keys=True, allow_nan=False) + "\n")
        return annotation

    def export_csv(self, annotator_id: str) -> bytes:
        annotations = self.load(annotator_id)
        fields = [
            "task_id",
            "annotator_id",
            "posture",
            "visible_translation",
            "gait",
            "visibility",
            "notes",
            "guide_version",
            "saved_at_utc",
            "revision",
        ]
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        for task_id in self.task_ids:
            if task_id in annotations:
                writer.writerow({field: annotations[task_id].get(field, "") for field in fields})
        return buffer.getvalue().encode("utf-8")


class AnnotationApplication:
    """State shared by the annotation HTTP request handlers."""

    def __init__(self, manifest_path: Path, output_dir: Path, static_dir: Path):
        self.manifest_path = manifest_path.resolve()
        self.tasks = pd.read_csv(self.manifest_path, dtype={"task_id": str})
        missing = set(BLIND_COLUMNS) - set(self.tasks)
        if missing:
            raise ValueError(f"Blind manifest lacks columns: {sorted(missing)}")
        if self.tasks["task_id"].duplicated().any():
            raise ValueError("Blind task IDs must be unique")
        self.by_id = self.tasks.set_index("task_id", drop=False)
        self.store = AnnotationStore(output_dir, self.tasks)
        self.static_dir = static_dir.resolve()

    def public_state(self, annotator_id: str) -> dict:
        annotator_id = validate_annotator_id(annotator_id)
        annotations = self.store.load(annotator_id)
        return {
            "annotator_id": annotator_id,
            "guide_version": ANNOTATION_GUIDE_VERSION,
            "total": len(self.tasks),
            "completed": len(annotations),
            "tasks": [
                {
                    "task_id": str(row.task_id),
                    "display_order": int(row.display_order),
                    "annotation": annotations.get(str(row.task_id)),
                }
                for row in self.tasks.sort_values("display_order").itertuples()
            ],
        }

    def render_image(self, task_id: str, view: str) -> bytes:
        if task_id not in self.by_id.index:
            raise KeyError(task_id)
        row = self.by_id.loc[task_id]
        image_path = Path(str(row["image_path"]))
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        x1, y1, x2, y2 = (
            float(row["bbox_xmin"]),
            float(row["bbox_ymin"]),
            float(row["bbox_xmax"]),
            float(row["bbox_ymax"]),
        )
        if view == "person":
            output = image.crop(
                (max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2))
            )
        elif view == "context":
            width, height = x2 - x1, y2 - y1
            crop = (
                max(0, x1 - 0.35 * width),
                max(0, y1 - 0.35 * height),
                min(image.width, x2 + 0.35 * width),
                min(image.height, y2 + 0.35 * height),
            )
            output = image.crop(crop)
            draw = ImageDraw.Draw(output)
            line_width = max(3, round(min(output.size) / 100))
            draw.rectangle(
                (x1 - crop[0], y1 - crop[1], x2 - crop[0], y2 - crop[1]),
                outline=(255, 190, 38),
                width=line_width,
            )
        elif view == "full":
            output = image.copy()
            draw = ImageDraw.Draw(output)
            line_width = max(3, round(min(output.size) / 100))
            draw.rectangle((x1, y1, x2, y2), outline=(255, 190, 38), width=line_width)
        else:
            raise ValueError("Unknown image view")
        buffer = io.BytesIO()
        output.save(buffer, format="JPEG", quality=92, optimize=True)
        return buffer.getvalue()


def make_handler(application: AnnotationApplication):
    """Create a request handler bound to one annotation application."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "VCOCOAnnotation/1.0"

        def log_message(self, format_string: str, *args) -> None:
            print(f"annotation-ui: {format_string % args}", flush=True)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            self._send(
                status,
                json.dumps(payload, allow_nan=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/state":
                    annotator_id = parse_qs(parsed.query).get("annotator_id", [""])[0]
                    self._json(HTTPStatus.OK, application.public_state(annotator_id))
                    return
                if parsed.path == "/api/export":
                    annotator_id = parse_qs(parsed.query).get("annotator_id", [""])[0]
                    body = application.store.export_csv(annotator_id)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.send_header(
                        "Content-Disposition", f'attachment; filename="{annotator_id}.csv"'
                    )
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path.startswith("/media/"):
                    parts = parsed.path.strip("/").split("/")
                    if len(parts) != 3:
                        raise KeyError(parsed.path)
                    body = application.render_image(parts[1], parts[2])
                    self._send(HTTPStatus.OK, body, "image/jpeg")
                    return
                relative = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
                static_path = (application.static_dir / relative).resolve()
                if application.static_dir not in static_path.parents:
                    raise KeyError(parsed.path)
                body = static_path.read_bytes()
                content_type = (
                    mimetypes.guess_type(static_path.name)[0] or "application/octet-stream"
                )
                self._send(HTTPStatus.OK, body, f"{content_type}; charset=utf-8")
            except (KeyError, FileNotFoundError):
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            except (ValueError, RuntimeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/annotation":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 2 or length > 65_536:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                annotation = application.store.save(payload)
                self._json(HTTPStatus.OK, {"saved": annotation})
            except (json.JSONDecodeError, ValueError, RuntimeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    return Handler


def serve_annotation_app(
    manifest_path: Path,
    output_dir: Path,
    static_dir: Path,
    *,
    host: str,
    port: int,
) -> None:
    application = AnnotationApplication(manifest_path, output_dir, static_dir)
    server = ThreadingHTTPServer((host, port), make_handler(application))
    print(f"V-COCO annotation interface: http://{host}:{port}", flush=True)
    server.serve_forever()
