"""POLAR dataset parsing, image views, and leakage-audit primitives."""

from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

SOURCE_TO_LABEL = {
    "sit": "sitting",
    "stand": "standing",
    "walk": "walking",
    "run": "running",
}
LABEL_TO_INDEX = {
    "sitting": 0,
    "standing": 1,
    "walking": 2,
    "running": 3,
}
THREE_CLASS_MAP = {
    "sitting": "sitting",
    "standing": "standing",
    "walking": "walking_running",
    "running": "walking_running",
}
EXPECTED_SPLIT_COUNTS = {"train": 21194, "val": 7065, "test": 7065}
EXPECTED_TARGET_COUNTS = {
    "train": {"sitting": 3043, "standing": 2740, "walking": 2002, "running": 2230},
    "val": {"sitting": 1015, "standing": 980, "walking": 654, "running": 716},
    "test": {"sitting": 1043, "standing": 936, "walking": 641, "running": 739},
}


@dataclass(frozen=True, slots=True)
class PersonTarget:
    source_label: str
    label_4: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass(frozen=True, slots=True)
class PolarAnnotation:
    image_id: str
    filename: str
    original_name: str
    annotated_width: int
    annotated_height: int
    target: PersonTarget | None
    target_record_count: int
    target_label_count: int


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_ids(image_sets_dir: str | Path) -> dict[str, set[str]]:
    root = Path(image_sets_dir)
    splits = {}
    for split in ("train", "val", "test"):
        path = root / f"{split}.txt"
        identifiers = {
            line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        }
        if len(identifiers) != EXPECTED_SPLIT_COUNTS[split]:
            raise ValueError(
                f"POLAR {split} count changed: expected {EXPECTED_SPLIT_COUNTS[split]}, "
                f"found {len(identifiers)}"
            )
        splits[split] = identifiers
    if (splits["train"] & splits["val"]) or (splits["train"] & splits["test"]):
        raise ValueError("POLAR train identifiers overlap a later official split")
    if splits["val"] & splits["test"]:
        raise ValueError("POLAR validation and test identifiers overlap")
    return splits


def _enabled_target_labels(actions: dict) -> list[str]:
    return [name for name in SOURCE_TO_LABEL if int(actions.get(name, 0)) == 1]


def parse_annotation(path: str | Path) -> PolarAnnotation:
    annotation_path = Path(path)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    target_records: list[tuple[str, dict]] = []
    total_target_labels = 0
    for person in payload.get("persons", []):
        labels = _enabled_target_labels(person.get("actions", {}))
        total_target_labels += len(labels)
        for label in labels:
            target_records.append((label, person.get("bndbox", {})))

    target = None
    if len(target_records) == 1 and total_target_labels == 1:
        source_label, box = target_records[0]
        target = PersonTarget(
            source_label=source_label,
            label_4=SOURCE_TO_LABEL[source_label],
            xmin=int(box["xmin"]),
            ymin=int(box["ymin"]),
            xmax=int(box["xmax"]),
            ymax=int(box["ymax"]),
        )
    filename = str(payload["filename"])
    return PolarAnnotation(
        image_id=Path(filename).stem,
        filename=filename,
        original_name=str(payload.get("originalname", "")),
        annotated_width=int(payload["width"]),
        annotated_height=int(payload["height"]),
        target=target,
        target_record_count=len(target_records),
        target_label_count=total_target_labels,
    )


def parse_annotations(
    paths: Iterable[Path], *, workers: int = 8, show_progress: bool = False
) -> list[PolarAnnotation]:
    ordered_paths = list(paths)
    if workers < 1:
        raise ValueError("workers must be positive")
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        results = executor.map(parse_annotation, ordered_paths)
        return list(
            tqdm(
                results,
                total=len(ordered_paths),
                desc="parsing POLAR annotations",
                disable=not show_progress,
                unit="annotation",
            )
        )


def context_box(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
    context_fraction: float,
) -> tuple[int, int, int, int]:
    """Expand a box by a fraction of its size and clip it to image bounds."""

    if context_fraction < 0.0:
        raise ValueError("context_fraction cannot be negative")
    xmin, ymin, xmax, ymax = (float(value) for value in box)
    image_width, image_height = (int(value) for value in image_size)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("bounding box must have positive area")
    width = xmax - xmin
    height = ymax - ymin
    left = max(0, int(np.floor(xmin - context_fraction * width)))
    top = max(0, int(np.floor(ymin - context_fraction * height)))
    right = min(image_width, int(np.ceil(xmax + context_fraction * width)))
    bottom = min(image_height, int(np.ceil(ymax + context_fraction * height)))
    if right <= left or bottom <= top:
        raise ValueError("clipped bounding box has no area")
    return left, top, right, bottom


def image_view(image: Image.Image, row: pd.Series | dict, view: str) -> Image.Image:
    """Return the declared full-frame or person-context view."""

    if view == "full_frame":
        return image
    context_by_view = {"person_context_10": 0.10, "person_context_25": 0.25}
    if view not in context_by_view:
        raise ValueError(f"Unknown POLAR view: {view}")
    box = (
        float(row["bbox_xmin"]),
        float(row["bbox_ymin"]),
        float(row["bbox_xmax"]),
        float(row["bbox_ymax"]),
    )
    return image.crop(context_box(box, image.size, context_by_view[view]))


def _inspect_image(path: Path) -> dict:
    encoded = path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    with Image.open(io.BytesIO(encoded)) as image:
        image.load()
        width, height = image.size
        perceptual_hash = str(imagehash.phash(image.convert("RGB"), hash_size=8))
        mode = str(image.mode)
        image_format = str(image.format)
    return {
        "sha256": digest,
        "phash": perceptual_hash,
        "actual_width": int(width),
        "actual_height": int(height),
        "image_mode": mode,
        "image_format": image_format,
        "decode_ok": True,
        "decode_error": "",
    }


def _safe_inspect_image(path: Path) -> dict:
    try:
        return _inspect_image(path)
    except Exception as error:  # pragma: no cover - exercised only by corrupt source data
        return {
            "sha256": "",
            "phash": "",
            "actual_width": -1,
            "actual_height": -1,
            "image_mode": "",
            "image_format": "",
            "decode_ok": False,
            "decode_error": f"{type(error).__name__}: {error}",
        }


def inspect_images(
    paths: Iterable[Path], *, workers: int = 8, show_progress: bool = False
) -> list[dict]:
    ordered_paths = list(paths)
    if workers < 1:
        raise ValueError("workers must be positive")
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        results = executor.map(_safe_inspect_image, ordered_paths)
        return list(
            tqdm(
                results,
                total=len(ordered_paths),
                desc="hashing and decoding POLAR",
                disable=not show_progress,
                unit="image",
            )
        )


def _split_lookup(splits: dict[str, set[str]]) -> dict[str, str]:
    lookup = {}
    for split, identifiers in splits.items():
        for identifier in identifiers:
            if identifier in lookup:
                raise ValueError(f"Duplicate official split identifier: {identifier}")
            lookup[identifier] = split
    return lookup


def build_manifest(
    annotations_dir: str | Path,
    images_dir: str | Path,
    image_sets_dir: str | Path,
    *,
    workers: int = 8,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, dict]:
    annotations_root = Path(annotations_dir).resolve()
    images_root = Path(images_dir).resolve()
    splits = load_split_ids(image_sets_dir)
    split_by_id = _split_lookup(splits)
    annotation_paths = sorted(annotations_root.glob("*.json"))
    if len(annotation_paths) != sum(EXPECTED_SPLIT_COUNTS.values()):
        raise ValueError(f"Expected 35,324 annotations, found {len(annotation_paths)}")

    rows = []
    non_target = 0
    ambiguous_target = 0
    annotations = parse_annotations(
        annotation_paths, workers=workers, show_progress=show_progress
    )
    for annotation_path, annotation in zip(annotation_paths, annotations, strict=True):
        split = split_by_id.get(annotation.image_id)
        if split is None:
            raise ValueError(f"Annotation is absent from official split lists: {annotation.image_id}")
        if annotation.target is None:
            if annotation.target_record_count:
                ambiguous_target += 1
            else:
                non_target += 1
            continue
        target = annotation.target
        rows.append(
            {
                "image_id": annotation.image_id,
                "split": split,
                "source_label": target.source_label,
                "label_4": target.label_4,
                "label_3": THREE_CLASS_MAP[target.label_4],
                "image_path": str((images_root / annotation.filename).resolve()),
                "annotation_path": str(annotation_path.resolve()),
                "original_name": annotation.original_name,
                "annotated_width": annotation.annotated_width,
                "annotated_height": annotation.annotated_height,
                "bbox_xmin": target.xmin,
                "bbox_ymin": target.ymin,
                "bbox_xmax": target.xmax,
                "bbox_ymax": target.ymax,
            }
        )
    frame = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    inspections = inspect_images(
        [Path(value) for value in frame["image_path"].astype(str)],
        workers=workers,
        show_progress=show_progress,
    )
    frame = pd.concat([frame, pd.DataFrame(inspections)], axis=1)
    frame["dimension_match"] = (
        (frame["annotated_width"] == frame["actual_width"])
        & (frame["annotated_height"] == frame["actual_height"])
    )
    clipped_xmin = frame["bbox_xmin"].clip(lower=0, upper=frame["actual_width"])
    clipped_ymin = frame["bbox_ymin"].clip(lower=0, upper=frame["actual_height"])
    clipped_xmax = frame["bbox_xmax"].clip(lower=0, upper=frame["actual_width"])
    clipped_ymax = frame["bbox_ymax"].clip(lower=0, upper=frame["actual_height"])
    frame["bbox_was_clipped"] = (
        (clipped_xmin != frame["bbox_xmin"])
        | (clipped_ymin != frame["bbox_ymin"])
        | (clipped_xmax != frame["bbox_xmax"])
        | (clipped_ymax != frame["bbox_ymax"])
    )
    frame["bbox_xmin"] = clipped_xmin.astype(int)
    frame["bbox_ymin"] = clipped_ymin.astype(int)
    frame["bbox_xmax"] = clipped_xmax.astype(int)
    frame["bbox_ymax"] = clipped_ymax.astype(int)
    frame["bbox_valid"] = (
        (frame["bbox_xmax"] > frame["bbox_xmin"])
        & (frame["bbox_ymax"] > frame["bbox_ymin"])
    )
    frame["bbox_area_fraction"] = (
        (frame["bbox_xmax"] - frame["bbox_xmin"])
        * (frame["bbox_ymax"] - frame["bbox_ymin"])
        / (frame["actual_width"] * frame["actual_height"])
    )
    frame["eligible"] = frame["decode_ok"] & frame["bbox_valid"]
    frame["exclusion_reason"] = ""
    frame.loc[~frame["decode_ok"], "exclusion_reason"] = "decode_failure"
    frame.loc[frame["decode_ok"] & ~frame["bbox_valid"], "exclusion_reason"] = (
        "invalid_person_box"
    )

    observed = (
        frame.groupby(["split", "label_4"], observed=True).size().unstack(fill_value=0)
    )
    for split, expected in EXPECTED_TARGET_COUNTS.items():
        actual = {label: int(observed.loc[split, label]) for label in LABEL_TO_INDEX}
        if actual != expected:
            raise ValueError(f"POLAR target counts changed for {split}: {actual} != {expected}")
    audit = {
        "annotations": len(annotation_paths),
        "target_rows": len(frame),
        "non_target_annotations": non_target,
        "ambiguous_target_annotations": ambiguous_target,
        "decode_failures": int((~frame["decode_ok"]).sum()),
        "dimension_mismatches": int((~frame["dimension_match"]).sum()),
        "clipped_boxes": int(frame["bbox_was_clipped"].sum()),
        "invalid_boxes": int((~frame["bbox_valid"]).sum()),
        "target_counts": {
            split: {label: int(observed.loc[split, label]) for label in LABEL_TO_INDEX}
            for split in ("train", "val", "test")
        },
    }
    return frame, audit


def exact_cross_split_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = frame[frame["sha256"].astype(str).str.len() == 64]
    for digest, group in valid.groupby("sha256", sort=True):
        if group["split"].nunique() < 2:
            continue
        records = group.sort_values(["split", "image_id"]).to_dict("records")
        for left_index, left in enumerate(records):
            for right in records[left_index + 1 :]:
                if left["split"] == right["split"]:
                    continue
                rows.append(
                    {
                        "sha256": digest,
                        "left_image_id": left["image_id"],
                        "left_split": left["split"],
                        "left_label": left["label_4"],
                        "right_image_id": right["image_id"],
                        "right_split": right["split"],
                        "right_label": right["label_4"],
                    }
                )
    return pd.DataFrame(rows)


def near_phash_cross_split_pairs(
    frame: pd.DataFrame, *, max_distance: int = 6
) -> pd.DataFrame:
    """Find all cross-split 64-bit pHash pairs within a Hamming radius."""

    if not 0 <= max_distance < 8:
        raise ValueError("max_distance must be between zero and seven")
    valid = frame[frame["phash"].astype(str).str.fullmatch(r"[0-9a-fA-F]{16}")].copy()
    valid = valid.reset_index(drop=True)
    values = [int(value, 16) for value in valid["phash"].astype(str)]
    splits = valid["split"].astype(str).tolist()
    identifiers = valid["image_id"].astype(str).tolist()
    labels = valid["label_4"].astype(str).tolist()
    paths = valid["image_path"].astype(str).tolist()

    chunks = max_distance + 1
    base_width, remainder = divmod(64, chunks)
    partitions = []
    offset = 0
    for chunk in range(chunks):
        width = base_width + (1 if chunk < remainder else 0)
        partitions.append((offset, width))
        offset += width

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    near: dict[tuple[int, int], dict] = {}
    for index, value in enumerate(values):
        for partition_index, (bit_offset, width) in enumerate(partitions):
            segment = (value >> bit_offset) & ((1 << width) - 1)
            key = (partition_index, segment)
            for other in buckets[key]:
                if splits[index] == splits[other]:
                    continue
                pair = (min(index, other), max(index, other))
                if pair in near:
                    continue
                distance = (values[index] ^ values[other]).bit_count()
                if distance <= max_distance:
                    left, right = pair
                    near[pair] = {
                        "left_image_id": identifiers[left],
                        "left_split": splits[left],
                        "left_label": labels[left],
                        "left_path": paths[left],
                        "right_image_id": identifiers[right],
                        "right_split": splits[right],
                        "right_label": labels[right],
                        "right_path": paths[right],
                        "phash_distance": int(distance),
                    }
            buckets[key].append(index)
    columns = [
        "left_image_id",
        "left_split",
        "left_label",
        "left_path",
        "right_image_id",
        "right_split",
        "right_label",
        "right_path",
        "phash_distance",
    ]
    return pd.DataFrame(near.values(), columns=columns).sort_values(
        ["phash_distance", "left_image_id", "right_image_id"], ignore_index=True
    )


def normalized_pair_similarity(left_path: str | Path, right_path: str | Path) -> dict:
    """Compare pHash candidates after deterministic grayscale normalization."""

    arrays = []
    for path in (left_path, right_path):
        with Image.open(path) as image:
            normalized = image.convert("L").resize((128, 128), Image.Resampling.BICUBIC)
            arrays.append(np.asarray(normalized, dtype=np.float32) / 255.0)
    left, right = arrays
    mae = float(np.mean(np.abs(left - right)))
    left_flat = left.ravel()
    right_flat = right.ravel()
    if float(left_flat.std()) == 0.0 or float(right_flat.std()) == 0.0:
        correlation = 1.0 if np.array_equal(left, right) else 0.0
    else:
        correlation = float(np.corrcoef(left_flat, right_flat)[0, 1])
    return {"normalized_mae": mae, "normalized_correlation": correlation}


def cross_split_embedding_pairs(
    features: np.ndarray,
    frame: pd.DataFrame,
    *,
    minimum_cosine: float = 0.985,
    top_k: int = 20,
    chunk_size: int = 512,
    device: str = "cpu",
) -> pd.DataFrame:
    """Retrieve high-cosine cross-split neighbours without using labels."""

    import torch

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(frame):
        raise ValueError("features must be [rows, dimensions] and align with frame")
    if not 0.0 < minimum_cosine <= 1.0:
        raise ValueError("minimum_cosine must be in (0, 1]")
    if top_k < 1 or chunk_size < 1:
        raise ValueError("top_k and chunk_size must be positive")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms == 0.0).any() or not np.isfinite(norms).all():
        raise ValueError("embedding rows must have finite non-zero norm")
    values = values / norms
    tensor = torch.as_tensor(values, dtype=torch.float32, device=torch.device(device))
    split_values = frame["split"].astype(str).to_numpy()
    split_categories = {value: index for index, value in enumerate(sorted(set(split_values)))}
    split_codes = torch.as_tensor(
        [split_categories[value] for value in split_values], dtype=torch.int16, device=tensor.device
    )
    requested_k = min(int(top_k), max(1, len(frame) - 1))
    pairs: dict[tuple[int, int], float] = {}
    for start in range(0, len(frame), int(chunk_size)):
        stop = min(start + int(chunk_size), len(frame))
        similarities = tensor[start:stop] @ tensor.T
        same_split = split_codes[start:stop, None] == split_codes[None, :]
        similarities.masked_fill_(same_split, -1.0)
        scores, indices = torch.topk(similarities, k=requested_k, dim=1)
        scores = scores.cpu().numpy()
        indices = indices.cpu().numpy()
        for local_index, (row_scores, row_indices) in enumerate(
            zip(scores, indices, strict=True)
        ):
            left_index = start + local_index
            for score, right_index in zip(row_scores, row_indices, strict=True):
                if float(score) < minimum_cosine:
                    continue
                pair = tuple(sorted((left_index, int(right_index))))
                pairs[pair] = max(float(score), pairs.get(pair, -1.0))
    rows = []
    records = frame.reset_index(drop=True).to_dict("records")
    for (left_index, right_index), cosine in pairs.items():
        left = records[left_index]
        right = records[right_index]
        rows.append(
            {
                "left_image_id": left["image_id"],
                "left_split": left["split"],
                "left_label": left.get("label_4", ""),
                "left_path": left["image_path"],
                "right_image_id": right["image_id"],
                "right_split": right["split"],
                "right_label": right.get("label_4", ""),
                "right_path": right["image_path"],
                "embedding_cosine": cosine,
            }
        )
    columns = [
        "left_image_id",
        "left_split",
        "left_label",
        "left_path",
        "right_image_id",
        "right_split",
        "right_label",
        "right_path",
        "embedding_cosine",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["embedding_cosine", "left_image_id", "right_image_id"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def enrich_near_pairs(pairs: pd.DataFrame, *, workers: int = 8) -> pd.DataFrame:
    if pairs.empty:
        return pairs.assign(normalized_mae=pd.Series(dtype=float), normalized_correlation=pd.Series(dtype=float))
    path_pairs = list(zip(pairs["left_path"], pairs["right_path"], strict=True))

    def compare(values: tuple[str, str]) -> dict:
        return normalized_pair_similarity(*values)

    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        similarities = list(executor.map(compare, path_pairs))
    return pd.concat([pairs.reset_index(drop=True), pd.DataFrame(similarities)], axis=1)


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def source_related_pairs(
    near_pairs: pd.DataFrame, *, minimum_correlation: float = 0.90
) -> pd.DataFrame:
    required = {
        "left_image_id",
        "right_image_id",
        "phash_distance",
        "normalized_correlation",
    }
    missing = required - set(near_pairs.columns)
    if missing:
        raise ValueError(f"Near-pair table is missing columns: {sorted(missing)}")
    selected = near_pairs[
        (near_pairs["phash_distance"] <= 6)
        & (near_pairs["normalized_correlation"] >= float(minimum_correlation))
    ].copy()
    selected["confirmation_rule"] = (
        f"phash_distance<=6_and_normalized_correlation>={minimum_correlation:.3f}"
    )
    return selected.sort_values(
        ["normalized_correlation", "left_image_id", "right_image_id"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def embedding_confirmed_source_pairs(
    embedding_pairs: pd.DataFrame,
    *,
    minimum_cosine: float = 0.985,
    minimum_correlation: float = 0.90,
) -> pd.DataFrame:
    required = {
        "left_image_id",
        "right_image_id",
        "embedding_cosine",
        "normalized_correlation",
    }
    missing = required - set(embedding_pairs.columns)
    if missing:
        raise ValueError(f"Embedding-pair table is missing columns: {sorted(missing)}")
    selected = embedding_pairs[
        (embedding_pairs["embedding_cosine"] >= float(minimum_cosine))
        & (embedding_pairs["normalized_correlation"] >= float(minimum_correlation))
    ].copy()
    selected["confirmation_rule"] = (
        f"embedding_cosine>={minimum_cosine:.3f}_and_"
        f"normalized_correlation>={minimum_correlation:.3f}"
    )
    return selected.sort_values(
        ["embedding_cosine", "left_image_id", "right_image_id"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def quarantine_components(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=["image_id", "quarantine_group"])
    disjoint = _DisjointSet()
    for row in pairs.itertuples(index=False):
        disjoint.union(str(row.left_image_id), str(row.right_image_id))
    components: dict[str, list[str]] = defaultdict(list)
    for image_id in sorted(disjoint.parent):
        components[disjoint.find(image_id)].append(image_id)
    ordered_components = sorted((sorted(values) for values in components.values()), key=lambda x: x[0])
    rows = []
    for index, image_ids in enumerate(ordered_components, start=1):
        group = f"source_leakage_{index:04d}"
        rows.extend({"image_id": image_id, "quarantine_group": group} for image_id in image_ids)
    return pd.DataFrame(rows).sort_values(["quarantine_group", "image_id"], ignore_index=True)


def apply_quarantine(
    frame: pd.DataFrame, quarantine: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if quarantine["image_id"].duplicated().any():
        raise ValueError("Each quarantined image must belong to one connected component")
    output = frame.copy()
    group_by_id = quarantine.set_index("image_id")["quarantine_group"].to_dict()
    output["quarantine_group"] = output["image_id"].map(group_by_id).fillna("")
    source_mask = output["quarantine_group"].astype(str).str.len() > 0
    output.loc[source_mask, "eligible"] = False
    output.loc[source_mask, "exclusion_reason"] = "cross_split_source_related"
    output["primary_included"] = output["eligible"].astype(bool)
    clean = output[output["primary_included"]].copy().reset_index(drop=True)
    return output.reset_index(drop=True), clean


def legacy_overlap(frame: pd.DataFrame, legacy_manifest: str | Path) -> pd.DataFrame:
    legacy = pd.read_csv(legacy_manifest, dtype=str)
    rows = []
    by_sha = defaultdict(list)
    by_phash = defaultdict(list)
    for record in legacy.to_dict("records"):
        by_sha[str(record.get("sha256", ""))].append(record)
        by_phash[str(record.get("phash", "")).lower()].append(record)
    for record in frame.to_dict("records"):
        matches = []
        for candidate in by_sha.get(str(record["sha256"]), []):
            matches.append((candidate, "sha256", 0))
        polar_hash = str(record["phash"]).lower()
        if len(polar_hash) == 16:
            polar_value = int(polar_hash, 16)
            for digest, candidates in by_phash.items():
                if len(digest) != 16:
                    continue
                distance = (polar_value ^ int(digest, 16)).bit_count()
                if distance <= 6:
                    for candidate in candidates:
                        matches.append((candidate, "phash", distance))
        seen = set()
        for candidate, match_type, distance in matches:
            key = (str(candidate.get("image_id", "")), match_type)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "polar_image_id": record["image_id"],
                    "polar_split": record["split"],
                    "polar_label": record["label_4"],
                    "legacy_image_id": candidate.get("image_id", ""),
                    "legacy_split": candidate.get("split", ""),
                    "legacy_label": candidate.get("label", ""),
                    "match_type": match_type,
                    "phash_distance": distance,
                }
            )
    columns = [
        "polar_image_id",
        "polar_split",
        "polar_label",
        "legacy_image_id",
        "legacy_split",
        "legacy_label",
        "match_type",
        "phash_distance",
    ]
    return pd.DataFrame(rows, columns=columns)


def manifest_summary(frame: pd.DataFrame, base_audit: dict) -> dict:
    summary = dict(base_audit)
    summary.update(
        {
            "eligible_rows": int(frame["eligible"].sum()),
            "exact_duplicate_groups": int(
                frame.loc[frame["sha256"].astype(str).str.len() == 64, "sha256"]
                .value_counts()
                .gt(1)
                .sum()
            ),
            "bbox_area_fraction": {
                "mean": float(frame["bbox_area_fraction"].mean()),
                "median": float(frame["bbox_area_fraction"].median()),
                "p05": float(frame["bbox_area_fraction"].quantile(0.05)),
                "p95": float(frame["bbox_area_fraction"].quantile(0.95)),
            },
        }
    )
    return summary


def annotation_to_dict(annotation: PolarAnnotation) -> dict:
    """Expose a JSON-safe representation for audit tools and tests."""

    return asdict(annotation)
