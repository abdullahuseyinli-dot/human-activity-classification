"""Validate and lock temporal metadata and development feature caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hac.polar import sha256_file
from hac.vcoco_v3_temporal import validate_temporal_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provider-provenance", type=Path, required=True)
    parser.add_argument(
        "--split-provenance",
        type=Path,
        help="Defaults to the manifest name with .provenance.json suffix",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=Path("experiments/polimi_itw_s_ontology.json"),
    )
    parser.add_argument(
        "--grid", type=Path, default=Path("experiments/vcoco_v3_temporal_grid.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/temporal_manifest_lock.json"),
    )
    return parser.parse_args()


def resolve_feature_path(manifest_path: Path, value: str) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def inspect_development_feature(path: Path, expected_frames: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Temporal development feature file is missing: {path}")
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Temporal features must use the declared NPZ schema: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = {"tight", "context", "geometry"}
        if not required.issubset(payload.files):
            raise ValueError(f"Temporal feature file is missing arrays: {path}")
        tight = payload["tight"]
        context = payload["context"]
        geometry = payload["geometry"]
        if tight.ndim != 2 or context.ndim != 2 or geometry.ndim != 2:
            raise ValueError(f"Temporal feature arrays must be two-dimensional: {path}")
        if tight.shape[0] != expected_frames or context.shape[0] != expected_frames:
            raise ValueError(f"Temporal visual feature count differs from the manifest: {path}")
        if geometry.shape != (expected_frames, 6):
            raise ValueError(f"Temporal geometry must have shape [frames, 6]: {path}")
        if tight.shape[1] != context.shape[1]:
            raise ValueError(f"Tight and context embedding widths differ: {path}")
        for name, values in (("tight", tight), ("context", context), ("geometry", geometry)):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Temporal {name} features contain non-finite values: {path}")
        evidence = {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "frames": int(expected_frames),
            "embedding_dimensions": int(tight.shape[1]),
            "pose_available": "pose" in payload.files,
        }
        if "pose" in payload.files:
            pose = payload["pose"]
            if pose.ndim != 3 or pose.shape[0] != expected_frames or pose.shape[2] not in {2, 3}:
                raise ValueError(f"Temporal pose array has an invalid shape: {path}")
            if not np.all(np.isfinite(pose)):
                raise ValueError(f"Temporal pose features contain non-finite values: {path}")
            evidence["pose_shape"] = list(map(int, pose.shape))
    return evidence


def inspect_packed_development_feature(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Packed temporal feature store is missing: {path}")
    store = json.loads(path.read_text(encoding="utf-8"))
    if store.get("status") != "VCOCO_V3_PACKED_TEMPORAL_FEATURE_STORE_COMPLETE":
        raise RuntimeError(f"Packed temporal feature store is incomplete: {path}")
    if store.get("confirmation_archive_opened") is not False:
        raise RuntimeError("A development store reports an early confirmation-archive open")
    arrays = {}
    for name in ("tight", "context", "geometry"):
        declaration = store.get("arrays", {}).get(name, {})
        array_path = (path.parent / str(declaration.get("path", ""))).resolve()
        if not array_path.is_file():
            raise FileNotFoundError(f"Packed temporal {name} array is missing: {array_path}")
        if sha256_file(array_path) != declaration.get("sha256"):
            raise RuntimeError(f"Packed temporal {name} array changed after caching")
        arrays[name] = np.load(array_path, mmap_mode="r")
    tight = arrays["tight"]
    context = arrays["context"]
    geometry = arrays["geometry"]
    expected_samples = int(store["samples"])
    expected_frames = int(store["frames_per_sample"])
    expected_dimensions = int(store["feature_dimensions"])
    if tight.shape != (expected_samples, expected_frames, expected_dimensions):
        raise RuntimeError("Packed tight-feature shape differs from its declaration")
    if context.shape != tight.shape:
        raise RuntimeError("Packed tight and context feature shapes differ")
    if geometry.shape != (expected_samples, expected_frames, 6):
        raise RuntimeError("Packed geometry must have shape [samples, frames, 6]")
    for name, values in arrays.items():
        for start in range(0, len(values), 256):
            if not np.isfinite(values[start : start + 256]).all():
                raise RuntimeError(f"Packed temporal {name} contains non-finite values")
    return {
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "samples": expected_samples,
        "frames": expected_frames,
        "embedding_dimensions": expected_dimensions,
        "array_sha256": {
            name: str(store["arrays"][name]["sha256"])
            for name in ("tight", "context", "geometry")
        },
        "pose_available": False,
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    provider_path = args.provider_provenance.resolve()
    split_path = (
        args.split_provenance.resolve()
        if args.split_provenance is not None
        else manifest_path.with_suffix(".provenance.json")
    )
    ontology_path = args.ontology.resolve()
    grid_path = args.grid.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    if grid.get("status") != "DECLARED_BEFORE_TEMPORAL_FITTING":
        raise RuntimeError("The temporal grid is not in its pre-fit state")
    dataset = str(grid.get("dataset", {}).get("name"))
    provider = json.loads(provider_path.read_text(encoding="utf-8"))
    if dataset == "POLIMI-ITW-S":
        if provider.get("dataset") != dataset:
            raise RuntimeError("Provider provenance belongs to a different temporal dataset")
        if provider.get("access_authorized") is not True:
            raise RuntimeError("Authorized POLIMI dataset access has not been recorded")
        if provider.get("provider_files_verified") is not True:
            raise RuntimeError("POLIMI provider files have not passed their checksum audit")
    elif dataset == "Okutama-Action":
        if provider.get("status") != "OKUTAMA_DEVELOPMENT_ARCHIVE_AND_CENTRES_AUDITED":
            raise RuntimeError("The Okutama development archive has not passed its audit")
        if provider.get("confirmation_archive_opened") is not False:
            raise RuntimeError("The Okutama confirmation archive was opened before locking")
    else:
        raise RuntimeError(f"Unsupported temporal dataset: {dataset}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("status") != "VCOCO_V3_TEMPORAL_SPLIT_ASSIGNED_BEFORE_MODEL_OUTCOMES":
        raise RuntimeError("The recording-grouped temporal split is not locked")
    if split.get("model_outcomes_read") != 0:
        raise RuntimeError("Temporal splits were assigned after model outcomes were read")
    if split["artifact_sha256"].get(manifest_path.name) != sha256_file(manifest_path):
        raise RuntimeError("The temporal manifest changed after split assignment")
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    if dataset == "POLIMI-ITW-S":
        if ontology.get("status") != "DECLARED_BEFORE_POLIMI_LABEL_USE":
            raise RuntimeError("The POLIMI ontology mapping is not locked")
        provider_column = str(ontology["provider_label_column"])
        mapping = ontology["target_mapping"]
    else:
        if ontology.get("status") != (
            "DECLARED_BEFORE_OKUTAMA_AGGREGATE_LABEL_AUDIT_OR_MODEL_FITTING"
        ):
            raise RuntimeError("The Okutama ontology mapping is not locked")
        if provider["source_sha256"].get("okutama_protocol") != sha256_file(ontology_path):
            raise RuntimeError("The Okutama protocol changed after the development audit")
        declared_archive = ontology["archives"]["development"]
        if provider["development_archive"].get("sha256") != declared_archive["sha256"]:
            raise RuntimeError("The audited Okutama archive differs from the declaration")
        provider_column = "provider_base_action"
        mapping = ontology["ontology"]["target_mapping"]

    frame = validate_temporal_manifest(
        pd.read_csv(
            manifest_path,
            dtype={"sample_id": str, "recording_id": str, "track_id": str},
        )
    )
    if provider_column not in frame:
        raise RuntimeError("The immutable provider-label column is missing")
    mapped = frame[provider_column].map(mapping)
    if mapped.isna().any() or not mapped.astype(str).equals(frame["label"].astype(str)):
        raise RuntimeError("Temporal labels do not match the declared provider ontology mapping")

    development_evidence = {}
    confirmation_paths = []
    embedding_dimensions = set()
    packed_evidence = {}
    for row in frame.itertuples(index=False):
        feature_path = resolve_feature_path(manifest_path, str(row.feature_path))
        if row.split == "confirmation":
            confirmation_paths.append(str(feature_path))
            continue
        if feature_path.suffix.lower() == ".json":
            key = str(feature_path)
            if key not in packed_evidence:
                packed_evidence[key] = inspect_packed_development_feature(feature_path)
            evidence = packed_evidence[key]
            if "feature_index" not in frame:
                raise RuntimeError("Packed temporal metadata requires feature_index")
            feature_index = int(row.feature_index)
            if not 0 <= feature_index < int(evidence["samples"]):
                raise RuntimeError("Packed temporal feature_index is outside the store")
            if int(row.frame_count) != int(evidence["frames"]):
                raise RuntimeError("Manifest frame count differs from the packed store")
        else:
            evidence = inspect_development_feature(feature_path, int(row.frame_count))
            development_evidence[str(row.sample_id)] = evidence
        embedding_dimensions.add(evidence["embedding_dimensions"])
    development_evidence.update(packed_evidence)
    if len(embedding_dimensions) != 1:
        raise RuntimeError("Temporal development embeddings do not share one width")

    split_counts = {
        name: {
            "samples": int(len(rows)),
            "recordings": int(rows["recording_id"].nunique()),
            "tracks": int(
                (rows["recording_id"].astype(str) + "::" + rows["track_id"].astype(str)).nunique()
            ),
            "class_counts": {
                str(label): int(count)
                for label, count in rows["label"].value_counts().sort_index().items()
            },
        }
        for name, rows in frame.groupby("split", sort=True)
    }
    result = {
        "status": "VCOCO_V3_TEMPORAL_MANIFEST_LOCKED",
        "dataset": dataset,
        "samples": len(frame),
        "embedding_dimensions": int(next(iter(embedding_dimensions))),
        "split_counts": split_counts,
        "development_feature_count": len(development_evidence),
        "confirmation_feature_paths_declared": len(confirmation_paths),
        "confirmation_feature_arrays_opened": 0,
        "confirmation_archive_opened": False,
        "model_outcomes_read": 0,
        "source_sha256": {
            "temporal_grid": sha256_file(grid_path),
            "ontology_mapping": sha256_file(ontology_path),
            "provider_provenance": sha256_file(provider_path),
            "split_provenance": sha256_file(split_path),
            "manifest": sha256_file(manifest_path),
        },
        "development_feature_sha256": development_evidence,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
