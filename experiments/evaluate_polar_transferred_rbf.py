"""Evaluate the locked RBF classifier on source-aligned DINOv2 features."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from screen_polar_embedding_classifiers import Candidate, calibrated_final_estimator

from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_features import load_aligned_feature_view

CLASS_NAMES = ["sitting", "standing", "walking", "running"]
MODEL_KIND = "dinov2_base"
VIEWS = ["full_frame", "person_context_10"]
REPRESENTATION = "last4_cls_mean_patch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("polar_study_protocol.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-folds", type=int, default=5)
    return parser.parse_args()


def git_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def support_vector_counts(estimator) -> list[int]:
    counts = []
    for calibrated in estimator.calibrated_classifiers_:
        classifier = calibrated.estimator.named_steps["classifier"]
        counts.append(int(np.asarray(classifier.n_support_).sum()))
    return counts


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    revision_at_start = git_revision(repository_root)
    protocol_path = args.protocol.resolve()
    protocol_hash = sha256_file(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "1.4.0":
        raise RuntimeError("Transferred RBF evaluation requires POLAR protocol 1.4.0")
    extension = protocol.get("source_aligned_representation_extension", {})
    rbf_lock = extension.get("classifiers", {}).get("transferred_calibrated_rbf_svm", {})
    if rbf_lock.get("C") != 10.0 or rbf_lock.get("retuned_on_multilayer_features") is not False:
        raise RuntimeError("Transferred RBF configuration differs from the protocol lock")
    if extension.get("test_rows_read") != 0 or extension.get("test_used_for_selection"):
        raise RuntimeError("Representation extension violates the held-out test gate")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest.resolve()
    manifest_hash = sha256_file(manifest_path)
    manifest = pd.read_csv(manifest_path, dtype={"image_id": str})
    manifest = manifest.sort_values("image_id").reset_index(drop=True)
    if set(manifest["split"].astype(str)) != {"train", "val"}:
        raise ValueError("Transferred RBF evaluation accepts development rows only")

    blocks = []
    cache_provenance = {}
    for view in VIEWS:
        block, provenance = load_aligned_feature_view(
            args.feature_root.resolve(), MODEL_KIND, view, manifest, manifest_hash
        )
        if provenance.get("representation") != REPRESENTATION:
            raise RuntimeError(f"Unexpected representation for {view}: {provenance}")
        blocks.append(block)
        cache_provenance[view] = provenance
    features = np.concatenate(blocks, axis=1)
    label_to_index = {label: index for index, label in enumerate(CLASS_NAMES)}
    labels = manifest["label_4"].map(label_to_index).to_numpy(dtype=int)
    train_mask = manifest["split"].astype(str).eq("train").to_numpy()
    val_mask = manifest["split"].astype(str).eq("val").to_numpy()

    candidate = Candidate(
        "rbf_svm",
        {"C": 10.0, "gamma_multiplier": 1.0, "class_weight": None},
    )
    estimator = calibrated_final_estimator(
        candidate,
        features.shape[1],
        args.seed,
        args.calibration_folds,
    )
    fit_started = time.perf_counter()
    estimator.fit(features[train_mask], labels[train_mask])
    fit_seconds = time.perf_counter() - fit_started
    inference_started = time.perf_counter()
    probabilities = np.asarray(estimator.predict_proba(features[val_mask]), dtype=float)
    inference_seconds = time.perf_counter() - inference_started
    if not np.array_equal(estimator.classes_, np.arange(len(CLASS_NAMES))):
        raise RuntimeError(f"Unexpected class ordering: {estimator.classes_.tolist()}")
    metrics = classification_metrics(labels[val_mask], probabilities)
    counts = support_vector_counts(estimator)

    model_path = output_dir / "development_transferred_rbf.joblib"
    joblib.dump(estimator, model_path, compress=3)
    np.savez_compressed(
        output_dir / "validation_transferred_rbf.npz",
        probabilities=probabilities,
        labels=labels[val_mask],
        image_ids=manifest.loc[val_mask, "image_id"].astype(str).to_numpy(),
        class_names=np.asarray(CLASS_NAMES),
    )
    result = {
        "status": "DEVELOPMENT_ONLY_TRANSFERRED_RBF",
        "protocol_version": "1.4.0",
        "manifest_sha256": manifest_hash,
        "model_kind": MODEL_KIND,
        "views": VIEWS,
        "representation": REPRESENTATION,
        "feature_dimensions": int(features.shape[1]),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(val_mask.sum()),
        "configuration": {
            "family": "rbf_svm",
            "C": 10.0,
            "gamma": float(1.0 / features.shape[1]),
            "gamma_multiplier": 1.0,
            "class_weight": "none",
            "calibration": "five_fold_sigmoid",
            "calibration_folds": int(args.calibration_folds),
            "seed": int(args.seed),
            "retuned_on_this_representation": False,
        },
        "metrics": metrics,
        "fit_seconds": float(fit_seconds),
        "inference_seconds": float(inference_seconds),
        "inference_rows_per_second": float(val_mask.sum() / inference_seconds),
        "support_vector_counts_by_calibration_fold": counts,
        "mean_support_vectors": float(np.mean(counts)),
        "model_bytes": int(model_path.stat().st_size),
        "model_sha256": sha256_file(model_path),
        "cache_provenance": cache_provenance,
        "protocol_sha256": protocol_hash,
        "git_revision_at_start": revision_at_start,
        "implementation_sha256": {
            "experiments/evaluate_polar_transferred_rbf.py": sha256_file(
                Path(__file__).resolve()
            ),
            "experiments/screen_polar_embedding_classifiers.py": sha256_file(
                Path(__file__).with_name("screen_polar_embedding_classifiers.py")
            ),
            "src/hac/metrics.py": sha256_file(repository_root / "src/hac/metrics.py"),
            "src/hac/polar_features.py": sha256_file(
                repository_root / "src/hac/polar_features.py"
            ),
        },
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "test_rows_read": 0,
        "test_used_for_selection": False,
    }
    write_json(output_dir / "transferred_rbf_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
