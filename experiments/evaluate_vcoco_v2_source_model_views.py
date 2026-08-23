"""Replay frozen POLAR DINO checkpoints under controlled V-COCO person views."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evaluate_polar_final import model_config_from_final
from torch import nn

from hac.augmentations import build_aspect_preserving_eval_transform, build_eval_transform
from hac.data import make_loader
from hac.metrics import classification_metrics
from hac.polar import sha256_file
from hac.polar_analysis import per_class_metrics
from hac.polar_models import build_polar_model
from hac.polar_training import TASK_LABELS, evaluate_classifier, normalize_probability_rows
from hac.transfer import image_cluster_paired_bootstrap

CLASS_NAMES = ("sitting", "standing", "walking_running")
VIEW_SPECS = {
    "context25_center": ("person_context_25", "legacy_center_crop"),
    "tight_center": ("person_tight", "legacy_center_crop"),
    "context25_pad": ("person_context_25", "aspect_preserving_pad"),
    "tight_pad": ("person_tight", "aspect_preserving_pad"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    return parser.parse_args()


def collapse_to_three(probabilities: np.ndarray) -> np.ndarray:
    return normalize_probability_rows(
        np.column_stack(
            [probabilities[:, 0], probabilities[:, 1], probabilities[:, 2] + probabilities[:, 3]]
        )
    )


def load_baseline(path: Path, rows: pd.DataFrame) -> np.ndarray:
    with np.load(path, allow_pickle=True) as payload:
        index = {str(value): row for row, value in enumerate(payload["person_ids"])}
        order = np.asarray([index[value] for value in rows["person_id"]], dtype=int)
        return normalize_probability_rows(payload["person_dinov2_base_top4"][order])


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.bootstrap_resamples < 1:
        raise ValueError("Invalid loader or bootstrap arguments")
    lock_path = args.protocol_lock.resolve()
    protocol_hash = sha256_file(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("Source-model view replay requires the locked V-COCO v2 protocol")
    manifest_paths = {
        "train": args.train_manifest.resolve(),
        "val": args.val_manifest.resolve(),
    }
    for split, path in manifest_paths.items():
        if sha256_file(path) != lock["artifact_sha256"][f"vcoco_{split}_clean.csv"]:
            raise RuntimeError(f"Locked {split} manifest drift")
    frames = []
    for split, path in manifest_paths.items():
        frame = pd.read_csv(path, dtype={"person_id": str, "image_id": str})
        frame["split"] = split
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    if rows["person_id"].duplicated().any():
        raise RuntimeError("Development person identifiers are not unique")
    inference = rows.copy()
    inference["coco_image_id"] = inference["image_id"]
    inference["image_id"] = inference["person_id"]
    inference["label_4"] = inference["label_4"].fillna("").replace("", "walking")
    inference["label"] = inference["label_4"]
    class_names_4 = list(TASK_LABELS["label_4"])
    class_to_index = {name: index for index, name in enumerate(class_names_4)}
    loaders = {}
    for name, (view, preprocess) in VIEW_SPECS.items():
        transform = (
            build_eval_transform()
            if preprocess == "legacy_center_crop"
            else build_aspect_preserving_eval_transform()
        )
        loaders[name] = make_loader(
            inference,
            class_to_index,
            transform,
            batch_size=args.batch_size,
            shuffle=False,
            seed=0,
            workers=args.workers,
            view=view,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    criterion = nn.CrossEntropyLoss()
    by_view: dict[str, list[np.ndarray]] = {name: [] for name in VIEW_SPECS}
    checkpoint_rows = []
    reference_configuration = None
    for checkpoint_path in [path.resolve() for path in args.checkpoint]:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        configuration = payload["configuration"]
        if configuration["model_kind"] != "dinov2_base" or payload["class_names"] != class_names_4:
            raise RuntimeError(f"Unexpected source checkpoint task: {checkpoint_path}")
        if reference_configuration is None:
            reference_configuration = configuration
        elif configuration != reference_configuration:
            raise RuntimeError("Source checkpoint configurations differ")
        model = build_polar_model(
            model_config_from_final(configuration), num_classes=len(class_names_4), pretrained=False
        )
        model.load_state_dict(payload["model_state_dict"])
        model = model.to(device)
        for name, loader in loaders.items():
            evaluation = evaluate_classifier(model, loader, criterion, device)
            if evaluation["image_ids"] != inference["person_id"].astype(str).tolist():
                raise RuntimeError(f"Prediction order drift for {checkpoint_path} and {name}")
            by_view[name].append(collapse_to_three(evaluation["probabilities"]))
        checkpoint_rows.append(
            {
                "checkpoint": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "epoch": int(payload["epoch"]),
            }
        )
        del model, payload
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ensemble = {
        name: normalize_probability_rows(np.mean(probabilities, axis=0))
        for name, probabilities in by_view.items()
    }
    val_mask = rows["split"].eq("val").to_numpy()
    val_rows = rows[val_mask].reset_index(drop=True)
    label_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    val_labels = val_rows["label_3"].map(label_to_index).to_numpy(dtype=int)
    baseline = load_baseline(args.baseline_predictions.resolve(), val_rows)
    summary_rows = []
    class_rows = []
    uncertainty = {}
    for name, probabilities in ensemble.items():
        val_probabilities = probabilities[val_mask]
        summary_rows.append({"view": name, **classification_metrics(val_labels, val_probabilities)})
        class_rows.extend(
            {"view": name, **row}
            for row in per_class_metrics(val_labels, val_probabilities, CLASS_NAMES)
        )
        uncertainty[name] = image_cluster_paired_bootstrap(
            val_labels,
            val_probabilities,
            baseline,
            val_rows["image_id"].astype(str).to_numpy(),
            resamples=args.bootstrap_resamples,
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["macro_f1", "log_loss", "view"], ascending=[False, True, True], ignore_index=True
    )
    reproduced = ensemble["context25_center"][val_mask]
    reproduction = {
        "maximum_absolute_probability_difference": float(np.abs(reproduced - baseline).max()),
        "mean_absolute_probability_difference": float(np.abs(reproduced - baseline).mean()),
        "argmax_agreement": float((reproduced.argmax(axis=1) == baseline.argmax(axis=1)).mean()),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "source_model_view_summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output_dir / "source_model_view_per_class.csv", index=False)
    (output_dir / "source_model_view_uncertainty.json").write_text(
        json.dumps(uncertainty, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "source_model_view_predictions.npz",
        person_ids=rows["person_id"].to_numpy(),
        image_ids=rows["image_id"].to_numpy(),
        split=rows["split"].to_numpy(),
        class_names=np.asarray(CLASS_NAMES),
        **ensemble,
    )
    result = {
        "status": "VCOCO_V2_SOURCE_MODEL_VIEW_REPLAY_COMPLETE",
        "selection_scope": "target_validation_selected_preprocessing_of_source_only_weights",
        "best_validation_result": summary.iloc[0].to_dict(),
        "baseline_reproduction": reproduction,
        "source_checkpoints": checkpoint_rows,
        "configuration": reference_configuration,
        "protocol_lock_sha256": protocol_hash,
        "test_rows_read": 0,
        "test_predictions_run": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
