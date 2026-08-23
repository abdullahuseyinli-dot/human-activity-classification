"""Evaluate one locked-development neural checkpoint and export aligned probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evaluate_polar_final import model_config_from_final
from torch import nn

from hac.augmentations import build_aspect_preserving_eval_transform, build_eval_transform
from hac.data import make_loader
from hac.polar import sha256_file
from hac.polar_analysis import per_class_metrics
from hac.polar_models import build_polar_model
from hac.polar_training import TASK_LABELS, evaluate_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock_path = args.protocol_lock.resolve()
    protocol_hash = sha256_file(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "VCOCO_V2_PROTOCOL_LOCKED_BEFORE_NEW_MODEL_FITTING":
        raise RuntimeError("Neural evaluation requires the locked V-COCO v2 protocol")
    request_path = args.request.resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("status") != "DEVELOPMENT_TRAINING_REQUEST":
        raise RuntimeError("Unexpected neural training request status")
    if request.get("protocol_lock", {}).get("sha256") != protocol_hash:
        raise RuntimeError("Training request protocol drift")
    manifest_path = args.manifest.resolve()
    if request.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("Training request manifest drift")
    frame = pd.read_csv(manifest_path, dtype={"image_id": str, "source_image_group": str})
    validation = frame[frame["split"].eq("val")].copy().sort_values("image_id", ignore_index=True)
    configuration = request["configuration"]
    task = configuration["task"]
    class_names = list(TASK_LABELS[task])
    class_to_index = {name: index for index, name in enumerate(class_names)}
    validation["label"] = validation[task].astype(str)
    transform = (
        build_aspect_preserving_eval_transform()
        if configuration.get("evaluation_preprocess") == "aspect_preserving_pad"
        else build_eval_transform()
    )
    loader = make_loader(
        validation,
        class_to_index,
        transform,
        batch_size=configuration["batch_size"],
        shuffle=False,
        seed=configuration["seed"],
        workers=configuration["workers"],
        view=configuration["view"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_polar_model(
        model_config_from_final(configuration), num_classes=len(class_names), pretrained=False
    ).to(device)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("request_sha256") != request.get("request_sha256"):
        raise RuntimeError("Checkpoint request hash drift")
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluation = evaluate_classifier(model, loader, nn.CrossEntropyLoss(), device)
    if evaluation["image_ids"] != validation["image_id"].astype(str).tolist():
        raise RuntimeError("Neural validation prediction order drift")
    per_class = pd.DataFrame(
        per_class_metrics(evaluation["labels"], evaluation["probabilities"], class_names)
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(output_dir / "validation_per_class.csv", index=False)
    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        person_ids=validation["person_id"].astype(str).to_numpy(),
        image_ids=validation["source_image_group"].astype(str).to_numpy(),
        labels=evaluation["labels"],
        class_names=np.asarray(class_names),
        probabilities=evaluation["probabilities"],
        logits=evaluation["logits"],
    )
    result = {
        "status": "VCOCO_V2_NEURAL_CHECKPOINT_DEVELOPMENT_EVALUATION_COMPLETE",
        "metrics": evaluation["metrics"],
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "request_sha256": sha256_file(request_path),
        "protocol_lock_sha256": protocol_hash,
        "validation_rows": len(validation),
        "test_rows_read": 0,
        "test_predictions_run": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
