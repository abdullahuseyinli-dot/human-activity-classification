"""Selection-controlled recovery runner for the human-activity experiment.

This utility reuses the training primitives from the earlier internal-CV
notebook while enforcing the fixed test partition. Candidate
selection never evaluates the test set.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_CELLS = (3, 5, 8, 10, 16, 21, 23, 25, 27)


CANDIDATES = {
    "conv_head_only_d10": {
        "model_kind": "convnext_small",
        "augmentation_strength": "mild",
        "batch_size": 4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "dropout": 0.10,
        "mixup_alpha": 0.0,
        "unfreeze_strategy": "head_only",
        "backbone_lr": 3e-5,
    },
    "conv_last_stage_d10": {
        "model_kind": "convnext_small",
        "augmentation_strength": "mild",
        "batch_size": 4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "dropout": 0.10,
        "mixup_alpha": 0.0,
        "unfreeze_strategy": "last_stage",
        "backbone_lr": 3e-5,
    },
    "conv_base_d0": {
        "model_kind": "convnext_small",
        "augmentation_strength": "mild",
        "batch_size": 4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "dropout": 0.0,
        "mixup_alpha": 0.0,
        "unfreeze_strategy": "full_backbone",
        "backbone_lr": 3e-5,
    },
    "conv_base_d10": {
        "model_kind": "convnext_small",
        "augmentation_strength": "mild",
        "batch_size": 4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "dropout": 0.10,
        "mixup_alpha": 0.0,
        "unfreeze_strategy": "full_backbone",
        "backbone_lr": 3e-5,
    },
    "conv_no_erasing_d10": {
        "model_kind": "convnext_small",
        "augmentation_strength": "mild_no_random_erasing",
        "batch_size": 4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "dropout": 0.10,
        "mixup_alpha": 0.0,
        "unfreeze_strategy": "full_backbone",
        "backbone_lr": 3e-5,
    },
    "conv_mixup_d10": {
        "model_kind": "convnext_small",
        "augmentation_strength": "mild",
        "batch_size": 4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "dropout": 0.10,
        "mixup_alpha": 0.20,
        "unfreeze_strategy": "full_backbone",
        "backbone_lr": 3e-5,
    },
    "conv_randaugment_d10": {
        "model_kind": "convnext_small",
        "augmentation_strength": "mild_randaugment_light",
        "batch_size": 4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "dropout": 0.10,
        "mixup_alpha": 0.0,
        "unfreeze_strategy": "full_backbone",
        "backbone_lr": 3e-5,
    },
    "conv_randaugment_d0": {
        "model_kind": "convnext_small",
        "augmentation_strength": "mild_randaugment_light",
        "batch_size": 4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "dropout": 0.0,
        "mixup_alpha": 0.0,
        "unfreeze_strategy": "full_backbone",
        "backbone_lr": 3e-5,
    },
    "conv_fixed_val_d10": {
        "model_kind": "convnext_small",
        "augmentation_strength": "mild",
        "batch_size": 4,
        "head_lr": 3e-4,
        "weight_decay": 1e-4,
        "label_smoothing": 0.05,
        "dropout": 0.10,
        "mixup_alpha": 0.0,
        "unfreeze_strategy": "full_backbone",
        "backbone_lr": 3e-5,
    },
    "dino_top2_d0": {
        "model_kind": "dinov2_small",
        "augmentation_strength": "moderate",
        "batch_size": 4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "dropout": 0.0,
        "mixup_alpha": 0.0,
        "unfreeze_strategy": "top_blocks",
        "top_n_blocks": 2,
        "backbone_lr": 5e-6,
    },
    "dino_probe_d10": {
        "model_kind": "dinov2_small",
        "augmentation_strength": "moderate",
        "batch_size": 4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "dropout": 0.10,
        "mixup_alpha": 0.0,
        "unfreeze_strategy": "probe_only",
        "backbone_lr": 5e-6,
    },
    "dino_full_d10": {
        "model_kind": "dinov2_small",
        "augmentation_strength": "moderate",
        "batch_size": 4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "dropout": 0.10,
        "mixup_alpha": 0.0,
        "unfreeze_strategy": "full_backbone",
        "backbone_lr": 5e-6,
    },
    "dino_top2_d10": {
        "model_kind": "dinov2_small",
        "augmentation_strength": "moderate",
        "batch_size": 4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "dropout": 0.10,
        "mixup_alpha": 0.0,
        "unfreeze_strategy": "top_blocks",
        "top_n_blocks": 2,
        "backbone_lr": 5e-6,
    },
    "dino_top2_mild_d10": {
        "model_kind": "dinov2_small",
        "augmentation_strength": "mild",
        "batch_size": 4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "dropout": 0.10,
        "mixup_alpha": 0.0,
        "unfreeze_strategy": "top_blocks",
        "top_n_blocks": 2,
        "backbone_lr": 5e-6,
    },
    "dino_top2_mixup_d10": {
        "model_kind": "dinov2_small",
        "augmentation_strength": "moderate",
        "batch_size": 4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "dropout": 0.10,
        "mixup_alpha": 0.20,
        "unfreeze_strategy": "top_blocks",
        "top_n_blocks": 2,
        "backbone_lr": 5e-6,
    },
    "dino_top2_d20": {
        "model_kind": "dinov2_small",
        "augmentation_strength": "moderate",
        "batch_size": 4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "dropout": 0.20,
        "mixup_alpha": 0.0,
        "unfreeze_strategy": "top_blocks",
        "top_n_blocks": 2,
        "backbone_lr": 5e-6,
    },
    "dino_fixed_val_d10": {
        "model_kind": "dinov2_small",
        "augmentation_strength": "mild",
        "batch_size": 4,
        "head_lr": 3e-4,
        "weight_decay": 1e-5,
        "label_smoothing": 0.10,
        "dropout": 0.10,
        "mixup_alpha": 0.0,
        "unfreeze_strategy": "top_blocks",
        "top_n_blocks": 2,
        "backbone_lr": 5e-6,
    },
    "dino_top4_d10": {
        "model_kind": "dinov2_small",
        "augmentation_strength": "moderate",
        "batch_size": 4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "dropout": 0.10,
        "mixup_alpha": 0.0,
        "unfreeze_strategy": "top_blocks",
        "top_n_blocks": 4,
        "backbone_lr": 5e-6,
    },
    "dino_top2_wd5e4_d10": {
        "model_kind": "dinov2_small",
        "augmentation_strength": "moderate",
        "batch_size": 4,
        "head_lr": 1e-3,
        "weight_decay": 5e-4,
        "label_smoothing": 0.0,
        "dropout": 0.10,
        "mixup_alpha": 0.0,
        "unfreeze_strategy": "top_blocks",
        "top_n_blocks": 2,
        "backbone_lr": 5e-6,
    },
}


STAGES = {
    "smoke": {"n_splits": 2, "epochs": 1, "patience": 1},
    "coarse": {"n_splits": 3, "epochs": 12, "patience": 3},
    "confirm": {"n_splits": 5, "epochs": 20, "patience": 5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-notebook", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), default="coarse")
    parser.add_argument(
        "--candidate",
        action="append",
        choices=sorted(CANDIDATES),
        help="Candidate to run. Repeat the flag for multiple candidates.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pipeline_cell_source(notebook: dict, source_index: int) -> str:
    tagged = [
        cell
        for cell in notebook.get("cells", [])
        if int(cell.get("metadata", {}).get("pipeline_source_index", -1)) == source_index
    ]
    if len(tagged) == 1:
        return "".join(tagged[0].get("source", []))
    if len(tagged) > 1:
        raise RuntimeError(f"Duplicate pipeline cell tag: {source_index}")
    if source_index >= len(notebook.get("cells", [])):
        raise RuntimeError(f"Missing pipeline source cell: {source_index}")
    return "".join(notebook["cells"][source_index].get("source", []))


def execute_cell(namespace: dict, source: str, label: str) -> None:
    print(f"[setup] executing {label}", flush=True)
    exec(compile(source, label, "exec"), namespace)


def patch_config_cell(source: str, project_root: Path, artifact_root: Path, manifest: Path) -> str:
    source = source.replace(
        "ROOT = Path.cwd()",
        f"ROOT = Path({str(project_root)!r}).resolve()",
        1,
    )
    source = source.replace(
        "MANIFEST_PATH = None",
        f"MANIFEST_PATH = Path({str(manifest)!r}).resolve()",
        1,
    )
    source = source.replace(
        "IMAGES_ROOT = None",
        f"IMAGES_ROOT = Path({str(project_root)!r}).resolve()",
        1,
    )
    source = source.replace(
        'ART = ROOT / "artifacts_v3_cv"',
        f"ART = Path({str(artifact_root)!r}).resolve()",
        1,
    )
    for flag in (
        "RUN_RAW_DINO_EDA",
        "RUN_SEARCH",
        "RUN_TRAINING",
        "RUN_CALIBRATION",
        "RUN_ENSEMBLE",
        "RUN_EMBEDDINGS",
        "RUN_EXPLAINABILITY",
        "RUN_CALIBRATION_BEFORE_AFTER",
        "RUN_PER_CLASS_COMPARISON",
        "RUN_ROBUSTNESS_SUITE",
        "RUN_ERROR_TAXONOMY_SUMMARY",
        "RUN_AUGMENTATION_ABLATIONS",
        "RUN_PERSON_CENTRIC_BRANCH",
        "RUN_PERSON_CENTRIC_HEAD_ONLY_TRAIN",
    ):
        source = source.replace(f"{flag} = True", f"{flag} = False")
    return source


def patch_cv_cell(source: str) -> str:
    """Retain OOF features and a two-view flip-TTA prediction bundle."""
    replacements = (
        (
            "pred_frames, logits_frames, probs_frames, features_frames = [], [], [], []",
            "pred_frames, logits_frames, probs_frames, features_frames = [], [], [], []\n"
            "    tta_logits_frames, tta_probs_frames = [], []",
        ),
        (
            "fold_out = evaluate_model(model, fold_loader, criterion, return_features=False)",
            "fold_out = evaluate_model(model, fold_loader, criterion, return_features=True)\n\n"
            "        flip_tf = transforms.Compose([\n"
            "            build_eval_transform(),\n"
            "            transforms.RandomHorizontalFlip(p=1.0),\n"
            "        ])\n"
            "        flip_loader = make_loader(\n"
            '            fold_val_df, flip_tf, int(cfg["batch_size"]), shuffle=False, seed=seed\n'
            "        )\n"
            "        fold_flip_out = evaluate_model(\n"
            "            model, flip_loader, criterion, return_features=False\n"
            "        )\n"
            '        if fold_out["image_ids"] != fold_flip_out["image_ids"]:\n'
            '            raise RuntimeError("TTA image ordering mismatch")\n'
            '        if not np.array_equal(fold_out["labels"], fold_flip_out["labels"]):\n'
            '            raise RuntimeError("TTA label ordering mismatch")\n'
            '        fold_tta_probs = 0.5 * (fold_out["probs"] + fold_flip_out["probs"])\n'
            "        fold_tta_logits = np.log(np.clip(fold_tta_probs, 1e-12, 1.0))",
        ),
        (
            "probs_frames.append(probs_fold)",
            "probs_frames.append(probs_fold)\n\n"
            "        tta_logits_fold = pd.DataFrame(\n"
            '            fold_tta_logits, columns=[f"logit_{i}" for i in range(fold_tta_logits.shape[1])]\n'
            "        )\n"
            '        tta_logits_fold["cv_row_id"] = pred_df["cv_row_id"].values\n'
            "        tta_logits_frames.append(tta_logits_fold)\n"
            "        tta_probs_fold = pd.DataFrame(\n"
            '            fold_tta_probs, columns=[f"prob_{i}" for i in range(fold_tta_probs.shape[1])]\n'
            "        )\n"
            '        tta_probs_fold["cv_row_id"] = pred_df["cv_row_id"].values\n'
            "        tta_probs_frames.append(tta_probs_fold)",
        ),
        (
            "oof_output = _build_oof_output(pool_df_, seed, pred_frames, logits_frames, probs_frames, features_frames if len(features_frames) else None)",
            "oof_output = _build_oof_output(pool_df_, seed, pred_frames, logits_frames, probs_frames, features_frames if len(features_frames) else None)\n"
            "    oof_tta_output = _build_oof_output(\n"
            "        pool_df_, seed, pred_frames, tta_logits_frames, tta_probs_frames,\n"
            "        features_frames if len(features_frames) else None,\n"
            "    )\n"
            '    oof_tta_output["pred_df"]["y_pred"] = oof_tta_output["probs"].argmax(axis=1)\n'
            '    oof_tta_output["pred_df"]["confidence"] = oof_tta_output["probs"].max(axis=1)',
        ),
        (
            '"oof_output": oof_output,',
            '"oof_output": oof_output,\n        "oof_tta_output": oof_tta_output,',
        ),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f"CV source patch expected one occurrence: {old[:72]!r}")
        source = source.replace(old, new, 1)
    return source


def restore_fixed_split(namespace: dict) -> None:
    df = namespace["df"].copy()
    if "original_split" not in df.columns:
        raise RuntimeError("The manifest does not expose the original fixed split.")

    original = (
        df["original_split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"valid": "val", "validation": "val"})
    )
    unexpected = sorted(set(original) - {"train", "val", "test"})
    if unexpected:
        raise RuntimeError(f"Unexpected original split labels: {unexpected}")

    df["original_split"] = original
    df["split"] = np.where(original.eq("test"), "test", "non_test_pool")
    df["protocol_split"] = df["split"]
    df["frozen_test_flag"] = original.eq("test")
    df["non_test_pool_flag"] = ~df["frozen_test_flag"]
    df["protocol_row_id"] = np.arange(len(df))
    df["frozen_test_seed"] = np.nan
    df["frozen_test_target_count"] = int(df["frozen_test_flag"].sum())

    if int(df["frozen_test_flag"].sum()) != 43:
        raise RuntimeError("Expected the fixed test split to contain 43 rows.")
    if int(df["non_test_pool_flag"].sum()) != 242:
        raise RuntimeError("Expected train+validation to form a 242-row development pool.")

    namespace["df"] = df
    dirs = namespace["DIRS"]
    df.to_csv(dirs["manifests"] / "standardized_manifest_fixed_test.csv", index=False)

    protocol = {
        "protocol": "fixed_test_plus_internal_stratified_cv",
        "selection_partition": "original train + validation",
        "selection_rows": 242,
        "final_test_partition": "original test",
        "final_test_rows": 43,
        "test_used_during_candidate_selection": False,
        "test_image_ids_sha256": hashlib.sha256(
            "\n".join(sorted(df.loc[df["frozen_test_flag"], "image_id"].astype(str))).encode(
                "utf-8"
            )
        ).hexdigest(),
    }
    with (dirs["configs"] / "selection_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)


def install_transform_variant(namespace: dict) -> None:
    original_builder = namespace["build_train_transform"]
    transforms_module = namespace["transforms"]

    def build_train_transform_variant(strength: str = "moderate"):
        if strength not in {"mild_no_random_erasing", "mild_randaugment_light"}:
            return original_builder(strength)
        base = original_builder("mild")
        if strength == "mild_no_random_erasing":
            retained = [
                transform
                for transform in base.transforms
                if not isinstance(transform, transforms_module.RandomErasing)
            ]
            return transforms_module.Compose(retained)

        operations = list(base.transforms)
        operations.insert(
            2,
            transforms_module.RandAugment(num_ops=2, magnitude=7),
        )
        return transforms_module.Compose(operations)

    namespace["build_train_transform"] = build_train_transform_variant


def install_regularized_model_builder(namespace: dict) -> None:
    nn = namespace["nn"]

    class DropoutLinear(nn.Linear):
        def __init__(self, in_features, out_features, bias=True, dropout=0.0):
            super().__init__(in_features, out_features, bias=bias)
            self.dropout_p = float(dropout)
            self.dropout = nn.Dropout(p=self.dropout_p)

        def forward(self, inputs):
            return nn.functional.linear(self.dropout(inputs), self.weight, self.bias)

    class ConvNeXtFeatureAdapter(nn.Module):
        """Expose pooled ConvNeXt features through the shared model interface."""

        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        @property
        def classifier(self):
            return self.backbone.classifier

        def forward(self, pixel_values, return_features=False, output_attentions=False):
            activations = self.backbone.features(pixel_values)
            activations = self.backbone.avgpool(activations)
            activations = self.backbone.classifier[0](activations)
            features = self.backbone.classifier[1](activations)
            logits = self.backbone.classifier[2](features)
            if return_features and output_attentions:
                return logits, features, None
            if return_features:
                return logits, features
            return logits

    def build_model_and_strategy_regularized(model_kind: str, cfg: dict):
        dropout = float(cfg.get("dropout", 0.0))
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), received {dropout}")

        if model_kind == "convnext_small":
            model = namespace["models"].convnext_small(
                weights=namespace["ConvNeXt_Small_Weights"].IMAGENET1K_V1
            )
            in_features = model.classifier[2].in_features
            model.classifier[2] = DropoutLinear(
                in_features,
                namespace["N_CLASSES"],
                dropout=dropout,
            )
            model = namespace["freeze_convnext"](model, cfg["unfreeze_strategy"])
            model = ConvNeXtFeatureAdapter(model)
        elif model_kind == "dinov2_small":
            model = namespace["Dinov2Classifier"](
                backbone_name=namespace["PROJECT_CONFIG"]["ssl_backbone_name"],
                dropout=dropout,
            )
            strategy = cfg["unfreeze_strategy"]
            if strategy == "probe_only":
                model = namespace["freeze_dino"](model, "probe_only")
            elif strategy == "top_blocks":
                model = namespace["freeze_dino"](
                    model,
                    "top_blocks",
                    top_n_blocks=int(cfg["top_n_blocks"]),
                )
            elif strategy == "full_backbone":
                model = namespace["freeze_dino"](model, "full_backbone")
            else:
                raise ValueError(strategy)
        else:
            raise ValueError(model_kind)

        model._mixup_alpha = float(cfg.get("mixup_alpha", 0.0))
        model._experiment_dropout = dropout
        return model

    namespace["DropoutLinear"] = DropoutLinear
    namespace["ConvNeXtFeatureAdapter"] = ConvNeXtFeatureAdapter
    namespace["build_model_and_strategy"] = build_model_and_strategy_regularized


def install_mixup_training(namespace: dict) -> None:
    torch = namespace["torch"]

    def train_one_epoch_regularized(
        model,
        loader,
        optimizer,
        criterion,
        scaler=None,
        grad_accum_steps=1,
    ):
        model.train()
        losses = []
        grad_norms = []
        mixup_alpha = float(getattr(model, "_mixup_alpha", 0.0))
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            inputs = batch["pixel_values"].to(namespace["DEVICE"], non_blocking=True)
            labels = batch["label"].to(namespace["DEVICE"], non_blocking=True)

            if mixup_alpha > 0.0:
                mix_weight = float(np.random.beta(mixup_alpha, mixup_alpha))
                permutation = torch.randperm(inputs.size(0), device=inputs.device)
                model_inputs = mix_weight * inputs + (1.0 - mix_weight) * inputs[permutation]
                labels_b = labels[permutation]
            else:
                mix_weight = 1.0
                model_inputs = inputs
                labels_b = labels

            with torch.cuda.amp.autocast(enabled=namespace["USE_AMP"]):
                logits = model(model_inputs)
                if mixup_alpha > 0.0:
                    raw_loss = mix_weight * criterion(logits, labels) + (
                        1.0 - mix_weight
                    ) * criterion(logits, labels_b)
                else:
                    raw_loss = criterion(logits, labels)
                window_start = (step // grad_accum_steps) * grad_accum_steps
                window_size = min(grad_accum_steps, len(loader) - window_start)
                loss = raw_loss / window_size

            if namespace["USE_AMP"]:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(loader))
            if should_step:
                if namespace["USE_AMP"]:
                    scaler.unscale_(optimizer)
                grad_norms.append(namespace["compute_grad_norm"](model))

                if namespace["USE_AMP"]:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(float(raw_loss.detach().item()))

        return (
            float(np.mean(losses)),
            float(np.mean(grad_norms)) if grad_norms else np.nan,
        )

    namespace["train_one_epoch"] = train_one_epoch_regularized


def initialise_pipeline(args: argparse.Namespace) -> dict:
    source_notebook = args.source_notebook.resolve()
    manifest = args.manifest.resolve()
    artifact_root = args.artifact_root.resolve()
    project_root = Path(__file__).resolve().parents[1]

    if not source_notebook.is_file():
        raise FileNotFoundError(source_notebook)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    artifact_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    notebook = load_notebook(source_notebook)
    namespace: dict = {"__name__": "__main__"}

    for cell_index in PIPELINE_CELLS:
        source = pipeline_cell_source(notebook, cell_index)
        if cell_index == 5:
            source = patch_config_cell(source, project_root, artifact_root, manifest)
        if cell_index == 27:
            source = patch_cv_cell(source)
        execute_cell(namespace, source, f"{source_notebook.name}:cell-{cell_index}")
        if cell_index == 8:
            restore_fixed_split(namespace)

    install_transform_variant(namespace)
    install_regularized_model_builder(namespace)
    install_mixup_training(namespace)
    namespace["SEARCH_CFG"] = {
        "search_mode": "focused_recovery",
        "stage_a_max_trials": 0,
        "stage_b_top_k": 0,
        "pilot_epochs": 12,
        "full_epochs": 20,
        "pilot_patience": 3,
        "full_patience": 5,
    }

    protocol_record = {
        "runner": Path(__file__).name,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "source_notebook": source_notebook.name,
        "source_notebook_sha256": sha256_file(source_notebook),
        "manifest_sha256": sha256_file(manifest),
        "python": sys.version,
        "torch": namespace["torch"].__version__,
        "torchvision": namespace["torchvision"].__version__,
        "device": str(namespace["DEVICE"]),
    }
    with (namespace["DIRS"]["configs"] / "runtime_provenance.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(protocol_record, handle, indent=2)

    for name in ("convnext_probe", "dino_probe"):
        namespace.pop(name, None)
    gc.collect()
    if namespace["torch"].cuda.is_available():
        namespace["torch"].cuda.empty_cache()
    return namespace


def result_payload(candidate_id: str, cfg: dict, stage: str, seed: int, cv_out: dict) -> dict:
    metrics = cv_out["oof_output"]["metrics"]
    fold_summaries = cv_out["fold_summaries"]
    return {
        "candidate_id": candidate_id,
        "model_kind": cfg["model_kind"],
        "stage": stage,
        "seed": int(seed),
        "selection_metric": "pooled_oof_macro_f1",
        "oof_accuracy": float(metrics["accuracy"]),
        "oof_macro_f1": float(metrics["macro_f1"]),
        "oof_weighted_f1": float(metrics["weighted_f1"]),
        "oof_balanced_accuracy": float(metrics["balanced_accuracy"]),
        "oof_log_loss": float(metrics["log_loss"]),
        "oof_brier_score": float(metrics["brier_score"]),
        "oof_ece": float(metrics["ece"]),
        "derived_final_epochs": int(cv_out["derived_final_epochs"]),
        "cv_n_splits": int(cv_out["cv_n_splits"]),
        "fold_best_epochs": [int(row["best_epoch"]) for row in fold_summaries],
        "fold_macro_f1": [float(row["best_val_macro_f1"]) for row in fold_summaries],
        "config": {key: value for key, value in cfg.items() if key != "model_kind"},
        "test_evaluated": False,
    }


def write_summary(artifact_root: Path) -> None:
    records = []
    for path in sorted((artifact_root / "candidate_results").glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    if not records:
        return

    rows = []
    for record in records:
        row = {
            key: value
            for key, value in record.items()
            if key not in {"config", "fold_best_epochs", "fold_macro_f1"}
        }
        row.update({f"cfg_{key}": value for key, value in record["config"].items()})
        row["fold_best_epochs"] = json.dumps(record["fold_best_epochs"])
        row["fold_macro_f1"] = json.dumps(record["fold_macro_f1"])
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        ["model_kind", "stage", "oof_macro_f1", "oof_log_loss"],
        ascending=[True, True, False, True],
    )
    summary.to_csv(artifact_root / "candidate_selection_summary.csv", index=False)


def run_candidates(args: argparse.Namespace, namespace: dict) -> None:
    stage_cfg = STAGES[args.stage]
    candidate_ids = args.candidate or list(CANDIDATES)
    artifact_root = args.artifact_root.resolve()
    result_dir = artifact_root / "candidate_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    namespace["CV_N_SPLITS"] = int(stage_cfg["n_splits"])
    namespace["CV_FOLDS"], namespace["cv_fold_assignment_df"] = namespace["get_cv_folds"](
        namespace["non_test_pool_df"],
        n_splits=int(stage_cfg["n_splits"]),
        seed=int(args.seed),
    )

    for candidate_id in candidate_ids:
        cfg_with_model = dict(CANDIDATES[candidate_id])
        model_kind = cfg_with_model.pop("model_kind")
        cfg = cfg_with_model
        result_path = result_dir / f"{args.stage}_{candidate_id}_seed_{args.seed}.json"
        if result_path.exists() and not args.force:
            print(f"[skip] {candidate_id}: existing result {result_path.name}", flush=True)
            continue

        print(
            f"[run] stage={args.stage} candidate={candidate_id} model={model_kind} "
            f"folds={stage_cfg['n_splits']} epochs={stage_cfg['epochs']} patience={stage_cfg['patience']}",
            flush=True,
        )
        print(json.dumps(cfg, indent=2), flush=True)

        cv_out = namespace["run_cv_for_cfg"](
            model_kind=model_kind,
            cfg=cfg,
            pool_df_=namespace["non_test_pool_df"],
            seed=int(args.seed),
            epochs=int(stage_cfg["epochs"]),
            patience=int(stage_cfg["patience"]),
            run_name_prefix=f"{args.stage}_{candidate_id}_seed_{args.seed}",
            n_splits=int(stage_cfg["n_splits"]),
            split_seed=int(args.seed),
            save_fold_predictions=False,
        )
        payload = result_payload(
            candidate_id,
            {"model_kind": model_kind, **cfg},
            args.stage,
            args.seed,
            cv_out,
        )
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(
            f"[result] {candidate_id}: OOF macro-F1={payload['oof_macro_f1']:.6f}, "
            f"log-loss={payload['oof_log_loss']:.6f}, final_epochs={payload['derived_final_epochs']}",
            flush=True,
        )

        del cv_out
        gc.collect()
        if namespace["torch"].cuda.is_available():
            namespace["torch"].cuda.empty_cache()
        write_summary(artifact_root)


def main() -> None:
    args = parse_args()
    namespace = initialise_pipeline(args)
    run_candidates(args, namespace)
    write_summary(args.artifact_root.resolve())
    print(f"[done] candidate evidence: {args.artifact_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
