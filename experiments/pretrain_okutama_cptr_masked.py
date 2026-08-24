"""Label-free masked target-video adaptation for the CPTR temporal encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from hac.cptr import CenterQueryEncoder
from hac.cptr_features import CPTRFeatureDataset
from hac.polar import sha256_file
from hac.polar_training import warmup_cosine_scheduler
from hac.training import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiments/okutama_cptr_protocol.json")
    )
    parser.add_argument(
        "--protocol-lock", type=Path, default=Path(".runs/cptr/protocol_lock.json")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".runs/vcoco_v3/temporal/development_manifest.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".runs/cptr/masked_pretraining/seed-42")
    )
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class MaskedTemporalAdapter(nn.Module):
    def __init__(self, input_dim: int, protocol: dict) -> None:
        super().__init__()
        architecture = protocol["architecture"]
        model_dim = int(architecture["model_dim"])
        self.encoder = CenterQueryEncoder(
            input_dim,
            model_dim=model_dim,
            layers=int(architecture["temporal_layers"]),
            heads=int(architecture["attention_heads"]),
            feedforward_dim=int(architecture["feedforward_dim"]),
            dropout=float(architecture["dropout"]),
            maximum_length=32,
        )
        self.mask_token = nn.Parameter(torch.zeros(input_dim))
        nn.init.normal_(self.mask_token, std=input_dim**-0.5)
        self.decoder = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, input_dim),
        )
        self.order_head = nn.Linear(model_dim, 1)

    def forward(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        masked_positions: torch.Tensor,
        centre_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked_values = torch.where(
            masked_positions[:, :, None],
            self.mask_token[None, None],
            values,
        )
        encoded, repaired_mask = self.encoder.encode_tokens(
            masked_values,
            valid_mask,
            centre_index=centre_indices,
        )
        pooled = self.encoder.pool_tokens(
            encoded,
            repaired_mask,
            centre_index=centre_indices,
        )
        return self.decoder(encoded), self.order_head(pooled).squeeze(1)


def masked_positions(
    valid_mask: torch.Tensor,
    centre_indices: torch.Tensor,
    *,
    fraction: float,
) -> torch.Tensor:
    random_values = torch.rand(valid_mask.shape, device=valid_mask.device)
    selected = (random_values < fraction) & valid_mask
    selected[torch.arange(len(selected), device=selected.device), centre_indices] = False
    rows_without_mask = ~selected.any(dim=1)
    for row in torch.nonzero(rows_without_mask, as_tuple=False).flatten().tolist():
        eligible = torch.nonzero(valid_mask[row], as_tuple=False).flatten()
        eligible = eligible[eligible != centre_indices[row]]
        if len(eligible):
            selected[row, eligible[0]] = True
    return selected


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("Masked CPTR adaptation requires CUDA; CPU fallback is disabled")
    protocol_path = args.protocol.resolve()
    lock_path = args.protocol_lock.resolve()
    manifest_path = args.manifest.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "OKUTAMA_CPTR_PROTOCOL_LOCKED_BEFORE_FIT":
        raise RuntimeError("The CPTR protocol is not locked")
    if lock["source_sha256"]["protocol"] != sha256_file(protocol_path):
        raise RuntimeError("The protocol changed after locking")
    if lock["source_sha256"]["development_manifest"] != sha256_file(manifest_path):
        raise RuntimeError("The development manifest changed after locking")
    frame = pd.read_csv(
        manifest_path,
        dtype={"sample_id": str, "recording_id": str, "track_id": str},
    )
    frame = frame[frame["split"].eq("train")].reset_index(drop=True)
    if len(frame) != 4977:
        raise RuntimeError("The fixed label-free adaptation partition changed")
    base_store_path = Path(str(frame.iloc[0]["feature_path"])).resolve()
    base_store = json.loads(base_store_path.read_text(encoding="utf-8"))
    input_dim = 2 * int(base_store["feature_dimensions"]) + 6
    adaptation = protocol["masked_target_adaptation"]
    epochs = int(adaptation["epochs"])
    mask_fraction = float(adaptation["mask_fraction"])
    seed_everything(args.seed)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats(device)
    dataset = CPTRFeatureDataset(frame, manifest_directory=manifest_path.parent)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=int(protocol["training"]["batch_size"]),
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        generator=generator,
    )
    model = MaskedTemporalAdapter(input_dim, protocol).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=math.ceil(len(loader)) * epochs,
        warmup_fraction=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=4096.0)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "status": "OKUTAMA_CPTR_MASKED_PRETRAINING_REQUEST",
        "seed": int(args.seed),
        "samples": int(len(frame)),
        "labels_used": False,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "mask_fraction": mask_fraction,
        "epochs": epochs,
        "source_sha256": {
            "protocol": sha256_file(protocol_path),
            "protocol_lock": sha256_file(lock_path),
            "manifest": sha256_file(manifest_path),
            "base_store": sha256_file(base_store_path),
            "runner": sha256_file(Path(__file__).resolve()),
            "model_module": sha256_file(Path(__file__).resolve().parents[1] / "src/hac/cptr.py"),
        },
    }
    request_hash = hashlib.sha256(
        json.dumps(request, sort_keys=True).encode("utf-8")
    ).hexdigest()
    request_path = output_dir / "request.json"
    write_json(request_path, {**request, "request_sha256": request_hash})
    history: list[dict] = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        reconstruction_values: list[float] = []
        semantic_values: list[float] = []
        order_values: list[float] = []
        total_values: list[float] = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            values = batch["short_features"].to(device, non_blocking=True)
            valid = batch["short_valid_mask"].to(device, non_blocking=True)
            centres = batch["short_centre_index"].to(device, non_blocking=True).long()
            reversed_rows = torch.rand(len(values), device=device) < 0.5
            if torch.any(reversed_rows):
                values = values.clone()
                valid = valid.clone()
                centres = centres.clone()
                values[reversed_rows] = values[reversed_rows].flip(1)
                valid[reversed_rows] = valid[reversed_rows].flip(1)
                centres[reversed_rows] = values.shape[1] - 1 - centres[reversed_rows]
            selected = masked_positions(valid, centres, fraction=mask_fraction)
            targets = F.layer_norm(values.float(), (input_dim,))
            with torch.autocast(device_type="cuda"):
                reconstruction, order_logits = model(values, valid, selected, centres)
                reconstruction_loss = F.smooth_l1_loss(
                    reconstruction[selected].float(),
                    targets[selected],
                )
                semantic_loss = 1.0 - F.cosine_similarity(
                    reconstruction[selected].float(),
                    targets[selected],
                    dim=1,
                ).mean()
                order_loss = F.binary_cross_entropy_with_logits(
                    order_logits.float(),
                    reversed_rows.float(),
                )
                loss = reconstruction_loss + 0.25 * semantic_loss + 0.10 * order_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            reconstruction_values.append(float(reconstruction_loss.detach()))
            semantic_values.append(float(semantic_loss.detach()))
            order_values.append(float(order_loss.detach()))
            total_values.append(float(loss.detach()))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(total_values)),
            "reconstruction_loss": float(np.mean(reconstruction_values)),
            "semantic_cosine_loss": float(np.mean(semantic_values)),
            "temporal_order_loss": float(np.mean(order_values)),
        }
        history.append(row)
        write_json(output_dir / "history.json", history)
        print(json.dumps(row, sort_keys=True), flush=True)

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "status": "OKUTAMA_CPTR_MASKED_PRETRAINING_COMPLETE",
            "request_sha256": request_hash,
            "encoder_state_dict": model.encoder.state_dict(),
            "model_state_dict": model.state_dict(),
            "epochs": epochs,
            "labels_used": False,
        },
        checkpoint_path,
    )
    summary = {
        "status": "OKUTAMA_CPTR_MASKED_PRETRAINING_COMPLETE",
        "seed": int(args.seed),
        "samples": int(len(frame)),
        "labels_used": False,
        "calibration_samples_read": 0,
        "confirmation_samples_read": 0,
        "final_losses": history[-1],
        "training_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "runtime_seconds": time.perf_counter() - started,
        "request_sha256": request_hash,
        "artifact_sha256": {
            "checkpoint.pt": sha256_file(checkpoint_path),
            "history.json": sha256_file(output_dir / "history.json"),
            "request.json": sha256_file(request_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
