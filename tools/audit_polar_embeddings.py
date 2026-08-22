"""Audit POLAR split proximity with frozen upstream DINOv2 embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel

from hac.augmentations import build_eval_transform
from hac.polar import (
    cross_split_embedding_pairs,
    normalized_pair_similarity,
    sha256_file,
)

PINNED_REVISIONS = {
    "facebook/dinov2-small": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
}


class AuditDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = build_eval_transform()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        from PIL import Image

        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as image:
            pixels = self.transform(image.convert("RGB"))
        return {"pixel_values": pixels, "index": index}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--revision")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minimum-cosine", type=float, default=0.985)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=512)
    return parser.parse_args()


@torch.inference_mode()
def extract_features(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    rows = []
    for batch in tqdm(loader, desc="extracting frozen DINOv2 features", unit="batch"):
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(pixel_values=inputs)
            features = (
                output.pooler_output
                if getattr(output, "pooler_output", None) is not None
                else output.last_hidden_state[:, 0]
            )
        rows.append(features.float().cpu().numpy())
    return np.concatenate(rows, axis=0)


def enrich_candidates(candidates: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    by_id = frame.set_index("image_id")
    rows = []
    for candidate in tqdm(
        candidates.itertuples(index=False), total=len(candidates), desc="checking candidates"
    ):
        left = by_id.loc[candidate.left_image_id]
        right = by_id.loc[candidate.right_image_id]
        similarity = normalized_pair_similarity(left["image_path"], right["image_path"])
        left_phash = int(str(left["phash"]), 16)
        right_phash = int(str(right["phash"]), 16)
        rows.append(
            {
                **candidate._asdict(),
                "phash_distance": (left_phash ^ right_phash).bit_count(),
                **similarity,
                "left_original_name": left["original_name"],
                "right_original_name": right["original_name"],
            }
        )
    return pd.DataFrame(rows)


def model_files(model_id: str, revision: str) -> tuple[Path, list[dict]]:
    root = Path(
        snapshot_download(repo_id=model_id, revision=revision, local_files_only=True)
    ).resolve()
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
        if path.suffix in {".json", ".safetensors", ".bin"}
    ], root


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest, dtype={"image_id": str, "phash": str})
    frame = frame.sort_values("image_id").reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        AuditDataset(frame),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    revision = args.revision or PINNED_REVISIONS.get(args.model)
    if not revision:
        raise ValueError("A pinned --revision is required for an unrecognized model")
    model = AutoModel.from_pretrained(args.model, revision=revision).to(device)
    features_path = output_dir / "dinov2_full_frame_features.npy"
    if features_path.is_file():
        features = np.load(features_path)
        if features.shape[0] != len(frame):
            raise RuntimeError("Cached embedding rows do not match the manifest")
    else:
        features = extract_features(model, loader, device)
        np.save(features_path, features)
    pd.DataFrame({"row": np.arange(len(frame)), "image_id": frame["image_id"]}).to_csv(
        output_dir / "dinov2_feature_rows.csv", index=False
    )
    pairs = cross_split_embedding_pairs(
        features,
        frame,
        minimum_cosine=args.minimum_cosine,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        device=str(device),
    )
    pairs = enrich_candidates(pairs, frame)
    pairs.to_csv(output_dir / "dinov2_cross_split_candidates.csv", index=False)
    files, snapshot_root = model_files(args.model, revision)
    provenance = {
        "status": "PRE_SUPERVISED_FIT_AUDIT",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "model": args.model,
        "model_revision": revision,
        "model_snapshot": str(snapshot_root),
        "model_files": files,
        "device": str(device),
        "torch": torch.__version__,
        "minimum_cosine": args.minimum_cosine,
        "top_k": args.top_k,
        "candidate_pairs": len(pairs),
        "embedding_shape": list(features.shape),
        "test_labels_used": False,
        "supervised_model_fitted": False,
    }
    (output_dir / "dinov2_embedding_audit.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
