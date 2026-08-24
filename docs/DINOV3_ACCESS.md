# DINOv3 model access and provenance

The matched representation screen uses Meta's gated DINOv3 ViT-B/16 checkpoint from
`facebook/dinov3-vitb16-pretrain-lvd1689m`. The model revision is pinned to
`5931719e67bbdb9737e363e781fb0c67687896bc`.

## Recorded checkpoint

The checkpoint requires approved Hugging Face access. The configuration and weights
were downloaded through the authenticated client on 24 August 2026 and retained only
in the local Hugging Face cache.

| Artifact | Recorded value |
| --- | --- |
| Model identifier | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| Revision | `5931719e67bbdb9737e363e781fb0c67687896bc` |
| `config.json` SHA-256 | `3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec` |
| `model.safetensors` bytes | 342,662,192 |
| `model.safetensors` SHA-256 | `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b` |
| Architecture | ViT-B/16, 12 layers, hidden size 768, image size 224 |

No credential, access token, checkpoint, or provider download URL is stored in the
repository. DINOv3 code and weights remain under Meta's DINOv3 License; see
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Matched evaluation

The checkpoint was evaluated only after the human-pilot, nested-fusion, spatial, and
representation locks were in place. DINOv2-B, DINOv3-B, and SigLIP2-B used the same
person views, folds, nested grouped protocol, classifier family, and selection budget.

| Representation | Macro-F1 | Accuracy | Locomotion F1 |
| --- | ---: | ---: | ---: |
| DINOv2-B | 0.8395 | 0.8636 | 0.7006 |
| DINOv3-B | 0.8367 | 0.8601 | 0.7013 |
| SigLIP2-B | 0.8358 | 0.8563 | 0.7172 |

DINOv3-B did not pass either predeclared promotion rule. DINOv2-B remained the locked
frame backbone for the Okutama temporal experiment. The DINOv3 result is retained in
`results/vcoco_v3/source_tag_development_metrics.csv` together with its paired
promotion decision.

## Reproducibility check

An authorized account can verify the small pinned configuration without downloading
the weights again:

```powershell
hf auth whoami
hf download facebook/dinov3-vitb16-pretrain-lvd1689m config.json `
  --revision 5931719e67bbdb9737e363e781fb0c67687896bc
```

The returned file must match the configuration hash above. A different revision,
unpinned mirror, or differently sized checkpoint is not an equivalent replay.

The official DINOv3 repository documents Transformers support from version 4.56.0.
The recorded environment uses a compatible Transformers release, PyTorch 2.11.0 with
CUDA 12.8, and an NVIDIA GeForce RTX 4060 Laptop GPU.
