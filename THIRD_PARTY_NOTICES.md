# Third-party notices

The repository's MIT License covers the original source code and documentation.
It does not replace the terms of source images, dataset annotations, framework
code, or pretrained model parameters.

## COCO source images

`data/manifest.csv` records COCO image URLs, labels, fixed splits, and checksums.
Full source datasets and source photographs are not included in the current tree.
The earlier v1 and v2 tags contained four qualitative composites made from COCO
photographs:

- `assets/champion_error_gallery.png`;
- `assets/convnext_small_faithfulness_gallery.jpg`;
- `assets/dinov2_small_faithfulness_gallery.jpg`;
- `assets/probability_blend_faithfulness_gallery.jpg`.

These files remain part of the historical Git record and byte-identical copies are
retained with nonpublic run evidence, but they are excluded from the current
distributable tree and any archive built from it. They are not covered by this
repository's MIT License. The corresponding image identifiers and source URLs remain in
`data/manifest.csv`, `results/champion_error_analysis.csv`, and
`results/faithfulness_test_per_image.csv`, so the numerical and error-analysis record
does not depend on redistributing the photographs. Under the
[COCO terms of use](https://cocodataset.org/#termsofuse),
COCO annotations and the COCO website are licensed under CC BY 4.0. The COCO
Consortium does not own the image copyrights, so every image remains subject to
its original Flickr/source terms. Those per-image terms include noncommercial and
no-derivatives conditions, which is why the qualitative composites are not part of
the current distributable tree.

Suggested dataset citation:

> Lin, T.-Y. et al. (2014). Microsoft COCO: Common Objects in Context. ECCV.
> https://doi.org/10.1007/978-3-319-10602-1_48

## POLAR dataset

The scale study uses POLAR version 1 from Mendeley Data under the licence stated
on the [dataset record](https://doi.org/10.17632/hvnsh7rwz7.1). The release is
downloaded and verified locally; no POLAR image is committed or redistributed.
Although the dataset record is marked CC BY 4.0, its metadata identifies Getty
source filenames. Getty and other upstream image rights are not relicensed by
this project. Users must obtain the release from its publisher and assess whether
their intended use is permitted.

Suggested citations:

> Ma, W., & Liang, S. (2021). POLAR: Posture-level Action Recognition Dataset.
> Mendeley Data, V1. https://doi.org/10.17632/hvnsh7rwz7.1

> Ma, W., & Liang, S. (2019). POLAR: Posture-level Action Recognition Dataset.
> ICSAI, 427-433. https://doi.org/10.1109/ICSAI48974.2019.9010160

## V-COCO external dataset

The external-transfer audit uses V-COCO annotations and locally obtained COCO images.
No V-COCO annotation archive or source image is committed. V-COCO is provided for
research use through its [official repository](https://github.com/s-gupta/v-coco), and
the underlying images retain the COCO/source-image terms described above.

Suggested citation:

> Gupta, S., & Malik, J. (2015). Visual Semantic Role Labeling. arXiv:1505.04474.
> https://arxiv.org/abs/1505.04474

## Okutama-Action dataset

The temporal extension uses locally obtained Okutama-Action frame archives and
provider annotations. No archive, frame, person crop, or annotation row is committed
or redistributed. Users must obtain the dataset from its provider and comply with the
terms supplied with that release. The tracked repository contains only aggregate
metrics and archive hashes.

Suggested citation:

> Barekatain, M. et al. (2017). Okutama-Action: An Aerial View Video Dataset for
> Concurrent Human Action Detection. CVPR Workshops, 28-35.
> https://openaccess.thecvf.com/content_cvpr_2017_workshops/w34/html/Barekatain_Okutama-Action_An_Aerial_CVPR_2017_paper.html

## DINOv2-Small and DINOv2-Base

The DINOv2-Small and DINOv2-Base pretrained models are developed by Meta AI and
distributed under
the [Apache License 2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE).
The upstream [model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)
documents the model's intended uses, limitations, and training-data context.

## DINOv3-Base

The matched representation screen uses the gated
[`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
checkpoint. DINOv3 code and weights are supplied under Meta's
[`DINOv3 License`](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md),
not this repository's MIT License. The checkpoint remains in the local Hugging Face
cache and is not committed or redistributed.

## SigLIP and SigLIP2 Base models

The frozen representation controls use
[`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224),
and the CPTR center-frame control uses
[`google/siglip-base-patch16-224`](https://huggingface.co/google/siglip-base-patch16-224).
Their model cards record the Apache License 2.0. No SigLIP checkpoint is committed.

## ConvNeXt-Small and torchvision

ConvNeXt-Small is loaded with torchvision's `IMAGENET1K_V1` pretrained weights.
The torchvision source is distributed under the
[BSD 3-Clause License](https://github.com/pytorch/vision/blob/main/LICENSE).
Torchvision notes that pretrained models can also be subject to terms derived
from their training data; those upstream rights are not relicensed here.

## OpenCV

Camera-motion estimation uses the headless Python distribution of
[OpenCV](https://github.com/opencv/opencv-python). OpenCV 4.5.0 and later are
distributed under the [Apache License 2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE).

## PyTorch Grad-CAM

The attribution evaluator uses the ROAD noisy-linear imputer from
[PyTorch Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam), distributed
under the [MIT License](https://github.com/jacobgil/pytorch-grad-cam/blob/master/LICENSE).
The Grad-CAM, HiResCAM, integrated-gradients, and transformer-rollout maps in
this repository are implemented locally so they can differentiate the exact
calibrated ensemble score used by the locked experiment.

No third-party framework source or pretrained model checkpoint is committed to
this repository.
