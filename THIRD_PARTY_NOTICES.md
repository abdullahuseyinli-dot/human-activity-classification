# Third-party notices

The repository's MIT License covers the original source code and documentation.
It does not replace the terms of source images, dataset annotations, framework
code, or pretrained model parameters.

## COCO source images

`data/manifest.csv` records COCO image URLs, labels, fixed splits, and checksums;
the image files are downloaded by the user and are not redistributed by this
repository. Under the [COCO terms of use](https://cocodataset.org/#termsofuse),
COCO annotations and the COCO website are licensed under CC BY 4.0. The COCO
Consortium does not own the image copyrights, so every image remains subject to
its original Flickr/source terms. Users are responsible for confirming that
their intended use is permitted.

Suggested dataset citation:

> Lin, T.-Y. et al. (2014). Microsoft COCO: Common Objects in Context. ECCV.
> https://doi.org/10.1007/978-3-319-10602-1_48

## DINOv2-Small

The DINOv2-Small pretrained model is developed by Meta AI and distributed under
the [Apache License 2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE).
The upstream [model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)
documents the model's intended uses, limitations, and training-data context.

## ConvNeXt-Small and torchvision

ConvNeXt-Small is loaded with torchvision's `IMAGENET1K_V1` pretrained weights.
The torchvision source is distributed under the
[BSD 3-Clause License](https://github.com/pytorch/vision/blob/main/LICENSE).
Torchvision notes that pretrained models can also be subject to terms derived
from their training data; those upstream rights are not relicensed here.

## PyTorch Grad-CAM

The attribution evaluator uses the ROAD noisy-linear imputer from
[PyTorch Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam), distributed
under the [MIT License](https://github.com/jacobgil/pytorch-grad-cam/blob/master/LICENSE).
The Grad-CAM, HiResCAM, integrated-gradients, and transformer-rollout maps in
this repository are implemented locally so they can differentiate the exact
calibrated ensemble score used by the locked experiment.

No third-party framework source or pretrained model checkpoint is committed to
this repository.
