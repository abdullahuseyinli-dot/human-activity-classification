# Source-Overlap-Controlled Human Activity Classification

[![Quality gates](https://github.com/abdullahuseyinli-dot/human-activity-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/abdullahuseyinli-dot/human-activity-classification/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/Code-MIT-0F766E.svg)](LICENSE)
[![Study report: v2.0.0](https://img.shields.io/badge/study_report-v2.0.0-0F766E.svg)](output/pdf/vcoco_v2_external_transfer_v2.0.0.pdf)

A leakage-audited study of still-image human activity classification across POLAR and
V-COCO. The repository covers source-overlap control, transfer learning, frozen
representations, person-centric multiview features, factorized targets, calibrated
classifiers, selection locks, one-time test gates, attribution sanity checks, and
bounded fault injection.

## V-COCO v2: person-level external transfer

The current [technical report](output/pdf/vcoco_v2_external_transfer_v2.0.0.pdf),
[Markdown source](docs/VCOCO_V2_EXTERNAL_TRANSFER.md),
[release notes](docs/releases/POLAR_STUDY_V2.0.0.md), and
[SHA-256 manifest](results/polar_study_v2.0.0_manifest.json) document the complete
follow-up study.

![Official V-COCO test comparison](assets/vcoco_v2_official_test_comparison.png)

The selected system uses two aspect-preserving DINOv2-B person views, cross-fitted
base probabilities, and five box-geometry features. It was trained on the official
V-COCO training split, selected on validation, refitted on
6,640 development people, and evaluated
after the test labels were opened once.

| Official-test method | Macro-F1 | Accuracy | Balanced accuracy | Log loss | ECE |
| --- | --- | --- | --- | --- | --- |
| Scale-conditioned DINO stack | 0.8663 | 0.8795 | 0.8636 | 0.2902 | 0.0076 |
| Historical source-only DINO | 0.7071 | 0.7010 | 0.7674 | 1.6525 | 0.2288 |

The official-test gain is **+0.1592 macro-F1**, with
a 95% image-cluster bootstrap interval of
**[+0.1454, +0.1735]**.
Validation macro-F1 was **0.8594** and test macro-F1 was
**0.8663** on 6,077 people from
3,708 images.

| Class | Stack F1 | Baseline F1 | Change | Support |
| --- | --- | --- | --- | --- |
| Sitting | 0.9496 | 0.9033 | +0.0462 | 1,882 |
| Standing | 0.8791 | 0.6209 | +0.2582 | 2,962 |
| Walking/running | 0.7702 | 0.5970 | +0.1732 | 1,233 |

### Measured sources of improvement

- **Person scale and crop construction:** the stack gains
  **+0.2202 macro-F1** in the smallest box-area quartile,
  compared with **+0.0938** in the largest quartile.
  Aspect-preserving person views remove most of the historical dependence on person
  height and box area.
- **Multiview complementarity:** the stack improves over the best single-view DINO
  control by **+0.0118**, with a 95% interval of
  **[+0.0037, +0.0201]**.
- **Target structure:** factorizing seated/upright posture and
  stationary/locomoting motion improves over the matched flat classifier by
  **+0.0111**, interval
  **[+0.0056, +0.0166]**.
- **Neural adaptation:** LP-FT with AugMix is close to the selected stack; their
  development difference is **+0.0014**, interval
  **[-0.0075, +0.0102]**.
  The frozen stack therefore keeps the simpler final fit.
- **Calibrated abstention:** at 90% coverage, the stack reaches
  **0.9163 accuracy** and **0.9011
  macro-F1**. At 70% coverage, it reaches **0.9605 accuracy**
  and **0.9459 macro-F1**.

![V-COCO person-scale gains](assets/vcoco_v2_scale_gain.png)

![V-COCO selective prediction](assets/vcoco_v2_selective_prediction.png)

## POLAR v1: source benchmark

The original four-class benchmark compares ConvNeXt and DINOv2 adaptation, linear and
nonlinear classifiers on frozen representations, and a development-locked probability
ensemble.

![Held-out POLAR comparison](assets/polar_test_comparison.png)

The ensemble reached **0.940 macro-F1** (95% stratified
bootstrap interval **[0.931,
0.948]**) and **0.946
accuracy** on 3,329 held-out images. Its smallest paired gain over a component
was **+0.013 macro-F1**, interval
**[+0.007,
+0.019]**.

| Predeclared candidate | Macro-F1 | 95% CI | Accuracy | Log loss | ECE |
| --- | --- | --- | --- | --- | --- |
| Locked probability ensemble | 0.940 | [0.931, 0.948] | 0.946 | 0.156 | 0.029 |
| DINOv2-B multilayer + calibrated RBF SVM | 0.927 | [0.918, 0.936] | 0.934 | 0.228 | 0.027 |
| DINOv2-B multilayer + logistic regression | 0.926 | [0.916, 0.935] | 0.932 | 0.176 | 0.013 |
| DINOv2-B, top four blocks adapted | 0.925 | [0.916, 0.934] | 0.933 | 0.217 | 0.042 |
| DINOv2-S, full adaptation | 0.913 | [0.903, 0.923] | 0.921 | 0.239 | 0.030 |
| ConvNeXt-S, full adaptation | 0.891 | [0.880, 0.902] | 0.899 | 0.308 | 0.043 |

The secondary three-class mapping, with walking and running combined, reached
**0.961 macro-F1** and **0.962 accuracy**.

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Running | 0.957 | 0.961 | 0.959 | 736 |
| Sitting | 0.992 | 0.985 | 0.989 | 1,034 |
| Standing | 0.925 | 0.939 | 0.932 | 921 |
| Walking | 0.887 | 0.873 | 0.880 | 638 |

![Locked POLAR confusion matrix](assets/polar_confusion_matrix.png)

### What changed the POLAR result

- The frozen DINOv2-B validation curve rose from
  **0.849** at 242 training
  images to **0.915** at
  9,958.
- The locked neural configurations use dropout 0.10, no MixUp, no label smoothing,
  and either mild or moderate augmentation. Interventions that reduced validation
  performance remain visible in the evidence tables.
- The final blend fixed its development-selected weights at ConvNeXt-S
  20%, DINOv2-S
  15%, adapted DINOv2-B
  25%, logistic regression
  20%, and calibrated RBF SVM
  20%.

![POLAR data-scale curve](assets/polar_scale_curve.png)

### Linear versus nonlinear final-stage classifiers

The calibrated RBF SVM is the strongest standalone POLAR component at 0.927
macro-F1. The multinomial logistic model reaches 0.926 with better log loss
(0.176 versus 0.228), fits in 13.9 seconds, and serializes to
0.4 MB. The RBF pipeline takes
60.4 minutes and serializes to
870.9 MB. The SVM contributes a small nonlinear
margin; logistic regression is the lighter calibrated endpoint.

## Attribution and bounded fault response

The attribution audit uses a deterministic 256-image cohort balanced by class and
person-box-area quartile. ConvNeXt Grad-CAM has a targeted-versus-random deletion gap
of **0.163**, concentrates
**2.37x** more attribution in the
person box than uniform area, and produces a person-minus-context probability drop of
**0.234**. DINOv2-B
integrated gradients show **1.10x**
area-normalized lift but retain high correlations after target and parameter
randomization, so the maps are used as coarse localization diagnostics.

![Attribution faithfulness](assets/polar_faithfulness.png)

![Attribution sanity checks](assets/polar_attribution_sanity.png)

Bit-flip experiments are reported separately. At a 0.1% input bit-flip rate,
prediction agreement with the clean models is
**0.984** for ConvNeXt-S and
**0.980** for DINOv2-B. Sixteen flips
per quantized classifier-weight matrix retain
**1.000** and
**1.000** agreement, respectively, on
the same cohort.

![Fault robustness](assets/polar_fault_robustness.png)

## Evaluation controls and evidence lineage

### POLAR

- Clean split: 9,958 train, 3,327 validation, and
  3,329 test images.
- 125 images in
  61 confirmed cross-split source-related
  components were quarantined before supervised fitting.
- Candidate selection, blend weights, epochs, and classifier settings were fixed on
  development evidence before the test cache opened.
- The portable summaries share selection-lock SHA-256 fa3fb7c80a073d29048afb8e0b8da1fb17f5ade9721630347c37523714cca187.

### V-COCO v2

- Official memberships remain intact after quarantining 60 test images used in the
  earlier external audit.
- Source-image groups stay together during cross-validation and image-cluster
  bootstrapping.
- The selected stack, evaluator, dependencies, and historical baseline were bound
  before the single official test-label open.
- Protocol lock: 3a90d6720a6cf5250b995820801199eca611706d514e7b5de2c83bea03f5a143.
- Selection lock: 4c0d13c05d537e28066000092e88c37f66b937111516715c8f1756cab162e10d.

Checkpoints, image paths, dense probabilities, and large feature tensors remain
outside Git. The tracked evidence is path-sanitized and hash-indexed.

## Reports and release files

| Artifact | Purpose |
| --- | --- |
| [V-COCO v2 report](output/pdf/vcoco_v2_external_transfer_v2.0.0.pdf) | Current person-level transfer study |
| [V-COCO v2 evidence](results/vcoco_v2/README.md) | Locks, metrics, uncertainty, and mechanism tables |
| [POLAR v1 report](output/pdf/polar_public_report_v1.0.0.pdf) | Original four-class source benchmark |
| [Executed notebook](human_activity_classification.ipynb) | Compact, code-backed result walkthrough |
| [Result lineage](docs/RESULT_LINEAGE.md) | Source benchmark artifact map |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | Dataset and model terms |

## Repository layout

~~~text
.
|-- human_activity_classification.ipynb  # executed evidence narrative
|-- src/hac/                             # reusable data, model, metric, and audit code
|-- experiments/                         # staged fitting and evaluation runners
|-- tools/                               # dataset, export, validation, and figure utilities
|-- results/                             # portable locks, metrics, uncertainty, and hashes
|-- assets/                              # publication figures
|-- docs/                                # protocols, reports, release notes, and lineage
|-- tests/                               # fast invariants and evidence-contract tests
+-- .github/workflows/ci.yml             # Linux quality gates
~~~

## Reproduce the environment

Python 3.11 is the recorded runtime. Install the PyTorch build appropriate for the
machine, then install the project:

~~~bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,dev]"
~~~

The staged commands and acceptance boundaries are documented in
[experiments/README.md](experiments/README.md). Validate a checkout with:

~~~bash
python -m ruff check .
python -m compileall -q src experiments tools
python -m pytest
python tools/validate_repository.py
python tools/build_study_release_manifest.py --check
~~~

## Citation, references, and license

GitHub's citation panel reads [CITATION.cff](CITATION.cff). The Zenodo DOI will be
added to the citation metadata after the v2 deposit.

> Huseyinli, A. (2026). *Improving Person-Level External Transfer on V-COCO*
> (Version 2.0.0) [Technical report].
> https://github.com/abdullahuseyinli-dot/human-activity-classification

- [POLAR dataset](https://doi.org/10.17632/hvnsh7rwz7.1)
- [V-COCO](https://arxiv.org/abs/1505.04474)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [ConvNeXt](https://arxiv.org/abs/2201.03545)
- [SigLIP2](https://arxiv.org/abs/2502.14786)
- [AugMix](https://arxiv.org/abs/1912.02781)
- [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292)

Original code and documentation are MIT licensed. Dataset images, annotations, and
pretrained weights retain their upstream terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
