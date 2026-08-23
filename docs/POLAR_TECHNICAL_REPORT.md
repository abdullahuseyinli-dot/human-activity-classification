# Leakage-Safe Transfer Learning for Still-Image Posture Recognition

## A locked POLAR study with external-transfer and attribution audits

**Abdulla Huseyinli**<br>
Technical report, version 2.0.0 — 23 August 2026

[Rendered PDF](../output/pdf/polar_technical_report.pdf)

## Abstract

This study evaluates transfer learning for four-class still-image posture recognition
on a cleaned subset of POLAR. Its purpose is not only to obtain a high in-domain score,
but to determine which gains survive a strict selection boundary and what those gains
do not establish. A content audit quarantined 125 images involved in confirmed
cross-split source relationships before supervised fitting. Model adaptation,
regularization, frozen-feature classifiers, seed averaging, and ensemble weights were
selected on 13,285 development images. Nine neural fits and three classifier probes were
then completed and hash-verified before a one-time evaluation on 3,329 held-out images.

The predeclared five-component probability ensemble achieved 0.9399 macro-F1 (95%
stratified-bootstrap CI [0.9312, 0.9481]) and 0.9456 accuracy. Its paired macro-F1
interval was positive against every component. A calibrated RBF SVM was the strongest
standalone component at 0.9274 macro-F1, but a logistic probe offered much better model
size, fit time, log loss, and calibration. On an independently audited V-COCO subset,
the locked three-class ensemble fell from 0.9611 in-domain macro-F1 to 0.6669. This gap
shows that the strong POLAR result does not directly imply deployment generalization.
ConvNeXt Grad-CAM passed the attribution sanity checks more convincingly than DINOv2-B
integrated gradients, whose maps remained highly correlated after target and parameter
randomization. The report therefore treats explanations and bit-flip behavior as
bounded diagnostics, not proof of causal reasoning or system reliability.

## 1. Research questions

The experiment addresses five questions:

1. How much does training-set scale contribute to posture classification performance?
2. When is partial or full backbone adaptation preferable to a frozen representation?
3. Does an RBF SVM add useful nonlinear separation at the final feature stage?
4. Does a development-locked blend provide repeatable gains over its components?
5. Do in-domain accuracy, external transfer, attribution faithfulness, and fault
   robustness tell the same story?

The study is a reproducible benchmark on a cleaned four-class POLAR subset. It is not a
claim of state of the art: no published result was found with the same subset, source
quarantine, fixed partitions, and macro-F1 definition.

## 2. Data and leakage control

### 2.1 Primary task

POLAR v1 contains 35,324 annotated images across nine posture-level actions. This study
uses images labeled sit, stand, walk, or run and maps them to sitting, standing,
walking, and running. A secondary three-class task combines walking and running.

| Clean split | Sitting | Standing | Walking | Running | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 3,021 | 2,720 | 1,994 | 2,223 | 9,958 |
| Validation | 1,000 | 967 | 648 | 712 | 3,327 |
| Test | 1,034 | 921 | 638 | 736 | 3,329 |

### 2.2 Source-related image quarantine

Byte hashes found no exact cross-split duplicates. Perceptual retrieval nevertheless
identified color/monochrome variants, alternate crops, and related frames. Candidate
pairs required a 64-bit perceptual-hash distance of at most six and independently
resized grayscale correlation of at least 0.90. Frozen DINOv2 retrieval was used only
to propose additional candidates; embedding proximity alone could not exclude a row.

The audit confirmed 64 source-related pairs across 61 connected components. All 125
images in those components were quarantined from every primary split. The quarantine
was locked before supervised fitting and did not move test images into development.

### 2.3 External data

V-COCO train/validation supplied an independent three-class transfer audit. The local
clean manifest contains 6,640 person annotations across 4,123 images. Image-level
evaluation uses 3,761 images whose relevant people agree on one mapped label; mixed
images remain in the person-level analysis. A cross-dataset audit compared 16,614 clean
POLAR images with all 4,123 V-COCO images and found zero exact matches, zero perceptual
candidates within the declared threshold, and zero confirmed source-related pairs.

## 3. Experimental design

### 3.1 Development-only selection

The official training split fitted candidates and the validation split made every
selection decision. Search proceeded in bounded stages:

- frozen probes screened label mappings, full/person-context views, and class balance;
- adaptation compared head-only, last-stage/top-four-block, and full-backbone training;
- regularization compared dropout, augmentation, MixUp, label smoothing, class weights,
  weight decay, and random-erasing removal as controlled interventions;
- seeds 42, 52, and 62 confirmed competitive candidates;
- DINOv2-B entered as a predeclared capacity extension;
- classifier and ensemble settings were selected from development predictions only.

The primary selector was macro-F1. Differences below 0.002 were treated as a practical
tie, resolved by seed stability, log loss, expected calibration error (ECE), and cost.

### 3.2 Locked neural models

| Model | Input view | Adaptation | Augmentation | Dropout | Fixed epochs |
| --- | --- | --- | --- | ---: | ---: |
| ConvNeXt-S | Full frame | Full backbone; layer decay 0.70 | Mild | 0.10 | 12 |
| DINOv2-S | Person + 25% context | Full backbone; layer decay 0.75 | Moderate | 0.10 | 12 |
| DINOv2-B | Person + 25% context | Top four blocks | Mild | 0.10 | 7 |

Each neural component averages the probabilities of three independently seeded final
fits. Epoch counts are the median best epochs from the corresponding confirmation
runs. Final fitting combines clean train and validation data only after selection.

### 3.3 Frozen DINOv2-B classifiers

The official multilayer representation concatenates normalized CLS tokens from the
last four transformer layers and the normalized final-layer mean patch token. Features
from the full frame and a person-plus-10%-context view are concatenated to 7,680
dimensions.

Two final classifiers were retained:

- standardized multinomial logistic regression, `C=0.001`, balanced class weights;
- RBF SVM, `C=10`, `gamma=1/7680`, followed by five-fold sigmoid calibration.

The RBF settings were transferred from the declared development screen rather than
retuned on the final representation.

### 3.4 Ensemble and test gate

The pre-test ensemble is an arithmetic probability blend:

| Component | Weight |
| --- | ---: |
| ConvNeXt-S full adaptation | 0.20 |
| DINOv2-S full adaptation | 0.15 |
| DINOv2-B top-four adaptation | 0.25 |
| DINOv2-B multilayer logistic | 0.20 |
| DINOv2-B multilayer RBF SVM | 0.20 |

Before the test gate opened, all nine neural checkpoints and three fitted probes were
verified against their request, source, and artifact hashes. Their combined fit runtime
was 11,557 seconds. The official test cache then opened once. Test scores never changed
a model, weight, threshold, epoch count, or attribution method.

### 3.5 Metrics

The primary metric is macro-F1. Accuracy, balanced accuracy, weighted F1, per-class
precision/recall/F1, log loss, multiclass Brier score, and 15-bin ECE are secondary.
Uncertainty uses 10,000 class-stratified bootstrap resamples with seed 20260822. Model
comparisons use paired resamples of the same test images.

## 4. Locked POLAR results

| Candidate | Macro-F1 | 95% CI | Accuracy | Log loss | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Locked ensemble | **0.9399** | [0.9312, 0.9481] | **0.9456** | **0.1564** | 0.0291 |
| DINOv2-B multilayer RBF SVM | 0.9274 | [0.9179, 0.9362] | 0.9342 | 0.2280 | 0.0269 |
| DINOv2-B multilayer logistic | 0.9258 | [0.9164, 0.9348] | 0.9324 | 0.1764 | **0.0133** |
| DINOv2-B top four | 0.9252 | [0.9156, 0.9343] | 0.9327 | 0.2173 | 0.0420 |
| DINOv2-S full | 0.9131 | [0.9028, 0.9226] | 0.9210 | 0.2389 | 0.0298 |
| ConvNeXt-S full | 0.8914 | [0.8803, 0.9020] | 0.8994 | 0.3081 | 0.0429 |

![Held-out comparison](../assets/polar_test_comparison.png)

The ensemble's paired macro-F1 gain was +0.0484 over ConvNeXt-S, +0.0268 over
DINOv2-S, +0.0147 over adapted DINOv2-B, +0.0141 over logistic regression, and +0.0125
over the RBF SVM. All five paired 95% intervals were strictly positive. The smallest was
the RBF comparison, [0.0065, 0.0186].

The secondary collapsed three-class system reached 0.9611 macro-F1 and 0.9622
accuracy, exceeding the direct three-class logistic probe's 0.9531 macro-F1.

### 4.1 Class behavior

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Sitting | 0.992 | 0.985 | 0.989 | 1,034 |
| Standing | 0.925 | 0.939 | 0.932 | 921 |
| Walking | 0.887 | 0.873 | 0.880 | 638 |
| Running | 0.957 | 0.961 | 0.959 | 736 |

![Locked confusion matrix](../assets/polar_confusion_matrix.png)

Walking is the limiting class. Its largest error path is standing (8.2% of walking
examples), while 4.4% are predicted as running. This pattern is plausible for a single
frame, where gait phase and motion cues are partially absent.

## 5. Where the gains came from

### 5.1 More data produced the largest isolated gain

The frozen DINOv2-B learning curve rose monotonically from 0.8487 validation macro-F1
with 242 training images to 0.9150 with 9,958. The 6.64 percentage-point change is much
larger than any single regularization intervention in the seed-42 screen.

| Training images | Validation macro-F1 |
| ---: | ---: |
| 242 | 0.8487 |
| 500 | 0.8718 |
| 1,000 | 0.8810 |
| 3,000 | 0.8980 |
| 9,958 | 0.9150 |

![Learning curve](../assets/polar_scale_curve.png)

### 5.2 Regularization helped by preventing regressions

On DINOv2-S seed 42, the mild-augmentation/dropout-0.10 baseline reached 0.9184
validation macro-F1. Dropout 0.20 reached 0.9191 and moderate augmentation reached
0.9182, effectively tied at this resolution. Removing dropout fell to 0.9113; MixUp
0.20 reached 0.9120; label smoothing 0.05 reached 0.9142; inverse-frequency weighting
reached 0.9082; and removing random erasing reached 0.9124. These results support a
conservative recipe: retain dropout and image augmentation, but do not add every
regularizer by default. Multi-seed confirmation and complementarity favored the
moderate DINOv2-S variant for the final blend.

### 5.3 Capacity and representation were complementary

DINOv2-B top-four adaptation improved substantially over DINOv2-S and transferred best
to V-COCO. The frozen official multilayer representation supported strong linear and
nonlinear classifiers. The ensemble then gained another 1.25 macro-F1 points over its
best standalone component, with paired uncertainty excluding zero. The gain is
therefore attributable to complementary probability errors, not simply to selecting the
best test model after the fact.

## 6. SVM as the final-stage classifier

The RBF SVM addresses the final-stage classifier question positively but with a clear
engineering tradeoff.

| Probe | Test macro-F1 | Test log loss | External image macro-F1 | Fit time | Serialized size |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multinomial logistic | 0.9258 | **0.1764** | 0.6392 | **13.9 s** | **0.4 MB** |
| Calibrated RBF SVM | **0.9274** | 0.2280 | **0.6504** | 60.4 min | 870.9 MB |

The RBF boundary adds 0.16 macro-F1 points in-domain and transfers slightly better than
the linear probe, but it uses about 5,038 support vectors per calibration fold and is
more than 2,000 times larger. It is useful as a research component and contributes
diversity to the blend. Logistic regression is the stronger default for low-latency,
well-calibrated deployment.

## 7. External transfer

| V-COCO image-level candidate | Macro-F1 | 95% CI | Accuracy | Log loss |
| --- | ---: | ---: | ---: | ---: |
| DINOv2-B top four | **0.6732** | [0.6609, 0.6849] | **0.6751** | 1.7509 |
| Locked ensemble, collapsed | 0.6669 | [0.6550, 0.6787] | 0.6663 | **1.2512** |
| DINOv2-B multilayer RBF | 0.6504 | [0.6380, 0.6623] | 0.6453 | 2.0801 |
| DINOv2-S full | 0.6455 | [0.6329, 0.6579] | 0.6472 | 1.6232 |
| Direct three-class probe | 0.6430 | [0.6309, 0.6545] | 0.6405 | 1.2775 |
| DINOv2-B multilayer logistic | 0.6392 | [0.6274, 0.6507] | 0.6363 | 1.4022 |
| ConvNeXt-S full | 0.6208 | [0.6073, 0.6337] | 0.6272 | 1.8414 |

![External validation](../assets/polar_external_validation.png)

The 29.4-point macro-F1 gap between the in-domain collapsed ensemble and its external
image-level result is the clearest limitation in the study. V-COCO standing recall is
only 0.412 for the locked ensemble, while walking/running recall is 0.929 but precision
is 0.349. The mapped datasets therefore differ in image composition and annotation
semantics, not only pixel style. The fact that adapted DINOv2-B transfers better than
the in-domain-optimal blend suggests that task diversity and external robustness are
not aligned objectives.

## 8. Attribution audit

The audit samples 256 test images deterministically, balanced across four classes and
four source-box-area quartiles. ConvNeXt uses Grad-CAM; DINOv2-B uses 16-step integrated
gradients. Both target the locked predicted class. Perturbation operates on an 8 × 8
grid with a nested random-deletion control. Localization, equal-area person/context
occlusion, full/crop consistency, alternative-target sensitivity, and two levels of
parameter randomization are reported separately.

| Metric | ConvNeXt-S Grad-CAM | DINOv2-B integrated gradients |
| --- | ---: | ---: |
| Deletion AUC ↓ | 0.615 | 0.874 |
| Random deletion AUC | 0.778 | 0.932 |
| Random-minus-targeted gap ↑ | **0.163** | 0.058 |
| Insertion AUC ↑ | 0.896 | **0.946** |
| Person attribution mass | 0.691 | **0.848** |
| Person-mass / bbox-area lift | **2.368** | 1.095 |
| Pointing-game rate | **0.859** | 0.828 |
| Person-minus-context probability drop | **0.234** | 0.012 |
| Alternative-target Spearman ρ ↓ | **-0.491** | 0.934 |
| Randomized-head Spearman ρ ↓ | **0.183** | 0.871 |
| Randomized-adapted-layers Spearman ρ ↓ | **0.135** | 0.708 |

![Faithfulness curves and localization](../assets/polar_faithfulness.png)

![Attribution sanity checks](../assets/polar_attribution_sanity.png)

ConvNeXt provides the stronger faithfulness evidence: targeted deletion is more
selective than random deletion, person evidence has area-normalized lift, and maps
change after target or parameter randomization. DINOv2-B integrated gradients place
substantial mass within person boxes, but those boxes also occupy much of its cropped
view. More importantly, the maps retain high rank correlation when the target changes
or learned parameters are reset. DINO integrated gradients therefore fail to establish
strong class-specific causal faithfulness here. They remain useful for coarse
localization only.

Two rows per family had projected person boxes covering the full analyzed frame, so an
equal-area context region did not exist. Those rows remain in every other metric and
are explicitly excluded from the matched-context statistic. The maximum difference
between recomputed attribution-path and locked probabilities was 0.00176.

## 9. Fault-injection audit

Input faults flip exact bits in the post-resize uint8 RGB tensor before normalization.
Parameter faults symmetrically quantize each classifier weight matrix to int8, flip
exact bits, and restore the original float weights after inference. Three declared fault
seeds are averaged.

| Condition | ConvNeXt-S agreement | DINOv2-B agreement |
| --- | ---: | ---: |
| Input bit-flip rate 0.00001 | 0.996 | 1.000 |
| Input bit-flip rate 0.0001 | 0.996 | 0.996 |
| Input bit-flip rate 0.001 | 0.984 | 0.980 |
| 16 classifier-matrix flips per seed model | 1.000 | 1.000 |

![Fault robustness](../assets/polar_fault_robustness.png)

The result shows local prediction stability under these bounded corruptions. It does
not measure random faults throughout the backbone, persistent memory corruption, or
hardware failure, and it is not a substitute for a safety case.

## 10. Interpretation

Three conclusions are well supported.

First, data scale was the largest measured performance amplifier. Second, pretrained
DINOv2 representations supported both efficient linear classification and a small
nonlinear SVM gain, while selective adaptation improved external transfer. Third, the
best in-domain result required complementary models; it was not obtained by a single
regularization trick.

Two conclusions are not supported. The study does not establish an exact POLAR state
of the art because no directly comparable published benchmark was found. It also does
not establish deployment readiness: the external gap is large, and DINO integrated
gradients fail strong randomization sanity.

This makes the work most suitable as a public technical report and reproducible
portfolio article. A formal preprint would be more compelling after replication on the
remaining POLAR labels or a second independently collected posture dataset, ideally
with subject/session identities and a predeclared domain-generalization intervention.

## 11. Reproducibility and availability

The repository publishes source code, protocols, selection locks, portable metrics,
uncertainty, exclusion records, figure builders, and artifact hashes. Raw images,
checkpoints, fitted classifiers, local paths, and dense prediction/attribution arrays
remain local because of rights, size, and privacy boundaries. The tracked final evidence
manifest can be validated without those bulky artifacts.

The code targets Python 3.11 and was run with PyTorch 2.11.0+cu126 on an NVIDIA GeForce
RTX 4060 Laptop GPU. GitHub Actions independently runs lint, compilation, tests, and the
portable repository validator on Linux with CPU PyTorch.

## References

1. Ma, W., & Liang, S. POLAR: Posture-level Action Recognition Dataset. Mendeley Data,
   V1. [doi:10.17632/hvnsh7rwz7.1](https://doi.org/10.17632/hvnsh7rwz7.1).
2. Ma, W., & Liang, S. POLAR: Posture-level Action Recognition Dataset. ICSAI 2019.
   [doi:10.1109/ICSAI48974.2019.9010160](https://doi.org/10.1109/ICSAI48974.2019.9010160).
3. Oquab, M. et al. DINOv2: Learning Robust Visual Features without Supervision.
   [arXiv:2304.07193](https://arxiv.org/abs/2304.07193).
4. Liu, Z. et al. A ConvNet for the 2020s.
   [arXiv:2201.03545](https://arxiv.org/abs/2201.03545).
5. Gupta, S., & Malik, J. Visual Semantic Role Labeling.
   [arXiv:1505.04474](https://arxiv.org/abs/1505.04474).
6. Sundararajan, M., Taly, A., & Yan, Q. Axiomatic Attribution for Deep Networks.
   [arXiv:1703.01365](https://arxiv.org/abs/1703.01365).
7. Selvaraju, R. R. et al. Grad-CAM: Visual Explanations from Deep Networks via
   Gradient-based Localization. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391).
8. Adebayo, J. et al. Sanity Checks for Saliency Maps.
   [arXiv:1810.03292](https://arxiv.org/abs/1810.03292).
9. Petsiuk, V., Das, A., & Saenko, K. RISE: Randomized Input Sampling for Explanation
   of Black-box Models. [arXiv:1806.07421](https://arxiv.org/abs/1806.07421).
10. Rong, Y. et al. Consistent and Efficient Evaluation of Feature Attribution Methods.
    [PMLR 162](https://proceedings.mlr.press/v162/rong22a.html).
