---
title: When a Still Image Is Not Enough
subtitle: Motion identifiability and budgeted temporal inference across V-COCO and Okutama-Action
short_title: Motion Identifiability and Budgeted Temporal Inference
author: Abdulla Huseyinli
date: 24 August 2026
status: Independent technical report
version: 3.0.0
document_type: Technical report
subject: Person-level human activity classification under domain and temporal shift
keywords: human activity recognition, motion identifiability, temporal inference, V-COCO, Okutama-Action
repository: https://github.com/abdullahuseyinli-dot/human-activity-classification
---

# Abstract

This study examines a specific failure mode in person-level activity classification:
some motion labels cannot be identified reliably from a single image. The earlier
person-level V-COCO system improved three-class macro-F1 through scale-aware crops,
factorized posture/motion targets, and calibrated multiview features, but its remaining
errors were concentrated in the standing-versus-locomotion boundary. The follow-up
separates label ambiguity, representation quality, spatial evidence, temporal evidence,
and target-domain adaptation.

A fixed single-rater pilot records posture, visible translation, gait, and visibility
for 130 blind presentations representing 126 unique V-COCO people. A matched
representation screen compares DINOv2-B, DINOv3-B, and SigLIP2-B. Spatial controls test
context, resolution, boundary, masking, blur, geometry, and box perturbation. The
external experiment then trains static and short-clip models on Okutama-Action under
scenario-grouped development, calibration, and sealed confirmation partitions.

On 1,771 confirmation examples from five scenarios, an eight-frame, 0.5-second model
reaches 0.7854 macro-F1 and 0.7708 accuracy. The matched static model reaches 0.7458 and
0.7301. The paired macro-F1 gain is 0.0396, with a 95% scenario-cluster bootstrap
interval of [0.0202, 0.0568]. A predeclared router applies the clip model to 50% of the
examples and reaches 0.7817 macro-F1; its gain over static is 0.0360 [0.0144, 0.0604].
Static distillation does not improve confirmation macro-F1. Within the evaluated model
families, missing temporal information explains a substantial share of the remaining
standing-versus-locomotion errors; additional static capacity alone does not resolve
them.

# 1. Research question

The project began with a source-overlap-controlled four-class POLAR benchmark and a
person-level V-COCO transfer study. The V-COCO model used two DINOv2-B person views,
box geometry, cross-fitted base probabilities, and factorized posture/motion targets.
Its official-test macro-F1 was 0.8663. Walking/running remained the weakest class at
0.7702 F1, and several apparently incorrect predictions were difficult to resolve from
one frame.

The extension asks four questions:

1. How much of the remaining gap reflects the label ontology rather than model error?
2. Do stronger frozen representations or additional spatial evidence resolve the
   standing-versus-locomotion boundary?
3. Does short temporal context improve the same three-class task on an independent
   tracked video dataset?
4. Can a static model identify which examples deserve the additional clip inference?

The target classes are sitting, standing, and walking/running. Walking and running are
combined because their shared property is visible locomotion; the provider gait subtype
is retained for analysis.

# 2. Evaluation design

## 2.1 Separation of evidence

The experiment has three distinct evidence sources:

- V-COCO source tags support nested development screens over 6,640 people from 4,123
  images.
- The 130-presentation human pilot is a descriptive mechanism audit. Its labels do not
  fit, select, calibrate, or rank models.
- Okutama-Action supplies tracked aerial video for the temporal experiment. Provider
  train is divided by scenario into train, validation, and calibration partitions;
  provider test is the confirmation partition.

The consumed V-COCO official test is not reopened. The Okutama confirmation archive is
opened only after the representation, temporal architecture, seed ensemble, epoch
counts, calibration temperatures, routing budgets, and prediction-set thresholds have
been fixed and hashed.

## 2.2 Okutama partitions

| Partition | Samples | Scenarios | Role |
| --- | ---: | ---: | --- |
| Train | 4,977 | 11 | Model fitting and recording-grouped cross-fitting |
| Validation | 1,383 | 3 | Teacher and student selection |
| Calibration | 1,979 | 3 | Temperature scaling, routing calibration, prediction-set thresholds |
| Confirmation | 1,771 | 5 | One locked evaluation |

Synchronized drone views from the same scenario remain together. No tracked person
crosses partitions. The confirmation set contains 351 sitting, 693 standing, 691
walking, and 36 running examples after the declared action-boundary and tracking
filters.

## 2.3 Metrics and uncertainty

Macro-F1 is primary because the class distribution is uneven. Accuracy, balanced
accuracy, locomotion F1, log loss, Brier score, and expected calibration error are also
reported. Confirmation differences use 10,000 paired resamples over scenario clusters.
The planned macro-F1 and locomotion tests are Holm-adjusted across the evaluated
families and clip budgets. Subgroup tables retain sample counts so small slices remain
visible.

# 3. Human motion audit

The annotation interface hides source tags, model predictions, confidence, and sampling
cohort. The rater records four independent axes:

| Axis | Values |
| --- | --- |
| Posture | seated, upright, other, indeterminate |
| Visible translation | stationary, locomoting, transition, not inferable |
| Gait | walking, running, not applicable, indeterminate |
| Visibility | clear, occluded, truncated, too small |

The fixed prefix contains 130 presentations, 126 unique content rows, and 121 unique
images. Four completed hidden repeat pairs agree exactly on every axis. That repeat
check measures only within-rater consistency; no interrater estimate is available.

The audit exposed two ontology boundaries. First, 32 of the 126 unique people were
locomoting but had gait marked `not applicable`: visible translation existed, but it
was not walking or running. Second, 41 used the `other` posture category rather than
seated or upright. Only 91 rows resolved cleanly into the model's three classes.

Within the probability sample, the source tag agreed with the human-resolved class on
52 of 61 resolvable rows, while the model prediction agreed on 48. Within the
error-enriched sample, the source agreed on 22 of 30 and the prediction on 8. Of the
31 reviewed locomotion-to-standing errors, 18 supported the source tag, four supported
the prediction, and nine did not resolve into the three-class ontology. The pilot
therefore shows both effects: some source tags are a poor fit for the visual question,
but many selected errors are genuine model failures.

# 4. Static evidence experiments

## 4.1 Factorized fusion

All V-COCO development comparisons use nested, source-image-grouped folds. The best
cached-feature candidate is a factorized DINO/SigLIP reliability stack.

| Development family | Macro-F1 | Accuracy | Locomotion F1 |
| --- | ---: | ---: | ---: |
| DINO + SigLIP factorized reliability stack | 0.8697 | 0.8870 | 0.7616 |
| DINO + SigLIP linear SVM control | 0.8641 | 0.8827 | 0.7470 |
| DINO factorized probability stack | 0.8560 | 0.8762 | 0.7331 |
| DINO flat probability stack | 0.8543 | 0.8747 | 0.7292 |

The reliability stack passes the declared general and locomotion promotion rules. It
shows that model complementarity and target structure still help, but this source-tag
result does not answer whether a single image contains enough motion evidence.

## 4.2 Representation screen

The frozen comparison holds folds, views, classifier family, and tuning budget fixed.

| Representation | Macro-F1 | Accuracy | Locomotion F1 |
| --- | ---: | ---: | ---: |
| DINOv2-B | 0.8395 | 0.8636 | 0.7006 |
| DINOv3-B | 0.8367 | 0.8601 | 0.7013 |
| SigLIP2-B | 0.8358 | 0.8563 | 0.7172 |

DINOv3-B does not improve macro-F1 over DINOv2-B in this matched setup. SigLIP2-B has
the highest locomotion point estimate, but its paired interval does not satisfy the
specialist rule. DINOv2-B is consequently fixed as the Okutama frame backbone.

## 4.3 Spatial controls

The spatial screen runs 25,380 CUDA logistic fits with no iteration-limit failures.
The best point estimate, two-view DINOv2 features at 224 and 336 pixels with
reliability fusion, reaches 0.8421 macro-F1. Adaptive context, a 448-pixel third view,
masked context, blurred context, box geometry, and box perturbation do not pass the
promotion threshold. Parameter-efficient neural adaptation is therefore not started.

This negative gate matters: the experiment does not spend additional capacity on a
family whose controlled spatial comparison failed to establish a useful gain.

# 5. External transfer and temporal models

## 5.1 Source-only and few-shot transfer

The source-only DINOv2 model uses no Okutama labels during fitting or selection. It
reaches 0.6168 macro-F1 over the complete development partition and 0.5735 on sealed
confirmation. The low scores on both Okutama partitions are consistent with a
substantial transfer gap; this design does not apportion that gap among viewpoint,
person scale, scene composition, and provider annotation policy.

Few-shot target heads are evaluated on validation over five fixed seeds. At 32 labels
per class, the target-only factorized linear head averages 0.6295 macro-F1 and 0.6911
locomotion F1. Source-probability recalibration averages 0.6194 and 0.6607. Lower
budgets vary substantially across seeds. Small target samples are useful for diagnosis,
but they do not replace scenario-diverse target training.

## 5.2 Architecture and training

Each frame contributes a tight DINOv2-B embedding, a 25% context embedding, and box
geometry. The static model is a 256-unit factorized classifier. Temporal candidates
sample either eight or 16 frames over 0.5 or 1.0 seconds. The selected teacher projects
features to 256 dimensions, uses two pre-normalized Transformer encoder layers with
four attention heads and a 512-unit feed-forward block, then applies masked attention
pooling and factorized posture/motion heads.

The locked training configuration uses dropout 0.10, AdamW at 2e-4, weight decay 0.01,
automatic mixed precision, class-balanced loss, label smoothing 0.02, a batch size of
64, gradient clipping at 1.0, warmup, early stopping, and five seeds. All 100 temporal
development, cross-fit, student, and final fits ran on an NVIDIA RTX 4060 Laptop GPU
with PyTorch 2.11.0 and CUDA 12.8. CPU fallback is disabled.

## 5.3 Teacher selection and cross-fitting

The selected teacher samples eight frames over 0.5 seconds. On validation, its ensemble
reaches 0.7806 macro-F1, compared with 0.7414 for the static ensemble. The 8-frame,
1.0-second candidate is close at 0.7794, and the 16-frame candidate reaches 0.7759.
The half-second model is retained under the primary macro-F1 ordering.

Recording-grouped cross-fitting creates teacher distributions for every one of the
4,977 training rows without predicting a row from a model that fitted its recording.
The teacher reaches 0.7165 out-of-fold macro-F1, compared with 0.6672 for static. A
temporal-benefit target is positive when the teacher improves true-class log likelihood
by at least 0.2; 1,016 rows, or 20.4%, meet that definition.

## 5.4 Static students and routing

The distilled student combines supervised loss with cross-fitted teacher
distributions. The identifiability-conditioned student also predicts whether the clip
teacher is likely to improve the example. Its validation routing score has average
precision 0.2508 against a positive prevalence of 0.1338, a gain of 0.1170. This passes
the validation gate for fixed-budget routing.

On confirmation, the distilled student is essentially level with static in macro-F1:
the difference is -0.0002 with a 95% interval of [-0.0187, 0.0127]. Its locomotion F1
is 0.7350, 0.0089 above static, but the interval also crosses zero. Distillation can
reshape the static decision boundary; it cannot reconstruct motion that the center
frame does not contain.

# 6. Confirmation results

## 6.1 Aggregate comparison

| Method | Clip fraction | Macro-F1 | Accuracy | Locomotion F1 | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source-only static transfer | 0% | 0.5735 | 0.5731 | 0.5036 | 0.9946 |
| Target-supervised static | 0% | 0.7458 | 0.7301 | 0.7261 | 0.6404 |
| Distilled static student | 0% | 0.7456 | 0.7307 | 0.7350 | 0.6331 |
| Routed student + teacher | 10% | 0.7580 | 0.7436 | 0.7449 | 0.6206 |
| Routed student + teacher | 25% | 0.7645 | 0.7516 | 0.7553 | 0.6055 |
| Routed student + teacher | 50% | 0.7817 | 0.7679 | 0.7692 | 0.5822 |
| Temporal teacher | 100% | 0.7854 | 0.7708 | 0.7692 | 0.5747 |

The temporal teacher improves macro-F1 by 0.0396 [0.0202, 0.0568] and accuracy by
0.0407. Its per-class F1 changes are +0.0289 for sitting, +0.0469 for standing, and
+0.0431 for walking/running; all three paired class intervals are above zero. The
Holm-adjusted macro-F1 p-value is 0.0016.

The 50% route reaches 0.7817 macro-F1, 0.0037 below the full teacher, and exactly
matches its 0.7692 locomotion F1. Its gain over static is 0.0360 [0.0144, 0.0604], with
a Holm-adjusted macro-F1 p-value of 0.0012. It retains 90.7% of the full temporal
macro-F1 gain while sending half of the examples to the clip model. The 10% and 25%
point estimates improve monotonically, but their cluster intervals cross zero.

## 6.2 Scenario and mechanism slices

The teacher improves macro-F1 in four of five confirmation scenarios. Scenario 1.9 is
nearly unchanged at -0.0022. Gains are larger in scenario 2.1 (+0.0636) and scenario
2.3 (+0.0571) than in scenario 1.8 (+0.0215). The pattern rules out a result driven by
one single scenario, while also showing that temporal value depends on scene
conditions.

| Slice | Samples | Teacher change | 50% route change |
| --- | ---: | ---: | ---: |
| Tiny people | 828 | +0.0499 | +0.0480 |
| Small people | 938 | +0.0332 | +0.0287 |
| Transition windows | 105 | +0.0318 | +0.0438 |
| Non-transition windows | 1,666 | +0.0397 | +0.0349 |
| Any window occlusion | 65 | +0.0171 | +0.0498 |
| No window occlusion | 1,706 | +0.0402 | +0.0354 |
| Drone 1 | 1,058 | +0.0289 | +0.0307 |
| Drone 2 | 713 | +0.0558 | +0.0437 |

The 50% route is particularly useful on the predefined transition and occlusion
slices, where it exceeds the full-teacher point gain. One plausible explanation is
that the router keeps the static student on cases where the full clip is distracting
and invokes temporal processing where motion evidence is useful. These subgroup
differences are mechanism evidence, not separately powered selection tests; the
occlusion slice contains only 65 examples.

## 6.3 Calibration and prediction sets

Temperature scaling reduces calibration-set log loss for every family. On
confirmation, the teacher has ECE 0.0292 and the 50% route 0.0281. Split-conformal APS
sets achieve empirical coverage 1.0 at both declared error levels, but their mean sizes
are 2.98 and 2.92 out of three classes. The sets are valid but too wide to support a
useful selective endpoint under this domain shift.

# 7. Findings

The experiments support five conclusions.

1. **Short temporal evidence resolves errors that static scaling and context do not.**
   The matched spatial families fail promotion, while the temporal model produces a
   positive confirmation interval and improves every class F1.
2. **The missing information cannot be distilled reliably into the same center
   frame.** The distilled student improves the development point estimate but is level
   with static on confirmation.
3. **Temporal processing can be allocated selectively.** A 50% clip budget preserves
   90.7% of the full teacher's macro-F1 gain and all of its locomotion F1 gain.
4. **A newer representation alone is not the deciding factor here.** DINOv3-B and
   SigLIP2-B do not displace DINOv2-B under the matched source-tag screen.
5. **The label boundary contains real ontology mismatch.** The pilot includes non-gait
   locomotion and non-seated/non-upright postures that cannot be represented cleanly by
   the three output classes, alongside errors that clearly support the source tag.

Together, these results shift the engineering question from "which still-image model
is larger?" to "when is the label identifiable from a still image, and when should the
system acquire temporal evidence?"

# 8. Limitations and next measurements

The annotation pilot has one rater and an enriched sampling design. It is adequate for
mechanism discovery, not a population estimate or a human-agreement benchmark. A
future annotation release should add an independent second pass and blinded
adjudication.

Okutama is aerial, low-resolution, and provider-labeled. The experiment evaluates
tracked person classification, not the dataset's original spatiotemporal action
detection metric. Results should be reproduced on a second independent video domain
with continuous tracks and compatible actions.

Only 36 confirmation examples carry the provider `Running` subtype, and only five
scenario clusters are available for uncertainty. The scenario bootstrap is more
appropriate than treating 1,771 rows as independent, but a larger number of held-out
scenarios would tighten the estimate.

Normalized pose is unavailable in the complete development cache, so the planned
pose/velocity SVM remains unresolved. Future work can add measured pose only when it is
available consistently across the full split. No pose values were imputed because
imputation would not provide observed pose evidence.

The current router ranks temporal benefit but does not optimize measured wall-clock
latency, energy, or memory. The next system experiment should measure those quantities
on uncached frames and compare fixed budgets under identical hardware.

# 9. Reproducibility and artifact boundary

The portable package in [`results/vcoco_v3`](../results/vcoco_v3/README.md) contains
the confirmation tables, paired uncertainty, subgroup evidence, routing curves,
development screens, annotation aggregates, protocol lineage, and SHA-256 inventory.
The execution order is recorded in
[`VCOCO_V3_EXECUTION_RUNBOOK.md`](VCOCO_V3_EXECUTION_RUNBOOK.md).

Dataset archives, images, model weights, checkpoints, embeddings, dense probabilities,
and private annotation rows are not committed. Their hashes remain in the local run
evidence. The exported package can be rebuilt with:

```powershell
.\.venv\Scripts\python.exe tools\export_vcoco_v3_results.py
.\.venv\Scripts\python.exe tools\check_vcoco_v3_readiness.py
```

# References

1. Barekatain, M., Marti, M., Shih, H.-F., Murray, S., Nakayama, K., Matsuo, Y., and
   Prendinger, H. "Okutama-Action: An Aerial View Video Dataset for Concurrent Human
   Action Detection." CVPR Workshops, 2017.
2. Gupta, S. and Malik, J. "Visual Semantic Role Labeling." 2015.
3. Oquab, M. et al. "DINOv2: Learning Robust Visual Features without Supervision."
   2023.
4. Simeoni, O. et al. "DINOv3." 2025.
5. Tschannen, M. et al. "SigLIP 2: Multilingual Vision-Language Encoders with Improved
   Semantic Understanding, Localization, and Dense Features." 2025.
