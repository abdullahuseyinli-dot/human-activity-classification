# POLAR scale study protocol

Protocol version: **1.2.0**
Locked: **2026-08-22, before image download, duplicate inspection, model fitting, or
access to POLAR test labels by the study runner**

### Amendment 1.2.0: layer-wise decay for full adaptation

Locked on **2026-08-22 after the frozen probe and head-only engineering baseline,
before partial/full adaptation and without loading any POLAR test row**. Official
fine-tuning implementations for transformer and ConvNeXt backbones decay learning
rates toward earlier layers, preserving general low-level features while allowing
task-specific change near the output. Full-backbone DINOv2 candidates therefore
use layer decay 0.75, matching the published BEiT-family recipe, and full-backbone
ConvNeXt candidates use the official ConvNeXt value 0.70. Head and partial-depth
candidates retain the two-rate policy so adaptation depth remains identifiable.

Primary implementation references:

- [Microsoft UNILM/BEiT fine-tuning](https://github.com/microsoft/unilm/blob/master/dit/classification/README.md)
- [Meta ConvNeXt fine-tuning](https://github.com/facebookresearch/ConvNeXt/blob/main/TRAINING.md)
- [DINOv2 paper and official code](https://github.com/facebookresearch/dinov2)

### Amendment 1.1.0: source-related split contamination

Locked on **2026-08-22 before any feature extraction or model fitting**. The
content audit reproduced every declared count and found no byte-identical
cross-split files. It did, however, reveal colour/monochrome renderings,
alternate crops, and adjacent frames from the same Getty capture across official
boundaries. These are materially easier than independent images and can reward
source or scene memorization.

The primary clean protocol therefore joins candidate pairs when their 64-bit
pHash distance is at most six and the Pearson correlation of their independently
resized 128 by 128 grayscale pixels is at least 0.90. Every image in a connected
cross-split component is quarantined from all primary partitions. The threshold,
pair table, component membership, and changed class counts are preserved. The
unfiltered official split may be reported only as a sensitivity baseline labelled
`OFFICIAL_UNFILTERED_SOURCE_OVERLAP`; it cannot select a model or support the
primary claim.

As a second audit, frozen upstream DINOv2-Small full-frame embeddings may retrieve
cross-split neighbours with cosine similarity at least 0.985. Embedding proximity
alone never removes an image. A retrieved pair is added to quarantine only when
independent source evidence or normalized-pixel inspection confirms the same
rendered asset, capture sequence, or scene. That decision and evidence must be
recorded before supervised fitting.

## Research question

This extension asks how data scale, adaptation depth, regularization, and
human-versus-context framing affect transfer learning for still-image activity
recognition. It also tests whether DINOv2 and ConvNeXt make complementary errors
and whether their explanations localize evidence on the annotated person.

The existing 285-image COCO experiment is retained as a **legacy endpoint**. Its
43-image test split has already been inspected and is not used for new model
selection or presented as a pristine replication.

## Primary dataset and task

The primary dataset is POLAR version 1, published with DOI
[`10.17632/hvnsh7rwz7.1`](https://doi.org/10.17632/hvnsh7rwz7.1). The source release
contains 35,324 images, nine action labels, person boxes, and official
train/validation/test identifiers. This study selects records whose single target
person is labelled `sit`, `stand`, `walk`, or `run`.

Expected target counts from the annotation-only audit are:

| Official split | sit | stand | walk | run | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 3,043 | 2,740 | 2,002 | 2,230 | 10,015 |
| validation | 1,015 | 980 | 654 | 716 | 3,365 |
| test | 1,043 | 936 | 641 | 739 | 3,359 |

Two predeclared label spaces are reported:

1. **Four class (primary):** sitting, standing, walking, running.
2. **Three class (secondary):** sitting, standing, walking/running. The secondary
   result compares direct three-class training with summing the walking and
   running probabilities of the selected four-class system.

The official train split is used for fitting and the official validation split
for every selection decision. Official test examples remain inaccessible to the
training process until the final lock is complete. Train and validation may be
combined only after the lock, for the single final fit.

## Source integrity and redistribution boundary

The expected multipart image archive is:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `JPEGImages.z01` | 734,003,200 | `2c70432788d221627ac5a7eeb67a044acf7e39e3f51323cd253e2fa3f946831d` |
| `JPEGImages.z02` | 734,003,200 | `ae23efcbd37dc96b3bc052ff703a51ec594c52a0101e8bdb356c382dd1bc3865` |
| `JPEGImages.z03` | 734,003,200 | `ce622c898f7898e288619632c3e84b490cd40683cb771e0f185fcfb545d68b6a` |
| `JPEGImages.z04` | 734,003,200 | `3e7ef5f09fcf6bb7ef9160b43f4a23ee7036308231eba553e61535b07d4860a0` |
| `JPEGImages.zip` | 153,773,946 | `a911de85a38f5d428408f2f377f839caa568d023f6c4b7c11cae46e2e4de4846` |

The Mendeley record is CC BY 4.0, while annotation metadata identifies Getty as
the upstream source for image names. Consequently, this repository will not
redistribute POLAR images. It will publish only download/audit code, provenance,
content hashes, exclusion records, aggregate metrics, and non-reconstructive
derived evidence.

## Leakage and duplicate audit

Before training, the builder must:

- verify every archive checksum and every selected image can be decoded;
- verify official split identifiers are disjoint and reproduce the declared
  counts;
- compute image SHA-256, dimensions, perceptual hash, and annotation fingerprints;
- compare hashes across official splits and against the legacy COCO manifest;
- flag cross-split perceptual-hash distances of six bits or fewer for review;
- record, never conceal, corrupt files, missing files, label conflicts, and
  quarantine decisions.

An exact duplicate, a confirmed same-image derivative, or a source-related pair
meeting Amendment 1.1.0 across an official boundary is excluded from **all**
affected study splits. Perceptual-hash proximity alone is not enough: confirmation
also requires the locked normalized-pixel rule, a visual check, or matching source
evidence. Exclusions are immutable after fitting begins. No test record may be
moved into train or validation.

## Views and localization controls

The following deterministic views may be selected on validation data:

- full frame;
- target-person box with 10% proportional context;
- target-person box with 25% proportional context;
- a late probability fusion of full-frame and the best person-context view.

Boxes are clipped to the image boundary. Degenerate boxes are rejected and
recorded. Evaluation views use deterministic resize/center-crop operations.
Training views apply spatial augmentation only after the declared crop is formed.

## Models and search stages

The required model families are ImageNet-pretrained ConvNeXt-Small and
DINOv2-Small. Pretrained weights are identified by exact upstream model name and
locally recorded file hashes. DINOv2-Base may enter only as a predeclared capacity
extension if it fits the same hardware and validation protocol; it cannot replace
the required Small comparison.

Search is deliberately staged:

1. **Probe screen:** frozen-backbone features compare label space, view, and
   imbalance handling cheaply.
2. **Adaptation screen:** head-only, partial (last ConvNeXt stage or top four
   DINOv2 blocks), and full-backbone training are compared with seed 42.
3. **Regularization screen:** validation compares dropout `{0.0, 0.1, 0.2}`,
   mild versus moderate augmentation, MixUp `{0.0, 0.2}`, label smoothing
   `{0.0, 0.05}`, and unweighted versus inverse-frequency cross-entropy. These
   are staged contrasts, not a full Cartesian search.
4. **Confirmation:** the selected candidate and any candidate within 0.5
   macro-F1 percentage points are rerun with seeds 42, 52, and 62.
5. **Capacity/ensemble check:** an eligible DINOv2-Base run and validation-locked
   DINOv2/ConvNeXt probability blends are compared if resources permit.

Early stopping observes validation macro-F1 with a minimum of three epochs and
patience four. The maximum is 20 epochs for full adaptation and 30 for probes or
partial adaptation. Automatic mixed precision and gradient accumulation are
allowed, but effective batch size and optimizer steps are recorded. AdamW is the
default optimizer. Search ranges are head learning rate `1e-4` to `1e-3`, backbone
learning rate `2e-6` to `5e-5`, and weight decay `1e-4` to `5e-2`; a one-cycle or
cosine schedule with warm-up may be selected on validation and then fixed.

The primary selector is validation macro-F1. Differences below 0.2 percentage
points are treated as a practical tie and broken, in order, by lower multi-seed
standard deviation, lower log loss, lower expected calibration error, and lower
training cost. Test performance never breaks a tie.

## Data-scale experiment

Each class is shuffled once with seed 20260822. Nested, stratified prefixes are
formed at total training sizes 242, 500, 1,000, 3,000, and the complete clean
training split. The same selected architecture is fitted at each scale with seeds
42, 52, and 62. If a requested size exceeds the clean split after quarantine, the
complete split is used and the actual size is reported. Validation is fixed.

The learning-curve analysis reports performance versus both images and optimizer
steps. It does not attribute a difference to data volume when training compute,
label space, or augmentation also changed.

## Final lock and test gate

Before any study code loads a POLAR test label or test image, it must write a
machine-readable lock containing:

- clean-manifest hash and exclusion-list hash;
- source-code and environment hashes;
- selected label mapping, views, models, hyperparameters, seeds, epochs, and
  stopping rule;
- class-balance and calibration choices;
- ensemble weights and any test-time augmentation;
- the exact metrics and uncertainty procedure below.

The runner must first complete every locked train/validation fit and the final
train+validation fits, then open the test manifest once. Failed final fits remain
visible and are not replaced selectively. A protocol amendment is allowed before
the gate only when timestamped, justified, and committed; it starts a new protocol
version and cannot cite test evidence.

## Metrics and uncertainty

Primary: macro-F1. Secondary: accuracy, balanced accuracy, per-class precision /
recall / F1, multiclass log loss, Brier score, and 15-bin expected calibration
error. Confusion matrices use counts and row-normalized rates.

Final confidence intervals use 10,000 stratified bootstrap resamples with seed
20260822. Model deltas use paired resampling of the same examples. Multi-seed
predictions are averaged before primary test scoring; seed dispersion is reported
separately. Claims such as “better” require a positive paired interval or are
described as point-estimate differences.

## Complementarity, robustness, and faithfulness

Ensemble analysis reports prediction disagreement, double-fault rate,
oracle upper bound, uniquely corrected errors, probability correlation, log-loss
delta, calibration delta, and paired uncertainty. Blend weights are selected on
validation in increments of 0.05.

The existing ROAD, deletion/insertion, matched-random deletion, parameter
randomization, target sensitivity, flip equivariance, and seed-stability checks
are retained. POLAR adds annotation-grounded measures:

- fraction of positive attribution mass inside the person box;
- energy-based pointing game (peak inside box);
- person occlusion versus matched-area context occlusion;
- full-frame versus person-context prediction consistency;
- results stratified by person area and class.

These are localization and perturbation tests, not proof of causal or human-like
reasoning. Attribution-method choices are made on validation and frozen before
test evaluation.

## External validation and claim limits

V-COCO train/validation is an independent COCO-domain source for sitting,
standing, and walking/running. The external evaluation uses person-level target
annotations, excludes mixed-label images for an image-level comparison, and
excludes every SHA-256/perceptual duplicate of the legacy COCO corpus. Any known
overlap with V-COCO test is listed and excluded.

The final article may claim state of the art only after finding a directly
comparable published result with the same POLAR release, target subset, splits,
label space, and metric. Otherwise it reports a reproducible benchmark and
explicitly labels cross-paper numbers as non-comparable.
