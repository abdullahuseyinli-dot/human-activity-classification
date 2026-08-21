# Experiment protocol

## Objective

Compare ImageNet-pretrained ConvNeXt-Small and self-supervised DINOv2-Small on a
three-class human-activity dataset without using the final test partition for
model, epoch, regularization, calibration, or ensemble selection.

The primary metric is macro-F1 because each activity should contribute equally
to selection. Log-loss and between-fold variability are deterministic
tie-breakers. Accuracy, balanced accuracy, per-class recall, Brier score, and
expected calibration error are reported as supporting measures.

## Data contract

The standardized manifest contains 285 unique image records:

| Partition | Sitting | Standing | Walking/running | Total |
|---|---:|---:|---:|---:|
| Train | 66 | 64 | 69 | 199 |
| Validation | 14 | 14 | 15 | 43 |
| Test | 15 | 14 | 14 | 43 |

The original train and validation partitions form a 242-image development
pool. The original 43-image test partition remains fixed. Candidate selection
uses stratified cross-validation within the development pool only.

Before training, the pipeline verifies that every manifest path exists, image
identifiers are unique, files can be decoded, labels are recognized, and no
exact or perceptual-hash duplicate group crosses the development/test boundary.
The manifest hash and the ordered test-image identifier hash are written into
the run provenance.

## Intervention design

Changes are evaluated as controlled comparisons instead of being stacked into
one untraceable recipe.

- Freeze depth: head-only, partial-backbone, and deeper fine-tuning controls
  test whether adapting pretrained representations helps this small dataset.
- Head dropout: 0.0, 0.1, and 0.2 are compared where appropriate. ConvNeXt's
  native stochastic-depth schedule remains enabled.
- Augmentation: mild geometric/photometric transforms are the baseline;
  random-erasing removal and a light RandAugment policy are direct ablations.
- MixUp: alpha 0.2 is evaluated as a separate candidate.
- Optimizer regularization: weight decay, label smoothing, and discriminative
  head/backbone learning rates are recorded for every candidate.

These choices follow the mechanisms described in the original
[Dropout](https://www.jmlr.org/papers/v15/srivastava14a.html),
[stochastic depth](https://arxiv.org/abs/1603.09382),
[MixUp](https://arxiv.org/abs/1710.09412),
[RandAugment](https://arxiv.org/abs/1909.13719), and
[label smoothing](https://arxiv.org/abs/1906.02629) papers. Promotion still
depends on this dataset's out-of-fold evidence, not on a generic recipe.

## Selection gates

1. A three-fold coarse screen rejects clearly weak interventions.
2. The strongest candidates in each model family are rerun with five folds and
   a wider early-stopping budget.
   A targeted interaction discovered during confirmation may enter the same
   five-fold gate directly when it changes only one already-screened factor and
   is declared before any test inference; it receives no selection advantage.
3. A machine-readable lock selects one configuration per family by pooled OOF
   macro-F1, then lower fold-score standard deviation, lower OOF log-loss, and
   candidate identifier.
4. The final seeds (`42`, `52`, and `62`) are declared before test inference.
5. For each seed, five-fold OOF training derives a median best-epoch count;
   the selected model is then retrained on all 242 development images for that
   fixed number of epochs. Because the CV optimizer uses plateau-driven LR
   reductions, full-pool training replays the median fold-derived LR state for
   each epoch instead of silently reverting to a constant learning rate.
6. Temperature scaling is fit on OOF logits and transferred unchanged to the
   corresponding full-pool model.
7. Seed averaging, the ConvNeXt/DINOv2 blend weight, and supplementary SVM
   probe hyperparameters are selected from OOF evidence. Only then are the
   locked methods evaluated on the fixed test split.
8. Center-crop inference and a two-view horizontal-flip average are compared on
   OOF predictions; the selected evaluation policy is locked per model family
   before its corresponding test probabilities are read.

## Reporting constraints

The 43-image test set is small. Final results therefore include stratified
bootstrap intervals and retain seed-level metrics. A higher point estimate is
not presented as a statistically certain improvement when the paired interval
includes zero. Historical notebook outputs produced under a different split or
training lineage are not mixed with corrected results.

The dataset does not expose subject or capture-session identifiers. The
experiment can prove manifest-level and duplicate-level separation, but it
cannot claim subject-independent generalization without those metadata.
