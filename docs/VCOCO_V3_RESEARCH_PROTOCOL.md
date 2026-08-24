# Motion Identifiability in Person-Level Activity Recognition

Version 3 development protocol, amendment 2, 24 August 2026

## Purpose

This study investigates the remaining standing-versus-locomotion error after the
locked V-COCO v2 result. The objective is to distinguish three mechanisms that a
single aggregate score cannot separate:

1. disagreement between source action tags and visually supported labels;
2. recoverable errors in person, scale, boundary, and context modeling;
3. motion that cannot be identified reliably from one frame.

The study does not make ranking claims. Model and architecture comparisons are
reported only within their declared data, supervision, input, and evaluation
protocols.

## Frozen reference

The `polar-study-v2.0.0` release and commit
`11b7b9160785466841588b91ac815943dd94dad5` are immutable references. The v2
official V-COCO test set has been evaluated and is considered consumed. Its existing
predictions may be used for error analysis and blinded annotation sampling, but no
new candidate may use those labels or outcomes for model selection or fresh
confirmation.

Future V-COCO development uses train and validation data under nested grouped
cross-validation. A new non-COCO holdout is required for final confirmation.
The human pilot is a descriptive ontology and error-mechanism audit of the already
consumed v2 result; its labels are not used to select v3 candidates. Until a separate
harmonized development set exists, nested development selection uses the immutable
source tags and reports this endpoint by name. Human-harmonized performance remains
the primary endpoint for an eligible independent confirmation set.

## Research hypothesis

The remaining standing-locomotion gap is an interaction between source-label
semantics, person observability, and motion identifiability. Factorized targets and
reliability-conditioned person/context evidence should improve visually identifiable
cases. Residual cases that are not identifiable from one image should benefit from a
short temporal clip or produce a calibrated set-valued prediction.

## Development pilot and acceptance gates

### Fixed development pilot

The primary descriptive audit is the first 130 presentations in the existing blind
manifest order. It uses one annotator, keeps model predictions and source tags hidden,
and preserves any later response without including it in the primary analysis. Hidden
repeat pairs within the fixed prefix provide a descriptive consistency check. This
pilot can unlock source-tag-only development because its labels never fit, select, or
calibrate a candidate.

This amendment separates feasibility of model development from eligibility for a
human-harmonized result. It does not convert the single-rater pilot into an agreement
study.

### Gate 1: human-harmonized endpoints

- At least two people annotate the same pilot independently.
- Source tags, model predictions, confidence, and cohort membership remain hidden.
- A third prediction- and source-blind pass adjudicates every item with any
  disagreement.
- Posture, visible translation, gait, and visibility are recorded separately.
- Source annotations are preserved and are never overwritten.
- Each of the four axes must reach Krippendorff's alpha of at least 0.80 across the
  280 unique items. Each rater must also reach at least 90% exact agreement on every
  axis across 20 hidden repeat items. Otherwise the guide is revised and the pilot
  is repeated before a human-harmonized endpoint is reported.

### Gate 2: development promotion

A general candidate must improve grouped development macro-F1 by at least 0.01,
with the paired cluster-bootstrap lower bound above zero and no class F1 regression
larger than 0.01. A locomotion specialist must improve locomotion F1 by at least
0.02 with its lower bound above zero and remain within 0.005 macro-F1 of the reference.

### Gate 3: representation and neural training

Frozen, cached-feature experiments run before neural adaptation. A neural family
advances only when its matched multiview comparison passes Gate 2. Screening uses at
least three seeds; the final comparison uses five.

### Gate 4: confirmation

The complete pipeline, thresholds, calibration, preprocessing, checkpoints, and
subgroup hypotheses are hash-locked before the new holdout is opened. The final
comparison uses one authorized evaluation and 10,000 paired cluster-bootstrap
resamples.

## Temporal mechanism study

The temporal stage uses scenario-grouped Okutama-Action partitions fixed before model
outcomes are read. A factorized short-clip teacher compares 8-frame and 16-frame
windows against a matched static model. Static students receive only
recording-grouped out-of-fold teacher distributions during development. One student
predicts the activity directly; the second also estimates whether temporal evidence
is likely to improve the static prediction.

The routing experiment reports fixed clip-budget fractions rather than choosing a
budget on confirmation data. Its eligibility is decided on validation data from the
average-precision gain over target prevalence and a recording-cluster bootstrap. The
calibration partition then fixes the routing-score calibration, temperature scaling,
and adaptive prediction-set thresholds.

When normalized 2-D pose is available, a linear pose/velocity SVM is included as a
mechanism control. Its hyperparameters are selected on validation data and its score
calibrator is trained from recording-grouped out-of-fold development predictions. The
standardized SVM and calibrator both use the declared CUDA optimizers.
Missing pose is recorded as unavailable and does not change the visual pipeline.

## Human annotation axes

Annotators label what is supported by the pixels, rather than reproducing an existing
dataset tag.

| Axis | Values |
| --- | --- |
| Posture | seated, upright, other, indeterminate |
| Visible translation | stationary, locomoting, transition, not inferable |
| Gait | walking, running, not applicable, indeterminate |
| Visibility | clear, occluded, truncated, too small |

`Not inferable` means the frame does not contain enough evidence to decide whether the
person is translating through the scene. It is different from `stationary`.

Gait is interpreted jointly with visible translation. `Stationary` with gait marked
`not applicable` means that the person is not changing location; localized actions
such as throwing or hitting do not make the person locomoting. `Locomoting` with gait
marked `not applicable` means that visible translation is present but its mode is
neither walking nor running, for example crawling, cycling, skating, or swimming.
`Indeterminate` is reserved for visible translation where walking and running cannot
be distinguished from the frame.

## Experiment order

1. Lock this protocol and its dependencies.
2. Finalize the fixed 130-presentation descriptive pilot without using its labels for
   candidate selection.
3. Test factorized DINO and DINO-SigLIP probability stacks under nested grouped
   cross-validation.
4. Run matched geometry, boundary, context, resolution, and box-perturbation
   experiments.
5. Screen frozen representations, then run matched multiview parameter-efficient
   adaptation for promoted candidates.
6. Compare a static factorized model, a temporal-teacher distilled student, an
   identifiability-conditioned student, and a true short-clip model.
7. Lock one candidate and confirm it on an independent recording- or track-grouped
   dataset.

## External confirmation

The complete POLIMI-ITW-S release remains the preferred shopping-mall extension, but
its academic access workflow and 392.1 GB footprint exceed the available storage. The
replacement study therefore uses Okutama-Action. Its provider train and test archives
fit locally, include person boxes and stable tracks, contain the four required source
actions, and provide continuous video around action transitions.

Okutama Sitting and Standing map directly to the corresponding study classes. Walking
and Running map to `walking_running`, while the provider subtype remains available for
subgroup analysis. The synchronized Drone1 and Drone2 recordings for each scenario
stay in the same development split. Provider train supplies train, validation, and
calibration partitions; provider test remains sealed until the complete temporal
pipeline is locked.

Source-only, few-shot, fully supervised static, short-clip, distilled-student, and
budgeted-routing results are kept separate. Okutama provider-label macro-F1 is the
external endpoint; the descriptive single-rater V-COCO pilot is not used to fit or
select an Okutama model.

## Reporting

Source-only, unlabeled adaptation, few-shot adaptation, and full target supervision
are reported in separate tables. Primary endpoints are harmonized macro-F1,
locomotion F1, and worst-class F1. Source-tag metrics, calibration, risk-coverage,
person scale, visibility, boundary contact, and scene occupancy are secondary.

All people from one image remain together. Video experiments resample and split by
recording or track. Development analyses are marked exploratory; only the locked new
holdout comparison is confirmatory.
