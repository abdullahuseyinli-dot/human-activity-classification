# Scientific validation plan

## Purpose

This plan defines the evidence needed to extend the current human-activity studies
beyond their recorded datasets and splits. It separates three decisions that should
not be collapsed into one:

1. whether an experiment is complete enough to report;
2. whether a candidate model is strong enough to replace its baseline;
3. whether the evidence supports a broader scientific claim.

A complete experiment may produce a positive, neutral, or negative result. The sign
of the result determines model promotion and claim wording; it does not determine
whether the work is reported.

## Current claim boundary

The retained results support statements about the declared POLAR, V-COCO, and
Okutama-Action protocols. They do not yet establish performance across camera types,
viewpoints, environments, or annotation policies that were not evaluated.

The Okutama temporal experiment provides evidence that short clips improve the
matched static classifier under its locked aerial-video protocol. Its provider test
partition has already been consumed for that comparison. The later CPTR work is a
development study: the center-and-parts branch improved the fixed validation split
but did not improve recording-grouped out-of-fold macro-F1. It remains a candidate
mechanism, not a replacement for the established temporal model.

The 130-presentation V-COCO annotation pilot is a blinded, single-rater mechanism
audit with enriched sampling. It can describe ontology and identifiability failures,
but it cannot estimate population prevalence or inter-rater reliability. The planned
POLIMI-ITW-S replication has a predeclared ontology, but the dataset has not been
acquired or evaluated.

## Limitation and evidence-gate matrix

| Area | Current limitation | Evidence required to close it | Decision gate |
| --- | --- | --- | --- |
| Annotation reliability | One rater completed the enriched 130-presentation audit. | Complete the declared blinded agreement cohort with independent ratings, hidden repeats, and disagreement adjudication. Preserve the original responses as a separate descriptive pass. | Each axis reaches the reliability and repeat-agreement thresholds in the locked annotation protocol before a human-harmonized endpoint is reported. |
| Gait and activity ontology | The three-class target combines walking and running and cannot represent every non-gait locomotion or posture. | Preserve provider labels; record posture, translation, gait, and visibility separately; publish the translation-by-gait cross-tab, quarantine reasons, and walking/running subtype counts by independent group. | No relabeling after model outcomes are seen. Unsupported states remain outside the primary endpoint and are reported rather than forced into a class. |
| Independent replication | The completed temporal evidence comes from one aerial, low-resolution video dataset. | Obtain authorized POLIMI-ITW-S access, verify the provider files, lock the grouped split and analysis before outcomes, and run the full matched baseline set. | The independent holdout is opened once after preprocessing, models, thresholds, and primary contrasts are hash-locked. |
| Grouped inference and power | Row-wise sample counts overstate precision when people share recordings or scenarios. The sealed temporal result has five confirmation scenarios; Okutama CPTR has 11 cross-fit recordings and only three fixed-validation recordings. The existing temporal interval is therefore precision-limited at the scenario level. | Use recording- or track-grouped inference, a prospective design analysis, paired group-level intervals, and an exact paired randomization test when the number of groups permits it. | The effective unit, target effect, power or attainable precision, and stopping rule are fixed before fitting. An underpowered fixed dataset is reported as precision-limited. |
| Matched baselines | A task-specific temporal model should be compared with both simpler pooling and a video-native representation. | Evaluate a static center-frame model, simple frozen-feature temporal pooling, the established temporal teacher, CPTR, and one revision-pinned video-native encoder under the same tracks, windows, splits, seeds, and selection budget. | Every promoted architecture must beat its declared matched baseline under the same grouped protocol; unmatched literature numbers remain context only. |
| CPTR under occlusion | CPTR was slightly positive on clear OOF windows but materially negative on windows containing occlusion. | Diagnose and repair the failure using provider-train development data only, with group-held-out predictions and an unchanged baseline. | No calibration or independent-confirmation labels may guide the repair. Promotion requires the predeclared aggregate, subgroup, per-class, and worst-group gates. |
| Runtime and resource cost | Reported cached-batch timing does not measure the complete system from decoded frames. | Benchmark uncached decoding, crop extraction, feature encoding, temporal inference, routing, and post-processing; record latency, throughput, peak memory, and energy on fixed hardware. | Static, always-temporal, routed-temporal, and CPTR paths use the same harness, input cohort, warm-up, and synchronization policy. |
| Reproducibility | Portable summaries do not by themselves prove a clean-machine replay. | Rebuild the public evidence package from a fresh clone or container using checksum-verified inputs and a fully recorded environment. | All published tables, figures, locks, and manifests reproduce within declared numerical tolerances, with no undeclared local path or manual result edit. |
| Multiplicity | Sequential development can make a favorable screen look stronger than a matched comparison. | Lock one primary endpoint and contrast, enumerate candidate families, retain all outcomes, and apply the declared family-wise correction. | Only locked matched evaluations support selection. Sequential screens remain labeled as development history. |

## 1. Annotation reliability

The existing 130 presentations remain a mechanism audit. They are not reused as if
they were independent ratings. The next annotation stage follows Gate 1 of
[`VCOCO_V3_RESEARCH_PROTOCOL.md`](VCOCO_V3_RESEARCH_PROTOCOL.md):

- two people independently annotate the same fixed cohort while source tags, model
  predictions, confidence, and sampling role remain hidden;
- the agreement cohort contains 280 unique items, with 20 hidden repeat items per
  rater;
- posture, visible translation, gait, and visibility are scored as separate axes;
- a third source- and prediction-blind pass adjudicates every disagreement;
- Krippendorff's alpha must be at least 0.80 on each axis, and each rater must achieve
  at least 90% exact agreement on every axis across the hidden repeats.

If a threshold is missed, the annotation guide is revised and a new pilot is run. The
threshold is not relaxed after inspecting disagreements. Agreement estimates include
intervals and the full confusion table for each axis; adjudicated labels never
overwrite the original rater records.

## 2. Gait and ontology coverage

Translation and gait answer different questions. A stationary person with gait marked
`not applicable` is not changing location. A locomoting person with gait marked `not
applicable` is visibly translating by some other mode, such as cycling or crawling.
`Transition` is reserved for a change between states, and `not inferable` means the
pixels do not support a movement decision.

Every future dataset keeps its provider label unchanged alongside the mapped target.
Walking and running form the primary `walking_running` class, with the two provider
subtypes reported separately. Non-gait locomotion, unsupported postures, ambiguous
labels, and incompatible multilabel cases are preserved in an exclusion table with a
reason code; they are not silently assigned to the nearest target class.

Before a gait-specific comparison is run, the study records the number of walking and
running examples and independent track or recording groups in each split. A blinded
design analysis then determines whether subtype inference has useful precision. If it
does not, the subtype table remains descriptive and the combined locomotion endpoint
stays primary.

## 3. Independent replication on POLIMI-ITW-S

POLIMI-ITW-S is the preferred independent replication because it changes the camera
geometry and environment from aerial outdoor video to a shopping-mall setting while
retaining person boxes, skeletons, and the four relevant activities. Acquisition is
conditional on the provider's authorization and a dedicated volume with enough space
for the 392.1 GB release and retained working evidence. The repository does not treat
an application as access and does not redistribute provider data.

Before any label-bearing analysis:

1. record the provider agreement, archive inventory, checksums, and immutable raw-data
   location;
2. apply the already declared mapping in
   [`polimi_itw_s_ontology.json`](../experiments/polimi_itw_s_ontology.json) without
   changing provider labels;
3. group synchronized views and all observations from a person track so that no group
   crosses development and evaluation partitions;
4. lock development, calibration, and independent holdout manifests, including the
   ordered identifier hashes;
5. lock preprocessing, candidate families, tuning budget, primary endpoint, subgroup
   analyses, and missing-data policy;
6. open the independent holdout once and retain the complete evaluation record.

The primary endpoint is provider-label three-class macro-F1. Accuracy, log loss,
calibration, worst-class F1, and walking-versus-running results are secondary. If
POLIMI-ITW-S cannot be acquired, any substitute is named in a protocol amendment
before its labels or model outcomes are inspected and must provide continuous tracks,
compatible activities, and a genuinely independent capture domain.

## 4. Grouped inference and prospective precision

The independent unit is the recording, scenario, or person track specified by the
dataset protocol, not an individual frame. All split generation, resampling, and
uncertainty calculations preserve that unit.

The replication design fixes a primary contrast and a minimum relevant macro-F1
change of 0.010 before model fitting. Using only prior development evidence, a
simulation estimates power and interval width across the available group count,
class balance, and observed between-group variation. The target is at least 80% power
at a two-sided 0.05 family-wise error rate. When the provider's fixed cohort cannot
meet that target, the attainable precision is recorded before evaluation and the
result is described as precision-limited rather than being analyzed as independent
frames.

Final uncertainty uses at least 10,000 paired resamples of independent groups. An exact
paired group-swap test is also reported when enumeration is feasible. Per-group deltas,
the worst-group change, and class-specific effects remain visible beside the aggregate
result.

## 5. Matched model comparison

The replication includes five roles:

1. the frozen center-frame static classifier;
2. a simple temporal control that pools the same frozen per-frame features;
3. the established 8-frame, 0.5-second temporal teacher;
4. the CPTR candidate selected on development data;
5. one revision-pinned video-native encoder selected for hardware compatibility before
   target performance is measured.

All roles receive the same person tracks, center instants, temporal extent, target
labels, grouped folds, seed set, and stopping information available to their model
class. Native input normalization and pretraining supervision are disclosed. Parameter
count, trainable parameter count, input resolution, frame count, tuning trials, and
measured resource cost accompany the accuracy table. A larger tuning budget or extra
labels are reported as a separate condition, not folded into the matched comparison.

## 6. CPTR occlusion study

The CPTR repair remains confined to Okutama provider-train development data. The
existing calibration partition stays closed until a new development lock is complete,
and the previously consumed Okutama confirmation data is not used to select or tune a
CPTR revision.

The work proceeds in this order:

1. reproduce the existing grouped OOF visibility split and audit occlusion labels,
   temporal support, person scale, part confidence, track continuity, and scenario;
2. test a deterministic fallback that suppresses the part residual when its measured
   support is inadequate;
3. test reliability calibration for the part and center gates without changing the
   frozen baseline;
4. test temporally coherent block occlusion and quality-conditioned modality dropout
   as one-factor interventions;
5. run every retained intervention with the same five recording-grouped folds and
   seeds 42 through 46;
6. select at most one repair before any calibration or independent-replication data is
   opened.

The primary comparison remains aggregate macro-F1. The predeclared occlusion subgroup
is a required safety check: its paired group-level interval must exclude a harmful
effect, while per-class and worst-recording limits remain in force. A subgroup gain
cannot promote a model that fails the aggregate gate. If no repair passes, the frozen
temporal model remains the default and the negative result is retained.

## 7. Runtime, energy, and memory

Resource measurements begin with source-video decoding, not cached embeddings. The
benchmark reports cold-start time separately, performs a fixed warm-up, synchronizes
CUDA around timed regions, and uses one immutable input cohort for every model path.
It measures:

- median and 95th-percentile per-person latency;
- clips or tracked people processed per second;
- peak allocated and reserved GPU memory;
- host peak resident memory;
- total energy and energy per classified center frame, sampled through the available
  hardware telemetry with its sampling interval recorded.

The static-only, always-temporal, fixed-budget routed, and CPTR paths are measured on
the same machine with the GPU, driver, CUDA, PyTorch, precision mode, batch size,
input resolution, and frame count recorded. Accuracy and resource results are kept in
separate columns so that a faster path is not presented as more accurate, or vice
versa.

## 8. Reproducibility check

A release candidate is replayed from a fresh clone or clean container that has no
access to the original machine's caches. External data and model files are supplied
through documented runtime paths and verified against their recorded checksums. The
replay must:

- build the environment from the tracked dependency specification;
- validate every manifest, split lock, source revision, and evidence hash;
- reproduce portable tables and figures from retained run outputs;
- reproduce primary metrics within the declared floating-point tolerance;
- pass tests, formatting, repository validation, and release-manifest checks;
- contain no private path, credential, untracked manual correction, or required file
  that exists only on the development machine.

The machine and environment record includes operating system, CPU, GPU, driver, CUDA,
Python, PyTorch, package lock, deterministic settings, and the exact commands used.
A failed replay remains part of the evidence and must be resolved or listed as an
unclosed limitation.

## 9. Multiplicity and analysis discipline

Each study amendment names one primary endpoint, one primary contrast, the candidate
families, the promotion order, and the independent evaluation unit. Secondary metrics
and subgroup analyses are labeled as such before fitting. Holm correction is applied
across promoted model families as declared in the CPTR protocol; unplanned analyses
are marked exploratory.

All screened candidates, stopped runs, regressions, and protocol amendments remain in
the execution record. Only comparisons sharing the locked code, data, folds, seeds,
and training budget are interpreted as matched ablations. The independent holdout is
not reopened to settle a development choice.

## Publication decision rule

The expanded study is ready for an archival research release when all of the following
are true:

1. the annotation agreement gate is complete, or all human-harmonized claims have
   been removed and the remaining label boundary is stated explicitly;
2. an authorized independent replication has been completed under a locked grouped
   protocol, or the report is explicitly limited to development evidence;
3. grouped uncertainty, prospective precision, matched baselines, multiplicity, and
   complete resource measurements are reported;
4. a fresh-environment replay has passed and the evidence inventory is complete;
5. dataset rights, exclusions, failed runs, and protocol deviations remain visible.

These are evidence-completeness gates, not success gates. A CPTR revision is promoted
only if it passes its locked performance and safety criteria. If it does not, the
negative result is still publishable once the protocol and evidence are complete, and
the established temporal model remains the default. Broader claims are added only
when independent results support them.
