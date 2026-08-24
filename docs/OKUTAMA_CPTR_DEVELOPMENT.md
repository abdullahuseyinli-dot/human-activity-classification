---
title: Camera-Compensated Part-Trajectory Residual Development
subtitle: Recording-grouped evaluation of residual motion and body-region experts
short_title: CPTR Architecture Development
author: Abdulla Huseyinli
date: 24 August 2026
status: Development report
version: 3.0.0
document_type: Development report
subject: Camera-aware residual architecture for tracked human activity classification
keywords: human activity recognition, camera compensation, body-region tokens, grouped cross-validation
repository: https://github.com/abdullahuseyinli-dot/human-activity-classification
---

# Camera-compensated part-trajectory residual development

## Summary

This study investigates the remaining standing-versus-locomotion and transition
errors in the Okutama-Action temporal classifier. The implementation keeps the
established five-seed static and 0.5-second temporal models frozen, then adds bounded
residual experts for center-conditioned temporal context, person-box kinematics, and
confidence-masked body regions. Separate posture and motion gates decide how much of
each residual reaches the factorized classifier.

The strongest development component combined the center-conditioned short-window
encoder with seven body-region streams. Its five-seed probability ensemble improved
macro-F1 from 0.7806 to 0.7887 on the fixed validation split, including a standing-F1
increase from 0.7023 to 0.7243. That result did not carry across the full grouped
cross-fit: out-of-fold macro-F1 was 0.7144 for the new component and 0.7165 for the
matched temporal baseline. The existing temporal ensemble therefore remains the
default model, while the center-and-parts branch remains available for further work.

This distinction matters. A three-recording validation split showed a consistent
positive direction, but the 11-recording OOF evaluation exposed seed and scenario
sensitivity that a single split could not reveal.

## 1. Question and evaluation boundary

The experiment asks whether explicit motion structure can improve the classes that
remain difficult after the v3 temporal study without weakening its static fallback.
The design covers five mechanisms:

1. remove camera motion from person-box trajectories;
2. query short clips from their labeled center frame rather than globally pooling;
3. represent coarse body regions as confidence-weighted temporal tokens;
4. constrain learned evidence with motion-null, order, visibility, and reliability
   signals;
5. retain the frozen temporal model whenever a new component does not pass the fixed
   development gates.

All label-bearing work uses the Okutama provider-train archive. The locked development
manifest contains 8,339 samples partitioned by recording: 4,977 training samples from
11 recordings, 1,383 validation samples from three recordings, and 1,979 calibration
samples held closed during this study. Synchronized drone views stay in the same
group. The provider test archive used by the earlier v3 confirmation is not opened or
rescored.

The primary metric is three-class macro-F1 over sitting, standing, and combined
walking/running. Accuracy, log loss, calibration error, per-class F1, transition
subgroups, visibility subgroups, and recording-level results are retained alongside
it.

## 2. Model

### 2.1 Frozen anchors

Each run loads the matching-seed v3 static model and 8-frame, 0.5-second temporal
teacher. Their parameters remain frozen. The static model supplies factorized posture
and motion logits, while the temporal teacher supplies the established short-clip
residual. The new experts can add bounded residual evidence but do not silently
replace either anchor.

The frame representation is the revision-pinned `facebook/dinov2-base` checkpoint.
For each tracked person and frame, the cache contains a tight person crop, a context
crop, and six normalized box-geometry values. All candidate training ran with CUDA
automatic mixed precision on an NVIDIA GeForce RTX 4060 Laptop GPU.

### 2.2 Center-conditioned temporal encoder

The short encoder receives eight samples spanning 0.5 seconds and always includes
the labeled center frame. A signed temporal encoding makes position relative to that
center explicit. Two pre-norm Transformer layers encode the sequence, and the center
token queries the encoded window through multi-head attention. This differs from
unconditional sequence pooling: the prediction is explicitly about the tracked
person at the labeled instant.

An independently instantiated long-clock encoder was evaluated over 1.0 second. Both
shared-role and posture/motion-specialized variants were tested. Neither improved the
short encoder, so the selected component uses only the short clock.

### 2.3 Camera-compensated kinematics

The trajectory path constructs a 21-dimensional signal per frame and a 58-dimensional
summary. It includes normalized box center, scale and aspect, first and second temporal
derivatives, path length, net displacement, straightness, camera transform diagnostics,
validity, and camera-estimation quality.

Background-dominant sparse Lucas-Kanade tracks estimate a partial affine transform to
the center frame. RANSAC rejects inconsistent tracks; phase-correlation translation is
the declared fallback. Raw trajectories remain available whenever compensation is
unreliable. The cache covers all 8,339 development samples and records a mean camera
quality of 0.9908.

The recorded seed-42 trajectory screen moved from 0.7709 with raw box motion to 0.7723
with camera compensation. A later screen that added the compensated path to the
center-conditioned encoder was 0.0026 higher than its recorded reference, just below
the fixed 0.003 component gate. The integrated center, part, and trajectory screen
reached 0.7678, so trajectory features were retained as a control rather than included
in the five-seed candidate. These sequential screens span several implementation
snapshots and are used to document candidate development, not to estimate matched
causal effects.

### 2.4 Confidence-masked body-region encoder

Seven coarse regions are extracted from every tracked person crop: head/shoulders,
torso, pelvis, left and right upper sides, and left and right legs. Each region carries
a frozen DINOv2 token and a confidence derived from visibility and crop support.
Within each frame, a spatial Transformer pools only supported regions. A second
center-query temporal encoder then aggregates the part sequence. The expert's
reliability is the joint temporal-validity and part-confidence mean.

The full part cache has shape 8,339 x 17 x 7 x 768 and a mean confidence of 0.4771.
The subsequent body-region screen reached 0.7775 macro-F1, 0.0031 above the recorded
center-only reference. This comparison selected the branch for the matched five-seed
evaluation; it is part of the sequential development trace described below.

### 2.5 Gated residual fusion

Every learned expert produces separate two-logit posture and motion residuals. A
quality-conditioned gate receives frozen static features, static probability
diagnostics, center and window occlusion, box geometry, camera quality, part quality,
and trajectory straightness. Separate sigmoid gates control posture and motion paths.
Expert-role masks allow a modality to serve posture, motion, or both.

Residual heads are zero-initialized. Gate biases strongly favor the legacy temporal
path at initialization. During training, quality-conditioned modality dropout and
stochastic depth remove unreliable learned paths more often, while the frozen legacy
path remains available. The selected model has 4.11 million parameters, of which 2.15
million are trainable.

### 2.6 Auxiliary and counterfactual objectives

The primary loss factorizes sitting/upright posture from stationary/locomoting motion.
Auxiliary heads estimate transition windows, walking-versus-running subtype when the
provider label supports it, and window visibility. The declared counterfactuals are:

- repeat the center frame and zero relative motion;
- reverse temporal order for non-transition consistency;
- apply coherent camera-coordinate jitter;
- mask low-confidence part observations.

The first counterfactual formulation weakened the aggregate result. A refined version
penalized only the learned residual under motion nulling and preserved the frozen
legacy response. It raised the seed-42 transition subgroup from 0.5236 to 0.5549
macro-F1, but its overall macro-F1 was 0.7694 versus 0.7775 for the simpler model. The
transition effect is retained as a follow-up result, not merged into the selected
classifier.

### 2.7 Adaptation and robustness controls

The study also evaluates:

- label-free masked reconstruction of frozen target-video features with a temporal
  order objective;
- a frozen center-frame SigLIP posture specialist;
- top-two-block DINOv2 query/value LoRA with 53,764 trainable parameters;
- scenario-weighted GroupDRO;
- a continuous utility router and the required static-confidence heuristics.

Masked pretraining converged, but its downstream macro-F1 was 0.7712. SigLIP reached
0.7720, GroupDRO 0.7752, and the LoRA specialist 0.7187. The learned router did not
predict continuous true-class gain reliably; its strongest mandatory control was the
inverse static-margin rule. None displaced the center-and-parts component.

The model and dataset interfaces also support a confidence-aware keypoint stream. No
complete measured pose stream was available across the development split, so the
optional pose input remained disabled.

## 3. Execution sequence

| Stage | Seed-42 validation macro-F1 | Recorded comparison | Decision |
| --- | ---: | ---: | --- |
| Frozen temporal reference | 0.7489 | - | Reference |
| Raw trajectory | 0.7709 | +0.0219 vs reference | Camera-control input |
| Compensated trajectory | 0.7723 | +0.0014 vs raw | Measured control |
| Center-conditioned short residual | 0.7744 | +0.0254 vs reference | Retained |
| Shared dual clock | 0.7672 | -0.0072 vs short | Rejected |
| Specialized dual clock | 0.7681 | -0.0062 vs short | Rejected |
| Short plus trajectory | 0.7770 | +0.0026 vs short | Below component gate |
| Short plus body regions | 0.7775 | +0.0031 vs short | Five-seed candidate |
| Short, regions, and trajectory | 0.7678 | -0.0097 vs short plus regions | Rejected |
| Refined counterfactual objective | 0.7694 | -0.0081 | Transition diagnostic |
| Masked initialization | 0.7712 | -0.0062 | Rejected |
| SigLIP posture specialist | 0.7720 | -0.0055 | Rejected |
| GroupDRO | 0.7752 | -0.0023 | Rejected |
| Top-block LoRA | 0.7187 | -0.0302 vs reference | Rejected |

All branch outcomes and local artifacts were retained in the declared execution order.
The component table is a sequential development trace: its rows do not all share the
same model, feature, and training source revisions. The portable
`component_ablation.csv` records each summary, request, and source hash so that this
history remains inspectable. The five-seed promotion comparison and all 25 grouped
cross-fit runs use one consistent model, feature, runner, grid, and data-store snapshot;
those matched evaluations, rather than the early screens, determine the final decision.

## 4. Five-seed and grouped results

### 4.1 Fixed validation split

| Model | Macro-F1 | Accuracy | Log loss | Sitting F1 | Standing F1 | Locomotion F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen temporal ensemble | 0.7806 | 0.7780 | 0.5632 | 0.8232 | 0.7023 | 0.8163 |
| Center and parts ensemble | 0.7887 | 0.7867 | 0.5576 | 0.8204 | 0.7243 | 0.8213 |
| Change | +0.0081 | +0.0087 | -0.0056 | -0.0029 | +0.0220 | +0.0050 |

The three validation recordings all have positive macro-F1 deltas: +0.0069, +0.0055,
and +0.0197. With only three exchangeable groups, however, the exact paired
recording-swap test has eight permutations and a two-sided p-value of 0.25. The narrow
cluster-bootstrap interval should therefore not be interpreted without the exact
test and the broader cross-fit below.

### 4.2 Recording-grouped cross-fit

For each of five folds, all samples from a recording stay together. Each fold is run
with seeds 42 through 46 using fixed epoch counts selected before cross-fit. The 25
models cover every one of the 4,977 training rows exactly once per seed.

| Model | Macro-F1 | Accuracy | Log loss | Sitting F1 | Standing F1 | Locomotion F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen temporal ensemble | 0.7165 | 0.7131 | 0.7539 | 0.7368 | 0.6767 | 0.7360 |
| Center and parts ensemble | 0.7144 | 0.7123 | 0.7503 | 0.7295 | 0.6773 | 0.7366 |
| Change | -0.0020 | -0.0008 | -0.0035 | -0.0073 | +0.0006 | +0.0006 |

The 95% paired recording-cluster interval for the macro-F1 change is [-0.0063,
+0.0024], with bootstrap p = 0.396. The exact 2,048-permutation recording-swap test
gives p = 0.471. Recording deltas range from -0.0136 to +0.0098. These results do not
support replacing the frozen temporal ensemble.

## 5. Faithfulness and failure analysis

The five-seed validation ensemble was evaluated under interventions without refitting:

| Intervention | Macro-F1 | Change from real sequence | Prediction changes |
| --- | ---: | ---: | ---: |
| Real sequence | 0.7887 | 0.0000 | 0.0% |
| Repeat center / motion null | 0.7446 | -0.0441 | 12.5% |
| Remove part stream | 0.7810 | -0.0076 | 1.4% |
| Reverse temporal order | 0.7860 | -0.0027 | 2.2% |
| Deterministic temporal shuffle | 0.7891 | +0.0005 | 2.1% |
| Zero geometry | 0.7879 | -0.0008 | 0.6% |
| Geometry only | 0.1789 | -0.6098 | 66.7% |
| Coherent camera jitter | 0.7887 | 0.0000 | 0.0% |

Motion nulling reduces mean true-class log probability by 0.1140 and changes 12.5% of
predictions. The real sequence rescues 107 cases and harms 43 relative to that
counterfactual. Among the evaluated interventions, temporal appearance has the largest
measured effect. Part tokens add a smaller validation contribution, while explicit
geometry and camera jitter have little direct effect in the selected candidate.

The mean learned gates also remain conservative. Legacy gate means are 0.975 for
posture and 0.932 for motion; center-short means are 0.223 and 0.267; part means are
0.084 and 0.133. Mean part reliability is 0.509.

Visibility is the largest failure mode observed in this evaluation. On clear OOF windows, the candidate is
+0.0015 macro-F1 above baseline. On windows containing occlusion it is -0.0308 below.
The validation split contains only 25 occluded windows, compared with 412 in OOF, so
its aggregate gain underrepresents this failure mode. This pattern is more consistent
with visibility-dependent instability than with transition status: the OOF transition
delta is -0.0021, close to the overall delta.

## 6. Decision

The fixed promotion rule required at least +0.010 validation macro-F1, a positive
paired cluster interval, no per-class regression larger than 0.010, and no material
worst-recording regression. The candidate gains +0.0081 on validation, falls -0.0020
in grouped OOF, and has a -0.0136 worst-recording delta. It does not pass the aggregate,
cross-fit, or worst-recording gates.

The development lock therefore keeps `v3_temporal_8f_050s_five_seed_ensemble` as the
default. Calibration remains unopened. The strongest new component and all failed
controls remain retained and auditable, with the occlusion-conditioned failure providing a
specific target for the next iteration.

The retained cross-fit requests bind the model, feature module, runner, candidate grid,
manifest, feature stores, and baseline predictions. Their original request schema did
not include the shared training helper, and the original plan-lock schema did not bind
the candidate grid or its adaptive-grid lock. A post-run contract audit verified that
all 25 requests share the same recorded source hashes and that their grid hash matches
the retained adaptive-grid lock. Current lock and runner schemas include the omitted
bindings for future runs; no historical artifact or reported metric was rewritten.

## 7. Limitations and validation requirements

CPTR is a development result. The calibration partition remained unopened, the
previously consumed Okutama confirmation archive was not rescored, and no independent
CPTR confirmation has been run. The fixed-validation improvement therefore does not
override the slightly negative recording-grouped OOF result.

The development evidence also has a limited number of independent groups. The fixed
validation split contains three recordings, and the cross-fit contains 11. Exact
recording-level tests are reported, but the resulting intervals remain too wide to
support a small positive effect. Frame-level sample counts do not increase that
independent evidence.

The component screen is a sequential engineering trace, not a fully matched causal
ablation. Several rows were produced under different implementation snapshots; only
the five-seed promotion comparison and the 25 cross-fit runs share one bound code,
feature, grid, and data-store lineage. Future component claims require matched reruns
under a single lock.

The body-region stream uses confidence-masked pooled image regions rather than a
complete measured keypoint or pose sequence. Its strongest observed weakness is
occlusion: the candidate is +0.0015 macro-F1 above baseline on clear OOF windows and
-0.0308 on occluded windows. The next repair cycle must use provider-train development
data only, keep calibration closed, and test reliability-aware fallback and coherent
occlusion interventions with recording-grouped OOF predictions.

External validity is also open. Okutama is one aerial, low-resolution domain, and the
task is tracked-person classification rather than the dataset's original action-
detection task. An authorized POLIMI-ITW-S study remains the preferred independent
shopping-mall replication. Resource evidence is incomplete as well: cached-batch
timing does not measure end-to-end decoding, feature extraction, latency, energy, or
memory on uncached clips.

The required annotation, external-replication, grouped-inference, matched-baseline,
occlusion, resource, and clean-replay evidence is specified in
[`SCIENTIFIC_VALIDATION_PLAN.md`](SCIENTIFIC_VALIDATION_PLAN.md). Its publication rule
does not require CPTR to improve: a neutral or negative result remains reportable when
the protocol is complete, while model promotion still requires the locked performance
gates.

## 8. Reproduction

Create the environment described in the repository README and use a CUDA-enabled
PyTorch build. The following commands verify the retained final evidence after the
feature caches and component screens have been produced:

```powershell
# Verify the retained evidence without changing locks or completed runs.
.\.venv\Scripts\python.exe tools\lock_okutama_cptr_adaptive_grid.py --check
.\.venv\Scripts\python.exe tools\lock_okutama_cptr_stage2_grid.py --check
.\.venv\Scripts\python.exe tools\lock_okutama_cptr_stage3_grid.py --check
.\.venv\Scripts\python.exe tools\lock_okutama_cptr_stage4_grid.py --check
.\.venv\Scripts\python.exe tools\lock_okutama_cptr_crossfit_plan.py --check
.\.venv\Scripts\python.exe tools\lock_okutama_cptr_development.py --check

# Rebuild the portable export from retained runs, then validate the repository.
.\.venv\Scripts\python.exe tools\export_okutama_cptr_results.py
.\.venv\Scripts\python.exe tools\validate_repository.py
```

The complete protocol is in
[`experiments/okutama_cptr_protocol.json`](../experiments/okutama_cptr_protocol.json).
The portable tables and hashes are in
[`results/okutama_cptr`](../results/okutama_cptr/README.md). Large caches and
checkpoints stay in ignored local run directories.

<!-- pagebreak -->

## References

1. Barekatain, M. et al. "Okutama-Action: An Aerial View Video Dataset for Concurrent
   Human Action Detection." CVPR Workshops, 2017.
2. Oquab, M. et al. "DINOv2: Learning Robust Visual Features without Supervision."
   arXiv:2304.07193, 2023.
3. Zhai, X. et al. "Sigmoid Loss for Language Image Pre-Training." arXiv:2303.15343,
   2023.
4. Hu, E. J. et al. "LoRA: Low-Rank Adaptation of Large Language Models."
   arXiv:2106.09685, 2021.
5. Sagawa, S. et al. "Distributionally Robust Neural Networks for Group Shifts."
   arXiv:1911.08731, 2019.
6. Lucas, B. D. and Kanade, T. "An Iterative Image Registration Technique with an
   Application to Stereo Vision." IJCAI, 1981.
7. Fischler, M. A. and Bolles, R. C. "Random Sample Consensus: A Paradigm for Model
   Fitting with Applications to Image Analysis and Automated Cartography."
   Communications of the ACM, 1981. doi:10.1145/358669.358692.
