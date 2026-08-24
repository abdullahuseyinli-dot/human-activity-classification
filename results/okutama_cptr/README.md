# Okutama CPTR development evidence

This package records the camera-compensated part-trajectory residual (CPTR) development
study on the provider-train portion of Okutama-Action. The evaluated family combines
frozen v3 static and temporal experts with center-conditioned temporal residuals,
camera-aware box trajectories, confidence-masked body-region tokens, quality gates,
counterfactual objectives, target-video masked pretraining, and parameter-efficient
specialists.

The strongest component was the center-conditioned short-window plus body-region
model. Its five-seed ensemble reached **0.7887
macro-F1** on the fixed three-recording validation split, compared with
**0.7806** for the frozen temporal baseline.
The gain was concentrated in standing F1
(**0.7023 to
0.7243**).

The recording-grouped OOF result did not reproduce that gain: the center-plus-parts
residual reached
**0.7144**, while the matched temporal baseline
reached **0.7165** across 4,977 samples from 11
recordings. The existing temporal ensemble therefore remains the default model. The
center-and-parts branch is retained as a documented research component.

## Main comparison

| Evaluation | Baseline macro-F1 | Center + parts residual | Change | Recordings |
| --- | ---: | ---: | ---: | ---: |
| Fixed validation | 0.7806 | 0.7887 | +0.0081 | 3 |
| Five-fold grouped OOF | 0.7165 | 0.7144 | -0.0020 | 11 |

The exact recording-level prediction-swap test gives the OOF comparison directly;
`uncertainty.json` also contains 10,000-resample paired recording-cluster intervals.

## What the interventions show

- Repeating the center frame and removing relative temporal evidence reduced
  validation macro-F1 by **0.0441**
  and reduced mean true-class log probability by
  **0.1140**.
- Removing the body-region stream reduced macro-F1 by
  **0.0076** on validation.
- Reversing temporal order reduced macro-F1 by
  **0.0027**; deterministic camera
  jitter changed it by **0.0000**.
- Clear-window OOF performance was slightly above baseline, but occluded-window
  macro-F1 was lower. This quality-dependent instability is the largest failure mode
  observed in this evaluation.

The intervention reference result is 0.7887 macro-F1.
Classifier-forward latency is measured from cached DINOv2 and part features; feature
extraction is intentionally reported separately.

## Files

- `development_decision.json` - aggregate metrics, gates, fixed epochs, and hashes;
- `headline_metrics.csv` - validation and grouped-OOF model metrics;
- `component_ablation.csv` - sequential development trace with request and source hashes;
- `fold_seed_metrics.csv` - all 25 fold/seed runs plus ensembles;
- `subgroup_metrics.csv` and `recording_metrics.csv` - transition, visibility, and
  recording-level results;
- `faithfulness_metrics.csv` and `faithfulness_summary.json` - intervention and gate
  evidence;
- `uncertainty.json` - paired cluster intervals and exact group-swap tests;
- `provenance.json` and `evidence_manifest.json` - model revisions, CUDA runtime, and
  file hashes.

Dataset frames, pretrained weights, cached embeddings, checkpoints, and dense
prediction arrays remain outside Git under their original terms.
