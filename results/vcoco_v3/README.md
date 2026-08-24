# Motion-identifiability study evidence

This directory is the portable evidence package for the V-COCO source-label analysis
and the Okutama-Action temporal extension. Private annotation rows, dataset media,
feature tensors, dense probabilities, and checkpoints remain outside Git.

## Sealed confirmation result

The confirmation partition contains 1,771 tracked people from 10 videos in five
scenarios. The complete pipeline was fixed before the provider test archive was opened
once.

| Method | Clip fraction | Macro-F1 | Accuracy | Locomotion F1 |
| --- | ---: | ---: | ---: | ---: |
| Source-only static transfer | 0% | 0.5735 | 0.5731 | 0.5036 |
| Target-supervised static model | 0% | 0.7458 | 0.7301 | 0.7261 |
| Distilled static student | 0% | 0.7456 | 0.7307 | 0.7350 |
| Routed static/clip system | 50% | 0.7817 | 0.7679 | 0.7692 |
| Eight-frame temporal model | 100% | 0.7854 | 0.7708 | 0.7692 |

Against the matched static model, the temporal model gains 0.0396 macro-F1 with a
scenario-cluster bootstrap interval of [0.0202, 0.0568]. The 50% routing policy gains
0.0360 with an interval of [0.0144, 0.0604]. Both intervals use 10,000 paired
resamples over the five confirmation scenarios.

The 50% policy retains 90.7% of the temporal model's macro-F1 gain while invoking the
clip path for half of the samples. The 25% policy reaches 0.7645 macro-F1, but its
paired interval crosses zero; it is kept in the routing curve rather than promoted as
a confirmed improvement.

## What the development stages established

- The matched frozen-representation screen selected DINOv2-B at 0.8395 source-tag
  macro-F1. DINOv3-B reached 0.8367 and SigLIP2-B reached 0.8358 under the same nested
  grouped protocol.
- Multiresolution and context interventions improved point estimates, but none met the
  predeclared spatial promotion rule. Neural adaptation therefore did not run.
- The DINO/SigLIP factorized reliability stack reached 0.8697 nested source-tag
  macro-F1 and passed its development promotion test.
- The fixed 130-presentation annotation pilot contained 126 unique people. Thirty-two
  were visibly translating without a walking/running gait, and 41 used the `other`
  posture label. These counts describe the audited pilot and were not used for model
  selection.
- The selected short-clip model uses eight samples over 0.5 seconds. Its grouped
  out-of-fold macro-F1 was 0.7165, compared with 0.6672 for the static model.
- Static distillation improved the validation point estimate but did not reproduce a
  macro-F1 gain on confirmation. The result is retained as evidence that information
  absent from the center frame was not recovered by distillation alone.
- The optional pose/velocity control could not run because normalized pose was not
  available for every development sample. No pose values were imputed or inferred.

## File map

| Artifact | Contents |
| --- | --- |
| `evidence_manifest.json` | SHA-256 inventory and confirmation headline |
| `protocol_lineage.json` | CUDA amendment, pipeline lock, and test-open lineage |
| `confirmation_metrics.csv` | Aggregate metrics for static, temporal, student, source-only, and routed systems |
| `confirmation_uncertainty.json` | Paired scenario-cluster intervals and multiplicity-adjusted tests |
| `confirmation_per_class.csv` | Sitting, standing, and walking/running metrics |
| `confirmation_routing_curve.csv` | Fixed 0%, 10%, 25%, 50%, and 100% clip budgets |
| `confirmation_subgroups.csv` | Scale, transition, occlusion, drone, time, scenario, and gait-subtype results |
| `confirmation_subgroup_deltas.csv` | Subgroup changes relative to the matched static model |
| `calibration_*.csv` | Pre-confirmation temperature scaling and routing evidence |
| `temporal_development_metrics.csv` | Validation comparison used before final fitting |
| `temporal_crossfit_summary.json` | Recording-grouped teacher targets and temporal-benefit prevalence |
| `okutama_fewshot_summary.csv` | Five-seed target-only and source-recalibration curves |
| `source_tag_development_metrics.csv` | Nested fusion, spatial, and representation screens |
| `annotation_*.json` and aggregate CSVs | Path-free summaries of the fixed single-rater pilot |

The study narrative is in
[`docs/VCOCO_V3_MOTION_IDENTIFIABILITY.md`](../../docs/VCOCO_V3_MOTION_IDENTIFIABILITY.md).
Rebuild this directory from retained local evidence with
`python tools/export_vcoco_v3_results.py`.
