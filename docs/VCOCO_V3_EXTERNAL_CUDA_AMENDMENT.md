# Motion-identifiability study: external confirmation and CUDA amendment

This amendment records two decisions made after the original source-development
protocol was locked and before any Okutama target model was fitted or the provider
test archive was opened.

## External dataset

The original protocol named POLIMI-ITW-S as the preferred external extension while
access and storage were unresolved. Its 392.1 GB footprint exceeded the study's
storage budget. Okutama-Action is used instead because the authorized local archives
fit, contain sitting, standing, walking, and running, and provide tracked people with
continuous frames and a separate provider test partition.

Provider train is divided by scenario into development, validation, and calibration.
Synchronized drone views of a scenario remain together. Provider test is the single
confirmation partition and stays unopened until the temporal pipeline, calibration,
routing rule, and model hashes are locked.

## Compute backend

All model fits initiated under this amendment require CUDA and fail when CUDA is not
available. This includes frozen-feature heads, spatial and representation screens,
parameter-efficient adaptation, source-transfer and few-shot heads, temporal models,
distillation, routing calibration, and the optional pose control. CPU work is limited
to metrics, manifests, hashing, archive parsing, and other non-training operations.

Every CUDA estimator records its device, software version, optimization settings, and
convergence evidence. There is no automatic CPU fallback.

## Boundaries retained from the base protocol

- The consumed V-COCO test is not reopened or used as confirmation.
- Existing V-COCO source tags remain immutable selection labels.
- The fixed 130-presentation human pilot remains descriptive and does not fit or
  select a model.
- Source-only, few-shot, and fully target-supervised Okutama results remain separate.
- The provider test is opened only after a locked pipeline and is never used to tune
  a candidate.

The original protocol lock remains preserved. This amendment supplies the provenance
for the changed external dataset and execution backend without rewriting earlier run
history.
