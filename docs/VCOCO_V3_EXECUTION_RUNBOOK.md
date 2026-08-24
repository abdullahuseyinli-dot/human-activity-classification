# Motion-identifiability study v3 execution runbook

This runbook records the completed motion-identifiability experiment in execution
order. Commands are run from the repository root in PowerShell. Long-running stages
reuse only artifacts whose declared hashes still match.

## Environment and status

The local RTX 4060 environment uses PyTorch 2.11.0 with CUDA 12.8. Recreate it when
`.venv` is absent:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-v3-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Activate the environment and audit the current state:

```powershell
.\.venv\Scripts\Activate.ps1
python tools/check_vcoco_v3_readiness.py --probe-dinov3
```

The audit is read-only except for its generated report at
`.runs/vcoco_v3/readiness/summary.json`. It does not download checkpoint weights,
fit a model, or open confirmation features.

## 1. Finalize the fixed human pilot

Launch the local annotation interface at <http://127.0.0.1:8765> with:

```powershell
python tools/run_vcoco_v3_annotation.py
```

The development pilot uses the first 130 tasks in blind-manifest display order. Freeze
and audit that prefix with:

```powershell
python tools/finalize_vcoco_v3_pilot.py
```

Advance source-tag development only when the final summary reports
`VCOCO_V3_HUMAN_PILOT_AUDIT_COMPLETE`. The labels from this pilot do not fit, select,
or calibrate candidates.

Before reporting a human-harmonized endpoint, run two complete independent passes,
the agreement analysis, and a third blinded adjudication pass for every disagreement:

```powershell
python tools/analyze_vcoco_v3_annotations.py
python tools/run_vcoco_v3_adjudication.py
python tools/finalize_vcoco_v3_annotations.py --adjudicator adjudicator-1
```

### Optional partial-pass audit

Freeze the snapshot before unblinding it against the private sampling manifest. A
stopped pass can be summarized without changing the annotation gate:

```powershell
python tools/summarize_vcoco_v3_single_rater_pilot.py `
  --snapshot C:\path\to\frozen\rater.json `
  --expected-responses 60
```

A partial annotation pass can be frozen and summarized without changing the primary
annotation gate. The audit canonicalizes an observed hidden-repeat task to its original sampling
cohort when the original task has not also been answered. Repeat observations are
never double-counted, and repeat agreement is reported only when both presentations
were completed.

## 2. Nested cached-feature development

```powershell
python experiments/evaluate_vcoco_v3_nested_stacks.py
```

Required terminal status:
`VCOCO_V3_NESTED_CACHED_FUSION_DEVELOPMENT_COMPLETE`.

## 3. Spatial mechanisms

Plan the cache queue directly from `experiments/vcoco_v3_spatial_grid.json`, review
the resolved paths, then execute it. Existing verified v2 caches are reused.

```powershell
python experiments/run_vcoco_v3_feature_queue.py --stage spatial
python experiments/run_vcoco_v3_feature_queue.py --stage spatial --execute
python experiments/evaluate_vcoco_v3_spatial.py
```

Required terminal status: `VCOCO_V3_SPATIAL_DEVELOPMENT_COMPLETE`.

## 4. Matched frozen representations

Plan and run the unmatched representation caches. The DINOv2 and SigLIP2 caches are
reused after hash validation. The DINOv3 identifier and revision are fixed in
`experiments/vcoco_v3_representation_grid.json`; the checkpoint is downloaded only
at this stage.

```powershell
python experiments/run_vcoco_v3_feature_queue.py --stage representation
python experiments/run_vcoco_v3_feature_queue.py --stage representation --execute
python tools/lock_vcoco_v3_representations.py
python experiments/evaluate_vcoco_v3_representations.py
```

Required terminal status:
`VCOCO_V3_MATCHED_REPRESENTATION_DEVELOPMENT_COMPLETE`.

## 5. Conditional neural adaptation

```powershell
python tools/lock_vcoco_v3_neural.py
python experiments/run_vcoco_v3_neural_queue.py --stage inner
```

The queue command is a dry plan by default. Review the run count and paths, then add
`--execute`. Inner screening completes before candidate selection:

```powershell
python experiments/run_vcoco_v3_neural_queue.py --stage inner --execute
python tools/select_vcoco_v3_neural.py
python experiments/run_vcoco_v3_neural_queue.py --stage outer
python experiments/run_vcoco_v3_neural_queue.py --stage outer --execute
python experiments/finalize_vcoco_v3_neural.py
```

Outer fits use only the candidate selected within each outer fold. Probe checkpoints
remain eligible when LoRA adaptation reduces the validation metric.

If the spatial promotion rule is not met, the neural stage records
`VCOCO_V3_NEURAL_STAGE_NOT_ELIGIBLE` and the study continues with the strongest
eligible frozen representation.

## 6. Independent temporal dataset

The temporal extension uses Okutama-Action because the complete 392.1 GB POLIMI-ITW-S
release does not fit the available storage. Keep both provider archives outside Git.
The provider-test archive remains sealed until Section 8.

```powershell
$datasetRoot = 'C:\path\to\OkutamaAction'
$okutamaRoot = '.runs\vcoco_v3\okutama'
$temporalRoot = '.runs\vcoco_v3\temporal'
$featureRoot = "$okutamaRoot\features\dinov2_base"
$developmentArchive = "$datasetRoot\TrainSetFrames.zip"
$developmentAudit = "$okutamaRoot\development_audit\summary.json"
$protocolAmendment = '.runs\vcoco_v3\protocol\external_cuda_amendment_lock.json'
$manifest = "$temporalRoot\development_manifest.csv"
$manifestLock = "$temporalRoot\temporal_manifest_lock.json"
$temporalLock = "$temporalRoot\temporal_grid_lock.json"

python tools/audit_okutama_development.py --archive $developmentArchive
python tools/lock_vcoco_v3_external_cuda_amendment.py `
  --output $protocolAmendment
python experiments/cache_okutama_temporal_features.py `
  --partition development `
  --archive $developmentArchive `
  --protocol-amendment $protocolAmendment `
  --model-kind dinov2_base `
  --output-dir $featureRoot
python tools/build_vcoco_v3_temporal_split.py `
  --metadata "$featureRoot\development_metadata.csv" `
  --grid experiments\okutama_temporal_grid.json `
  --output $manifest
python tools/lock_vcoco_v3_temporal_manifest.py `
  --manifest $manifest `
  --provider-provenance $developmentAudit `
  --ontology experiments\okutama_action_protocol.json `
  --grid experiments\okutama_temporal_grid.json `
  --output $manifestLock
python tools/lock_vcoco_v3_temporal.py `
  --grid experiments\okutama_temporal_grid.json `
  --protocol-amendment $protocolAmendment `
  --manifest-lock $manifestLock `
  --output $temporalLock
```

The matched representation screen selected `dinov2_base`; the replay commands use that
locked backbone explicitly. The development split keeps synchronized drone views from
each scenario together. The manifest lock verifies every packed array and records that
no provider-test feature was opened.

Fit the source-only head before any Okutama label-based adaptation. The fitting command
does not accept the target manifest; its evaluator joins the resulting probabilities to
labels in a separate process.

```powershell
python experiments/fit_okutama_source_only_transfer.py `
  --target-store "$featureRoot\store.json" `
  --target-cache-summary "$featureRoot\summary.json"
python experiments/evaluate_okutama_source_only_transfer.py `
  --metadata $manifest `
  --target-store "$featureRoot\store.json"
python experiments/evaluate_okutama_fewshot_transfer.py `
  --grid experiments\okutama_temporal_grid.json `
  --manifest $manifest `
  --target-store "$featureRoot\store.json"
```

## 7. Temporal development and controls

Plan each queue before adding `--execute`. All model fitting is CUDA-only; a missing
CUDA runtime is a hard failure.

```powershell
$teacherLock = "$temporalRoot\teacher_selection_lock.json"
$studentSummary = "$temporalRoot\student_targets\summary.json"
$developmentSummary = "$temporalRoot\development_final\summary.json"
$queueArgs = @(
  '--grid', 'experiments\okutama_temporal_grid.json',
  '--temporal-lock', $temporalLock,
  '--manifest-lock', $manifestLock,
  '--manifest', $manifest,
  '--temporal-root', $temporalRoot
)

python experiments/run_vcoco_v3_temporal_queue.py --phase development @queueArgs
python experiments/run_vcoco_v3_temporal_queue.py --phase development @queueArgs --execute
python tools/select_vcoco_v3_temporal_teacher.py `
  --grid experiments\okutama_temporal_grid.json `
  --temporal-lock $temporalLock `
  --run-root "$temporalRoot\development" `
  --output $teacherLock
python experiments/run_vcoco_v3_temporal_queue.py `
  --phase crossfit @queueArgs --teacher-selection $teacherLock --execute
python tools/aggregate_vcoco_v3_temporal_crossfit.py `
  --grid experiments\okutama_temporal_grid.json `
  --temporal-lock $temporalLock `
  --teacher-selection $teacherLock `
  --manifest $manifest `
  --run-root "$temporalRoot\crossfit" `
  --output-dir "$temporalRoot\crossfit_aggregate"
python tools/build_vcoco_v3_temporal_student_targets.py `
  --grid experiments\okutama_temporal_grid.json `
  --temporal-lock $temporalLock `
  --teacher-selection $teacherLock `
  --crossfit-summary "$temporalRoot\crossfit_aggregate\summary.json" `
  --crossfit-targets "$temporalRoot\crossfit_aggregate\crossfit_targets.npz" `
  --development-run-root "$temporalRoot\development" `
  --output-dir "$temporalRoot\student_targets"
python experiments/run_vcoco_v3_temporal_queue.py `
  --phase students @queueArgs `
  --teacher-selection $teacherLock `
  --student-target-summary $studentSummary `
  --execute
python tools/finalize_vcoco_v3_temporal_development.py `
  --grid experiments\okutama_temporal_grid.json `
  --temporal-lock $temporalLock `
  --teacher-selection $teacherLock `
  --student-target-summary $studentSummary `
  --student-targets "$temporalRoot\student_targets\student_targets.npz" `
  --student-run-root "$temporalRoot\students" `
  --development-run-root "$temporalRoot\development" `
  --output-dir "$temporalRoot\development_final"
```

Fit the optional normalized-pose mechanism control. If pose arrays are unavailable,
the command records `VCOCO_V3_POSE_CONTROL_UNAVAILABLE` and leaves the rest of the
pipeline unchanged.

```powershell
python experiments/evaluate_vcoco_v3_pose_velocity_control.py `
  --grid experiments\okutama_temporal_grid.json `
  --temporal-lock $temporalLock `
  --manifest-lock $manifestLock `
  --development-summary $developmentSummary `
  --manifest $manifest `
  --output-dir "$temporalRoot\pose_control"
```

## 8. Seal and open confirmation once

Fit the final seed models on train plus validation using the locked epoch counts and
predict calibration only. Then seal calibration, routing, prediction-set thresholds,
model hashes, and the pose-control state:

```powershell
python experiments/run_vcoco_v3_temporal_queue.py `
  --phase final @queueArgs `
  --teacher-selection $teacherLock `
  --student-target-summary $studentSummary `
  --development-summary $developmentSummary `
  --execute
python tools/calibrate_vcoco_v3_temporal_pipeline.py `
  --grid experiments\okutama_temporal_grid.json `
  --temporal-lock $temporalLock `
  --development-summary $developmentSummary `
  --manifest-lock $manifestLock `
  --model-root "$temporalRoot\pipeline_models" `
  --pose-control-summary "$temporalRoot\pose_control\summary.json" `
  --output-dir "$temporalRoot\pipeline_lock"
```

Only after `pipeline_lock\summary.json` exists, open the provider test once, cache its
features on CUDA, and evaluate the locked models:

```powershell
$confirmationArchive = "$datasetRoot\TestSetFrames.zip"
$confirmationAudit = "$okutamaRoot\confirmation_audit"
$confirmationFeatures = "$okutamaRoot\confirmation_features\dinov2_base"
$confirmationSourceOnly = "$okutamaRoot\confirmation_source_only"
$pipelineLock = "$temporalRoot\pipeline_lock\summary.json"

python tools/audit_okutama_confirmation.py `
  --archive $confirmationArchive `
  --pipeline-lock $pipelineLock `
  --output-dir $confirmationAudit
python experiments/cache_okutama_temporal_features.py `
  --partition confirmation `
  --archive $confirmationArchive `
  --centres "$confirmationAudit\confirmation_centres.csv" `
  --audit-summary "$confirmationAudit\summary.json" `
  --pipeline-lock $pipelineLock `
  --protocol-amendment $protocolAmendment `
  --model-kind dinov2_base `
  --output-dir $confirmationFeatures
python experiments/fit_okutama_source_only_transfer.py `
  --target-partition confirmation `
  --target-store "$confirmationFeatures\store.json" `
  --pipeline-lock $pipelineLock `
  --target-cache-summary "$confirmationFeatures\summary.json" `
  --output-dir $confirmationSourceOnly
python experiments/evaluate_vcoco_v3_temporal_confirmation.py `
  --grid experiments\okutama_temporal_grid.json `
  --temporal-lock $temporalLock `
  --pipeline-lock $pipelineLock `
  --calibration "$temporalRoot\pipeline_lock\calibration.json" `
  --manifest "$confirmationFeatures\confirmation_metadata.csv" `
  --provider-confirmation-summary "$confirmationAudit\summary.json" `
  --confirmation-cache-summary "$confirmationFeatures\summary.json" `
  --source-only-summary "$confirmationSourceOnly\summary.json" `
  --source-only-predictions "$confirmationSourceOnly\predictions.npz" `
  --model-root "$temporalRoot\pipeline_models" `
  --pose-control-root "$temporalRoot\pose_control" `
  --output-dir "$temporalRoot\confirmation"
```

Both the provider audit and evaluator write single-open ledgers. A runtime failure can
resume only against the same locked pipeline, archive, and manifest.

## Validation before a commit

```powershell
python -m ruff check .
python -m pytest -q
python tools/check_vcoco_v3_readiness.py
git status --short
```
