# Human Activity Classification Study v3.0.0

Version 3.0.0 adds the motion-identifiability and tracked-video experiments to the
source-overlap and person-level transfer studies released in v1 and v2. The release
candidate contains the complete technical reports, implementation, portable aggregate
evidence, publication figures, and an executed evidence notebook.

## Sealed temporal result

The eight-frame, 0.5-second Okutama model reached 0.7854 macro-F1 and 0.7708 accuracy
on 1,771 tracked people from five confirmation scenarios. The matched static model
reached 0.7458 macro-F1 and 0.7301 accuracy. The paired macro-F1 change was +0.0396,
with a 95% scenario-cluster bootstrap interval of [+0.0202, +0.0568].

A fixed 50% routing policy reached 0.7817 macro-F1 and retained 90.7% of the temporal
gain while invoking clip inference for half of the examples. Its paired interval was
[+0.0144, +0.0604]. The pipeline, temperatures, routing budgets, prediction-set
thresholds, checkpoints, and evaluator were bound before the provider test archive
was opened once.

## Camera-compensated part-trajectory residual development

The follow-up architecture combined frozen temporal anchors with center-conditioned
residuals, camera-compensated trajectories, confidence-masked body-region tokens,
quality-aware gates, counterfactual objectives, masked target-video adaptation,
GroupDRO, a SigLIP specialist, and top-block LoRA.

The strongest component improved fixed-validation macro-F1 from 0.7806 to 0.7887,
with most of the gain in standing F1. Recording-grouped cross-fitting did not reproduce
the aggregate improvement: the component reached 0.7144 macro-F1 versus 0.7165 for the
matched temporal baseline. The component remains documented for further work, and the
existing temporal ensemble remains the default model.

## Scientific validation program

The [scientific validation plan](../SCIENTIFIC_VALIDATION_PLAN.md) binds the remaining
work to explicit evidence gates: independent annotation, external-domain replication,
recording-grouped inference, matched baselines, operational measurements, and a fresh
environment replay. These gates determine the permitted claim scope; they do not
require a positive result.

## Release artifacts

- `docs/VCOCO_V3_MOTION_IDENTIFIABILITY.md` and
  `output/pdf/vcoco_v3_motion_identifiability_v3.0.0.pdf`: temporal study report;
- `docs/OKUTAMA_CPTR_DEVELOPMENT.md` and
  `output/pdf/okutama_cptr_development_v3.0.0.pdf`: architecture development report;
- `results/vcoco_v3/`: confirmation metrics, paired uncertainty, routing, calibration,
  subgroup evidence, annotation aggregates, and protocol lineage;
- `results/okutama_cptr/`: component screens, grouped cross-fit, faithfulness,
  uncertainty, provenance, and the development decision;
- `docs/SCIENTIFIC_VALIDATION_PLAN.md`: limitation coverage and evidence gates;
- `human_activity_classification.ipynb`: executed evidence narrative;
- `results/human_activity_study_v3.0.0_manifest.json`: release-level SHA-256 inventory;
- `release/HUMAN_ACTIVITY_STUDY_V3.0.0_SHA256SUMS.txt`: checksums for the reports and
  manifest.

The v3 source tree excludes dataset media, pretrained checkpoints, and qualitative
source-image composites. The v1.0.0 and v2.0.0 tags, reports, manifests, and checksum
files remain unchanged.

## Validation

```bash
python -m ruff check .
python -m compileall -q src experiments tools
python -m pytest
python tools/validate_repository.py
python tools/build_study_release_manifest.py --check
python tools/build_v3_release_manifest.py --check
python tools/verify_v3_release_archive.py
```
