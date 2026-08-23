# Human Activity Classification Study v2.0.0

Release tag: `polar-study-v2.0.0`

Version 2.0.0 adds the locked person-level V-COCO study to the source-overlap-controlled
POLAR benchmark. The release includes the complete technical report, implementation,
portable aggregate evidence, and publication figures.

## Official test result

The selected scale-conditioned DINOv2-B stack reached 0.8663 macro-F1, 0.8795
accuracy, 0.2902 log loss, and 0.0076 expected calibration error on 6,077 people from
3,708 official-test images. The historical source-only DINO system reached 0.7071
macro-F1 and 0.7010 accuracy on the same rows. The paired macro-F1 gain was +0.1592,
with a 95% image-cluster bootstrap interval of [+0.1454, +0.1735].

All model and classifier choices were made on the official training and validation
splits. The final stack, evaluation code, historical checkpoints, and metric
implementation were hash-locked before the official test labels were opened once.

## What changed

- Replaced full-frame transfer with aspect-preserving person views at tight and 25%
  context scales.
- Combined cross-fitted multiview log probabilities with five person-box geometry
  features in a calibrated stack.
- Compared DINOv2-B, ConvNeXt-S, and SigLIP2-B under matched target-domain screens.
- Tested flat and factorized posture-motion targets under identical feature inputs.
- Evaluated linear-probe-then-fine-tune schedules with dropout 0.10, person-safe
  augmentation, and AugMix.
- Measured the effect of person scale, crop construction, background intervention,
  geometry, scene occupancy, and image-boundary contact.
- Added image-grouped few-shot curves and selective-prediction analysis.

The largest test gains occurred for the smallest person boxes (+0.2202 macro-F1) and
shortest people (+0.2325). A factorized posture-motion classifier improved over its
matched flat control by +0.0111 validation macro-F1. AugMix and mild partial
fine-tuning were statistically tied with the selected frozen stack, so neither was
added to the final estimator.

## Release artifacts

- `docs/VCOCO_V2_EXTERNAL_TRANSFER.md`: technical report source;
- `output/pdf/vcoco_v2_external_transfer_v2.0.0.pdf`: rendered report;
- `results/vcoco_v2/`: portable locks, metrics, uncertainty, mechanism tables, and
  evidence manifest;
- `assets/vcoco_v2_*`: figures in PNG and SVG formats;
- `results/polar_study_v2.0.0_manifest.json`: release-level SHA-256 inventory;
- `release/POLAR_STUDY_V2.0.0_SHA256SUMS.txt`: checksums for the report and manifest.

The v1.0.0 POLAR report, tag, manifest, and checksum file remain available as the
historical source benchmark.

## Validation

```bash
python -m ruff check .
python -m compileall -q src experiments tools
python -m pytest
python tools/validate_repository.py
python tools/build_study_release_manifest.py --check
```
