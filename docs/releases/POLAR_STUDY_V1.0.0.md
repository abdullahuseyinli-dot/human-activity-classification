# POLAR Study Report v1.0.0

Release tag: `polar-study-v1.0.0`

This release packages the final public report, reproducible analysis code, portable
aggregate evidence, and publication figures for the source-overlap-controlled POLAR
study. It does not include dataset images, trained checkpoints, fitted classifier
binaries, dense per-example predictions, or full-resolution attribution arrays.

## Headline result

The development-locked five-component ensemble reaches 0.9399 macro-F1 (95%
class-stratified bootstrap interval [0.9312, 0.9481]) and 0.9456 accuracy on 3,329
post-quarantine POLAR test images. Every paired macro-F1 interval favors the ensemble
over its corresponding constituent.

## What this release adds

- A one-time, hash-verified test evaluation following development-only model,
  classifier, epoch, and blend selection.
- A fixed nested-subset analysis showing validation macro-F1 increasing from 0.8487 at
  242 training images to 0.9150 at 9,958 images.
- A cost-aware comparison of frozen DINOv2-B logistic and calibrated RBF probes.
- A no-retuning V-COCO audit reported at both image and person level.
- A post-lock decomposition showing that 95.6% of ensemble errors lie between adjacent
  posture states.
- A person-scale analysis in which the ensemble's largest gain over the RBF probe occurs
  in the smallest-person quartile (+0.0305 macro-F1).
- An annotation-policy audit showing that 93.1% of mapped V-COCO locomotion people also
  carry a source `stand` action, explaining much of the apparent external error.
- Attribution diagnostics that separate raw person-box mass, area-normalized lift,
  matched occlusion, target sensitivity, and parameter sensitivity.
- Bounded random bit-flip diagnostics across input tensors and classifier matrices.

Post-lock analyses are labelled as hypothesis-generating and leave the selected models,
ensemble weights, thresholds, and test metrics unchanged.

## Release artifacts

- `docs/POLAR_PUBLIC_REPORT.md` - canonical report source.
- `output/pdf/polar_public_report_v1.0.0.pdf` - rendered report.
- `results/polar_study_v1.0.0_manifest.json` - SHA-256 manifest for the public artifact
  set.
- `results/polar_exploratory_summary.json` and `results/polar_exploratory_*.csv` -
  portable exploratory results.
- `experiments/analyze_polar_exploratory.py` - deterministic post-lock analysis and
  figure generation.
- `assets/polar_exploratory_*` - publication figures in PNG and SVG formats.

GitHub automatically provides source ZIP and tar archives for the release tag. The PDF
and checksum file are intended to be attached as explicit release assets.

## Validation

```bash
python -m ruff check .
python -m compileall -q src experiments tools
python -m pytest
python tools/validate_repository.py
python tools/build_study_release_manifest.py --check
```

## Versioning

The research report is version 1.0.0. The installable Python package remains version
2.0.0. The distinct `polar-study-v1.0.0` tag prevents the report release from being
mistaken for a backwards software-package version.
