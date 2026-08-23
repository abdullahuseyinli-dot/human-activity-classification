# Changelog

This file records versioned public study artifacts. The Python package has its own
version in `pyproject.toml`.

## Study Release 2.0.0 - 2026-08-24

- Added a split-preserving, person-level V-COCO study with training-split candidate
  fitting, validation-only selection, a locked train-plus-validation refit, and one
  official test-label open.
- Improved official-test macro-F1 from 0.7071 for the historical source-only DINO
  baseline to 0.8663 for the selected scale-conditioned DINO stack. The paired gain is
  +0.1592 with a 95% image-cluster bootstrap interval of [+0.1454, +0.1735].
- Added controlled screens for person crops, context width, aspect-ratio handling,
  DINOv2, ConvNeXt, SigLIP2, geometry, factorized targets, LP-FT, person-safe
  augmentation, AugMix, background intervention, and a pose diagnostic.
- Added official-test calibration, selective-prediction, per-class, confusion,
  person-scale, boundary, and scene-occupancy evidence.
- Added the v2 technical report, publication figures, executed notebook update,
  release notes, citation metadata, and versioned SHA-256 manifests.
- Retained the v1.0.0 POLAR report, manifest, checksum file, and Git tag as the
  historical source benchmark.

## POLAR Study Report 1.0.0 - 2026-08-23

- Added the complete independent technical report and its reproducible PDF builder.
- Added deterministic post-lock analyses covering error topology, model disagreement,
  person scale, annotation semantics, mixed-person scenes, selective prediction,
  attribution geometry, regularization tradeoffs, and class-conditioned fault response.
- Added 17 portable exploratory tables, a strict JSON summary with source hashes, and
  seven publication figure families in PNG and SVG formats.
- Added explicit separation between the primary locked result, predeclared auxiliary
  diagnostics, and post-lock hypothesis-generating analyses.
- Added GitHub release notes, Zenodo metadata, citation metadata, and a checksummed
  release manifest.
- Consolidated report navigation around the versioned v1.0.0 artifact and removed the
  superseded report duplicate.
- Removed a machine-local path from the portable failure ledger and made future exports
  sanitize repository-local failure messages.

The report version is 1.0.0. The installable Python package is version 2.0.0; the
historical report tag is `polar-study-v1.0.0`.
