# Changelog

This file records versioned public study artifacts. The Python package has its own
version in `pyproject.toml`.

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

The report version is 1.0.0. The installable Python package remains at version 2.0.0;
the intended Git tag is `polar-study-v1.0.0` to keep those version lines distinct.
