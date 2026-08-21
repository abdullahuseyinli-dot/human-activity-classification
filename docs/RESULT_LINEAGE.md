# Result lineage

This repository records a reproducible experiment lineage from a fingerprinted source
pipeline. The source fingerprint below anchors the retained evidence.

## Source fingerprints

| Source | SHA-256 |
|---|---|
| Executed source pipeline | `6468a7765b5d908541a7f30cc3e5cefa3f6cd8fba14e9fba952fd0c623801efc` |
| Cross-validation methodology source | `d455152bd24f90ddcca03297d45adaab8a7b2c877f4299104cd0bf61934fe861` |
| Standardized source manifest | `29886b9ed2048cc1fe1d5d862afbdc5978437f237e78293acc829f8558abc09c` |

## Lineage controls

- The written model-selection result and the executed DINOv2 final-training
  branch disagreed: the search selected partial top-block adaptation, while a
  later cell forced full-backbone fine-tuning.
- Some reported epoch counts and conclusions referred to earlier executions
  rather than the checkpoints represented by the visible output tables.
- Historical artifact directories contained results from different split
  lineages. Those values are retained locally as evidence but are not combined
  into the published benchmark.
- Configured head dropout was present in search dictionaries but was not
  propagated through every model builder. The corrected implementation records
  and applies it explicitly.
- The inherited final full-pool loop discarded validation-driven learning-rate
  reductions. The corrected final runner replays a fold-derived median LR
  schedule.
- ConvNeXt and DINOv2 exposed different feature-return interfaces. A tested
  adapter now provides the same `(logits, pooled_features)` contract without
  changing ConvNeXt logits.
- Local absolute paths, kernel metadata, embedded multi-megabyte training logs,
  project-specific presentation labels, and stale narrative cells are excluded
  from the published notebook.

## Evidence handling

Interrupted smoke runs and superseded checkpoints remain under the ignored
`.runs/` tree. They are not silently promoted or deleted. Tracked `results/`
files are exported only from the corrected lineage after selection locks and
validation checks pass.
