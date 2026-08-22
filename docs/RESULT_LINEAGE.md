# Result lineage

The tracked release consolidates model selection, final training, uncertainty
analysis, and promoted metrics into a single reproducible lineage. Source
fingerprints anchor the released evidence without requiring bulky checkpoints in
version control.

## Source fingerprints

| Source | SHA-256 |
|---|---|
| Executed source pipeline | `6468a7765b5d908541a7f30cc3e5cefa3f6cd8fba14e9fba952fd0c623801efc` |
| Cross-validation methodology source | `d455152bd24f90ddcca03297d45adaab8a7b2c877f4299104cd0bf61934fe861` |
| Standardized source manifest | `29886b9ed2048cc1fe1d5d862afbdc5978437f237e78293acc829f8558abc09c` |

## Lineage controls

- The selected freeze depth is passed unchanged into every DINOv2 training branch.
- Reported epoch counts and conclusions are derived from the checkpoints represented
  by the promoted result tables.
- Artifacts from different split lineages are kept separate and are never combined
  into the headline comparison.
- Configured head dropout is recorded and applied by every model builder.
- Full-pool training replays a fold-derived median learning-rate schedule.
- ConvNeXt and DINOv2 share a tested `(logits, pooled_features)` adapter contract
  without altering ConvNeXt logits.
- Attribution methods are selected from fold-held-out development predictions;
  the attribution lock is written before the fixed-test explanations are made.
- Faithfulness replays the exact calibrated inference function and fingerprints
  every contributing OOF and full-pool checkpoint.
- Raw DINOv2 attention rollout remains visible as a class-agnostic negative
  control and is never promoted as a faithful class explanation.
- Release notebooks use portable paths and kernels and exclude bulky training logs.

## Evidence handling

Interrupted smoke runs and superseded checkpoints remain under the ignored `.runs/`
tree. They are not promoted or deleted. Tracked `results/` files are exported only
after selection locks and validation checks pass.

The local faithfulness run additionally preserves dense perturbation traces and
attribution arrays. The tracked release contains the OOF selection cohort and
per-image metrics, the three lock/provenance chains, parameter-randomization and
stability checks, aggregate curves, and review figures. Local image paths and
checkpoint payloads are excluded from the export.
