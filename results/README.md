# Tracked evidence

This directory contains the compact, path-sanitized outputs required to audit
and render the portfolio notebook: candidate summaries, configuration locks,
seed-level metrics, OOF downstream rankings, test reports, bootstrap intervals,
the champion prediction index, and the locked attribution audit.

The attribution evidence includes the path-sanitized OOF selection cohort,
candidate and selected-method metrics, fixed-test perturbation summaries,
per-image audit values, probability replay checks, seed/TTA stability,
parameter-randomization checks, checkpoint fingerprints, and a dedicated
provenance record. Raw attribution arrays and dense perturbation traces remain
local because they are bulky and unnecessary for normal review.

Model checkpoints, raw logits, embeddings, local paths, and interrupted runs
remain under `.runs/` and are intentionally ignored by Git. A tracked metric is
exported only after the relevant selection lock and validation gate pass.
