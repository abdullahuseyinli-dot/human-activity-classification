# Tracked evidence

This directory contains the compact, path-sanitized outputs required to audit
and render the portfolio notebook: candidate summaries, configuration locks,
seed-level metrics, OOF downstream rankings, test reports, bootstrap intervals,
and the champion prediction index.

Model checkpoints, raw logits, embeddings, local paths, and interrupted runs
remain under `.runs/` and are intentionally ignored by Git. A tracked metric is
exported only after the relevant selection lock and validation gate pass.
