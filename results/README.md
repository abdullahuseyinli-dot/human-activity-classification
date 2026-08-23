# Tracked evidence

This directory contains compact, path-sanitized evidence for two independent studies.
Files prefixed with `polar_` belong to the primary POLAR benchmark. The remaining files
preserve the historical 285-image COCO benchmark.

For the POLAR release, begin with:

- `polar_final_selection_lock.json` — the pre-test model, ensemble, and evaluation lock;
- `polar_final_fit_manifest.json` — checkpoint/probe hashes, sizes, configurations, and
  runtimes recorded before test evaluation;
- `polar_test_summary.json` and `polar_test_uncertainty.json` — locked primary metrics
  and 10,000-resample uncertainty;
- `polar_external_summary.json` — no-retuning V-COCO transfer results;
- `polar_faithfulness_summary.json` — localization, perturbation, and randomization
  evidence;
- `polar_fault_summary.json` — separately reported bit-flip robustness;
- `polar_final_evidence_manifest.json` — hashes for every portable final artifact.

Checkpoints, fitted RBF/logistic binaries, embeddings, local paths, dense probabilities,
and full-resolution attribution arrays remain under ignored `.runs/` directories. They
are not deleted, but they are not suitable for a portable Git repository.
