# Tracked evidence

This directory contains compact, path-sanitized evidence for two distinct studies.
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
- `polar_final_evidence_manifest.json` — hashes for the locked primary and auxiliary
  evidence exported before the post-lock analysis.
- `polar_study_v1.0.0_manifest.json` — SHA-256 inventory for the public report,
  post-lock analysis supplement, and release metadata.

## Post-lock exploratory analysis

Files prefixed with `polar_exploratory_` are hypothesis-generating analyses of the
already completed predictions. They do not alter the locked candidate, ensemble
weights, or confirmatory interpretation.

- `polar_exploratory_summary.json` records the analysis role, fixed seed, 10,000-draw
  intervals, source hashes, and the principal findings.
- `polar_exploratory_pairwise.csv`, `polar_exploratory_blend_sensitivity.csv`, and
  `polar_exploratory_ensemble_structure.csv` describe paired comparisons, weight
  sensitivity, disagreement, and oracle headroom.
- `polar_exploratory_bbox_strata.csv`, `polar_exploratory_domain_composition.csv`,
  `polar_exploratory_external_semantics.csv`, and
  `polar_exploratory_mixed_scene_persons.csv` examine person scale, dataset
  composition, annotation semantics, and multi-person scenes.
- `polar_exploratory_scale_integrity.csv`, `polar_exploratory_scale_per_class.csv`,
  and `polar_exploratory_regularization.csv` document the fixed-subset learning curve
  and classification-calibration tradeoffs.
- The remaining exploratory tables cover selective prediction, geometry-only signal,
  faithfulness strata and error detection, and class-conditioned fault response.

The deterministic builder is `experiments/analyze_polar_exploratory.py`. Re-running it
requires the ignored dense predictions and sanitized local manifests under `.runs/`;
the portable aggregate outputs are tracked so a repository reader does not need those
large, rights-sensitive artifacts.

Checkpoints, fitted RBF/logistic binaries, embeddings, local paths, dense probabilities,
and full-resolution attribution arrays remain under ignored `.runs/` directories. They
are not deleted, but they are not suitable for a portable Git repository.
