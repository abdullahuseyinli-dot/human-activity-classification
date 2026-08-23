# Experiment runners

The POLAR workflow is staged so data inspection, model selection, final fitting, and
test evaluation have explicit boundaries.

1. `tools/prepare_polar.py` builds the target manifest, audits exact and perceptual
   cross-split relationships, and writes the immutable quarantine.
2. `cache_polar_features.py` and the probe runners screen views, label spaces, and
   frozen classifiers on development data.
3. `train_polar_candidate.py` runs adaptation and regularization candidates;
   `aggregate_polar_seed_predictions.py` and the confirmation runners establish
   multi-seed stability.
4. `analyze_polar_validation.py` measures complementarity and selects blend weights.
5. `lock_polar_final_selection.py` writes the final pre-test lock.
6. `fit_polar_final_model.py` and `fit_polar_final_probe.py` fit all declared models on
   the combined clean development set. Artifact verification must pass before the test
   gate can open.
7. `evaluate_polar_final.py` performs the single locked POLAR test evaluation.
8. `evaluate_vcoco_external.py` evaluates the fixed models on V-COCO without retuning,
   after `tools/audit_polar_vcoco_overlap.py` reports a clean cross-dataset audit.
9. `evaluate_polar_faithfulness.py` runs bbox-aware attribution, perturbation,
   target-sensitivity, and parameter-randomization checks on the fixed cohort.
10. `evaluate_polar_fault_robustness.py` evaluates exact input and classifier-weight
    bit flips as a separate robustness study.

The commands are intentionally explicit rather than wrapped in a single script. This
makes every acceptance gate visible and avoids an accidental second test-selection loop.
Long jobs are resumable when their implementation and artifact hashes match.

`pipeline_source.ipynb` belongs to the historical COCO benchmark. The root notebook is
the concise, executed study summary and reads only tracked evidence.
