# Experiment runners

The repository contains two versioned experiment sequences. Each sequence separates
data preparation, development-only model selection, final fitting, test access, and
portable evidence export.

## POLAR v1 source benchmark

1. `tools/prepare_polar.py` builds the target manifest, audits exact and perceptual
   cross-split relationships, and writes the quarantine.
2. `cache_polar_features.py` and the probe runners screen views, label spaces, and
   frozen classifiers on development data.
3. `train_polar_candidate.py` runs adaptation and regularization candidates;
   `aggregate_polar_seed_predictions.py` and the confirmation runners establish
   multi-seed stability.
4. `analyze_polar_validation.py` measures complementarity and selects blend weights.
5. `lock_polar_final_selection.py` writes the pre-test selection lock.
6. `fit_polar_final_model.py` and `fit_polar_final_probe.py` fit the declared models on
   the combined clean development set and verify their artifacts.
7. `evaluate_polar_final.py` performs the single locked POLAR test evaluation.
8. `evaluate_vcoco_external.py` runs the original no-retuning V-COCO audit after
   `tools/audit_polar_vcoco_overlap.py` verifies the cross-dataset boundary.
9. `evaluate_polar_faithfulness.py` runs bbox-aware attribution, perturbation,
   target-sensitivity, and parameter-randomization checks on the fixed cohort.
10. `evaluate_polar_fault_robustness.py` evaluates exact input and classifier-weight
    bit flips as a separate robustness study.

## V-COCO v2 person-level study

1. `tools/build_vcoco_v2_manifests.py` creates factorized person-level manifests while
   retaining the official train, validation, and test memberships.
2. `tools/lock_vcoco_v2_protocol.py` fixes the target mapping, overlap policy,
   development boundary, test gate, metrics, and promotion rule.
3. `tools/build_vcoco_v2_training_manifest.py` materializes the label-bearing training
   view from the locked source manifest.
4. `cache_vcoco_v2_features.py` extracts revision-pinned DINOv2, ConvNeXt, and SigLIP2
   features for declared person views without reading test labels.
5. `screen_vcoco_v2_features.py`, `evaluate_vcoco_v2_source_model_views.py`, and
   `screen_vcoco_v2_pose_rbf.py` run the controlled representation, crop, geometry,
   classifier, and pose screens.
6. `evaluate_vcoco_v2_factorized_fusion.py` compares factorized posture-motion targets
   with flat classifiers under matched inputs.
7. `run_vcoco_v2_fewshot_curve.py` measures image-grouped target-label efficiency.
8. `train_polar_candidate.py` and `evaluate_vcoco_v2_neural_checkpoint.py` run the
   linear-probe-then-fine-tune experiments with mild and AugMix policies.
9. `analyze_vcoco_v2_mechanisms.py` and `analyze_vcoco_v2_prediction_shift.py` quantify
   person-scale, context, background, geometry, and error-transition effects.
10. `fit_vcoco_v2_final_stack.py` replays the selected development result and refits the
    stack on the combined train and validation rows.
11. `tools/lock_vcoco_v2_final_selection.py` binds the selected estimator, inference
    sources, historical baseline, metrics, and official-test contract.
12. `cache_vcoco_v2_final_test_features.py` performs label-free test feature extraction;
    `evaluate_vcoco_v2_final_test.py` then opens the test labels once and writes the
    locked evaluation.
13. `tools/export_vcoco_v2_results.py` creates the path-sanitized evidence package, and
    `tools/render_vcoco_v2_figures.py` renders the tracked figures from that package.

The commands remain explicit so each gate has a visible input and output. Long-running
jobs resume only when the recorded implementation and artifact hashes still match.
`pipeline_source.ipynb` belongs to the historical COCO benchmark; the root notebook is
the current executed evidence summary.
