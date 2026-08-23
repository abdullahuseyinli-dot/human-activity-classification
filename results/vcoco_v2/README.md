# V-COCO v2 external-transfer evidence

This directory contains the portable evidence for the person-level V-COCO transfer
study. The selected system uses revision-pinned DINOv2-B features from a tight person
crop and a 25% context crop, then combines their cross-fitted probabilities with five
person-box geometry features.

The official test set contains 6,077 people in 3,708 source images. The selected stack
reached 0.8663 macro-F1 and 0.8795 accuracy. The historical source-only DINO system
reached 0.7071 macro-F1 and 0.7010 accuracy on the same rows. The paired macro-F1 gain
was +0.1592 with a 95% image-cluster bootstrap interval of [+0.1454, +0.1735].

Start with:

- `official_test_summary.json` for the primary result and lock lineage;
- `development_candidates.csv` for the controlled validation comparison;
- `official_test_per_class.csv` and `official_test_confusions.json` for class behavior;
- `official_test_selective_metrics.json` for risk-coverage behavior;
- `official_test_strata.csv` for person-scale, boundary, and scene-composition results;
- `final_selection_lock.json` for the selected method and pre-test decisions;
- `evidence_manifest.json` for the SHA-256 inventory.

The full analysis is in
[`docs/VCOCO_V2_EXTERNAL_TRANSFER.md`](../../docs/VCOCO_V2_EXTERNAL_TRANSFER.md).
Dense probabilities, checkpoints, feature tensors, and image manifests with local
paths remain in the ignored `.runs/` tree.
