# Historical COCO benchmark

The first version of this repository studied a checksum-verified 285-image COCO subset
with three labels: sitting, standing, and walking/running. It remains useful as a compact
example of out-of-fold model selection and attribution-method locking, but it is no
longer the primary cross-domain study.

## Preserved result

The development pool contained 242 images and the fixed test split contained 43. An
out-of-fold weighted probability blend reached 0.862 macro-F1 and 0.860 accuracy on the
test split, with a wide 95% stratified-bootstrap macro-F1 interval of [0.743, 0.954].
The interval and small test size are the main reason the larger POLAR study was added.

The tracked non-POLAR files in `results/` preserve:

- coarse and confirmation rankings;
- configuration and downstream-selection locks;
- seed-level and fixed-test metrics;
- paired bootstrap comparisons;
- OOF-selected attribution methods and fixed-test faithfulness checks.

## Relationship to the POLAR study

The studies do not share a performance claim. POLAR uses its own official splits,
quarantine policy, four-class target, model search, final fits, and one-time test gate.
The COCO manifest is used only as an additional overlap source during data audit. No
legacy test result influenced the POLAR selection.
