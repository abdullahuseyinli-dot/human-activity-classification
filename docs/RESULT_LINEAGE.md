# Result lineage

The promoted POLAR result has one evidence chain from the pre-fit data audit to the
portable release. Every selection decision was made on the clean development split.
The official test manifest was opened once, after all final fits and their artifact
hashes had been verified.

## Release chain

| Gate | Tracked evidence | Outcome |
| --- | --- | --- |
| Data lock | `polar_data_audit.json`, `polar_quarantine.csv` | 125 source-related images quarantined before fitting |
| Development protocol | `polar_study_protocol.json` | Models, searches, metrics, and label mappings declared |
| Development selection | `polar_validation_*`, `polar_classifier_*`, `polar_confirmation_*` | Candidates and ensemble weights fixed without test rows |
| Final selection lock | `polar_final_selection_lock.json` | Immutable SHA-256 `fa3fb7c80a073d29048afb8e0b8da1fb17f5ade9721630347c37523714cca187` |
| Final-fit gate | `polar_final_fit_manifest.json` | Nine neural fits and three probes hash-verified; zero test rows read |
| Test-access gate | `polar_test_access_gate.json` | Official test cache opened once after the lock |
| Locked evaluation | `polar_test_*` | 3,329 test images; no post-test model selection |
| External evaluation | `polar_external_*` | V-COCO evaluated without retuning after a clean overlap audit |
| Explanation audit | `polar_faithfulness_*` | Fixed 256-image cohort; no attribution-method selection on test |
| Fault audit | `polar_fault_*` | Separate input and classifier-weight bit-flip evaluation |
| Portable export | `polar_final_evidence_manifest.json` | Path-free tracked evidence with per-file hashes |

## Data fingerprints

| Artifact | SHA-256 |
| --- | --- |
| Clean POLAR development manifest | `2cc5bf62790f6064acc74f7113c341b5c2cd1124bcbfbc3d1e9eb19618a75b83` |
| Clean POLAR test manifest | `a8e01e1037e980edeb1886ef2bbbd22a07c316df044741c08e8e2afa8d32fce5` |
| Opened test-manifest cache | `35d2a73b33de866f52d857c266f57d3eee1259113f3da43b07bdd3e8dfa1142e` |
| Clean V-COCO person manifest | `0eb0afcc7e832babc495d3d9a74981077968f401097eedc34c6aeb9a0e644cb4` |
| POLAR/V-COCO overlap audit | `668e0951fc2d4128559a140a306e612fb32321e246617ca0923ee380fc1cc7d0` |

## Selection boundary

The following were fixed before the official test cache opened:

- label spaces and primary metric;
- image views, adaptation depths, augmentation, dropout, and fixed epoch counts;
- seeds 42, 52, and 62 for neural fits;
- the DINOv2 multilayer representation and linear/RBF classifier settings;
- seed averaging and the five component ensemble weights;
- bootstrap seed and resample count;
- the external mapping, attribution cohort, and fault-injection levels.

Test results did not break ties. The primary ensemble happened to be the strongest
observed test candidate, but its identity and weights remained the pre-test lock.

## Local versus tracked artifacts

Local `.runs/` evidence retains checkpoints, fitted classifier binaries, local image
paths, full probability arrays, full-resolution attribution maps, failures, and
interrupted runs. These files are intentionally ignored rather than deleted.

The tracked `results/` export excludes reconstructive or machine-specific artifacts.
Its manifest records the hash of every promoted table and JSON document. The final-fit
manifest publishes checkpoint hashes, sizes, configurations, and runtimes without
publishing the checkpoints themselves.

## Historical study

The original 285-image COCO experiment has a separate selection and test lineage. It is
preserved in the non-POLAR result files and summarized in
[`LEGACY_COCO_STUDY.md`](LEGACY_COCO_STUDY.md). Its already-inspected 43-image test set
was not reused to select the POLAR system, and its result is not the repository headline.
