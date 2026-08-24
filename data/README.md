# Dataset setup

No POLAR, V-COCO, Okutama-Action, or COCO image is redistributed in this repository.

## POLAR

Download POLAR v1 from the [Mendeley Data record](https://doi.org/10.17632/hvnsh7rwz7.1)
and verify the multipart archive hashes recorded in
`experiments/polar_study_protocol.json`. Then build the local audited manifests:

```bash
python tools/prepare_polar.py \
  --annotations-dir /path/to/Annotations \
  --images-dir /path/to/JPEGImages \
  --image-sets-dir /path/to/ImageSets \
  --output-dir .runs/polar_data \
  --legacy-manifest data/manifest.csv
```

The promoted audit contains 16,614 clean four-class images after quarantining 125
cross-split source-related images. Local manifests contain absolute image paths and stay
under `.runs/`; only hashes, counts, and exclusion records are exported.

## V-COCO external validation

The external evaluator uses V-COCO train/validation annotations mapped to sitting,
standing, and walking/running. `tools/build_vcoco_external_manifest.py` creates the
person-level manifest, and `tools/audit_polar_vcoco_overlap.py` must pass before model
predictions are allowed. Mixed-label images are retained for person-level evaluation
and excluded from the image-level comparison.

## Okutama-Action temporal extension

Keep the provider train and test frame archives outside the repository. The train
archive supplies the scenario-grouped development, validation, and calibration
partitions; the test archive is opened only against a completed temporal pipeline
lock. Archive hashes, class counts, exclusions, and aggregate metrics are exported,
while frames, person crops, annotations, tracks, features, and checkpoints stay under
ignored local storage. The complete command sequence is in
`docs/VCOCO_V3_EXECUTION_RUNBOOK.md`.

Commands that read an Okutama archive require an explicit `--archive` argument. The
portable CPTR protocol uses `data/external/OkutamaAction/TrainSetFrames.zip` as a
repository-relative example; callers may keep the archive elsewhere and pass its path
without editing a tracked file.

The CPTR architecture study reuses only the locked provider-train development
manifest. Its camera transforms, part-region tokens, specialist features, and 25
grouped cross-fit runs remain under `.runs/cptr/`. The candidate did not pass its
development gate, so the 1,979-row calibration partition was not opened for CPTR.
Portable aggregate evidence is in `results/okutama_cptr/`.

## Historical COCO manifest

`manifest.csv` records URLs, labels, fixed splits, dimensions, and checksums for the
older 285-image benchmark. `tools/download_dataset.py` downloads those files atomically
to the ignored `data/images/` directory and rejects checksum drift.

Source-image rights remain with their original owners. See `THIRD_PARTY_NOTICES.md`.
