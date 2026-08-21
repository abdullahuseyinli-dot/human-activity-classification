# Dataset setup

The repository tracks labels, fixed splits, source URLs, image dimensions, and
SHA-256 checksums in `manifest.csv`; it does not redistribute COCO image files.

From the repository root:

```bash
python tools/download_dataset.py --manifest data/manifest.csv
```

The downloader resumes valid files, stages each transfer before an atomic move,
and rejects checksum mismatches. Images are written to `data/images/`, which is
ignored by Git.

The source images come from the COCO image collection. Their original copyright
and terms continue to apply; see the [COCO dataset terms](https://cocodataset.org/#termsofuse).
