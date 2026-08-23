# POLAR Study Report v1.0.0

`polar_public_report_v1.0.0.pdf` is the canonical public research report. It extends the
repository overview with the complete experimental design, post-lock analysis,
statistical boundaries, integrated interpretation, and artifact map.

The PDF is rendered directly from its Markdown source, so narrative and PDF content do
not need to be maintained independently:

```bash
python tools/build_study_papers.py docs/POLAR_PUBLIC_REPORT.md \
  -o output/pdf/polar_public_report_v1.0.0.pdf
```

The source is `docs/POLAR_PUBLIC_REPORT.md`; layout is defined by
`tools/build_study_papers.py`. The release-level SHA-256 inventory is
`results/polar_study_v1.0.0_manifest.json`.

This is the only report PDF intended for the versioned public release.
