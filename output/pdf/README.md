# Versioned technical reports

`vcoco_v2_external_transfer_v2.0.0.pdf` is the current release report. It presents the
locked person-level V-COCO protocol, development experiments, official-test result,
paired uncertainty, calibration, selective prediction, and measured performance
mechanisms.

The PDF is rendered directly from the Markdown source:

```bash
python tools/build_study_papers.py docs/VCOCO_V2_EXTERNAL_TRANSFER.md \
  -o output/pdf/vcoco_v2_external_transfer_v2.0.0.pdf
```

`polar_public_report_v1.0.0.pdf` remains the source-overlap-controlled POLAR report
from the v1 release. Rebuild it from its tagged source with:

```bash
git checkout polar-study-v1.0.0
python tools/build_study_papers.py docs/POLAR_PUBLIC_REPORT.md \
  -o output/pdf/polar_public_report_v1.0.0.pdf
```

The current release inventory is `results/polar_study_v2.0.0_manifest.json`; checksums
for the PDF and manifest are recorded in
`release/POLAR_STUDY_V2.0.0_SHA256SUMS.txt`.
