# Versioned technical reports

`vcoco_v3_motion_identifiability_v3.0.0.pdf` is the current release report. It presents
the V-COCO mechanism study and the locked Okutama-Action static, temporal,
distillation, and fixed-budget routing experiments.

`okutama_cptr_development_v3.0.0.pdf` is the companion architecture development
report. It records the component sequence, five-seed validation result,
recording-grouped cross-fit, faithfulness interventions, and retained failure modes.

Both PDFs are rendered directly from their Markdown sources:

```bash
python tools/build_study_papers.py docs/VCOCO_V3_MOTION_IDENTIFIABILITY.md \
  -o output/pdf/vcoco_v3_motion_identifiability_v3.0.0.pdf
python tools/build_study_papers.py docs/OKUTAMA_CPTR_DEVELOPMENT.md \
  -o output/pdf/okutama_cptr_development_v3.0.0.pdf
```

`vcoco_v2_external_transfer_v2.0.0.pdf` remains the person-level V-COCO report from
the v2 release. Rebuild it from its tagged source with:

```bash
git checkout polar-study-v2.0.0
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

The current release inventory is `results/human_activity_study_v3.0.0_manifest.json`;
checksums for both reports and the manifest are recorded in
`release/HUMAN_ACTIVITY_STUDY_V3.0.0_SHA256SUMS.txt`. The v1 and v2 inventories remain
available with their tagged releases.
