# Figures

POLAR charts are generated from tracked, locked evidence with:

```bash
python tools/render_polar_final_figures.py --results-dir results --output-dir assets
```

Each chart is stored as PNG for GitHub/notebook rendering and SVG for vector-quality
reuse. The final set covers held-out model comparison, confusion, data scale, external
transfer, bbox-aware faithfulness, attribution sanity, and bit-flip robustness.

The motion-identifiability confirmation comparison and fixed-budget routing curve are
rendered from the portable Okutama tables with:

```bash
python tools/render_vcoco_v3_figures.py --results results/vcoco_v3 --output-dir assets
```

Files without the `polar_` prefix belong to the historical COCO benchmark. Four
qualitative galleries contain reduced-resolution COCO photographs and remain subject
to the original image terms; they are enumerated in `THIRD_PARTY_NOTICES.md`. The
`vcoco_v3_` files are aggregate charts and contain no source frame. Raw POLAR,
V-COCO, and Okutama-Action media are not copied into the repository.
