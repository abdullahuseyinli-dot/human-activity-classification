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

The current distributable assets contain aggregate charts and architecture diagrams;
they do not contain dataset photographs or video frames. The v1 and v2 tags included
four qualitative COCO composites. Those files remain identifiable in the historical
record and in retained local run evidence, but are excluded from the v3 distributable
tree because the source photographs retain their individual Flickr terms. See
`THIRD_PARTY_NOTICES.md` for the release boundary. Raw POLAR, V-COCO, and
Okutama-Action media are not copied into the repository.
