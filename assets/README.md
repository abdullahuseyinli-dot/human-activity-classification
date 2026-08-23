# Figures

POLAR charts are generated from tracked, locked evidence with:

```bash
python tools/render_polar_final_figures.py --results-dir results --output-dir assets
```

Each chart is stored as PNG for GitHub/notebook rendering and SVG for vector-quality
reuse. The final set covers held-out model comparison, confusion, data scale, external
transfer, bbox-aware faithfulness, attribution sanity, and bit-flip robustness.

Files without the `polar_` prefix belong to the historical COCO benchmark. Image
galleries contain only redistributable derivatives from that study; raw POLAR and
V-COCO images are never copied into the repository.
