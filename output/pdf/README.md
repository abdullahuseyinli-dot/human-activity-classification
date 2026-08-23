# Technical report artifact

`polar_technical_report.pdf` is the rendered release report for the locked POLAR
study. It is generated from tracked, hash-validated result exports and publication
figures; rebuilding it does not train models or reopen the held-out test set.

From the repository root:

```bash
python -m pip install -e ".[report]"
python tools/render_polar_final_figures.py
python tools/build_technical_report_pdf.py
```

The editable narrative is maintained in
[`docs/POLAR_TECHNICAL_REPORT.md`](../../docs/POLAR_TECHNICAL_REPORT.md). The PDF
builder is [`tools/build_technical_report_pdf.py`](../../tools/build_technical_report_pdf.py).
