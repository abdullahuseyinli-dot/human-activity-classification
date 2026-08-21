"""Extract the audited training primitives into a portable code-only notebook."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat

SOURCE_INDICES = (3, 5, 8, 10, 16, 21, 23, 25, 27)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sanitize(source: str, source_index: int) -> str:
    headings = {
        3: "# Imports shared by the experiment pipeline.",
        5: "# Paths, run flags, and deterministic seed values.",
        8: "# Standardize the manifest before downstream validation.",
        10: "# Check image metadata, decodability, duplicates, and split leakage.",
        16: "# Transform, dataset, loader, and cross-validation definitions.",
        21: "# Model builders and architecture summaries.",
        23: "# Label encoding and shared metric helpers.",
        25: "# Training, validation, and artifact serialization primitives.",
        27: "# Cross-validation and fixed-epoch full-pool training orchestration.",
    }
    lines = source.splitlines()
    lines[0] = headings[source_index]
    if source_index == 16:
        lines = [
            lines[0],
            "# The development pool uses internal CV; the original test split remains fixed.",
        ] + [line for line in lines[1:] if not line.startswith("#")]
    if source_index == 21:
        lines = [
            lines[0],
            "# Saved configuration files keep architecture metadata aligned with selected runs.",
        ] + [line for line in lines[1:] if not line.startswith("#")]
    sanitized = "\n".join(lines) + ("\n" if source.endswith("\n") else "")
    if source_index == 8:
        anchor = 'df = raw_df.copy()\nrename_map = {path_col: "image_path", label_col: "label"}'
        replacement = '''df = raw_df.copy()
# A portable manifest may retain both the public split and its explicit provenance.
# Verify the two views agree before reducing them to the loader's canonical schema.
if split_col is not None and split_col != "original_split" and "original_split" in df.columns:
    declared_original_split = _canonicalize_original_split(df["original_split"])
    declared_split = _canonicalize_original_split(df[split_col])
    if not declared_original_split.equals(declared_split):
        raise ValueError("split and original_split disagree in the input manifest.")
    df = df.drop(columns=["original_split"])

rename_map = {path_col: "image_path", label_col: "label"}'''
        if sanitized.count(anchor) != 1:
            raise RuntimeError("Could not locate the manifest canonicalization anchor.")
        sanitized = sanitized.replace(anchor, replacement, 1)
    return re.sub(
        r'"project_title": "[A-Z]{2}\d{3}[A-Z]\s+Human Activity Classification from Still Images"',
        '"project_title": "Human Activity Classification from Still Images"',
        sanitized,
    )


def main() -> None:
    args = parse_args()
    source_notebook = nbformat.read(args.source, as_version=4)
    cells = []
    for source_index in SOURCE_INDICES:
        original = source_notebook.cells[source_index]
        if original.cell_type != "code":
            raise RuntimeError(f"Expected code at source cell {source_index}")
        cell = nbformat.v4.new_code_cell(sanitize(original.source, source_index))
        cell.metadata["pipeline_source_index"] = source_index
        cell.metadata["tags"] = ["pipeline-source", f"source-cell-{source_index}"]
        cells.append(cell)

    output = nbformat.v4.new_notebook(cells=cells)
    output.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "source_provenance": {
            "description": "Audited code-only extraction; outputs intentionally omitted",
            "source_indices": list(SOURCE_INDICES),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(output, args.output)
    print(f"Wrote portable pipeline source: {args.output.resolve()}")


if __name__ == "__main__":
    main()
