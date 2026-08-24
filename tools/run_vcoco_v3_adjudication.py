"""Run the third blind pass for V-COCO v3 annotation disagreements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hac.vcoco_v3_annotation import serve_annotation_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agreement-summary",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/agreement/summary.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/agreement/adjudication_tasks.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".runs/vcoco_v3/annotation/adjudication")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.agreement_summary.resolve().read_text(encoding="utf-8"))
    if summary.get("status") != "VCOCO_V3_ANNOTATION_READY_FOR_BLINDED_ADJUDICATION":
        raise RuntimeError("The independent annotation gate is not ready for adjudication")
    if not (1 <= args.port <= 65535):
        raise ValueError("Port must be between 1 and 65535")
    static_dir = Path(__file__).resolve().parent / "annotation_ui"
    serve_annotation_app(
        args.manifest,
        args.output_dir,
        static_dir,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
