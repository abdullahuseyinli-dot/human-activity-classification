"""Run the local, resumable V-COCO v3 blinded annotation interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from hac.vcoco_v3_annotation import serve_annotation_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".runs/vcoco_v3/annotation/pilot/blind_tasks.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(".runs/vcoco_v3/annotation/pilot"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
