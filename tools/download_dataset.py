"""Download and checksum the image files declared by the portable manifest."""

from __future__ import annotations

import argparse
import hashlib
import time
import urllib.request
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_output(repo_root: Path, relative_path: str) -> Path:
    destination = (repo_root / Path(relative_path)).resolve()
    try:
        destination.relative_to(repo_root)
    except ValueError as exc:
        raise RuntimeError(f"Manifest path escapes the repository: {relative_path}") from exc
    return destination


def download_one(
    url: str, destination: Path, expected_sha256: str, retries: int, timeout: float
) -> str:
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return "present"
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "hac-repro/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with staged.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            observed = sha256_file(staged)
            if observed != expected_sha256:
                raise RuntimeError(
                    f"checksum mismatch: expected {expected_sha256}, observed {observed}"
                )
            staged.replace(destination)
            return "downloaded"
        except Exception:
            if staged.exists():
                staged.unlink()
            if attempt == retries:
                raise
            time.sleep(float(attempt))
    raise AssertionError("unreachable")


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    repo_root = manifest_path.parent.parent
    frame = pd.read_csv(manifest_path, dtype={"image_id": str, "sha256": str})
    required = {"image_path", "image_url", "sha256"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing manifest fields: {sorted(missing)}")

    counts = {"present": 0, "downloaded": 0}
    failures = []
    for index, row in frame.iterrows():
        destination = resolve_output(repo_root, str(row["image_path"]))
        try:
            status = download_one(
                str(row["image_url"]),
                destination,
                str(row["sha256"]).lower(),
                int(args.retries),
                float(args.timeout),
            )
            counts[status] += 1
        except Exception as exc:
            failures.append((int(index), str(row["image_url"]), str(exc)))
        completed = index + 1
        if completed % 25 == 0 or completed == len(frame):
            print(f"{completed}/{len(frame)} checked", flush=True)

    print(f"Valid existing files: {counts['present']}")
    print(f"Downloaded files: {counts['downloaded']}")
    if failures:
        preview = "\n".join(f"row {idx}: {url} ({error})" for idx, url, error in failures[:10])
        raise RuntimeError(f"{len(failures)} downloads failed:\n{preview}")


if __name__ == "__main__":
    main()
