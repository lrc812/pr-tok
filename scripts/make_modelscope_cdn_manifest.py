#!/usr/bin/env python3
"""Resolve ModelScope API URLs and build an aria2 manifest for another CN CDN."""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import urllib.request
from pathlib import Path


TRAIN_INDEX_RE = re.compile(r"train-(\d{5})-of-\d{5}\.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cdn", default="cdn-lfs-cn-2.modelscope.cn")
    parser.add_argument("--train-start", type=int, default=234)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def read_entries(path: Path) -> list[tuple[str, str, str]]:
    lines = path.read_text().splitlines()
    if len(lines) % 3:
        raise ValueError(f"Expected three lines per aria2 entry in {path}")
    entries = []
    for offset in range(0, len(lines), 3):
        url = lines[offset].strip()
        out_line = lines[offset + 1].strip()
        checksum_line = lines[offset + 2].strip()
        if not out_line.startswith("out="):
            raise ValueError(f"Invalid output line: {out_line}")
        if not checksum_line.startswith("checksum="):
            raise ValueError(f"Invalid checksum line: {checksum_line}")
        entries.append((url, out_line, checksum_line))
    return entries


def select_entry(entry: tuple[str, str, str], train_start: int) -> bool:
    output_name = entry[1].split("=", 1)[1]
    if output_name.startswith("validation-"):
        return True
    match = TRAIN_INDEX_RE.fullmatch(output_name)
    return bool(match and int(match.group(1)) >= train_start)


def resolve(entry: tuple[str, str, str], cdn: str) -> tuple[str, str, str]:
    request = urllib.request.Request(entry[0], method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        direct_url = response.geturl()
    if "cdn-lfs-cn-" not in direct_url:
        raise RuntimeError(f"Unexpected ModelScope redirect: {direct_url}")
    direct_url = re.sub(r"cdn-lfs-cn-\d+\.modelscope\.cn", cdn, direct_url)
    return direct_url, entry[1], entry[2]


def main() -> None:
    args = parse_args()
    entries = [
        entry
        for entry in read_entries(args.input)
        if select_entry(entry, args.train_start)
    ]
    with concurrent.futures.ThreadPoolExecutor(args.workers) as executor:
        resolved = list(
            executor.map(lambda entry: resolve(entry, args.cdn), entries)
        )
    lines = []
    for url, output, checksum in resolved:
        lines.extend((url, f"  {output}", f"  {checksum}"))
    args.output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(resolved)} entries to {args.output}")


if __name__ == "__main__":
    main()
