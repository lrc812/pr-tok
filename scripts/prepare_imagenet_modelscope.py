#!/usr/bin/env python3
"""Convert the ModelScope ImageNet-1K Parquet mirror to ImageFolder layout.

The mirror stores each sample as:
    image: {"bytes": ..., "path": "..._nXXXXXXXX.JPEG"}
    label: integer

This script extracts the synset from the embedded source filename and writes:
    <output>/train/<synset>/*.JPEG
    <output>/val/<synset>/*.JPEG

It is restartable. A shard marker is written only after every row in that shard
has been checked or extracted successfully.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


SYNSET_RE = re.compile(r"n\d{8}")
EXPECTED_SHARDS = {"train": 294, "validation": 14}
EXPECTED_IMAGES = {"train": 1_281_167, "validation": 50_000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ImageNet-1K from the ModelScope Parquet mirror."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/imagenet/modelscope_parquet"),
        help="Directory containing train-*.parquet and validation-*.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/imagenet"),
        help="Output root; train/ and val/ are created below it.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of Parquet shards extracted concurrently.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Process completed shards even while other shards are downloading.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the already extracted ImageFolder tree.",
    )
    return parser.parse_args()


def split_for_shard(path: Path) -> str:
    if path.name.startswith("train-"):
        return "train"
    if path.name.startswith("validation-"):
        return "validation"
    raise ValueError(f"Unrecognized shard name: {path.name}")


def output_split(split: str) -> str:
    return "val" if split == "validation" else split


def completed_shards(source: Path) -> list[Path]:
    shards = sorted(source.glob("train-*.parquet"))
    shards += sorted(source.glob("validation-*.parquet"))
    return [path for path in shards if not Path(f"{path}.aria2").exists()]


def find_synset(source_name: str) -> str:
    matches = SYNSET_RE.findall(source_name)
    if not matches:
        raise ValueError(f"No ImageNet synset in source filename: {source_name!r}")
    return matches[-1]


def extract_shard(task: tuple[str, str, str]) -> dict[str, Any]:
    shard_text, output_text, state_text = task
    shard = Path(shard_text)
    output = Path(output_text)
    state_dir = Path(state_text)
    split = split_for_shard(shard)
    destination = output / output_split(split)
    marker = state_dir / f"{shard.name}.json"

    parquet_file = pq.ParquetFile(shard)
    expected_rows = parquet_file.metadata.num_rows
    if marker.exists():
        state = json.loads(marker.read_text())
        if state.get("rows") == expected_rows:
            return {
                "shard": shard.name,
                "rows": expected_rows,
                "written": 0,
                "skipped": expected_rows,
                "labels": state.get("labels", {}),
                "marker": "reused",
            }

    written = 0
    skipped = 0
    rows = 0
    label_to_synset: dict[int, str] = {}
    for batch in parquet_file.iter_batches(
        batch_size=256, columns=["image", "label"], use_threads=False
    ):
        images = batch.column("image").to_pylist()
        labels = batch.column("label").to_pylist()
        for image, label in zip(images, labels):
            if image is None or image.get("bytes") is None or not image.get("path"):
                raise ValueError(f"Invalid image record in {shard.name}, row {rows}")
            source_name = Path(image["path"]).name
            synset = find_synset(source_name)
            label = int(label)
            previous = label_to_synset.setdefault(label, synset)
            if previous != synset:
                raise ValueError(
                    f"Label {label} maps to both {previous} and {synset} "
                    f"in {shard.name}"
                )

            class_dir = destination / synset
            class_dir.mkdir(parents=True, exist_ok=True)
            target = class_dir / source_name
            image_bytes = image["bytes"]
            if target.exists() and target.stat().st_size == len(image_bytes):
                skipped += 1
            else:
                target.write_bytes(image_bytes)
                written += 1
            rows += 1

    if rows != expected_rows:
        raise RuntimeError(
            f"{shard.name}: extracted {rows} rows, expected {expected_rows}"
        )

    state_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "shard": shard.name,
        "split": split,
        "rows": rows,
        "labels": {str(key): value for key, value in label_to_synset.items()},
    }
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, sort_keys=True) + "\n")
    os.replace(temporary, marker)
    return {
        "shard": shard.name,
        "rows": rows,
        "written": written,
        "skipped": skipped,
        "labels": state["labels"],
        "marker": "created",
    }


def verify_tree(output: Path, require_exact: bool) -> bool:
    observed: dict[str, tuple[int, int]] = {}
    ok = True
    for source_split, destination_split in (
        ("train", "train"),
        ("validation", "val"),
    ):
        root = output / destination_split
        class_dirs = [path for path in root.glob("n????????") if path.is_dir()]
        image_count = sum(1 for _ in root.glob("n????????/*.JPEG"))
        observed[source_split] = (len(class_dirs), image_count)
        expected = EXPECTED_IMAGES[source_split]
        count_ok = image_count == expected if require_exact else image_count <= expected
        ok = ok and count_ok and len(class_dirs) <= 1000
        print(
            f"{destination_split}: {len(class_dirs)} classes, "
            f"{image_count:,}/{expected:,} images"
        )
    return ok


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    state_dir = output / ".modelscope_extract_state"

    if args.verify_only:
        return 0 if verify_tree(output, require_exact=True) else 1

    shards = completed_shards(source)
    shard_counts = Counter(split_for_shard(path) for path in shards)
    active = sorted(path.name for path in source.glob("*.parquet.aria2"))
    print(
        "Completed shards: "
        f"train={shard_counts['train']}/{EXPECTED_SHARDS['train']}, "
        f"validation={shard_counts['validation']}/"
        f"{EXPECTED_SHARDS['validation']}; active={len(active)}"
    )
    if not args.allow_partial:
        for split, expected in EXPECTED_SHARDS.items():
            if shard_counts[split] != expected:
                raise SystemExit(
                    f"{split}: need {expected} complete shards, "
                    f"found {shard_counts[split]}. Re-run after download finishes "
                    "or pass --allow-partial."
                )

    tasks = [(str(path), str(output), str(state_dir)) for path in shards]
    total_rows = total_written = total_skipped = 0
    global_label_map: dict[str, str] = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, args.workers)
    ) as executor:
        futures = {
            executor.submit(extract_shard, task): Path(task[0]).name
            for task in tasks
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            result = future.result()
            total_rows += result["rows"]
            total_written += result["written"]
            total_skipped += result["skipped"]
            for label, synset in result["labels"].items():
                previous = global_label_map.setdefault(label, synset)
                if previous != synset:
                    raise RuntimeError(
                        f"Label {label} maps to both {previous} and {synset}"
                    )
            print(
                f"[{completed}/{len(tasks)}] {result['shard']}: "
                f"{result['rows']:,} rows, {result['written']:,} written, "
                f"{result['skipped']:,} existing"
            )

    print(
        f"Processed {total_rows:,} rows: {total_written:,} written, "
        f"{total_skipped:,} existing."
    )
    all_complete = all(
        shard_counts[split] == expected
        for split, expected in EXPECTED_SHARDS.items()
    )
    verified = verify_tree(output, require_exact=all_complete)
    if all_complete and len(global_label_map) != 1000:
        print(f"ERROR: expected 1000 label mappings, found {len(global_label_map)}")
        verified = False
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
