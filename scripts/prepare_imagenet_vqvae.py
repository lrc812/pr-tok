#!/usr/bin/env python3
"""Prepare an extracted ImageNet-1K tree for VQ tokenizer training.

No image is copied or converted. The script writes absolute-path manifests that
can be consumed by ``taming.data.custom.CustomTrain`` and ``CustomTest``.
ImageNet-1K provides class labels, not per-image natural-language captions.
An optional CSV can therefore only contain weak, class-template captions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


EXPECTED_COUNTS = {"train": 1_281_167, "val": 50_000}
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ROCO-style path lists for extracted ImageNet-1K."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/imagenet"),
        help="Root containing train/<synset>/ and val/<synset>/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Manifest output directory (default: DATA_ROOT).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow counts other than the official ImageNet-1K counts.",
    )
    parser.add_argument(
        "--write-caption-csv",
        action="store_true",
        help="Also write large per-image CSVs with class-template captions.",
    )
    parser.add_argument(
        "--caption-template",
        default="a photo of a {class_name}",
        help="Template for optional weak captions.",
    )
    return parser.parse_args()


def imagenet_categories() -> list[str] | None:
    try:
        from torchvision.models._meta import _IMAGENET_CATEGORIES
    except (ImportError, RuntimeError):
        return None
    categories = list(_IMAGENET_CATEGORIES)
    return categories if len(categories) == 1000 else None


def class_table(train_root: Path, val_root: Path) -> list[dict[str, object]]:
    train_synsets = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    val_synsets = sorted(path.name for path in val_root.iterdir() if path.is_dir())
    if train_synsets != val_synsets:
        only_train = sorted(set(train_synsets) - set(val_synsets))
        only_val = sorted(set(val_synsets) - set(train_synsets))
        raise RuntimeError(
            "Train/val synset directories differ: "
            f"only_train={only_train[:5]}, only_val={only_val[:5]}"
        )
    if len(train_synsets) != 1000:
        raise RuntimeError(
            f"Expected 1000 synset directories, found {len(train_synsets)}."
        )

    categories = imagenet_categories()
    classes = []
    for class_index, synset in enumerate(train_synsets):
        class_name = categories[class_index] if categories else synset
        classes.append(
            {
                "class_index": class_index,
                "synset": synset,
                "class_name": class_name,
            }
        )
    return classes


def atomic_target(path: Path) -> tuple[Path, object]:
    temporary = path.with_name(f".{path.name}.tmp")
    return temporary, temporary.open("w", encoding="utf-8", newline="")


def write_class_csv(output_dir: Path, classes: list[dict[str, object]]) -> Path:
    target = output_dir / "imagenet_classes.csv"
    temporary, handle = atomic_target(target)
    with handle:
        writer = csv.DictWriter(
            handle, fieldnames=["class_index", "synset", "class_name"]
        )
        writer.writeheader()
        writer.writerows(classes)
    os.replace(temporary, target)
    return target


def write_split(
    split: str,
    split_root: Path,
    output_dir: Path,
    classes: list[dict[str, object]],
    write_caption_csv: bool,
    caption_template: str,
) -> tuple[Path, Path | None, int]:
    list_target = output_dir / f"imagenet_{split}.txt"
    list_temporary, list_handle = atomic_target(list_target)

    csv_target = output_dir / f"imagenet_{split}_captions.csv"
    if write_caption_csv:
        csv_temporary, csv_handle = atomic_target(csv_target)
        csv_writer = csv.writer(csv_handle)
        csv_writer.writerow(
            [
                "image_path",
                "caption",
                "class_index",
                "synset",
                "class_name",
                "caption_source",
            ]
        )
    else:
        csv_temporary = csv_handle = csv_writer = None

    count = 0
    try:
        for item in classes:
            synset = str(item["synset"])
            class_dir = split_root / synset
            image_paths = sorted(
                path.resolve()
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            class_name = str(item["class_name"])
            caption = caption_template.format(
                class_name=class_name, synset=synset
            )
            for image_path in image_paths:
                list_handle.write(f"{image_path}\n")
                if csv_writer is not None:
                    csv_writer.writerow(
                        [
                            image_path,
                            caption,
                            item["class_index"],
                            synset,
                            class_name,
                            "class_template",
                        ]
                    )
                count += 1
    finally:
        list_handle.close()
        if csv_handle is not None:
            csv_handle.close()

    os.replace(list_temporary, list_target)
    if csv_temporary is not None:
        os.replace(csv_temporary, csv_target)
    return list_target, csv_target if write_caption_csv else None, count


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else data_root
    train_root = data_root / "train"
    val_root = data_root / "val"
    for path in (train_root, val_root):
        if not path.is_dir():
            raise SystemExit(f"Missing ImageNet split directory: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = class_table(train_root, val_root)
    class_csv = write_class_csv(output_dir, classes)
    outputs: dict[str, object] = {
        "data_root": str(data_root),
        "class_count": len(classes),
        "class_csv": str(class_csv),
        "caption_type": (
            "class_template" if args.write_caption_csv else "not_generated"
        ),
        "splits": {},
    }

    for split, split_root in (("train", train_root), ("val", val_root)):
        list_path, caption_path, count = write_split(
            split,
            split_root,
            output_dir,
            classes,
            args.write_caption_csv,
            args.caption_template,
        )
        expected = EXPECTED_COUNTS[split]
        if count != expected and not args.allow_partial:
            raise RuntimeError(
                f"{split}: found {count:,} images, expected {expected:,}. "
                "Use --allow-partial only if this is intentional."
            )
        outputs["splits"][split] = {
            "count": count,
            "path_list": str(list_path),
            "caption_csv": str(caption_path) if caption_path else None,
        }
        print(f"{split}: {count:,} images -> {list_path}")

    summary_path = output_dir / "imagenet_manifest.json"
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary.write_text(
        json.dumps(outputs, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)
    print(f"classes: {len(classes)} -> {class_csv}")
    print(f"summary: {summary_path}")
    print("No image files were copied or modified.")
    if not args.write_caption_csv:
        print(
            "Per-image captions were not generated; ImageNet-1K has class "
            "labels rather than natural captions."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
