#!/usr/bin/env python3
"""Prepare authorized CheXpert Plus files in this repository's ROCO layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "valid", "test")
PATH_COLUMNS = ("path_to_image", "image_path", "Path", "path", "file_path")
PATIENT_COLUMNS = ("patient_id", "patient", "Patient", "subject_id")
STUDY_COLUMNS = ("study_id", "study", "Study", "accession_number")
SPLIT_COLUMNS = ("split", "Split", "partition", "dataset_split")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_root = project_root / "data" / "chexpert_plus"
    parser = argparse.ArgumentParser(
        description="Prepare CheXpert Plus image/report pairs like the local ROCO data."
    )
    parser.add_argument("--raw-dir", type=Path, default=default_root / "raw")
    parser.add_argument("--output-dir", type=Path, default=default_root)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument(
        "--caption-mode",
        choices=("impression", "findings", "findings_impression"),
        default="impression",
        help="'impression' falls back to findings when impression is empty.",
    )
    parser.add_argument(
        "--split-source",
        choices=("patient-hash", "official", "auto"),
        default="patient-hash",
    )
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy", "none"),
        default="symlink",
        help="Symlinks avoid duplicating more than 200k image files.",
    )
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--one-image-per-study", action="store_true")
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument(
        "--absolute-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def normalized_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def choose_caption(row: pd.Series, mode: str) -> str:
    impression = normalized_text(row.get("section_impression"))
    findings = normalized_text(row.get("section_findings"))
    if mode == "impression":
        return impression or findings
    if mode == "findings":
        return findings or impression
    parts = []
    if findings:
        parts.append(f"Findings: {findings}")
    if impression:
        parts.append(f"Impression: {impression}")
    return " ".join(parts)


def auto_metadata_csv(raw_dir: Path) -> Path:
    preferred = sorted(raw_dir.rglob("df_chexpert_plus_*.csv"))
    if preferred:
        return preferred[-1]
    candidates = sorted(raw_dir.rglob("*.csv"))
    for candidate in candidates:
        try:
            columns = set(pd.read_csv(candidate, nrows=1).columns)
        except Exception:
            continue
        if (
            set(PATH_COLUMNS).intersection(columns)
            and {"section_findings", "section_impression"}.intersection(columns)
        ):
            return candidate
    raise FileNotFoundError(
        f"No CheXpert Plus metadata CSV under {raw_dir}; "
        "expected df_chexpert_plus_*.csv."
    )


def build_image_index(raw_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    by_relative: dict[str, Path] = {}
    by_basename: dict[str, Path] = {}
    duplicate_basenames: set[str] = set()
    for path in raw_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        resolved = path.resolve()
        relative = path.relative_to(raw_dir).as_posix()
        by_relative[relative] = resolved
        by_relative[relative.lower()] = resolved
        basename = path.name.lower()
        if basename in by_basename and by_basename[basename] != resolved:
            duplicate_basenames.add(basename)
        else:
            by_basename[basename] = resolved
    for basename in duplicate_basenames:
        by_basename.pop(basename, None)
    return by_relative, by_basename


def resolve_image(
    value: object,
    raw_dir: Path,
    by_relative: dict[str, Path],
    by_basename: dict[str, Path],
) -> Path | None:
    raw_value = normalized_text(value).replace("\\", "/")
    if not raw_value:
        return None
    input_path = Path(raw_value)
    direct_candidates = (
        input_path,
        raw_dir / input_path,
        raw_dir / "images" / input_path,
        raw_dir / "CheXpert-v1.0" / input_path,
        raw_dir / "CheXpert-v1.0-small" / input_path,
    )
    for candidate in direct_candidates:
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            return candidate.resolve()

    normalized = raw_value.lstrip("./")
    parts = normalized.split("/")
    suffixes = [normalized]
    suffixes.extend("/".join(parts[start:]) for start in range(1, min(5, len(parts))))
    for candidate in suffixes:
        found = by_relative.get(candidate) or by_relative.get(candidate.lower())
        if found is not None:
            return found
    return by_basename.get(input_path.name.lower())


def patient_key(row: pd.Series, column: str | None, raw_path: object) -> str:
    if column:
        value = normalized_text(row.get(column))
        if value:
            return value
    match = re.search(r"(?:^|/)(patient[^/]+)(?:/|$)", str(raw_path), re.I)
    return match.group(1).lower() if match else str(raw_path)


def study_key(
    row: pd.Series,
    patient: str,
    column: str | None,
    raw_path: object,
) -> str:
    if column:
        value = normalized_text(row.get(column))
        if value:
            return f"{patient}/{value}"
    match = re.search(r"(?:^|/)(study[^/]+)(?:/|$)", str(raw_path), re.I)
    return f"{patient}/{match.group(1).lower()}" if match else str(raw_path)


def hashed_split(key: str, seed: int, train_ratio: float, valid_ratio: float) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if fraction < train_ratio:
        return "train"
    if fraction < train_ratio + valid_ratio:
        return "valid"
    return "test"


def official_split(value: object) -> str | None:
    return {
        "train": "train",
        "training": "train",
        "val": "valid",
        "valid": "valid",
        "validation": "valid",
        "test": "test",
        "testing": "test",
    }.get(normalized_text(value).lower())


def expose_image(source: Path, destination: Path, mode: str) -> Path:
    if mode == "none":
        return source
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return destination
        raise FileExistsError(f"Refusing to replace existing file: {destination}")
    if mode == "symlink":
        destination.symlink_to(os.path.relpath(source, destination.parent))
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    return destination


def write_outputs(
    output_dir: Path,
    records: dict[str, list[dict[str, str]]],
    absolute_paths: bool,
) -> None:
    for split in SPLITS:
        rows = records[split]
        with (output_dir / f"{split}_captions_modified.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=("ID", "Caption"))
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "ID": (
                            str(Path(row["image_path"]).resolve())
                            if absolute_paths
                            else row["relative_path"]
                        ),
                        "Caption": row["caption"],
                    }
                )
        with (output_dir / f"{split}_captions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=("ID", "Caption"))
            writer.writeheader()
            writer.writerows(
                {"ID": row["relative_path"], "Caption": row["caption"]}
                for row in rows
            )
        with (output_dir / f"medical_{split}.txt").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in rows:
                path = (
                    str(Path(row["image_path"]).resolve())
                    if absolute_paths
                    else row["relative_path"]
                )
                handle.write(f"{path}\n")

    # Compatibility aliases for existing configs that call validation "test".
    shutil.copyfile(output_dir / "medical_valid.txt", output_dir / "medical_test.txt")
    shutil.copyfile(
        output_dir / "valid_captions_modified.csv",
        output_dir / "test_captions_modified.csv",
    )
    shutil.copyfile(
        output_dir / "valid_captions.csv", output_dir / "test_captions.csv"
    )


def main() -> int:
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)
    ratios = (args.train_ratio, args.valid_ratio, args.test_ratio)
    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must be non-negative and sum to 1: {ratios}")

    metadata_csv = (
        args.metadata_csv.expanduser().resolve()
        if args.metadata_csv
        else auto_metadata_csv(raw_dir)
    )
    frame = pd.read_csv(metadata_csv, low_memory=False)
    path_column = first_column(frame, PATH_COLUMNS)
    if path_column is None:
        raise ValueError(f"No image path column; found {list(frame.columns)}")
    patient_column = first_column(frame, PATIENT_COLUMNS)
    study_column = first_column(frame, STUDY_COLUMNS)
    split_column = first_column(frame, SPLIT_COLUMNS)

    print(f"Metadata: {metadata_csv}")
    print(f"Rows: {len(frame):,}; path column: {path_column}")
    print("Indexing downloaded JPEG/PNG files...")
    by_relative, by_basename = build_image_index(raw_dir)
    unique_images = len({str(path) for path in by_relative.values()})
    print(f"Indexed images: {unique_images:,}")
    if not unique_images:
        raise FileNotFoundError(
            "No JPEG/PNG images found. Download the PNG/JPEG image table, "
            "not only DICOM files."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (output_dir / f"{split}_images" / split).mkdir(parents=True, exist_ok=True)

    records: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    counters: Counter[str] = Counter()
    seen_studies: set[str] = set()
    for index, row in frame.iterrows():
        raw_path = row.get(path_column)
        caption = choose_caption(row, args.caption_mode)
        if not caption:
            counters["empty_caption"] += 1
            continue
        source = resolve_image(raw_path, raw_dir, by_relative, by_basename)
        if source is None:
            counters["missing_image"] += 1
            if counters["missing_image"] <= 10:
                print(f"Missing row {index}: {raw_path}", file=sys.stderr)
            continue
        if args.verify_images:
            try:
                with Image.open(source) as image:
                    image.verify()
            except Exception:
                counters["invalid_image"] += 1
                continue

        patient = patient_key(row, patient_column, raw_path)
        study = study_key(row, patient, study_column, raw_path)
        if args.one_image_per_study and study in seen_studies:
            counters["duplicate_study_view"] += 1
            continue
        seen_studies.add(study)

        supplied_split = official_split(row.get(split_column)) if split_column else None
        if args.split_source == "official":
            if supplied_split is None:
                raise ValueError("Official split requested but missing/unrecognized.")
            split = supplied_split
        elif args.split_source == "auto" and supplied_split:
            split = supplied_split
        else:
            split = hashed_split(
                patient, args.seed, args.train_ratio, args.valid_ratio
            )

        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem)
        image_name = f"chexpert_{int(index):06d}_{safe_stem}{source.suffix.lower()}"
        destination = expose_image(
            source, output_dir / f"{split}_images" / split / image_name, args.link_mode
        )
        try:
            relative = destination.relative_to(output_dir).as_posix()
        except ValueError:
            relative = str(destination)
        records[split].append(
            {
                "image_path": str(destination),
                "relative_path": relative,
                "caption": caption,
                "patient": patient,
                "study": study,
            }
        )
        counters["selected"] += 1

    if counters["missing_image"] and not args.allow_missing_images:
        raise FileNotFoundError(
            f"{counters['missing_image']:,} metadata rows could not be matched. "
            "Re-run with --allow-missing-images only if this is expected."
        )
    if any(not records[split] for split in SPLITS):
        raise RuntimeError(
            "Empty split: " + ", ".join(f"{s}={len(records[s])}" for s in SPLITS)
        )

    write_outputs(output_dir, records, args.absolute_paths)
    report = {
        "source": "Stanford AIMI CheXpert Plus",
        "metadata_csv": str(metadata_csv),
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "caption_mode": args.caption_mode,
        "split_source": args.split_source,
        "seed": args.seed,
        "link_mode": args.link_mode,
        "counts": {split: len(records[split]) for split in SPLITS},
        "unique_patients": {
            split: len({row["patient"] for row in records[split]})
            for split in SPLITS
        },
        "unique_studies": {
            split: len({row["study"] for row in records[split]})
            for split in SPLITS
        },
        "processing_counts": dict(counters),
    }
    with (output_dir / "prepare_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
