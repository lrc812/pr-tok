#!/usr/bin/env python3
"""Resumable downloader for the authorized CheXpert Plus PNG archives."""

from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import redivis


DATASET = "aimi.chexpert_plus:5yyj:v1_0"
PNG_ARCHIVE_TABLE = "png_compressed:wsd7"
MIB = 1024**2
GIB = 1024**3


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download the five CheXpert Plus PNG zip archives."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "chexpert_plus" / "raw",
    )
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--status-seconds", type=int, default=60)
    parser.add_argument("--retry-seconds", type=int, default=15)
    return parser.parse_args()


def human_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    index = 0
    while value >= 1024 and index < len(units) - 1:
        value /= 1024
        index += 1
    return f"{value:.2f} {units[index]}"


def download_one(
    remote_file,
    output_dir: Path,
    chunk_size: int,
    retry_seconds: int,
    print_lock: threading.Lock,
) -> Path:
    destination = output_dir / remote_file.name
    partial = destination.with_name(f"{destination.name}.part")
    expected_size = int(remote_file.size)

    if destination.exists():
        actual_size = destination.stat().st_size
        if actual_size == expected_size:
            with print_lock:
                print(f"[skip] {destination.name} is already complete", flush=True)
            return destination
        raise RuntimeError(
            f"{destination} has size {actual_size}, expected {expected_size}; "
            "move it aside before restarting."
        )

    while True:
        downloaded = partial.stat().st_size if partial.exists() else 0
        if downloaded > expected_size:
            raise RuntimeError(
                f"{partial} is larger than expected "
                f"({downloaded} > {expected_size})."
            )
        if downloaded == expected_size:
            os.replace(partial, destination)
            with print_lock:
                print(f"[done] {destination.name}", flush=True)
            return destination

        try:
            with remote_file.open("rb", start_byte=downloaded) as source:
                with partial.open("ab") as target:
                    while downloaded < expected_size:
                        chunk = source.read(min(chunk_size, expected_size - downloaded))
                        if not chunk:
                            break
                        target.write(chunk)
                        downloaded += len(chunk)
                    target.flush()
                    os.fsync(target.fileno())
        except Exception as error:
            with print_lock:
                print(
                    f"[retry] {destination.name} at {human_bytes(downloaded)}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
            time.sleep(retry_seconds)


def current_downloaded(output_dir: Path, remote_files) -> int:
    total = 0
    for remote_file in remote_files:
        destination = output_dir / remote_file.name
        partial = destination.with_name(f"{destination.name}.part")
        if destination.exists():
            total += min(destination.stat().st_size, int(remote_file.size))
        elif partial.exists():
            total += min(partial.stat().st_size, int(remote_file.size))
    return total


def monitor(
    output_dir: Path,
    remote_files,
    total_size: int,
    status_seconds: int,
    stop_event: threading.Event,
    print_lock: threading.Lock,
) -> None:
    previous_size = current_downloaded(output_dir, remote_files)
    previous_time = time.monotonic()
    while not stop_event.wait(status_seconds):
        now = time.monotonic()
        downloaded = current_downloaded(output_dir, remote_files)
        elapsed = max(now - previous_time, 1e-6)
        speed = max(downloaded - previous_size, 0) / elapsed
        remaining = max(total_size - downloaded, 0)
        eta_hours = remaining / speed / 3600 if speed > 0 else float("inf")
        eta = f"{eta_hours:.1f} h" if speed > 0 else "unknown"
        with print_lock:
            print(
                f"[status] {human_bytes(downloaded)} / {human_bytes(total_size)} "
                f"({100 * downloaded / total_size:.2f}%), "
                f"{human_bytes(speed)}/s, ETA {eta}",
                flush=True,
            )
        previous_size = downloaded
        previous_time = now


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = redivis.dataset(DATASET).get()
    if dataset.properties.get("accessLevel") != "data":
        raise PermissionError(
            "CheXpert Plus accessLevel is not 'data'. Complete the Redivis "
            "data-access agreement before downloading."
        )

    remote_files = sorted(
        dataset.table(PNG_ARCHIVE_TABLE).list_files(), key=lambda item: item.name
    )
    if len(remote_files) != 5:
        raise RuntimeError(f"Expected 5 PNG archives, found {len(remote_files)}")

    total_size = sum(int(remote_file.size) for remote_file in remote_files)
    print(
        f"Downloading {len(remote_files)} archives ({human_bytes(total_size)}) "
        f"to {args.output_dir}",
        flush=True,
    )

    print_lock = threading.Lock()
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor,
        args=(
            args.output_dir,
            remote_files,
            total_size,
            args.status_seconds,
            stop_event,
            print_lock,
        ),
        daemon=True,
    )
    monitor_thread.start()

    try:
        with ThreadPoolExecutor(
            max_workers=min(args.workers, len(remote_files))
        ) as executor:
            futures = [
                executor.submit(
                    download_one,
                    remote_file,
                    args.output_dir,
                    args.chunk_mib * MIB,
                    args.retry_seconds,
                    print_lock,
                )
                for remote_file in remote_files
            ]
            for future in as_completed(futures):
                future.result()
    finally:
        stop_event.set()
        monitor_thread.join()

    downloaded = current_downloaded(args.output_dir, remote_files)
    print(
        f"All PNG archives complete: {human_bytes(downloaded)} / "
        f"{human_bytes(total_size)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
