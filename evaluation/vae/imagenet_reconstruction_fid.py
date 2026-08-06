#!/usr/bin/env python3
"""Evaluate reconstruction FID for a standalone VQGAN on ImageNet val."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from evaluation.t2i.eval import calculate_fid_components
from evaluation.vae.standalone_vqgan import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_MODEL_DIR,
    ImageNetCenterCropDataset,
    autocast_context,
    collect_images,
    load_standalone_vqgan,
    resolve_device,
    select_images,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "vae" / "fid_results"


def inception_features(model, images_01):
    prediction = model(images_01)[0]
    if prediction.shape[2:] != (1, 1):
        prediction = adaptive_avg_pool2d(prediction, output_size=(1, 1))
    return prediction.flatten(1)


def summarize(values):
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def file_list_digest(files, root):
    value = "\n".join(str(path.relative_to(root)) for path in files)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_args(args):
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0 or args.max_images < 0 or args.preview_images < 0:
        raise ValueError("worker/image counts cannot be negative")
    if args.analysis_top_k < 0:
        raise ValueError("--analysis-top-k cannot be negative")
    if args.dims not in InceptionV3.BLOCK_INDEX_BY_DIM:
        choices = sorted(InceptionV3.BLOCK_INDEX_BY_DIM)
        raise ValueError(f"--dims must be one of {choices}, got {args.dims}")


def evaluate(args):
    validate_args(args)
    started_at = datetime.now().astimezone()
    start_time = time.perf_counter()
    image_dir = Path(args.image_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"ImageNet validation directory not found: {image_dir}")
    for path in (config_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    all_files = collect_images(image_dir)
    files = select_images(all_files, args.max_images, args.seed)
    if len(files) < 2:
        raise ValueError(f"FID requires at least two images, found {len(files)}")

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.benchmark = args.cudnn_benchmark

    print(f"Images:     {len(files):,} / {len(all_files):,}")
    print(f"Image dir:  {image_dir}")
    print(f"Config:     {config_path}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Device:     {device} ({args.precision})")

    vqgan, config, checkpoint_metadata = load_standalone_vqgan(
        config_path, checkpoint, device
    )
    image_size = args.image_size or int(config.model.params.ddconfig.resolution)
    if image_size % 16:
        raise ValueError("VQGAN f16 input size must be divisible by 16")
    dataset = ImageNetCenterCropDataset(files, image_size)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    block_index = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
    inception = InceptionV3([block_index]).eval().requires_grad_(False).to(device)
    real_activations = np.empty((len(dataset), args.dims), dtype=np.float64)
    reconstructed_activations = np.empty_like(real_activations)
    per_image_mse = np.empty(len(dataset), dtype=np.float64)
    per_image_psnr = np.empty(len(dataset), dtype=np.float64)
    preview_count = min(args.preview_images, len(dataset))
    preview_pairs = []

    offset = 0
    with torch.inference_mode():
        for images in tqdm(loader, desc="Reconstruct + Inception", unit="batch"):
            images = images.to(device, non_blocking=device.type == "cuda")
            with autocast_context(device, args.precision):
                reconstructed = vqgan(images)[0]
            reconstructed = reconstructed.float().clamp(-1.0, 1.0)
            real_01 = images.float().add(1.0).mul(0.5).clamp(0.0, 1.0)
            reconstructed_01 = reconstructed.add(1.0).mul(0.5)

            mse = (real_01 - reconstructed_01).square().mean(dim=(1, 2, 3))
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
            combined_features = inception_features(
                inception, torch.cat((real_01, reconstructed_01), dim=0)
            )
            real_features, reconstructed_features = combined_features.chunk(2)

            end = offset + images.shape[0]
            real_activations[offset:end] = real_features.cpu().numpy()
            reconstructed_activations[offset:end] = reconstructed_features.cpu().numpy()
            per_image_mse[offset:end] = mse.cpu().numpy()
            per_image_psnr[offset:end] = psnr.cpu().numpy()
            if len(preview_pairs) < 2 * preview_count:
                needed = preview_count - len(preview_pairs) // 2
                for original, reconstruction in zip(
                    real_01[:needed].cpu(), reconstructed_01[:needed].cpu()
                ):
                    preview_pairs.extend((original, reconstruction))
            offset = end

    if offset != len(dataset):
        raise RuntimeError(f"Expected {len(dataset)} samples, processed {offset}")

    print("Computing means, covariances, and FID matrix square root...")
    fid_analysis = calculate_fid_components(
        real_activations.mean(axis=0),
        np.cov(real_activations, rowvar=False),
        reconstructed_activations.mean(axis=0),
        np.cov(reconstructed_activations, rowvar=False),
        top_k=args.analysis_top_k,
    )

    elapsed = time.perf_counter() - start_time
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"imagenet_reconstruction_fid_{timestamp}.json"
    preview_path = None
    if preview_pairs:
        preview_path = output_dir / f"imagenet_reconstruction_preview_{timestamp}.png"
        save_image(preview_pairs, preview_path, nrow=2, padding=2)

    report = {
        "metric": {
            "name": "reconstruction_fid",
            "fid": fid_analysis["fid"],
            "feature_dimensions": args.dims,
            "standard_comparable_fid": args.dims == 2048,
            "analysis": fid_analysis,
            "real_distribution": "center-cropped ImageNet tokenizer inputs",
            "reconstructed_distribution": "quantized VQGAN reconstructions",
            "inception_implementation": "pytorch_fid.inception.InceptionV3",
        },
        "paired_metrics": {
            "mse_rgb_0_1": summarize(per_image_mse),
            "psnr_db_rgb_0_1": summarize(per_image_psnr),
        },
        "dataset": {
            "image_dir": str(image_dir),
            "discovered_image_count": len(all_files),
            "evaluated_image_count": len(files),
            "image_size": image_size,
            "preprocessing": "SmallestMaxSize + CenterCrop (albumentations)",
            "seed": args.seed,
            "max_images": args.max_images,
            "selected_file_list_sha256": file_list_digest(files, image_dir),
            "first_selected_files": [str(path.relative_to(image_dir)) for path in files[:10]],
        },
        "model": {
            "config": str(config_path),
            "checkpoint": str(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "target": str(config.model.target),
            "n_embed": int(config.model.params.n_embed),
            "embed_dim": int(config.model.params.embed_dim),
            **checkpoint_metadata,
        },
        "runtime": {
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": elapsed,
            "images_per_second": len(files) / elapsed,
            "device": str(device),
            "precision": args.precision,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
        "outputs": {
            "report": str(report_path),
            "preview": str(preview_path) if preview_path else None,
        },
        "arguments": vars(args),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Reconstruction FID ({args.dims} dims): {fid_analysis['fid']:.6f}")
    print(
        f"  components: mean={fid_analysis['mean_term']:.6f}, "
        f"covariance={fid_analysis['covariance_term']:.6f}"
    )
    print(f"  PSNR: {per_image_psnr.mean():.4f} dB")
    print(f"  Time: {elapsed:.1f}s ({len(files) / elapsed:.2f} img/s)")
    if preview_path:
        print(f"  Preview: {preview_path}")
    print(f"  Report:  {report_path}")
    return report_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate VQGAN reconstruction FID on ImageNet validation."
    )
    parser.add_argument("--config", default=str(DEFAULT_MODEL_DIR / "config.yaml"))
    parser.add_argument("--checkpoint", default=str(DEFAULT_MODEL_DIR / "model.ckpt"))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="Random subset size; 0 evaluates all 50,000 validation images.",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--dims", type=int, default=2048,
        help="Keep 2048 for a standard comparable FID.",
    )
    parser.add_argument("--analysis-top-k", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--preview-images", type=int, default=16)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
