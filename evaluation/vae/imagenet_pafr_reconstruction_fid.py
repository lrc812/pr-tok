#!/usr/bin/env python3
"""Evaluate PAFR reconstruction FID on ImageNet validation images."""
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
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

from evaluation.t2i.eval import calculate_fid_components
from pafr_vq.imagenet import ImageNetListDataset, build_imagenet_pafr, load_residual_state


def inception_features(model, images):
    prediction = model(images)[0]
    if prediction.shape[2:] != (1, 1): prediction = adaptive_avg_pool2d(prediction, (1, 1))
    return prediction.flatten(1)


def summarize(values):
    return {"count": int(values.size), "mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()), "max": float(values.max())}


def evaluate(args):
    if args.dims not in InceptionV3.BLOCK_INDEX_BY_DIM: raise ValueError(f"Invalid Inception dimensions: {args.dims}")
    for value in (args.val_list, args.base_config, args.base_checkpoint, args.pafr_checkpoint):
        if not Path(value).is_file(): raise FileNotFoundError(value)
    started = datetime.now().astimezone(); start_time = time.perf_counter()
    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda": torch.cuda.set_device(device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    full_dataset = ImageNetListDataset(args.val_list, args.image_size, train=False)
    if args.max_images and args.max_images < len(full_dataset):
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(full_dataset), generator=generator)[:args.max_images].sort().values.tolist()
        dataset = Subset(full_dataset, indices)
    else:
        indices = list(range(len(full_dataset))); dataset = full_dataset
    if len(dataset) < 2: raise ValueError("FID requires at least two images")
    loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    print(f"ImageNet validation images: {len(dataset):,}/{len(full_dataset):,}", flush=True)
    model = build_imagenet_pafr(args.base_config, args.base_checkpoint, device, args.scorer, args.residual_dim, args.residual_codebook_size, args.commitment_weight)
    if model.base.codebook_size != 16384: raise ValueError(f"Expected original VQGAN codebook 16384, found {model.base.codebook_size}")
    checkpoint = load_residual_state(model, args.pafr_checkpoint); model.eval().requires_grad_(False)
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]]).to(device).eval().requires_grad_(False)
    real_activations = np.empty((len(dataset), args.dims), dtype=np.float64); recon_activations = np.empty_like(real_activations)
    mse_values = np.empty(len(dataset), dtype=np.float64); psnr_values = np.empty(len(dataset), dtype=np.float64)
    preview = []; offset = 0
    with torch.inference_mode():
        for images in tqdm(loader, desc="ImageNet PAFR reconstruct + Inception", unit="batch"):
            images = images.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and args.precision != "fp32"):
                reconstruction = model(images, active_ratio=args.active_ratio)["reconstruction"]
            real = images.float().add(1).mul(.5).clamp(0, 1); restored = reconstruction.float().add(1).mul(.5).clamp(0, 1)
            mse = (real - restored).square().mean((1, 2, 3)); psnr = -10 * torch.log10(mse.clamp_min(1e-12))
            real_features, recon_features = inception_features(inception, torch.cat((real, restored))).chunk(2)
            end = offset + images.shape[0]
            real_activations[offset:end] = real_features.cpu().numpy(); recon_activations[offset:end] = recon_features.cpu().numpy(); mse_values[offset:end] = mse.cpu().numpy(); psnr_values[offset:end] = psnr.cpu().numpy()
            needed = max(args.preview_images - len(preview) // 2, 0)
            for original, reconstruction_image in zip(real[:needed].cpu(), restored[:needed].cpu()): preview.extend((original, reconstruction_image))
            offset = end
    fid = calculate_fid_components(real_activations.mean(0), np.cov(real_activations, rowvar=False), recon_activations.mean(0), np.cov(recon_activations, rowvar=False), top_k=args.analysis_top_k)
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True); stamp = started.strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"imagenet_pafr_reconstruction_fid_{stamp}.json"; preview_path = output_dir / f"imagenet_pafr_reconstruction_preview_{stamp}.png"
    if preview: save_image(preview, preview_path, nrow=2, padding=2)
    selected_paths = [str(full_dataset.paths[index]) for index in indices]
    report = {
        "metric": {"name": "reconstruction_fid", "fid": fid["fid"], "feature_dimensions": args.dims, "standard_comparable_fid": args.dims == 2048, "analysis": fid, "real_distribution": "center-cropped ImageNet validation images", "reconstructed_distribution": "original-VQGAN + PAFR reconstructions"},
        "paired_metrics": {"mse_rgb_0_1": summarize(mse_values), "psnr_db_rgb_0_1": summarize(psnr_values)},
        "dataset": {"list_file": str(Path(args.val_list).resolve()), "available_images": len(full_dataset), "evaluated_images": len(dataset), "image_size": args.image_size, "selected_paths_sha256": hashlib.sha256("\n".join(selected_paths).encode()).hexdigest()},
        "model": {"base_family": "original_compvis_vqgan_f16_16384", "base_config": str(Path(args.base_config).resolve()), "base_checkpoint": str(Path(args.base_checkpoint).resolve()), "base_codebook_size": model.base.codebook_size, "pafr_checkpoint": str(Path(args.pafr_checkpoint).resolve()), "pafr_epoch": checkpoint.get("epoch"), "pafr_step": checkpoint.get("step"), "active_ratio": args.active_ratio, "scorer": args.scorer, "residual_dim": args.residual_dim, "residual_codebook_size": args.residual_codebook_size},
        "runtime": {"started_at": started.isoformat(), "completed_at": datetime.now().astimezone().isoformat(), "elapsed_seconds": time.perf_counter() - start_time, "device": str(device), "precision": args.precision, "batch_size": args.batch_size},
        "outputs": {"report": str(report_path), "preview": str(preview_path) if preview else None}, "arguments": vars(args),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ImageNet PAFR rFID ({args.dims}d): {fid['fid']:.6f}", flush=True); print(f"PSNR: {psnr_values.mean():.4f} dB", flush=True); print(f"Report: {report_path}", flush=True)
    return report_path


def build_parser():
    parser = argparse.ArgumentParser(); parser.add_argument("--val-list", required=True); parser.add_argument("--base-config", required=True); parser.add_argument("--base-checkpoint", required=True); parser.add_argument("--pafr-checkpoint", required=True); parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "evaluation" / "vae" / "fid_results")); parser.add_argument("--device", default="auto"); parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16"); parser.add_argument("--image-size", type=int, default=256); parser.add_argument("--batch-size", type=int, default=12); parser.add_argument("--num-workers", type=int, default=8); parser.add_argument("--max-images", type=int, default=0); parser.add_argument("--seed", type=int, default=23); parser.add_argument("--dims", type=int, default=2048); parser.add_argument("--analysis-top-k", type=int, default=10); parser.add_argument("--preview-images", type=int, default=16); parser.add_argument("--scorer", choices=("random", "pixel", "sobel", "haar_dwt", "hybrid"), default="hybrid"); parser.add_argument("--active-ratio", type=float, default=.25); parser.add_argument("--residual-dim", type=int, default=64); parser.add_argument("--residual-codebook-size", type=int, default=1024); parser.add_argument("--commitment-weight", type=float, default=.25)
    return parser


if __name__ == "__main__": evaluate(build_parser().parse_args())
