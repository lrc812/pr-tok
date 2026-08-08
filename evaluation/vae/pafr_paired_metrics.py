#!/usr/bin/env python3
"""Compare a frozen VQGAN reconstruction with its PAFR reconstruction.

The script intentionally focuses on paired, full-reference metrics and global
token statistics.  It complements reconstruction FID, which is distributional
and can be poorly aligned with medical-image fidelity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchmetrics.functional.image import structural_similarity_index_measure
from tqdm import tqdm

from pafr_vq.imagenet import ImageNetListDataset, build_imagenet_pafr
from pafr_vq.rocov2 import RocoImageListDataset, build_rocov2_pafr, load_residual_state


def _select_dataset(dataset, maximum: int, seed: int):
    if not maximum or maximum >= len(dataset):
        return dataset, list(range(len(dataset)))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:maximum].sort().values.tolist()
    return Subset(dataset, indices), indices


def _gradient_l1(target: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    vertical = (target[..., 1:, :] - target[..., :-1, :]).sub(
        estimate[..., 1:, :] - estimate[..., :-1, :]
    ).abs().mean((1, 2, 3))
    horizontal = (target[..., :, 1:] - target[..., :, :-1]).sub(
        estimate[..., :, 1:] - estimate[..., :, :-1]
    ).abs().mean((1, 2, 3))
    return vertical + horizontal


def _image_metrics(target: torch.Tensor, estimate: torch.Tensor) -> dict[str, torch.Tensor]:
    difference = target - estimate
    mse = difference.square().mean((1, 2, 3))
    return {
        "l1_rgb_0_1": difference.abs().mean((1, 2, 3)),
        "mse_rgb_0_1": mse,
        "psnr_db_rgb_0_1": -10.0 * torch.log10(mse.clamp_min(1e-12)),
        "ssim_rgb_0_1": structural_similarity_index_measure(
            estimate, target, data_range=1.0, reduction="none"
        ),
        "gradient_l1_rgb_0_1": _gradient_l1(target, estimate),
    }


def _summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _codebook_summary(counts: torch.Tensor) -> dict[str, float | int]:
    counts = counts.double()
    total = counts.sum()
    probabilities = counts / total.clamp_min(1)
    entropy_nats = -(probabilities * probabilities.clamp_min(1e-300).log()).sum()
    nonzero = int((counts > 0).sum().item())
    return {
        "token_count": int(total.item()),
        "codebook_size": int(counts.numel()),
        "used_codes": nonzero,
        "dead_codes": int(counts.numel() - nonzero),
        "utilization": float(nonzero / counts.numel()),
        "perplexity": float(entropy_nats.exp().item()),
        "entropy_bits_per_token": float((entropy_nats / math.log(2.0)).item()),
    }


def _rate_summary(
    base_counts: torch.Tensor,
    residual_counts: torch.Tensor,
    positions: int,
    active_positions: int,
    pixels: int,
) -> dict[str, float | int]:
    base_size = base_counts.numel()
    residual_size = residual_counts.numel()
    mask_bits = (
        math.lgamma(positions + 1)
        - math.lgamma(active_positions + 1)
        - math.lgamma(positions - active_positions + 1)
    ) / math.log(2.0)
    fixed_base = positions * math.log2(base_size)
    fixed_residual = active_positions * math.log2(residual_size)
    base_stats = _codebook_summary(base_counts)
    residual_stats = _codebook_summary(residual_counts)
    entropy_base = positions * float(base_stats["entropy_bits_per_token"])
    entropy_residual = active_positions * float(residual_stats["entropy_bits_per_token"])
    return {
        "latent_positions": positions,
        "active_residual_positions": active_positions,
        "active_ratio": active_positions / positions,
        "mask_combinatorial_bits_per_image": mask_bits,
        "fixed_length_base_bits_per_image": fixed_base,
        "fixed_length_residual_bits_per_image": fixed_residual,
        "fixed_length_total_bits_per_image": fixed_base + fixed_residual + mask_bits,
        "fixed_length_total_bits_per_pixel": (fixed_base + fixed_residual + mask_bits) / pixels,
        "empirical_entropy_total_bits_per_image": entropy_base + entropy_residual + mask_bits,
        "empirical_entropy_total_bits_per_pixel": (entropy_base + entropy_residual + mask_bits) / pixels,
        "note": "Entropy rate is an oracle marginal estimate; it excludes entropy-model overhead and spatial dependence.",
    }


def evaluate(args: argparse.Namespace) -> Path:
    for value in (args.list_file, args.base_config, args.base_checkpoint, args.pafr_checkpoint):
        if not Path(value).is_file():
            raise FileNotFoundError(value)
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("invalid batch size or worker count")

    started = datetime.now().astimezone()
    start_time = time.perf_counter()
    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]

    if args.dataset == "imagenet":
        full_dataset = ImageNetListDataset(args.list_file, args.image_size, train=False)
        model = build_imagenet_pafr(
            args.base_config, args.base_checkpoint, device, args.scorer,
            args.residual_dim, args.residual_codebook_size, args.commitment_weight,
        )
    else:
        full_dataset = RocoImageListDataset(args.list_file, args.image_size)
        model = build_rocov2_pafr(
            args.base_config, args.base_checkpoint, device, args.scorer,
            args.residual_dim, args.residual_codebook_size, args.commitment_weight,
        )
    checkpoint = load_residual_state(model, args.pafr_checkpoint)
    model.eval().requires_grad_(False)

    dataset, selected_indices = _select_dataset(full_dataset, args.max_images, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    metrics = {
        branch: {name: [] for name in (
            "l1_rgb_0_1", "mse_rgb_0_1", "psnr_db_rgb_0_1",
            "ssim_rgb_0_1", "gradient_l1_rgb_0_1",
        )}
        for branch in ("base", "pafr")
    }
    improvement_counts = {"mse": 0, "psnr": 0, "ssim": 0, "gradient_l1": 0}
    base_counts = torch.zeros(model.base.codebook_size, dtype=torch.int64)
    residual_counts = torch.zeros(model.quantizer.codebook_size, dtype=torch.int64)
    active_total = 0
    position_total = 0
    latent_positions = None
    active_positions = None

    print(f"{args.dataset}: evaluating {len(dataset):,}/{len(full_dataset):,} images on {device}", flush=True)
    with torch.inference_mode():
        for images in tqdm(loader, desc=f"{args.dataset} paired metrics", unit="batch"):
            images = images.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=device.type == "cuda" and args.precision != "fp32",
            ):
                output = model(images, active_ratio=args.active_ratio)
            target = images.float().add(1).mul(0.5).clamp(0, 1)
            base = output["base_reconstruction"].float().add(1).mul(0.5).clamp(0, 1)
            pafr = output["reconstruction"].float().add(1).mul(0.5).clamp(0, 1)
            batch_metrics = {
                "base": _image_metrics(target, base),
                "pafr": _image_metrics(target, pafr),
            }
            for branch, values in batch_metrics.items():
                for name, value in values.items():
                    metrics[branch][name].append(value.detach().cpu().numpy())
            improvement_counts["mse"] += int((batch_metrics["pafr"]["mse_rgb_0_1"] < batch_metrics["base"]["mse_rgb_0_1"]).sum())
            improvement_counts["psnr"] += int((batch_metrics["pafr"]["psnr_db_rgb_0_1"] > batch_metrics["base"]["psnr_db_rgb_0_1"]).sum())
            improvement_counts["ssim"] += int((batch_metrics["pafr"]["ssim_rgb_0_1"] > batch_metrics["base"]["ssim_rgb_0_1"]).sum())
            improvement_counts["gradient_l1"] += int((batch_metrics["pafr"]["gradient_l1_rgb_0_1"] < batch_metrics["base"]["gradient_l1_rgb_0_1"]).sum())

            base_ids = output["base_token_ids"].detach().reshape(-1).cpu()
            residual_ids = output["residual_token_ids"].detach().reshape(-1).cpu()
            active = residual_ids != model.quantizer.null_code_id
            base_counts += torch.bincount(base_ids, minlength=base_counts.numel())
            residual_counts += torch.bincount(residual_ids[active], minlength=residual_counts.numel())
            active_total += int(active.sum())
            position_total += int(residual_ids.numel())
            if latent_positions is None:
                latent_positions = int(output["residual_mask"][0].numel())
                active_positions = int(output["residual_mask"][0].sum())

    arrays = {
        branch: {name: np.concatenate(chunks).astype(np.float64) for name, chunks in values.items()}
        for branch, values in metrics.items()
    }
    paired = {branch: {name: _summary(values) for name, values in branch_values.items()} for branch, branch_values in arrays.items()}
    paired["delta_pafr_minus_base"] = {
        name: _summary(arrays["pafr"][name] - arrays["base"][name])
        for name in arrays["base"]
    }
    paired["fraction_images_improved"] = {
        name: count / len(dataset) for name, count in improvement_counts.items()
    }

    selected_paths = [str(full_dataset.paths[index]) for index in selected_indices]
    report = {
        "dataset": {
            "name": args.dataset,
            "list_file": str(Path(args.list_file).resolve()),
            "available_images": len(full_dataset),
            "evaluated_images": len(dataset),
            "image_size": args.image_size,
            "preprocessing": "project center_crop_arr + RGB conversion + [-1,1] normalization",
            "selected_paths_sha256": hashlib.sha256("\n".join(selected_paths).encode()).hexdigest(),
        },
        "model": {
            "base_config": str(Path(args.base_config).resolve()),
            "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
            "base_codebook_size": model.base.codebook_size,
            "pafr_checkpoint": str(Path(args.pafr_checkpoint).resolve()),
            "pafr_epoch": checkpoint.get("epoch"),
            "pafr_step": checkpoint.get("step"),
            "active_ratio": args.active_ratio,
            "scorer": args.scorer,
            "residual_dim": args.residual_dim,
            "residual_codebook_size": args.residual_codebook_size,
        },
        "paired_metrics": paired,
        "codebooks": {
            "base": _codebook_summary(base_counts),
            "residual": _codebook_summary(residual_counts),
            "observed_active_ratio": active_total / position_total,
        },
        "rate": _rate_summary(
            base_counts, residual_counts, int(latent_positions), int(active_positions), args.image_size ** 2
        ),
        "runtime": {
            "started_at": started.isoformat(),
            "completed_at": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": time.perf_counter() - start_time,
            "device": str(device),
            "precision": args.precision,
            "batch_size": args.batch_size,
        },
        "metric_notes": {
            "psnr_ssim": "Full-reference RGB metrics on [0,1]; higher is better.",
            "gradient_l1": "Paired finite-difference error on [0,1]; lower is better and emphasizes edge fidelity.",
            "rfid": "Not computed here; use the dataset-specific reconstruction-FID script for distributional fidelity.",
        },
        "arguments": vars(args),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"base: PSNR={paired['base']['psnr_db_rgb_0_1']['mean']:.4f}, SSIM={paired['base']['ssim_rgb_0_1']['mean']:.6f}", flush=True)
    print(f"pafr: PSNR={paired['pafr']['psnr_db_rgb_0_1']['mean']:.4f}, SSIM={paired['pafr']['ssim_rgb_0_1']['mean']:.6f}", flush=True)
    print(f"report: {output}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("rocov2", "imagenet"), required=True)
    parser.add_argument("--list-file", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--pafr-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--scorer", choices=("random", "pixel", "sobel", "haar_dwt", "hybrid"), default="hybrid")
    parser.add_argument("--active-ratio", type=float, default=0.25)
    parser.add_argument("--residual-dim", type=int, default=64)
    parser.add_argument("--residual-codebook-size", type=int, default=1024)
    parser.add_argument("--commitment-weight", type=float, default=0.25)
    parser.add_argument("--allow-tf32", action="store_true")
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
