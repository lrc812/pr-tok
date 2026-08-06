#!/usr/bin/env python3
"""Measure VQGAN codebook utilization on ImageNet validation and save simple distribution plots."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf.nodes import AnyNode
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from main import instantiate_from_config


DEFAULT_MODEL_DIR = PROJECT_ROOT / "pretrained_models" / "vqgan_f16_16384"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "imagenet" / "val"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_device(value: str):
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def autocast_context(device, precision: str):
    if precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 inference requires CUDA")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def safe_load_checkpoint(checkpoint: Path):
    allowed_globals = {
        "omegaconf.dictconfig.DictConfig": DictConfig,
        "omegaconf.nodes.AnyNode": AnyNode,
        "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint": __import__(
            "pytorch_lightning.callbacks.model_checkpoint", fromlist=["ModelCheckpoint"]
        ).ModelCheckpoint,
        "builtins.list": list,
        "omegaconf.base.ContainerMetadata": ContainerMetadata,
        "omegaconf.listconfig.ListConfig": ListConfig,
        "collections.defaultdict": defaultdict,
        "builtins.int": int,
        "builtins.dict": dict,
        "omegaconf.base.Metadata": Metadata,
    }
    referenced = set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint))
    unsupported = sorted(referenced.difference(allowed_globals))
    if unsupported:
        raise RuntimeError(
            "Checkpoint references globals outside the safe allowlist: "
            + ", ".join(unsupported)
        )
    with torch.serialization.safe_globals([allowed_globals[name] for name in sorted(referenced)]):
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(data).__name__}")
    return data


def find_logdir_project_yaml(log_dir: Path) -> Path:
    cfg_dir = log_dir / "configs"
    if not cfg_dir.is_dir():
        raise FileNotFoundError(f"configs directory not found: {cfg_dir}")
    ymls = sorted(cfg_dir.glob("*project.yaml"))
    if not ymls:
        ymls = sorted(cfg_dir.glob("*project*.yaml"))
    if not ymls:
        raise FileNotFoundError(f"No project yaml found in {cfg_dir}")
    return ymls[-1]


def find_logdir_checkpoint(log_dir: Path) -> Path:
    ckpt_dir = log_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoints directory not found: {ckpt_dir}")
    last = ckpt_dir / "last.ckpt"
    if last.is_file():
        return last
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No ckpt file found in {ckpt_dir}")
    return ckpts[-1]


def load_model_from_model_dir(model_dir: Path, device):
    config_path = model_dir / "config.yaml"
    checkpoint_path = model_dir / "model.ckpt"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    cfg = OmegaConf.load(config_path)
    model_config = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_config.params.lossconfig = {"target": "torch.nn.Identity"}
    model_config.params.pop("ckpt_path", None)
    model = instantiate_from_config(model_config)
    checkpoint_data = safe_load_checkpoint(checkpoint_path)
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing_core = [key for key in missing if not key.startswith("loss.")]
    unexpected_core = [key for key in unexpected if not key.startswith("loss.")]
    if missing_core or unexpected_core:
        raise RuntimeError(
            "Model/checkpoint mismatch outside omitted training loss: "
            f"missing={missing_core[:20]}, unexpected={unexpected_core[:20]}"
        )
    model.eval().requires_grad_(False).to(device)
    metadata = {
        "epoch": checkpoint_data.get("epoch"),
        "global_step": checkpoint_data.get("global_step"),
        "missing_keys": list(missing),
        "unexpected_training_loss_keys": list(unexpected),
    }
    return model, cfg, metadata, config_path, checkpoint_path


def load_model_from_logdir(log_dir: Path, device):
    config_path = find_logdir_project_yaml(log_dir)
    checkpoint_path = find_logdir_checkpoint(log_dir)
    cfg = OmegaConf.load(config_path)
    model = instantiate_from_config(cfg.model)
    checkpoint_data = safe_load_checkpoint(checkpoint_path)
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval().requires_grad_(False).to(device)
    metadata = {
        "epoch": checkpoint_data.get("epoch"),
        "global_step": checkpoint_data.get("global_step"),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    return model, cfg, metadata, config_path, checkpoint_path


def collect_images(image_dir: Path):
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def select_images(files: list[Path], count: int, seed: int):
    if count <= 0 or count >= len(files):
        return files
    indices = sorted(np.random.default_rng(seed).choice(len(files), size=count, replace=False))
    return [files[index] for index in indices]


class CenterCropImageDataset(Dataset):
    def __init__(self, files: list[Path], image_size: int):
        self.files = files
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                width, height = image.size
                scale = self.image_size / min(width, height)
                resized = (
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                )
                image = image.resize(resized, resample=Image.BICUBIC)
                left = max(0, (image.size[0] - self.image_size) // 2)
                top = max(0, (image.size[1] - self.image_size) // 2)
                image = image.crop((left, top, left + self.image_size, top + self.image_size))
                array = np.asarray(image, dtype=np.uint8)
        except Exception as error:
            raise RuntimeError(f"Failed to preprocess image: {path}") from error
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
        tensor = tensor.float().div_(127.5).sub_(1.0)
        return tensor


def entropy_from_counts(counts: np.ndarray, base: float = 2.0) -> tuple[float, float, np.ndarray]:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("counts sum to zero")

    probs = counts / total
    nonzero = probs > 0
    probs_nz = probs[nonzero]
    if base is None:
        entropy = -float(np.sum(probs_nz * np.log(probs_nz)))
        perplexity = float(np.exp(entropy))
    else:
        entropy = -float(np.sum(probs_nz * (np.log(probs_nz) / np.log(base))))
        perplexity = float(base**entropy)
    return entropy, perplexity, probs


def summarize(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _load_font(size: int = 22):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _save_image(img: Image.Image, path: Path) -> Path:
    img.save(path)
    return path


def save_heatmap(counts: np.ndarray, output_dir: Path, title: str) -> Path:
    side = int(math.isqrt(len(counts)))
    if side * side != len(counts):
        raise ValueError(f"codebook size {len(counts)} is not a perfect square")

    grid = np.asarray(counts, dtype=np.float32).reshape(side, side)
    grid = np.log1p(grid)
    grid = grid / max(float(grid.max()), 1e-12)
    gray = Image.fromarray(np.uint8(np.clip(grid * 255.0, 0, 255)), mode="L")
    colored = ImageOps.colorize(gray, black="#440154", white="#fde725").convert("RGB")

    scale = max(1, 1024 // side)
    heatmap = colored.resize((side * scale, side * scale), resample=Image.NEAREST)
    canvas = Image.new("RGB", (heatmap.width + 80, heatmap.height + 120), "white")
    canvas.paste(heatmap, (40, 60))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(24)
    small = _load_font(16)
    draw.text((40, 20), title, font=font, fill="black")
    draw.text((40, 40 + heatmap.height), "Dark = low usage, bright = high usage", font=small, fill="#444444")
    return _save_image(canvas, output_dir / "codebook_heatmap.png")


def save_histogram(counts: np.ndarray, output_dir: Path, title: str) -> Path:
    counts = np.asarray(counts, dtype=np.float64)
    hist, edges = np.histogram(counts, bins=80)

    width, height = 1400, 700
    left, top, right, bottom = 90, 70, 50, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _load_font(24)
    small = _load_font(16)
    draw.text((left, 20), title, font=font, fill="black")

    x0, y0 = left, top + plot_h
    draw.line((x0, top, x0, y0), fill="black", width=2)
    draw.line((x0, y0, x0 + plot_w, y0), fill="black", width=2)

    max_hist = int(hist.max()) if hist.size else 1
    max_hist = max(max_hist, 1)
    bar_w = plot_w / len(hist)
    for idx, value in enumerate(hist):
        x1 = x0 + idx * bar_w
        x2 = x0 + (idx + 1) * bar_w
        h = (value / max_hist) * (plot_h - 10)
        y1 = y0 - h
        draw.rectangle((x1, y1, x2, y0), fill="#2c7fb8", outline="#2c7fb8")

    for frac, label in [(0.0, "0"), (0.5, f"{float(edges[len(edges)//2]):.0f}"), (1.0, f"{float(edges[-1]):.0f}")]:
        x = x0 + frac * plot_w
        draw.line((x, y0, x, y0 + 6), fill="black", width=1)
        bbox = draw.textbbox((0, 0), label, font=small)
        draw.text((x - (bbox[2] - bbox[0]) / 2, y0 + 10), label, font=small, fill="black")

    draw.text((left, height - 50), "Usage count per code", font=small, fill="black")
    return _save_image(canvas, output_dir / "codebook_histogram.png")


def save_sorted_bar(counts: np.ndarray, output_dir: Path, title: str) -> Path:
    sorted_counts = np.sort(np.asarray(counts, dtype=np.float64))[::-1]
    if sorted_counts.size == 0:
        raise ValueError("empty counts")

    width, height = 1600, 700
    left, top, right, bottom = 90, 70, 50, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _load_font(24)
    small = _load_font(16)
    draw.text((left, 20), title, font=font, fill="black")

    x0, y0 = left, top + plot_h
    draw.line((x0, top, x0, y0), fill="black", width=2)
    draw.line((x0, y0, x0 + plot_w, y0), fill="black", width=2)

    max_count = float(sorted_counts.max())
    min_count = float(sorted_counts.min())
    pts = []
    for idx, value in enumerate(sorted_counts):
        x = x0 + (idx / max(1, sorted_counts.size - 1)) * plot_w
        y = y0 - (value / max_count) * (plot_h - 10)
        pts.append((x, y))
    if len(pts) > 1:
        draw.line(pts, fill="#08519c", width=2)
    else:
        draw.ellipse((pts[0][0] - 2, pts[0][1] - 2, pts[0][0] + 2, pts[0][1] + 2), fill="#08519c")

    for frac, label in [(0.0, "1"), (0.5, f"{sorted_counts.size // 2}"), (1.0, f"{sorted_counts.size}")]:
        x = x0 + frac * plot_w
        draw.line((x, y0, x, y0 + 6), fill="black", width=1)
        bbox = draw.textbbox((0, 0), label, font=small)
        draw.text((x - (bbox[2] - bbox[0]) / 2, y0 + 10), label, font=small, fill="black")

    draw.text((left, height - 50), f"Max={max_count:.0f}, Min={min_count:.0f}", font=small, fill="black")
    return _save_image(canvas, output_dir / "codebook_sorted_bar.png")


@torch.no_grad()
def compute_usage_counts(model, dataloader, device, precision: str) -> np.ndarray:
    n_embed = int(model.codebook_size) if hasattr(model, "codebook_size") else int(model.quantize.n_e)
    counts = torch.zeros(n_embed, dtype=torch.long, device="cpu")

    for batch in tqdm(dataloader, desc="Counting code usage", unit="batch"):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device, non_blocking=device.type == "cuda")
        with autocast_context(device, precision):
            _, _, info = model.encode(images)
        indices = info[-1].reshape(-1).detach().to("cpu")
        counts += torch.bincount(indices, minlength=n_embed)

    return counts.numpy()


def resolve_image_dir(image_dir: str | None) -> Path:
    candidates = []
    if image_dir is not None:
        candidates.append(Path(image_dir).expanduser().resolve())
    candidates.extend(
        [
            DEFAULT_IMAGE_DIR.resolve(),
            (PROJECT_ROOT / "data" / "imagenet" / "val").resolve(),
            Path("/root/.cache/autoencoders/data/ILSVRC2012_validation/data").resolve(),
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not find an ImageNet validation directory. Pass --image-dir explicitly.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None, help="Directory containing config.yaml and model.ckpt")
    parser.add_argument("--logdir", type=str, default=None, help="Training log directory containing configs/ and checkpoints/")
    parser.add_argument("--image-dir", type=str, default=None, help="ImageNet validation root")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    if bool(args.model_dir) == bool(args.logdir):
        raise ValueError("Pass exactly one of --model-dir or --logdir")

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    image_dir = resolve_image_dir(args.image_dir)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

    if args.model_dir:
        model_dir = Path(args.model_dir).expanduser().resolve()
        model, cfg, metadata, config_path, checkpoint_path = load_model_from_model_dir(model_dir, device)
        default_output = model_dir / "codebook_usage_imagenet"
    else:
        logdir = Path(args.logdir).expanduser().resolve()
        model, cfg, metadata, config_path, checkpoint_path = load_model_from_logdir(logdir, device)
        default_output = logdir / "codebook_usage_imagenet"

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config:     {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image dir:  {image_dir}")
    print(f"Device:     {device} ({args.precision})")

    files = collect_images(image_dir)
    files = select_images(files, args.max_images, args.seed)
    if len(files) < 2:
        raise ValueError(f"Need at least two images for a meaningful distribution, got {len(files)}")

    image_size = int(cfg.model.params.ddconfig.resolution)
    dataset = CenterCropImageDataset(files, image_size)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    counts = compute_usage_counts(model, loader, device=device, precision=args.precision)
    entropy, perplexity, probs = entropy_from_counts(counts, base=2.0)

    nonzero = int((counts > 0).sum())
    total_tokens = int(counts.sum())
    utilization = float(nonzero / counts.size)
    top_k = min(20, counts.size)
    top_indices = np.argsort(counts)[::-1][:top_k]
    top_values = counts[top_indices]
    bottom_values = np.sort(counts)[:top_k]

    heatmap_path = save_heatmap(counts, output_dir, "ImageNet VQGAN codebook usage heatmap")
    histogram_path = save_histogram(counts, output_dir, "ImageNet VQGAN codebook usage histogram")
    bar_path = save_sorted_bar(counts, output_dir, "ImageNet VQGAN codebook usage sorted bar")

    report = {
        "model": {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "n_embed": int(cfg.model.params.n_embed),
            "embed_dim": int(cfg.model.params.embed_dim),
            **metadata,
        },
        "dataset": {
            "image_dir": str(image_dir),
            "num_images": len(files),
            "image_size": image_size,
            "seed": args.seed,
            "max_images": args.max_images,
        },
        "usage": {
            "total_tokens": total_tokens,
            "nonzero_codes": nonzero,
            "codebook_size": int(counts.size),
            "utilization_ratio": utilization,
            "entropy_bits": entropy,
            "perplexity": perplexity,
            "normalized_entropy": float(entropy / np.log2(counts.size)),
            "count_summary": summarize(counts),
            "top_20_code_indices": top_indices.tolist(),
            "top_20_code_counts": top_values.tolist(),
            "bottom_20_code_counts": bottom_values.tolist(),
            "top_20_code_probabilities": probs[top_indices].tolist(),
        },
        "files": {
            "heatmap": str(heatmap_path),
            "histogram": str(histogram_path),
            "sorted_bar": str(bar_path),
            "counts_npy": str(output_dir / "codebook_counts.npy"),
            "report_json": str(output_dir / f"codebook_usage_report_{timestamp}.json"),
        },
    }

    np.save(output_dir / "codebook_counts.npy", counts)
    report_path = output_dir / f"codebook_usage_report_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved report:  {report_path}")
    print(f"Saved heatmap: {heatmap_path}")
    print(f"Saved hist:    {histogram_path}")
    print(f"Saved bar:     {bar_path}")
    print(f"Nonzero codes: {nonzero} / {counts.size}")
    print(f"Utilization:   {utilization:.6f}")
    print(f"Entropy bits:  {entropy:.6f}")
    print(f"Perplexity:    {perplexity:.6f}")
    print(f"Total tokens:  {total_tokens}")


if __name__ == "__main__":
    main()
