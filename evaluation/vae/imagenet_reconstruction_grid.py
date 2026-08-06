#!/usr/bin/env python3
"""Compare ImageNet inputs, continuous decodes, and VQ decodes."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

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


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "evaluation" / "vae" / "test_results"
COLUMN_LABELS = (
    "Original (tokenizer input)",
    "Decode without quantization",
    "Decode after quantization",
)


def batch_to_pil(images):
    arrays = (
        ((images.detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    return [Image.fromarray(array, mode="RGB") for array in arrays]


def make_canvas(image_size, rows, header_height=36):
    canvas = Image.new(
        "RGB", (3 * image_size, header_height + rows * image_size), "black"
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, label in enumerate(COLUMN_LABELS):
        left = column * image_size
        draw.text((left + 8, 11), label, fill="white", font=font)
        if column:
            draw.line((left, 0, left, canvas.height), fill="white", width=1)
    draw.line((0, header_height - 1, canvas.width, header_height - 1), fill="white")
    return canvas, header_height


def run(args):
    if args.num_images <= 0 or args.batch_size <= 0:
        raise ValueError("--num-images and --batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    image_dir = Path(args.image_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    for path in (config_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    all_files = collect_images(image_dir)
    if len(all_files) < args.num_images:
        raise ValueError(
            f"Requested {args.num_images} images, but found only {len(all_files)}"
        )
    files = select_images(all_files, args.num_images, args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model, config, checkpoint_metadata = load_standalone_vqgan(
        config_path, checkpoint, device
    )
    image_size = args.image_size or int(config.model.params.ddconfig.resolution)
    dataset = ImageNetCenterCropDataset(files, image_size, return_path=True)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    started_at = datetime.now().astimezone()
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = (
            DEFAULT_OUTPUT_ROOT
            / f"vqgan_f16_16384__imagenet_val__seed-{args.seed}__{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "reconstruction_comparison.png"
    report_path = output_dir / "run_info.json"
    canvas, header_height = make_canvas(image_size, len(dataset))

    row = 0
    selected_samples = []
    with torch.inference_mode():
        for images, paths in tqdm(loader, desc="Three-way reconstruction", unit="batch"):
            images = images.to(device, non_blocking=device.type == "cuda")
            with autocast_context(device, args.precision):
                continuous = model.quant_conv(model.encoder(images))
                quantized, _, _ = model.quantize(continuous)
                decoded_continuous = model.decode(continuous)
                decoded_quantized = model.decode(quantized)

            originals = batch_to_pil(images)
            continuous_images = batch_to_pil(decoded_continuous)
            quantized_images = batch_to_pil(decoded_quantized)
            for original, no_quant, quantized, source_path in zip(
                originals, continuous_images, quantized_images, paths
            ):
                top = header_height + row * image_size
                canvas.paste(original, (0, top))
                canvas.paste(no_quant, (image_size, top))
                canvas.paste(quantized, (2 * image_size, top))
                selected_samples.append(
                    {
                        "row": row,
                        "source_path": source_path,
                        "relative_path": str(Path(source_path).relative_to(image_dir)),
                    }
                )
                row += 1

    if row != len(dataset):
        raise RuntimeError(f"Expected {len(dataset)} rows, produced {row}")
    canvas.save(grid_path, format="PNG", compress_level=1)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "model": {
            "config": str(config_path),
            "checkpoint": str(checkpoint),
            "target": str(config.model.target),
            "n_embed": int(config.model.params.n_embed),
            "embed_dim": int(config.model.params.embed_dim),
            **checkpoint_metadata,
        },
        "dataset": {
            "image_dir": str(image_dir),
            "discovered_image_count": len(all_files),
            "sample_count": len(files),
            "seed": args.seed,
            "image_size": image_size,
            "preprocessing": "SmallestMaxSize + CenterCrop (albumentations)",
            "selected_samples": selected_samples,
        },
        "columns": list(COLUMN_LABELS),
        "reconstruction": {
            "continuous_path": "decoder(post_quant_conv(quant_conv(encoder(x))))",
            "quantized_path": "decoder(quantize(quant_conv(encoder(x))))",
            "precision": args.precision,
            "device": str(device),
        },
        "outputs": {
            "directory": str(output_dir),
            "comparison_image": str(grid_path),
            "metadata_json": str(report_path),
        },
        "arguments": vars(args),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved comparison image: {grid_path}")
    print(f"Saved run information:  {report_path}")
    return grid_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create a 3-column ImageNet reconstruction grid."
    )
    parser.add_argument("--config", default=str(DEFAULT_MODEL_DIR / "config.yaml"))
    parser.add_argument("--checkpoint", default=str(DEFAULT_MODEL_DIR / "model.ckpt"))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
