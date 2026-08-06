#!/usr/bin/env python3
"""Create a three-column visual check of a trained VQ tokenizer.

The dataset and its preprocessing are instantiated from the project YAML saved
in the supplied VQ log directory.  Each row of the output image contains:

    training-preprocessed input | continuous-latent decode | VQ reconstruction
"""

import argparse
import json
import random
import re
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from evaluation.load_vqgan import (
    _find_ckpt,
    _find_project_yaml,
    load_model_from_logdir,
)
from main import instantiate_from_config
from taming.data.utils import custom_collate


COLUMN_LABELS = (
    "Original (tokenizer input)",
    "Decode without quantization",
    "Decode with quantization",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample images through the dataset interface saved in a "
            "VQ log directory and create one three-column reconstruction grid."
        )
    )
    parser.add_argument(
        "--vq-log-dir",
        "--log-dir",
        dest="vq_log_dir",
        required=True,
        help="VQ training log directory containing configs/ and checkpoints/.",
    )
    parser.add_argument(
        "--dataset-split",
        choices=("auto", "train", "validation", "test"),
        default="auto",
        help="Dataset config to reuse. auto prefers test, then validation.",
    )
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, for example cuda:0 or cpu (default: auto).",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "fp16"),
        default="fp32",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Result directory. By default a timestamped directory is created "
            "under evaluation/vae/test_results/."
        ),
    )
    parser.add_argument(
        "--raw-continuous-latent",
        action="store_true",
        help=(
            "Do not L2-normalize the continuous latent for tokenizers whose "
            "quantizer uses normalized embeddings. By default normalization is "
            "kept, but nearest-code lookup is still bypassed."
        ),
    )
    return parser.parse_args()


def choose_dataset_config(config, requested_split):
    if "data" not in config or "params" not in config.data:
        raise KeyError("The project config has no data.params section")

    data_params = config.data.params
    if requested_split == "auto":
        candidates = ("test", "validation")
    else:
        candidates = (requested_split,)

    for split in candidates:
        if split in data_params and data_params[split] is not None:
            return split, data_params[split]

    available = [
        name
        for name in ("train", "validation", "test")
        if name in data_params and data_params[name] is not None
    ]
    raise KeyError(
        f"Could not resolve dataset split {requested_split!r}; "
        f"available splits: {available}"
    )


def resolve_device(device_name):
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 inference requires a CUDA device")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def prepare_model_input(model, batch, device):
    if hasattr(model, "get_input"):
        images = model.get_input(batch, model.image_key)
    else:
        images = batch[getattr(model, "image_key", "image")]
        if images.ndim == 3:
            images = images[..., None]
        images = images.permute(0, 3, 1, 2).contiguous().float()
    return images.to(device, non_blocking=device.type == "cuda")


def reconstruct_both_paths(model, images, raw_continuous_latent=False):
    """Decode one encoder result before and after nearest-code lookup."""
    continuous = model.quant_conv(model.encoder(images))
    quantized, _, _ = model.quantize(continuous)

    continuous_for_decode = continuous
    quantizer_normalizes = bool(
        getattr(model.quantize, "l2_normalized", False)
        or getattr(model, "latent_l2_normalized", False)
    )
    if quantizer_normalizes and not raw_continuous_latent:
        eps = float(
            getattr(
                model.quantize,
                "l2_normalize_eps",
                getattr(model, "latent_l2_normalize_eps", 1e-6),
            )
        )
        continuous_for_decode = F.normalize(
            continuous.float(), p=2, dim=1, eps=eps
        ).to(dtype=continuous.dtype)

    decoded_continuous = model.decode(continuous_for_decode)
    decoded_quantized = model.decode(quantized)
    return decoded_continuous, decoded_quantized, quantizer_normalizes


def batch_to_pil(images):
    images = (
        ((images.detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    result = []
    for array in images:
        if array.shape[-1] == 1:
            result.append(Image.fromarray(array[..., 0], mode="L").convert("RGB"))
        elif array.shape[-1] == 3:
            result.append(Image.fromarray(array, mode="RGB"))
        else:
            raise ValueError(f"Expected 1 or 3 image channels, got {array.shape[-1]}")
    return result


def make_canvas(width, height, row_count, header_height=36):
    canvas = Image.new("RGB", (3 * width, header_height + row_count * height), "black")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, label in enumerate(COLUMN_LABELS):
        left = column * width
        draw.text((left + 8, 11), label, fill="white", font=font)
        if column:
            draw.line((left, 0, left, canvas.height), fill=(255, 255, 255), width=1)
    draw.line((0, header_height - 1, canvas.width, header_height - 1), fill=(255, 255, 255))
    return canvas, header_height


def safe_name(value):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "unknown"


def json_ready(config):
    return OmegaConf.to_container(config, resolve=True)


def main(args):
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    vq_log_dir = Path(args.vq_log_dir).expanduser().resolve()
    if not vq_log_dir.is_dir():
        raise FileNotFoundError(f"VQ log directory does not exist: {vq_log_dir}")

    project_yaml = Path(_find_project_yaml(str(vq_log_dir))).resolve()
    checkpoint = Path(_find_ckpt(str(vq_log_dir))).resolve()
    config = OmegaConf.load(project_yaml)
    split, dataset_config = choose_dataset_config(config, args.dataset_split)
    dataset = instantiate_from_config(dataset_config)
    if len(dataset) < args.num_images:
        raise ValueError(
            f"Dataset split {split!r} contains {len(dataset)} images, but "
            f"{args.num_images} were requested"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    selected_indices = random.Random(args.seed).sample(
        range(len(dataset)), args.num_images
    )

    subset = Subset(dataset, selected_indices)
    generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(
        subset,
        batch_size=min(args.batch_size, args.num_images),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=custom_collate,
        generator=generator,
    )

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, _ = load_model_from_logdir(str(vq_log_dir), device=str(device))
    model.eval()

    run_started = datetime.now().astimezone()
    timestamp = run_started.strftime("%Y%m%d_%H%M%S")
    dataset_target = str(dataset_config.target)
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        run_name = "__".join(
            (
                safe_name(vq_log_dir.name),
                safe_name(dataset_target.rsplit(".", 1)[-1]),
                safe_name(split),
                f"seed-{args.seed}",
                timestamp,
            )
        )
        output_dir = PROJECT_ROOT / "evaluation" / "vae" / "test_results" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "reconstruction_comparison.png"
    info_path = output_dir / "run_info.json"

    canvas = None
    header_height = None
    row_offset = 0
    selected_samples = []
    quantizer_normalizes = False

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Reconstruct", unit="batch"):
            images = prepare_model_input(model, batch, device)
            with autocast_context(device, args.precision):
                decoded_continuous, decoded_quantized, batch_normalizes = (
                    reconstruct_both_paths(
                        model,
                        images,
                        raw_continuous_latent=args.raw_continuous_latent,
                    )
                )
            quantizer_normalizes = quantizer_normalizes or batch_normalizes

            originals_pil = batch_to_pil(images)
            continuous_pil = batch_to_pil(decoded_continuous)
            quantized_pil = batch_to_pil(decoded_quantized)
            if canvas is None:
                width, height = originals_pil[0].size
                canvas, header_height = make_canvas(
                    width, height, args.num_images
                )

            paths = batch.get("file_path_", [None] * len(originals_pil))
            for original, continuous, quantized, source_path in zip(
                originals_pil, continuous_pil, quantized_pil, paths
            ):
                expected_size = originals_pil[0].size
                if continuous.size != expected_size or quantized.size != expected_size:
                    raise ValueError(
                        "Decoder output size does not match tokenizer input size: "
                        f"input={expected_size}, continuous={continuous.size}, "
                        f"quantized={quantized.size}"
                    )
                top = header_height + row_offset * expected_size[1]
                canvas.paste(original, (0, top))
                canvas.paste(continuous, (expected_size[0], top))
                canvas.paste(quantized, (2 * expected_size[0], top))
                selected_samples.append(
                    {
                        "row": row_offset,
                        "dataset_index": selected_indices[row_offset],
                        "source_path": str(source_path) if source_path is not None else None,
                    }
                )
                row_offset += 1

    if canvas is None or row_offset != args.num_images:
        raise RuntimeError(
            f"Expected {args.num_images} reconstructed rows, produced {row_offset}"
        )
    canvas.save(grid_path, format="PNG", compress_level=1)

    continuous_mode = "raw_encoder_latent"
    if quantizer_normalizes and not args.raw_continuous_latent:
        continuous_mode = "l2_normalized_encoder_latent"

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "vq": {
            "log_dir": str(vq_log_dir),
            "project_config": str(project_yaml),
            "checkpoint": str(checkpoint),
            "model_target": str(config.model.target),
            "model_params": json_ready(config.model.get("params", {})),
        },
        "dataset": {
            "requested_split": args.dataset_split,
            "resolved_split": split,
            "target": dataset_target,
            "params": json_ready(dataset_config.get("params", {})),
            "size": len(dataset),
            "sample_count": args.num_images,
            "seed": args.seed,
            "selected_samples": selected_samples,
        },
        "reconstruction": {
            "columns": list(COLUMN_LABELS),
            "continuous_latent_mode": continuous_mode,
            "quantizer_uses_l2_normalization": quantizer_normalizes,
            "precision": args.precision,
            "device": str(device),
        },
        "output": {
            "directory": str(output_dir),
            "comparison_image": str(grid_path),
            "metadata_json": str(info_path),
            "image_width": canvas.width,
            "image_height": canvas.height,
        },
        "command_arguments": vars(args),
    }
    info_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved comparison image: {grid_path}")
    print(f"Saved run information:  {info_path}")


if __name__ == "__main__":
    main(parse_args())
