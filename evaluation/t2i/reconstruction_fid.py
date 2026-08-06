"""Evaluate VQ tokenizer reconstruction FID on two random image splits."""

import argparse
import gc
import json
import math
import random
import re
import sys
import tempfile
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d, normalize
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import structural_similarity_index_measure
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from tqdm import tqdm

from evaluation.calc_entropy import load_model_from_logdir
from evaluation.t2i.eval import ResizedImageDataset, calculate_fid_components


DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "2026-07-29T23-40-25_my_vqgan_experiment"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
FID_SIZE = (299, 299)


def collect_images(image_dir, recursive=False):
    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def safe_stem(path):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    return stem or "image"


class CenterCropPreprocessor:
    """Match taming.data.base.ImagePaths(size, random_crop=False)."""

    def __init__(self, size):
        self.size = size

    def __call__(self, path):
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            scale = self.size / min(width, height)
            resized_width = max(self.size, round(width * scale))
            resized_height = max(self.size, round(height * scale))
            image = image.resize(
                (resized_width, resized_height), Image.Resampling.BILINEAR
            )
            left = (resized_width - self.size) // 2
            top = (resized_height - self.size) // 2
            image = image.crop((left, top, left + self.size, top + self.size))
            return np.asarray(image, dtype=np.uint8)


class TokenizerInputDataset(Dataset):
    def __init__(self, files, image_size):
        self.files = files
        self.preprocess = CenterCropPreprocessor(image_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        array = self.preprocess(self.files[index])
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float()
        tensor = tensor / 127.5 - 1.0
        return tensor, index


class OriginalFidDataset(Dataset):
    """FID view of the exact center-cropped images given to the tokenizer."""

    def __init__(self, files, image_size):
        self.files = files
        self.preprocess = CenterCropPreprocessor(image_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        array = self.preprocess(self.files[index])
        image = Image.fromarray(array, mode="RGB")
        image = image.resize(FID_SIZE, Image.Resampling.BICUBIC)
        return to_tensor(image)


def loader_for(dataset, batch_size, num_workers, use_cuda):
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast(device_type=device.type, dtype=dtype)


def reconstruct_batch(tokenizer, images, bypass_quantization=False):
    """Reconstruct a batch through either discrete or continuous latents."""
    if not bypass_quantization:
        return tokenizer(images)[0]

    continuous = tokenizer.encoder(images)
    continuous = tokenizer.quant_conv(continuous)

    quantizer = tokenizer.quantize
    if getattr(quantizer, "l2_normalized", False):
        continuous = normalize(
            continuous.float(),
            p=2,
            dim=1,
            eps=float(getattr(quantizer, "l2_normalize_eps", 1e-6)),
        )

    return tokenizer.decode(continuous)



def paired_batch_metrics(original, reconstructed, perceptual_model):
    """Calculate per-image reconstruction metrics without extra image I/O."""
    original = original.float().clamp(-1.0, 1.0)
    reconstructed = reconstructed.float().clamp(-1.0, 1.0)
    original_01 = (original + 1.0) / 2.0
    reconstructed_01 = (reconstructed + 1.0) / 2.0

    mse = torch.mean(
        torch.square(original_01 - reconstructed_01), dim=(1, 2, 3)
    )
    psnr = -10.0 * torch.log10(torch.clamp(mse, min=1e-12))
    ssim = structural_similarity_index_measure(
        reconstructed_01,
        original_01,
        data_range=1.0,
        reduction="none",
    )
    lpips = perceptual_model(original, reconstructed).flatten(1).mean(1)
    return {
        "psnr_db": psnr.detach().cpu().reshape(-1).tolist(),
        "ssim": ssim.detach().cpu().reshape(-1).tolist(),
        "lpips_vgg": lpips.detach().cpu().reshape(-1).tolist(),
    }


def summarize_values(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty metric array")
    if not np.isfinite(array).all():
        raise ValueError("Paired reconstruction metrics contain non-finite values")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize_paired_metrics(metrics_by_split):
    metric_names = ("psnr_db", "ssim", "lpips_vgg")
    overall_values = {
        metric: [
            value
            for split_metrics in metrics_by_split.values()
            for value in split_metrics[metric]
        ]
        for metric in metric_names
    }
    report = {
        "definitions": {
            "psnr_db": (
                "RGB PSNR in dB on [0,1]; higher is better; MSE is floored "
                "at 1e-12 (maximum 120 dB)"
            ),
            "ssim": (
                "RGB SSIM on [0,1], Gaussian 11x11 window; higher is better"
            ),
            "lpips_vgg": (
                "Training-matched taming VGG-LPIPS on [-1,1]; lower is better"
            ),
        },
        "pairing": (
            "Each center-cropped source image is compared with its own reconstruction"
        ),
        "overall": {
            metric: summarize_values(values)
            for metric, values in overall_values.items()
        },
    }
    for split_name, split_metrics in metrics_by_split.items():
        report[split_name] = {
            metric: summarize_values(split_metrics[metric])
            for metric in metric_names
        }
    return report


def print_paired_metric_report(report):
    print("Paired reconstruction metrics (source image vs its reconstruction):")
    labels = {
        "psnr_db": "PSNR dB (higher is better)",
        "ssim": "SSIM (higher is better)",
        "lpips_vgg": "LPIPS-VGG (lower is better)",
    }
    for metric, label in labels.items():
        values = report["overall"][metric]
        print(
            "  {}: mean={:.6f}, std={:.6f}, median={:.6f}, "
            "p05={:.6f}, p95={:.6f}, n={}".format(
                label,
                values["mean"],
                values["std"],
                values["median"],
                values["p05"],
                values["p95"],
                values["count"],
            )
        )


def reconstruct_split(
    files,
    output_dir,
    tokenizer,
    perceptual_model,
    image_size,
    batch_size,
    num_workers,
    device,
    precision,
    description,
    bypass_quantization,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = TokenizerInputDataset(files, image_size)
    dataloader = loader_for(
        dataset, batch_size, num_workers, use_cuda=device.type == "cuda"
    )
    output_paths = [None] * len(dataset)
    paired_values = {"psnr_db": [], "ssim": [], "lpips_vgg": []}
    tokenizer.eval()
    with torch.inference_mode():
        for images, indices in tqdm(dataloader, desc=description, unit="batch"):
            images = images.to(device, non_blocking=device.type == "cuda")
            with autocast_context(device, precision):
                reconstructed = reconstruct_batch(
                    tokenizer,
                    images,
                    bypass_quantization=bypass_quantization,
                )
            reconstructed = reconstructed.float().clamp(-1.0, 1.0)
            if perceptual_model is not None:
                batch_metrics = paired_batch_metrics(
                    images, reconstructed, perceptual_model
                )
                for metric, values in batch_metrics.items():
                    paired_values[metric].extend(values)
            reconstructed_01 = (reconstructed + 1.0) / 2.0
            for image, index in zip(reconstructed_01.cpu(), indices.tolist()):
                target = (
                    output_dir / f"{index:07d}_{safe_stem(files[index])}.png"
                )
                save_image(image, target)
                output_paths[index] = target
    if any(path is None for path in output_paths):
        raise RuntimeError(f"Not every image was reconstructed in {description}")
    return output_paths, paired_values


def activation_statistics(
    dataset, model, batch_size, dims, device, num_workers, description
):
    dataloader = loader_for(
        dataset, batch_size, num_workers, use_cuda=device.type == "cuda"
    )
    activations = np.empty((len(dataset), dims), dtype=np.float64)
    start = 0
    model.eval()
    with torch.inference_mode():
        for images in tqdm(dataloader, desc=description, unit="batch"):
            images = images.to(device, non_blocking=device.type == "cuda")
            prediction = model(images)[0]
            if prediction.shape[2:] != (1, 1):
                prediction = adaptive_avg_pool2d(prediction, output_size=(1, 1))
            prediction = prediction.squeeze(3).squeeze(2).cpu().numpy()
            end = start + len(prediction)
            activations[start:end] = prediction
            start = end
    if start != len(dataset):
        raise RuntimeError(f"Expected {len(dataset)} activations, got {start}")
    return (
        np.mean(activations, axis=0),
        np.cov(activations, rowvar=False),
        len(dataset),
    )



def print_fid_analysis(label, analysis, console_top_k=5):
    """Print exact FID terms and compact feature-level diagnostics."""
    fid = analysis["fid"]
    mean_term = analysis["mean_term"]
    covariance_term = analysis["covariance_term"]
    mean_percent = analysis["mean_term_percent"]
    covariance_percent = analysis["covariance_term_percent"]
    print(f"FID({label}): {fid:.6f}")
    if mean_percent is None:
        component_percentages = "percentages unavailable because FID is zero"
    else:
        component_percentages = (
            f"mean={mean_percent:.2f}%, covariance={covariance_percent:.2f}%"
        )
    print(
        "  exact components: "
        f"mean={mean_term:.6f}, covariance={covariance_term:.6f} "
        f"({component_percentages})"
    )
    trace1 = analysis["covariance_trace_1"]
    trace2 = analysis["covariance_trace_2"]
    cross_trace = analysis["cross_covariance_sqrt_trace"]
    print(
        "  covariance traces: "
        f"trace_1={trace1:.6f}, trace_2={trace2:.6f}, "
        f"cross_sqrt_trace={cross_trace:.6f}"
    )

    mean_items = analysis["top_mean_shift_dimensions"][:console_top_k]
    if mean_items:
        summary = ", ".join(
            "d{}:{:.4g}".format(item["dimension"], item["contribution"])
            for item in mean_items
        )
        print(f"  top mean-shift dimensions (exact mean contributions): {summary}")

    variance_items = analysis["top_marginal_variance_mismatches"][:console_top_k]
    if variance_items:
        summary = ", ".join(
            "d{}:{:.4g}".format(
                item["dimension"], item["absolute_variance_gap"]
            )
            for item in variance_items
        )
        print(f"  top marginal-variance gaps (diagnostic): {summary}")

    correlation_items = analysis["top_correlation_mismatches"][:console_top_k]
    if correlation_items:
        summary = ", ".join(
            "d{}-d{}:{:.4g}".format(
                item["dimension_1"],
                item["dimension_2"],
                item["absolute_correlation_gap"],
            )
            for item in correlation_items
        )
        print(f"  top correlation gaps (diagnostic): {summary}")


def resolve_device(device_name):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def main(args):
    test_started_at = datetime.now().astimezone()
    test_timestamp = test_started_at.strftime("%Y%m%d_%H%M%S")
    image_dir = Path(args.image_dir).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    if not log_dir.is_dir():
        raise FileNotFoundError(log_dir)
    if args.analysis_top_k < 0:
        raise ValueError("--analysis-top-k must be non-negative")
    if not 0 < args.min_split_ratio <= args.max_split_ratio < 1:
        raise ValueError("Split ratios must satisfy 0 < min <= max < 1")

    files = collect_images(image_dir, recursive=args.recursive)
    if args.max_images > 0:
        files = files[: args.max_images]
    if len(files) < 4:
        raise ValueError(f"Need at least 4 images, found {len(files)}")

    rng = random.Random(args.seed)
    rng.shuffle(files)
    sampled_ratio = rng.uniform(args.min_split_ratio, args.max_split_ratio)
    minimum_count = max(2, math.ceil(len(files) * args.min_split_ratio))
    maximum_count = min(len(files) - 2, math.floor(len(files) * args.max_split_ratio))
    if minimum_count > maximum_count:
        raise ValueError(
            f"Cannot split {len(files)} images inside the requested ratio range"
        )
    split_index = min(
        max(round(len(files) * sampled_ratio), minimum_count), maximum_count
    )
    splits = {"split_a": files[:split_index], "split_b": files[split_index:]}
    actual_ratio = len(splits["split_a"]) / len(files)
    print(
        f"Images={len(files)}, seed={args.seed}, sampled_ratio={sampled_ratio:.6f}, "
        f"actual split={len(splits['split_a'])}/{len(splits['split_b'])} "
        f"({actual_ratio:.4%}/{1 - actual_ratio:.4%})"
    )

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.benchmark = args.cudnn_benchmark

    tokenizer, config = load_model_from_logdir(str(log_dir), device=device)
    image_size = args.image_size or int(config.model.params.ddconfig.resolution)
    print(f"Tokenizer input size: {image_size}x{image_size}")
    latent_path = (
        "continuous_normalized"
        if args.bypass_quantization
        else "quantized"
    )
    print("Test configuration:")
    print(f"  started_at: {test_started_at.isoformat()}")
    print(f"  vq_log_dir: {log_dir}")
    print(f"  real_images_dir: {image_dir}")
    print(f"  latent_path: {latent_path}")
    print(f"Reconstruction latent path: {latent_path}")
    if args.skip_paired_metrics:
        perceptual_model = None
        print("Paired PSNR/SSIM/LPIPS metrics disabled.")
    else:
        loss_module = getattr(tokenizer, "loss", None)
        perceptual_model = getattr(loss_module, "perceptual_loss", None)
        if perceptual_model is None:
            raise AttributeError(
                "Tokenizer has no loss.perceptual_loss model for LPIPS evaluation"
            )
        perceptual_model.eval()

    temp_parent = None
    if args.temp_dir:
        temp_parent = Path(args.temp_dir).expanduser().resolve()
        temp_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="reconstruction_fid_", dir=temp_parent
    ) as temporary:
        temporary = Path(temporary)
        print(f"Temporary reconstructions: {temporary}")
        reconstructions = {}
        paired_metrics_by_split = {}
        for name, split_files in splits.items():
            (
                reconstructions[name],
                paired_metrics_by_split[name],
            ) = reconstruct_split(
                split_files,
                temporary / name,
                tokenizer,
                perceptual_model,
                image_size,
                args.reconstruction_batch_size,
                args.num_workers,
                device,
                args.precision,
                f"Reconstruct {name}",
                args.bypass_quantization,
            )

        paired_metrics_report = None
        if perceptual_model is not None:
            paired_metrics_report = summarize_paired_metrics(
                paired_metrics_by_split
            )
            print_paired_metric_report(paired_metrics_report)

        del tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if args.dims not in InceptionV3.BLOCK_INDEX_BY_DIM:
            choices = sorted(InceptionV3.BLOCK_INDEX_BY_DIM)
            raise ValueError(f"--dims must be one of {choices}, got {args.dims}")
        block_index = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
        inception = InceptionV3([block_index], resize_input=False).to(device)

        reconstructed_a_stats = activation_statistics(
            ResizedImageDataset(reconstructions["split_a"]),
            inception,
            args.fid_batch_size,
            args.dims,
            device,
            args.num_workers,
            "Reconstructed A",
        )
        reconstructed_b_stats = activation_statistics(
            ResizedImageDataset(reconstructions["split_b"]),
            inception,
            args.fid_batch_size,
            args.dims,
            device,
            args.num_workers,
            "Reconstructed B",
        )
        original_a_stats = activation_statistics(
            OriginalFidDataset(splits["split_a"], image_size),
            inception,
            args.fid_batch_size,
            args.dims,
            device,
            args.num_workers,
            "Original A",
        )
        original_b_stats = activation_statistics(
            OriginalFidDataset(splits["split_b"], image_size),
            inception,
            args.fid_batch_size,
            args.dims,
            device,
            args.num_workers,
            "Original B",
        )

        analyses = {
            "reconstructed_a_vs_original_a": calculate_fid_components(
                reconstructed_a_stats[0],
                reconstructed_a_stats[1],
                original_a_stats[0],
                original_a_stats[1],
                top_k=args.analysis_top_k,
            ),
            "reconstructed_a_vs_reconstructed_b": calculate_fid_components(
                reconstructed_a_stats[0],
                reconstructed_a_stats[1],
                reconstructed_b_stats[0],
                reconstructed_b_stats[1],
                top_k=args.analysis_top_k,
            ),
            "reconstructed_a_vs_original_b": calculate_fid_components(
                reconstructed_a_stats[0],
                reconstructed_a_stats[1],
                original_b_stats[0],
                original_b_stats[1],
                top_k=args.analysis_top_k,
            ),
            "original_a_vs_original_b": calculate_fid_components(
                original_a_stats[0],
                original_a_stats[1],
                original_b_stats[0],
                original_b_stats[1],
                top_k=args.analysis_top_k,
            ),
        }
        comparison_labels = {
            "reconstructed_a_vs_original_a":
                "reconstructed_A, original_A [same images]",
            "reconstructed_a_vs_reconstructed_b":
                "reconstructed_A, reconstructed_B",
            "reconstructed_a_vs_original_b": "reconstructed_A, original_B",
            "original_a_vs_original_b": "original_A, original_B [split baseline]",
        }
        for key, analysis in analyses.items():
            print_fid_analysis(comparison_labels[key], analysis)

        report = {
            "test_configuration": {
                "test_timestamp": test_timestamp,
                "started_at": test_started_at.isoformat(),
                "completed_at": datetime.now().astimezone().isoformat(),
                "vq_log_dir": str(log_dir),
                "real_images_dir": str(image_dir),
                "generated_images_dir": str(temporary),
                "generated_split_dirs": {
                    name: str(temporary / name) for name in splits
                },
                "generated_images_are_temporary": True,
                "latent_path": latent_path,
                "args": vars(args),
            },
            "image_dir": str(image_dir),
            "log_dir": str(log_dir),
            "latent_path": latent_path,
            "bypass_quantization": args.bypass_quantization,
            "seed": args.seed,
            "sampled_split_ratio": sampled_ratio,
            "actual_split_ratio": actual_ratio,
            "split_a_count": len(splits["split_a"]),
            "split_b_count": len(splits["split_b"]),
            "reconstructed_a_vs_original_a_fid": analyses[
                "reconstructed_a_vs_original_a"
            ]["fid"],
            "reconstructed_a_vs_reconstructed_b_fid": analyses[
                "reconstructed_a_vs_reconstructed_b"
            ]["fid"],
            "reconstructed_a_vs_original_b_fid": analyses[
                "reconstructed_a_vs_original_b"
            ]["fid"],
            "original_a_vs_original_b_fid": analyses[
                "original_a_vs_original_b"
            ]["fid"],
            "fid_analysis": analyses,
            "paired_psnr_db_mean": (
                paired_metrics_report["overall"]["psnr_db"]["mean"]
                if paired_metrics_report is not None
                else None
            ),
            "paired_ssim_mean": (
                paired_metrics_report["overall"]["ssim"]["mean"]
                if paired_metrics_report is not None
                else None
            ),
            "paired_lpips_vgg_mean": (
                paired_metrics_report["overall"]["lpips_vgg"]["mean"]
                if paired_metrics_report is not None
                else None
            ),
            "paired_reconstruction_metrics": paired_metrics_report,
            "analysis_top_k": args.analysis_top_k,
            "analysis_explanation": {
                "exact_identity": "FID = mean_term + covariance_term",
                "mean_term": "Squared distance between Inception feature means",
                "covariance_term": (
                    "Difference in feature spread and covariance structure"
                ),
                "same_image_reconstruction": (
                    "reconstructed_A vs original_A isolates reconstruction shift"
                ),
                "split_baseline": (
                    "original_A vs original_B estimates finite-sample/split noise"
                ),
                "feature_rankings": (
                    "Variance/correlation rankings are diagnostic and do not sum to FID"
                ),
            },
            "image_size": image_size,
            "dims": args.dims,
        }


    if args.output_json:
        requested_output = Path(args.output_json).expanduser().resolve()
        output_suffix = requested_output.suffix or ".json"
        output_json = requested_output.with_name(
            f"{requested_output.stem}_{test_timestamp}{output_suffix}"
        )
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_json = (
            output_dir
            / f"reconstruction_fid_{latent_path}_{test_timestamp}.json"
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {output_json}")
    print("Temporary reconstruction directory removed.")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument(
        "--log-dir",
        "--vq-log-dir",
        dest="log_dir",
        default=str(DEFAULT_LOG_DIR),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--reconstruction-batch-size", type=int, default=16)
    parser.add_argument("--fid-batch-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dims", type=int, default=2048)
    parser.add_argument("--analysis-top-k", type=int, default=10)
    parser.add_argument("--skip-paired-metrics", action="store_true")
    parser.add_argument(
        "--bypass-quantization",
        action="store_true",
        help=(
            "Decode the L2-normalized continuous encoder latent before nearest-"
            "neighbor codebook lookup. The default keeps the quantized path."
        ),
    )
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--min-split-ratio", type=float, default=0.4)
    parser.add_argument("--max-split-ratio", type=float, default=0.6)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--temp-dir", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "evaluation" / "t2i" / "test_results"),
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
