"""Calculate FID between real and generated ROCO images."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_fid.inception import InceptionV3
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REAL_DIR = (
    PROJECT_ROOT / "data" / "roco" / "1" / "rocov2" / "valid_images" / "valid"
)
DEFAULT_GENERATED_DIR = (
    PROJECT_ROOT /"evaluation"/"t2i"/"images_2026-01-16T14-12-20_vqgan_with_larp"
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
INCEPTION_SIZE = (299, 299)


def list_images(path):
    return sorted(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
    )


class ResizedImageDataset(Dataset):
    """Resize images in memory so DataLoader can batch mixed image sizes."""

    def __init__(self, files, size=INCEPTION_SIZE):
        self.files = files
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image = image.resize(self.size, Image.Resampling.BICUBIC)
                return to_tensor(image)
        except Exception as error:
            raise RuntimeError(f"Failed to load image: {path}") from error


def calculate_activation_statistics(
    files, model, batch_size, dims, device, num_workers, description
):
    """Calculate Inception statistics without writing resized images."""
    dataset = ResizedImageDataset(files)
    use_cuda = device.type == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    activations = np.empty((len(dataset), dims), dtype=np.float64)
    start = 0
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=description, unit="batch"):
            batch = batch.to(device, non_blocking=use_cuda)
            prediction = model(batch)[0]
            if prediction.shape[2:] != (1, 1):
                prediction = adaptive_avg_pool2d(prediction, output_size=(1, 1))
            prediction = prediction.squeeze(3).squeeze(2).cpu().numpy()
            end = start + prediction.shape[0]
            activations[start:end] = prediction
            start = end

    if start != len(dataset):
        raise RuntimeError(f"Expected {len(dataset)} activations, got {start}")
    return np.mean(activations, axis=0), np.cov(activations, rowvar=False)


def calculate_fid_components(
    mu1,
    sigma1,
    mu2,
    sigma2,
    eps=1e-6,
    top_k=10,
    analyze_correlations=True,
):
    """Return exact FID terms and feature-level distribution diagnostics.

    Only ``mean_term`` and ``covariance_term`` are additive components of FID.
    Feature variance and correlation differences help locate distribution
    mismatches, but are not additional FID terms.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    if mu1.shape != mu2.shape:
        raise ValueError(f"Mean shapes differ: {mu1.shape} != {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(
            f"Covariance shapes differ: {sigma1.shape} != {sigma2.shape}"
        )

    difference = mu1 - mu2
    covmean = matrix_sqrt(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        print(
            "FID covariance product is nearly singular; "
            f"adding {eps} to both covariance diagonals."
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = matrix_sqrt((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            maximum = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {maximum}")
        covmean = covmean.real

    mean_contributions = np.square(difference)
    mean_term = float(np.sum(mean_contributions))
    trace1 = float(np.trace(sigma1))
    trace2 = float(np.trace(sigma2))
    cross_trace = float(np.trace(covmean))
    covariance_term = trace1 + trace2 - 2.0 * cross_trace
    fid = mean_term + covariance_term

    top_k = max(0, min(int(top_k), len(mu1)))
    mean_indices = (
        np.argsort(mean_contributions)[-top_k:][::-1]
        if top_k
        else np.empty(0, dtype=int)
    )
    top_mean_dimensions = [
        {
            "dimension": int(index),
            "contribution": float(mean_contributions[index]),
            "mean_1": float(mu1[index]),
            "mean_2": float(mu2[index]),
        }
        for index in mean_indices
    ]

    variance1 = np.diag(sigma1)
    variance2 = np.diag(sigma2)
    variance_gap = np.abs(variance1 - variance2)
    variance_indices = (
        np.argsort(variance_gap)[-top_k:][::-1]
        if top_k
        else np.empty(0, dtype=int)
    )
    top_variance_mismatches = [
        {
            "dimension": int(index),
            "absolute_variance_gap": float(variance_gap[index]),
            "variance_1": float(variance1[index]),
            "variance_2": float(variance2[index]),
        }
        for index in variance_indices
    ]

    correlation_frobenius_distance = None
    mean_absolute_correlation_gap = None
    top_correlation_mismatches = []
    if analyze_correlations and top_k and len(mu1) > 1:
        standard_deviation1 = np.sqrt(np.maximum(variance1, eps))
        standard_deviation2 = np.sqrt(np.maximum(variance2, eps))
        correlation1 = sigma1 / np.outer(standard_deviation1, standard_deviation1)
        correlation2 = sigma2 / np.outer(standard_deviation2, standard_deviation2)
        correlation_gap = np.abs(correlation1 - correlation2)
        np.fill_diagonal(correlation_gap, 0.0)
        correlation_frobenius_distance = float(np.linalg.norm(correlation_gap))
        upper_rows, upper_columns = np.triu_indices(len(mu1), k=1)
        upper_gaps = correlation_gap[upper_rows, upper_columns]
        mean_absolute_correlation_gap = float(np.mean(upper_gaps))
        pair_count = min(top_k, len(upper_gaps))
        if pair_count:
            pair_indices = np.argpartition(upper_gaps, -pair_count)[-pair_count:]
            pair_indices = pair_indices[np.argsort(upper_gaps[pair_indices])[::-1]]
            top_correlation_mismatches = [
                {
                    "dimension_1": int(upper_rows[index]),
                    "dimension_2": int(upper_columns[index]),
                    "absolute_correlation_gap": float(upper_gaps[index]),
                    "correlation_1": float(
                        correlation1[upper_rows[index], upper_columns[index]]
                    ),
                    "correlation_2": float(
                        correlation2[upper_rows[index], upper_columns[index]]
                    ),
                }
                for index in pair_indices
            ]

    denominator = fid if abs(fid) > np.finfo(float).eps else None
    return {
        "fid": float(fid),
        "mean_term": mean_term,
        "covariance_term": float(covariance_term),
        "mean_term_percent": (
            float(100.0 * mean_term / denominator) if denominator else None
        ),
        "covariance_term_percent": (
            float(100.0 * covariance_term / denominator) if denominator else None
        ),
        "covariance_trace_1": trace1,
        "covariance_trace_2": trace2,
        "cross_covariance_sqrt_trace": cross_trace,
        "top_mean_shift_dimensions": top_mean_dimensions,
        "top_marginal_variance_mismatches": top_variance_mismatches,
        "correlation_frobenius_distance": correlation_frobenius_distance,
        "mean_absolute_correlation_gap": mean_absolute_correlation_gap,
        "top_correlation_mismatches": top_correlation_mismatches,
        "diagnostic_note": (
            "Only mean_term and covariance_term sum to FID. Variance and "
            "correlation mismatch rankings are non-additive diagnostics."
        ),
    }


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Frechet distance compatible with both old and new SciPy."""
    return calculate_fid_components(
        mu1,
        sigma1,
        mu2,
        sigma2,
        eps=eps,
        top_k=0,
        analyze_correlations=False,
    )["fid"]


def matrix_sqrt(matrix):
    """Handle SciPy sqrtm before and after removal of the disp argument."""
    try:
        result = linalg.sqrtm(matrix, disp=False)
    except TypeError:
        result = linalg.sqrtm(matrix)
    return result[0] if isinstance(result, tuple) else result


def calculate_fid(
    real_files, generated_files, batch_size, device, dims, num_workers
):
    if dims not in InceptionV3.BLOCK_INDEX_BY_DIM:
        choices = sorted(InceptionV3.BLOCK_INDEX_BY_DIM)
        raise ValueError(f"--dims must be one of {choices}, got {dims}")
    block_index = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    # DataLoader workers already resize each image. Avoid a redundant resize
    # inside the Inception network and the associated interpolation cost.
    model = InceptionV3([block_index], resize_input=False).to(device)
    real_mean, real_covariance = calculate_activation_statistics(
        real_files,
        model,
        batch_size,
        dims,
        device,
        num_workers,
        "Real images",
    )
    generated_mean, generated_covariance = calculate_activation_statistics(
        generated_files,
        model,
        batch_size,
        dims,
        device,
        num_workers,
        "Generated images",
    )
    return calculate_frechet_distance(
        real_mean,
        real_covariance,
        generated_mean,
        generated_covariance,
    )


def validate_images(files):
    """Fail early with a useful filename if a directory contains bad images."""
    for path in files:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            raise RuntimeError(f"Invalid image: {path}") from error


def resolve_device(device_name):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def main(args):
    real_dir = Path(args.real_dir).expanduser().resolve()
    generated_dir = Path(args.generated_dir).expanduser().resolve()
    for path in (real_dir, generated_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)

    real_files = list_images(real_dir)
    generated_files = list_images(generated_dir)
    real_count = len(real_files)
    generated_count = len(generated_files)
    if real_count == 0 or generated_count == 0:
        raise RuntimeError(
            f"Image directory is empty: real={real_count}, generated={generated_count}"
        )
    print(
        f"Calculating FID: real={real_count} ({real_dir}), "
        f"generated={generated_count} ({generated_dir})"
    )
    if real_count != generated_count:
        print(
            "Warning: image counts differ. Generate the full validation set "
            "before reporting the final ROCO valid FID."
        )
    if args.validate_images:
        print("Validating image files...")
        validate_images(real_files)
        validate_images(generated_files)

    device = resolve_device(args.device)
    fid = calculate_fid(
        real_files,
        generated_files,
        batch_size=args.batch_size,
        device=device,
        dims=args.dims,
        num_workers=args.num_workers,
    )
    print(f"FID: {fid:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", default=str(DEFAULT_REAL_DIR))
    parser.add_argument("--generated-dir", default=str(DEFAULT_GENERATED_DIR))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dims", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--validate-images",
        action="store_true",
        help="Decode-check every image before FID (safer but adds an extra pass).",
    )
    main(parser.parse_args())
