from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .haar import HaarDWT2D


def _per_sample_normalize(values: Tensor, eps: float = 1e-6) -> Tensor:
    flat = values.flatten(1)
    low, high = flat.min(-1, keepdim=True).values, flat.max(-1, keepdim=True).values
    return ((flat - low) / (high - low).clamp_min(eps)).view_as(values)


class ResidualScorer(nn.Module):
    """Scores image regions from pixel, Sobel, Haar, random, or hybrid error."""

    supported = {"random", "pixel", "sobel", "haar_dwt", "hybrid"}

    def __init__(
        self, scorer_type: str = "haar_dwt", pixel_weight: float = 1.0,
        sobel_weight: float = 1.0, dwt_weight: float = 1.0, eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if scorer_type not in self.supported:
            raise ValueError(f"Unknown scorer {scorer_type}; choose {sorted(self.supported)}")
        self.scorer_type, self.eps = scorer_type, eps
        self.weights = {"pixel": pixel_weight, "sobel": sobel_weight, "haar_dwt": dwt_weight}
        self.dwt = HaarDWT2D()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_x.t().view(1, 1, 3, 3), persistent=False)
        self.last_components: dict[str, Tensor] = {}

    @staticmethod
    def _pool(values: Tensor, latent_hw: tuple[int, int]) -> Tensor:
        return F.adaptive_avg_pool2d(values.mean(dim=1, keepdim=True), latent_hw).squeeze(1)

    def _sobel_error(self, images: Tensor, base_recon: Tensor) -> Tensor:
        channels = images.shape[1]
        kernel_x = self.sobel_x.to(images).repeat(channels, 1, 1, 1)
        kernel_y = self.sobel_y.to(images).repeat(channels, 1, 1, 1)
        def grad(x: Tensor) -> Tensor:
            gx = F.conv2d(x, kernel_x, padding=1, groups=channels)
            gy = F.conv2d(x, kernel_y, padding=1, groups=channels)
            return torch.sqrt(gx.square() + gy.square() + self.eps)
        return (grad(images) - grad(base_recon)).abs()

    def _dwt_error(self, images: Tensor, base_recon: Tensor) -> Tensor:
        bands_a, bands_b = self.dwt(images), self.dwt(base_recon)
        return sum((a - b).abs() for a, b in zip(bands_a[1:], bands_b[1:])) / 3.0

    def forward(self, images: Tensor, base_recon: Tensor, latent_hw: tuple[int, int]) -> Tensor:
        if images.shape != base_recon.shape:
            raise ValueError("images and base_recon must have the same shape")
        if self.scorer_type == "random":
            score = torch.rand(images.shape[0], *latent_hw, device=images.device, dtype=images.dtype)
            self.last_components = {"random": score.detach()}
            return score
        components = {
            "pixel": self._pool((images - base_recon).abs(), latent_hw),
            "sobel": self._pool(self._sobel_error(images, base_recon), latent_hw),
            "haar_dwt": self._pool(self._dwt_error(images, base_recon), latent_hw),
        }
        normalized = {name: _per_sample_normalize(value, self.eps) for name, value in components.items()}
        self.last_components = {name: value.detach() for name, value in normalized.items()}
        if self.scorer_type == "hybrid":
            score = sum(self.weights[name] * normalized[name] for name in normalized)
        else:
            score = normalized[self.scorer_type]
        return _per_sample_normalize(score, self.eps)
