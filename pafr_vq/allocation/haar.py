"""Fixed, differentiable Haar wavelets without an external dependency."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class HaarDWT2D(nn.Module):
    """One-level channel-wise Haar DWT with replicate padding for odd sizes.

    The output bands have spatial shape ``ceil(H / 2), ceil(W / 2)``. Kernels
    are buffers, so the transform remains device/dtype portable and has no
    trainable parameters.
    """

    def __init__(self, padding: str = "replicate") -> None:
        super().__init__()
        if padding not in {"replicate", "reflect", "constant"}:
            raise ValueError(f"Unsupported padding mode: {padding}")
        self.padding = padding
        kernels = torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0]], [[-1.0, -1.0], [1.0, 1.0]],
             [[-1.0, 1.0], [-1.0, 1.0]], [[1.0, -1.0], [-1.0, 1.0]]]
        ) / 2.0
        self.register_buffer("kernels", kernels.unsqueeze(1), persistent=False)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if images.ndim != 4:
            raise ValueError("HaarDWT2D expects [B, C, H, W]")
        _, channels, height, width = images.shape
        pad_h, pad_w = height % 2, width % 2
        if pad_h or pad_w:
            images = F.pad(images, (0, pad_w, 0, pad_h), mode=self.padding)
        kernel = self.kernels.to(dtype=images.dtype).repeat(channels, 1, 1, 1)
        bands = F.conv2d(images, kernel, stride=2, groups=channels)
        bands = bands.view(images.shape[0], channels, 4, bands.shape[-2], bands.shape[-1])
        return tuple(bands[:, :, i] for i in range(4))  # type: ignore[return-value]
