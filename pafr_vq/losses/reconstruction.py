from __future__ import annotations
import torch
from torch import Tensor
import torch.nn.functional as F
from pafr_vq.allocation.haar import HaarDWT2D


def _gradient_l1(a: Tensor, b: Tensor) -> Tensor:
    return (a[..., 1:, :] - a[..., :-1, :]).sub(b[..., 1:, :] - b[..., :-1, :]).abs().mean() + \
        (a[..., :, 1:] - a[..., :, :-1]).sub(b[..., :, 1:] - b[..., :, :-1]).abs().mean()


def reconstruction_losses(target: Tensor, reconstruction: Tensor, dwt: HaarDWT2D | None = None) -> dict[str, Tensor]:
    dwt = dwt or HaarDWT2D().to(target)
    target_bands, recon_bands = dwt(target), dwt(reconstruction)
    frequency = sum(F.l1_loss(a, b) for a, b in zip(target_bands[1:], recon_bands[1:])) / 3.0
    return {"l1": F.l1_loss(reconstruction, target), "mse": F.mse_loss(reconstruction, target),
            "frequency": frequency, "gradient": _gradient_l1(target, reconstruction)}
