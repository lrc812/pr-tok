from __future__ import annotations
import math
import torch
from torch import Tensor


def combination_bits(num_positions: int, budget: int, device: torch.device | None = None) -> Tensor:
    if not 0 <= budget <= num_positions:
        raise ValueError("budget must be in [0, num_positions]")
    if budget == 0 or budget == num_positions:
        return torch.zeros((), dtype=torch.float64, device=device)
    value = torch.tensor(float(num_positions), dtype=torch.float64, device=device)
    return (torch.lgamma(value + 1) - torch.lgamma(torch.tensor(float(budget), dtype=torch.float64, device=device) + 1)
            - torch.lgamma(torch.tensor(float(num_positions - budget), dtype=torch.float64, device=device) + 1)) / math.log(2.0)


def bitrate_metrics(base_ids: Tensor, residual_mask: Tensor, base_codebook_size: int,
                    residual_codebook_size: int) -> dict[str, Tensor]:
    batch, total = base_ids.shape[0], base_ids[0].numel()
    active = residual_mask.reshape(batch, -1).sum(-1).float()
    mask_bits = torch.stack([combination_bits(total, int(item.item()), base_ids.device) for item in active])
    base_bits = torch.full_like(active, total * math.log2(base_codebook_size))
    residual_bits = active * math.log2(residual_codebook_size)
    total_bits = base_bits + residual_bits + mask_bits
    return {"base_bits_per_image": base_bits.mean(), "residual_code_bits_per_image": residual_bits.mean(),
            "mask_combinatorial_bits_per_image": mask_bits.mean(), "total_bits_per_image": total_bits.mean(),
            "bits_per_pixel": total_bits.mean() / (total * 1.0), "active_ratio": active.mean() / total,
            "dense_bitmap_mask_bits_per_image": torch.full_like(active, float(total)).mean()}
