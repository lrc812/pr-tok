from __future__ import annotations

import math
import torch
from torch import Tensor


class BudgetSelector:
    """Exact per-image hard top-k selection with a reproducible random baseline."""

    def __init__(self, deterministic: bool = False, seed: int | None = None) -> None:
        self.deterministic = deterministic
        self.seed = seed

    @staticmethod
    def _budget(scores: Tensor, active_ratio: float | None, budget: int | None) -> int:
        if (active_ratio is None) == (budget is None):
            raise ValueError("Provide exactly one of active_ratio or budget")
        total = scores.shape[-2] * scores.shape[-1]
        if active_ratio is not None:
            if not 0.0 <= active_ratio <= 1.0:
                raise ValueError("active_ratio must be in [0, 1]")
            budget = int(round(total * active_ratio))
        assert budget is not None
        return max(0, min(int(budget), total))

    def __call__(
        self, scores: Tensor, active_ratio: float | None = None, budget: int | None = None,
        random: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if scores.ndim != 3:
            raise ValueError("scores must have shape [B, H, W]")
        chosen = self._budget(scores, active_ratio, budget)
        batch, height, width = scores.shape
        flat = scores.reshape(batch, -1)
        if chosen == 0:
            indices = torch.empty(batch, 0, dtype=torch.long, device=scores.device)
        elif random:
            generator = None
            if self.deterministic or self.seed is not None:
                generator = torch.Generator(device=scores.device)
                generator.manual_seed(0 if self.seed is None else self.seed)
            noise = torch.rand(flat.shape, device=scores.device, generator=generator)
            indices = noise.topk(chosen, dim=-1).indices
        else:
            # stable=True makes equal scores reproducible across runs/devices.
            indices = torch.argsort(flat, dim=-1, descending=True, stable=True)[:, :chosen]
        mask = torch.zeros_like(flat, dtype=torch.bool)
        if chosen:
            mask.scatter_(1, indices, True)
        return mask.view(batch, height, width), indices
