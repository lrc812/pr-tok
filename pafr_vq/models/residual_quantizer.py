from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class ResidualVectorQuantizer(nn.Module):
    """VQ whose codebook receives gradients only from selected locations."""
    def __init__(self, codebook_size: int = 1024, dim: int = 64, commitment_weight: float = .25, soft_temperature: float = 1.) -> None:
        super().__init__(); self.codebook_size, self.dim, self.null_code_id = codebook_size, dim, codebook_size; self.commitment_weight, self.soft_temperature = commitment_weight, soft_temperature
        self.codebook = nn.Embedding(codebook_size, dim); nn.init.normal_(self.codebook.weight, std=dim ** -.5)
    def forward(self, latents: Tensor, active_mask: Tensor) -> dict[str, Tensor]:
        if latents.ndim != 4 or active_mask.shape != latents.shape[:1] + latents.shape[2:]: raise ValueError("latents [B,C,H,W] and active_mask [B,H,W] required")
        b, _, h, w = latents.shape; flat, active = latents.permute(0, 2, 3, 1).reshape(-1, self.dim), active_mask.reshape(-1).bool()
        ids = torch.full((flat.shape[0],), self.null_code_id, dtype=torch.long, device=latents.device); quantized = torch.zeros_like(flat); logits = flat.new_zeros(flat.shape[0], self.codebook_size); probs = logits.clone()
        if active.any():
            vectors = flat[active]; distance = vectors.square().sum(-1, keepdim=True) + self.codebook.weight.square().sum(-1) - 2 * vectors @ self.codebook.weight.t(); active_logits = (-distance / max(self.soft_temperature, 1e-6)).to(flat.dtype)
            active_ids = distance.argmin(-1); code = self.codebook(active_ids).to(vectors.dtype); ids[active] = active_ids; quantized[active] = (vectors + (code - vectors).detach()).to(quantized.dtype); logits[active] = active_logits.to(logits.dtype); probs[active] = active_logits.softmax(-1).to(probs.dtype)
            commitment, codebook_loss = F.mse_loss(vectors, code.detach()) * self.commitment_weight, F.mse_loss(code, vectors.detach()); counts = torch.bincount(active_ids, minlength=self.codebook_size).float()
        else:
            commitment = codebook_loss = latents.new_zeros(()); counts = latents.new_zeros(self.codebook_size)
        usage = counts / counts.sum().clamp_min(1); entropy = -(usage * usage.clamp_min(1e-12).log()).sum()
        return {"hard_ids": ids.view(b, h, w), "quantized": quantized.view(b, h, w, self.dim).permute(0, 3, 1, 2).contiguous(), "soft_probs": probs.view(b,h,w,self.codebook_size), "distance_logits": logits.view(b,h,w,self.codebook_size), "commitment_loss": commitment, "codebook_loss": codebook_loss, "usage": usage, "usage_entropy": entropy, "perplexity": entropy.exp(), "active_code_utilization": (counts > 0).float().mean(), "dead_codes": (counts == 0).sum()}
