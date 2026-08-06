from torch import Tensor, nn
import torch.nn.functional as F

class ResidualEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, residual_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(in_channels, hidden_dim, 3, padding=1), nn.SiLU(), nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.SiLU(), nn.Conv2d(hidden_dim, residual_dim, 1))
    def forward(self, residual: Tensor, latent_hw: tuple[int, int]) -> Tensor:
        return F.adaptive_avg_pool2d(self.net(residual), latent_hw)
