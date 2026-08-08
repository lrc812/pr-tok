from torch import Tensor, nn
import torch.nn.functional as F

class ResidualDecoder(nn.Module):
    """Small deterministic CNN that predicts an additive image residual."""
    def __init__(self, base_dim: int, residual_dim: int, out_channels: int = 3, hidden_dim: int = 96, zero_init_output: bool = True) -> None:
        super().__init__(); self.base_projection, self.residual_projection = nn.Conv2d(base_dim, hidden_dim, 1), nn.Conv2d(residual_dim, hidden_dim, 1)
        self.blocks = nn.Sequential(nn.SiLU(), nn.Conv2d(hidden_dim,hidden_dim,3,padding=1),nn.SiLU(),nn.Conv2d(hidden_dim,hidden_dim,3,padding=1),nn.SiLU()); self.output = nn.Conv2d(hidden_dim,out_channels,3,padding=1)
        if zero_init_output: nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
    @property
    def parameter_count(self) -> int: return sum(p.numel() for p in self.parameters())
    def forward(self, base_quantized_features: Tensor, residual_quantized_features: Tensor, mask: Tensor, output_size: tuple[int,int]) -> Tensor:
        features = self.base_projection(base_quantized_features) + self.residual_projection(residual_quantized_features * mask.unsqueeze(1).to(residual_quantized_features.dtype))
        return F.interpolate(self.output(self.blocks(features)), size=output_size, mode="bilinear", align_corners=False)
