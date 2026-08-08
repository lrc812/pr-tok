import torch
from pafr_vq.allocation import ResidualScorer
def test_all_scorers_are_finite_and_latent_shaped():
 x=torch.ones(2,3,32,32);recon=x.clone()
 for name in ("pixel","sobel","haar_dwt","hybrid","random"):
  score=ResidualScorer(name)(x,recon,(4,4));assert score.shape==(2,4,4);assert torch.isfinite(score).all()
