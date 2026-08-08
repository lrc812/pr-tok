import torch
from pafr_vq import TinyBaseTokenizer,PAFRTokenizer
def test_frozen_base_and_zero_init_decoder():
 base=TinyBaseTokenizer(token_dim=8,codebook_size=16);model=PAFRTokenizer(base,residual_dim=8,residual_codebook_size=16);x=torch.randn(2,3,32,32);out=model(x,budget=4);assert torch.allclose(out["reconstruction"],out["base_reconstruction"],atol=1e-6);out["losses"]["total"].backward();assert all(p.grad is None for p in base.parameters())
