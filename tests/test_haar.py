import torch
from pafr_vq.allocation import HaarDWT2D
def test_haar_constant_high_frequency_and_gradients():
 x=torch.ones(2,3,7,9,requires_grad=True);ll,lh,hl,hh=HaarDWT2D()(x);assert ll.shape[-2:]==(4,5);assert max(float(v.abs().max()) for v in (lh,hl,hh)) < 1e-6;(ll.square().mean()+lh.square().mean()).backward();assert x.grad is not None
