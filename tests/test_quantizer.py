import torch
from pafr_vq.models.residual_quantizer import ResidualVectorQuantizer
def test_quantizer_active_only_and_empty_mask():
 q=ResidualVectorQuantizer(8,4);z=torch.randn(2,4,3,3,requires_grad=True);mask=torch.zeros(2,3,3,dtype=torch.bool);out=q(z,mask);assert torch.isfinite(out["commitment_loss"]) and (out["hard_ids"]==8).all();mask[:,0,0]=True;out=q(z,mask);assert (out["hard_ids"]!=8).sum()==2;assert torch.allclose(out["soft_probs"][mask].sum(-1),torch.ones(2));(out["commitment_loss"]+out["codebook_loss"]).backward();assert q.codebook.weight.grad is not None
