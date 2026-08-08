import torch
from pafr_vq import TinyBaseTokenizer,PAFRTokenizer
from pafr_vq.utils.checkpoint import save_checkpoint,load_checkpoint
def test_smoke_training_and_checkpoint(tmp_path):
 model=PAFRTokenizer(TinyBaseTokenizer(token_dim=8,codebook_size=16),residual_dim=8,residual_codebook_size=16);opt=torch.optim.Adam(model.parameters(),1e-3);x=torch.randn(2,3,32,32)
 for _ in range(2):out=model(x,budget=4);opt.zero_grad();out["losses"]["total"].backward();opt.step();assert torch.isfinite(out["losses"]["total"])
 path=tmp_path/"state.pt";save_checkpoint(path,model,opt,2,{});clone=PAFRTokenizer(TinyBaseTokenizer(token_dim=8,codebook_size=16),residual_dim=8,residual_codebook_size=16);load_checkpoint(path,clone);assert torch.equal(next(model.parameters()),next(clone.parameters()))
