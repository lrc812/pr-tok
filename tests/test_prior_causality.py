import torch
from pafr_vq.models.residual_prior import ResidualARPrior
def test_future_teacher_tokens_do_not_change_past_logits():
 torch.manual_seed(0);model=ResidualARPrior(16,8,model_dim=16,depth=1,heads=4,dropout=0.);model.eval();base=torch.randint(0,16,(1,2,2));pos=torch.tensor([[0,1,2,3]]);mask=torch.ones(1,4,dtype=torch.bool);a=torch.tensor([[1,2,3,4]]);b=a.clone();b[0,3]=7
 assert torch.allclose(model(base,pos,a,mask)[:,:3],model(base,pos,b,mask)[:,:3])
