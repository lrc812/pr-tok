from __future__ import annotations
import torch
from torch import Tensor,nn
class MaskPrior(nn.Module):
    """Bidirectional base-token encoder; residual masks do not require raster causality."""
    def __init__(self,base_codebook_size:int,model_dim:int=256,depth:int=4,heads:int=8,dropout:float=.1,max_positions:int=4096)->None:
        super().__init__(); self.token_embedding=nn.Embedding(base_codebook_size,model_dim); self.position_embedding=nn.Embedding(max_positions,model_dim); layer=nn.TransformerEncoderLayer(model_dim,heads,model_dim*4,dropout,batch_first=True,norm_first=True); self.encoder=nn.TransformerEncoder(layer,depth); self.head=nn.Linear(model_dim,1)
    def forward(self,base_token_ids:Tensor,condition:Tensor|None=None)->Tensor:
        b,h,w=base_token_ids.shape; pos=torch.arange(h*w,device=base_token_ids.device); x=self.token_embedding(base_token_ids.reshape(b,-1))+self.position_embedding(pos); x=x+condition.unsqueeze(1) if condition is not None else x
        return self.head(self.encoder(x)).squeeze(-1).view(b,h,w)
    @torch.no_grad()
    def infer_mask(self,base_token_ids:Tensor,budget:int|None=None,threshold:float=0.)->Tensor:
        logits=self(base_token_ids); flat=logits.flatten(1)
        if budget is None:return logits>threshold
        mask=torch.zeros_like(flat,dtype=torch.bool);mask.scatter_(1,flat.topk(budget,dim=1).indices,True);return mask.view_as(logits)
