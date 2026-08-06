from __future__ import annotations
import torch
from torch import Tensor,nn
class ResidualARPrior(nn.Module):
    def __init__(self,base_codebook_size:int,residual_codebook_size:int,model_dim:int=256,depth:int=4,heads:int=8,dropout:float=.1,max_positions:int=4096)->None:
        super().__init__();self.residual_codebook_size=residual_codebook_size;self.base_embed=nn.Embedding(base_codebook_size,model_dim);self.code_embed=nn.Embedding(residual_codebook_size+1,model_dim);self.pos_embed=nn.Embedding(max_positions,model_dim);layer=nn.TransformerEncoderLayer(model_dim,heads,model_dim*4,dropout,batch_first=True,norm_first=True);self.base_encoder=nn.TransformerEncoder(layer,depth);self.residual_encoder=nn.TransformerEncoder(layer,depth);self.head=nn.Linear(model_dim,residual_codebook_size)
    def forward(self,base_token_ids:Tensor,positions:Tensor,residual_ids:Tensor,attention_mask:Tensor,condition:Tensor|None=None)->Tensor:
        b=base_token_ids.shape[0];base=self.base_encoder(self.base_embed(base_token_ids.reshape(b,-1))).mean(1,keepdim=True);bos=torch.full((b,1),self.residual_codebook_size,dtype=torch.long,device=base.device);teacher=torch.cat([bos,residual_ids[:,:-1]],1);x=self.code_embed(teacher)+self.pos_embed(positions)+base;x=x+condition.unsqueeze(1) if condition is not None else x;length=x.shape[1];causal=torch.triu(torch.ones(length,length,device=x.device,dtype=torch.bool),diagonal=1);return self.head(self.residual_encoder(x,mask=causal,src_key_padding_mask=~attention_mask.bool()))
