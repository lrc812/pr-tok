from __future__ import annotations
import torch
from torch import Tensor
from torch.utils.data import Dataset

class SyntheticImages(Dataset[Tensor]):
    """Deterministic small images containing gradients, edges, and checkerboards."""
    def __init__(self, length: int=32, image_size: int=32, seed: int=42) -> None: self.length,self.image_size,self.seed=length,image_size,seed
    def __len__(self)->int:return self.length
    def __getitem__(self,index:int)->Tensor:
        g=torch.Generator().manual_seed(self.seed+index); size=self.image_size; yy,xx=torch.meshgrid(torch.linspace(-1,1,size),torch.linspace(-1,1,size),indexing="ij")
        base=torch.stack([xx,yy,torch.sin((index%7+2)*xx*3.14)*torch.cos((index%5+2)*yy*3.14)]); noise=torch.rand((3,size,size),generator=g)*.12-.06
        edge=((xx*(index%3+1)+yy*(index%4+1))>0).float().unsqueeze(0)
        return (base*.55+noise+edge*.25).clamp(-1,1)
