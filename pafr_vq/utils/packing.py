from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class PackedTokens:
    ids: Tensor
    positions: Tensor
    attention_mask: Tensor
    latent_hw: tuple[int, int]

def pack_active_tokens(dense_ids: Tensor, mask: Tensor) -> PackedTokens:
    if dense_ids.shape != mask.shape or dense_ids.ndim != 3: raise ValueError("dense_ids and mask must be [B,H,W]")
    b,h,w=dense_ids.shape; flat_ids,flat_mask=dense_ids.reshape(b,-1),mask.reshape(b,-1).bool(); maximum=int(flat_mask.sum(-1).max().item()) if b else 0
    ids=torch.zeros(b,maximum,dtype=dense_ids.dtype,device=dense_ids.device); positions=torch.zeros(b,maximum,dtype=torch.long,device=dense_ids.device); attention=torch.zeros(b,maximum,dtype=torch.bool,device=dense_ids.device)
    for index in range(b):
        pos=torch.nonzero(flat_mask[index],as_tuple=False).flatten(); length=pos.numel(); positions[index,:length],ids[index,:length],attention[index,:length]=pos,flat_ids[index,pos],True
    return PackedTokens(ids,positions,attention,(h,w))

def unpack_active_tokens(packed: PackedTokens, mask: Tensor, null_id: int) -> Tensor:
    if mask.ndim != 3 or mask.shape[0] != packed.ids.shape[0] or tuple(mask.shape[-2:]) != packed.latent_hw: raise ValueError("mask incompatible with packed tokens")
    output=torch.full(mask.shape,null_id,dtype=packed.ids.dtype,device=packed.ids.device); flat=output.reshape(output.shape[0],-1)
    for index in range(output.shape[0]):
        valid=packed.attention_mask[index]; flat[index,packed.positions[index,valid]]=packed.ids[index,valid]
    return output
