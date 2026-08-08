from torch import Tensor
import torch
def codebook_metrics(ids:Tensor,codebook_size:int,null_id:int|None=None)->dict[str,Tensor]:
    flat=ids.reshape(-1);flat=flat[flat != null_id] if null_id is not None else flat;counts=torch.bincount(flat,minlength=codebook_size).float() if flat.numel() else torch.zeros(codebook_size,device=ids.device);p=counts/counts.sum().clamp_min(1);entropy=-(p*p.clamp_min(1e-12).log()).sum();return {"utilization":(counts>0).float().mean(),"perplexity":entropy.exp(),"usage_entropy":entropy,"dead_codes":(counts==0).sum()}
