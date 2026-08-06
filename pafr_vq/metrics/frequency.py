from torch import Tensor
import torch
from pafr_vq.allocation.haar import HaarDWT2D
def frequency_metrics(target:Tensor,reconstruction:Tensor)->dict[str,Tensor]:
    dwt=HaarDWT2D().to(target);a,b=dwt(target),dwt(reconstruction);names=("ll","lh","hl","hh");result={f"dwt_{n}_l1":(x-y).abs().mean() for n,x,y in zip(names,a,b)}
    result["sobel_l1"]=(target[...,1:,:]-target[...,:-1,:]).sub(reconstruction[...,1:,:]-reconstruction[...,:-1,:]).abs().mean();return result
