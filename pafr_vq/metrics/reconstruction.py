from torch import Tensor
import torch
import torch.nn.functional as F
def reconstruction_metrics(target:Tensor,reconstruction:Tensor)->dict[str,Tensor]:
    mse=F.mse_loss(reconstruction,target);return {"l1":F.l1_loss(reconstruction,target),"mse":mse,"psnr":-10*torch.log10(mse.clamp_min(1e-10))}
