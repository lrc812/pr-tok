from __future__ import annotations
from torch import Tensor, nn
import torch
from pafr_vq.allocation import ResidualScorer, BudgetSelector, HaarDWT2D
from pafr_vq.losses import reconstruction_losses, bitrate_metrics
from .base_adapter import BaseTokenizerAdapter
from .residual_encoder import ResidualEncoder
from .residual_quantizer import ResidualVectorQuantizer
from .residual_decoder import ResidualDecoder

class PAFRTokenizer(nn.Module):
    def __init__(self, base: BaseTokenizerAdapter, residual_dim: int=64, residual_codebook_size: int=1024, scorer: ResidualScorer|None=None, selector: BudgetSelector|None=None, freeze_base: bool=True, loss_weights: dict[str,float]|None=None) -> None:
        super().__init__(); self.base,self.scorer,self.selector=base,scorer or ResidualScorer(),selector or BudgetSelector(); self.residual_encoder=ResidualEncoder(residual_dim=residual_dim); self.quantizer=ResidualVectorQuantizer(residual_codebook_size,residual_dim); self.residual_decoder=ResidualDecoder(base.token_dim,residual_dim); self.dwt=HaarDWT2D(); self.loss_weights={"l1":1.,"frequency":1.,"gradient":.5,"vq":1.,**(loss_weights or {})}; self.base_frozen=freeze_base
        if freeze_base: self.base.freeze()
    def train(self, mode: bool=True):
        super().train(mode)
        if self.base_frozen: self.base.eval()
        return self
    def forward(self, images: Tensor, scorer_type: str|None=None, active_ratio: float|None=.25, budget: int|None=None, dense: bool=False) -> dict[str,object]:
        with torch.no_grad() if self.base_frozen else torch.enable_grad(): base=self.base.reconstruct(images)
        base_recon,base_features=base["reconstruction"],base["quantized_features"]; latent_hw=tuple(base_features.shape[-2:]); scorer=self.scorer
        if scorer_type is not None and scorer_type != scorer.scorer_type: scorer=ResidualScorer(scorer_type,**scorer.weights).to(images)
        if budget is not None:
            active_ratio = None
        scores=scorer(images,base_recon,latent_hw); mask,indices=self.selector(scores,active_ratio=active_ratio,budget=budget,random=scorer.scorer_type=="random")
        if dense: mask=torch.ones_like(mask); indices=torch.arange(mask[0].numel(),device=mask.device).expand(mask.shape[0],-1)
        residual_latents=self.residual_encoder(images-base_recon,latent_hw); quant=self.quantizer(residual_latents,mask); delta=self.residual_decoder(base_features,quant["quantized"],mask,images.shape[-2:]); reconstruction=(base_recon+delta).clamp(-1,1)
        rec_losses=reconstruction_losses(images,reconstruction,self.dwt); vq=quant["commitment_loss"]+quant["codebook_loss"]; total=sum(self.loss_weights.get(name,0.)*value for name,value in rec_losses.items())+self.loss_weights["vq"]*vq
        rates=bitrate_metrics(base["token_ids"],mask,self.base.codebook_size,self.quantizer.codebook_size); rates["bits_per_pixel"]=rates["total_bits_per_image"]/(images.shape[-1]*images.shape[-2])
        error=(images-base_recon).abs().mean(1); covered=torch.nn.functional.interpolate(mask.unsqueeze(1).float(),size=images.shape[-2:],mode="nearest").squeeze(1)
        metrics={**rates,"residual_energy_capture":(error*covered).sum()/error.sum().clamp_min(1e-8),"decoder_parameters":torch.tensor(float(self.residual_decoder.parameter_count),device=images.device),**{key:quant[key] for key in ("perplexity","usage_entropy","active_code_utilization","dead_codes")}}
        return {"base_token_ids":base["token_ids"],"base_reconstruction":base_recon,"residual_scores":scores,"residual_mask":mask,"selected_indices":indices,"residual_token_ids":quant["hard_ids"],"residual_quantized":quant["quantized"],"delta_reconstruction":delta,"reconstruction":reconstruction,"losses":{**rec_losses,"vq":vq,"total":total},"metrics":metrics,"quantizer":quant}
