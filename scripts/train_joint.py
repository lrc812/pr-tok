#!/usr/bin/env python
"""Minimal Phase-3 soft predictability post-training with a frozen residual prior."""
from __future__ import annotations
import argparse,torch
from pafr_vq.experiment import load_config,device_from_config,make_loader,make_tokenizer
from pafr_vq.models.residual_prior import ResidualARPrior
from pafr_vq.utils.packing import pack_active_tokens
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--max-steps",type=int,default=10);p.add_argument("--predictability-weight",type=float,default=.1);args=p.parse_args();cfg=load_config(args.config);device=device_from_config(cfg["device"]);model=make_tokenizer(cfg,device);pc=cfg["prior"];prior=ResidualARPrior(model.base.codebook_size,model.quantizer.codebook_size,**pc).to(device).eval()
 for param in prior.parameters():param.requires_grad_(False)
 opt=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=cfg["train"]["learning_rate"])
 for step,images in zip(range(args.max_steps),make_loader(cfg)):
  out=model(images.to(device),active_ratio=cfg["selector"]["active_ratio"]);packed=pack_active_tokens(out["residual_token_ids"],out["residual_mask"]);logits=prior(out["base_token_ids"],packed.positions,packed.ids,packed.attention_mask);prob=out["quantizer"]["soft_probs"].reshape(images.shape[0],-1,model.quantizer.codebook_size);target=torch.gather(prob,1,packed.positions.unsqueeze(-1).expand(-1,-1,model.quantizer.codebook_size));soft=-(target*logits.log_softmax(-1)).sum(-1);soft=(soft*packed.attention_mask).sum()/packed.attention_mask.sum().clamp_min(1);loss=out["losses"]["total"]+args.predictability_weight*soft
  if not torch.isfinite(loss):raise FloatingPointError("non-finite joint loss")
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["train"]["grad_clip"]);opt.step();print({"step":step,"loss":float(loss.detach()),"soft_predictability":float(soft.detach())})
if __name__=="__main__":main()
