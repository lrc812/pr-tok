#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from pafr_vq.experiment import load_config,device_from_config,make_loader,make_tokenizer,write_json
from pafr_vq.models.mask_prior import MaskPrior
from pafr_vq.models.residual_prior import ResidualARPrior
from pafr_vq.losses.predictability import mask_prediction_metrics,masked_code_nll
from pafr_vq.utils.packing import pack_active_tokens
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--max-steps",type=int);args=p.parse_args();cfg=load_config(args.config);device=device_from_config(cfg["device"]);tokenizer=make_tokenizer(cfg,device).eval();prior_cfg=cfg["prior"];mask_prior=MaskPrior(tokenizer.base.codebook_size,**prior_cfg).to(device);code_prior=ResidualARPrior(tokenizer.base.codebook_size,tokenizer.quantizer.codebook_size,**prior_cfg).to(device);opt=torch.optim.AdamW(list(mask_prior.parameters())+list(code_prior.parameters()),lr=cfg["train"]["learning_rate"]);last={}
 for step,images in zip(range(args.max_steps or cfg["train"]["max_steps"]),make_loader(cfg)):
  with torch.no_grad():out=tokenizer(images.to(device),active_ratio=cfg["selector"]["active_ratio"]);packed=pack_active_tokens(out["residual_token_ids"],out["residual_mask"])
  logits=mask_prior(out["base_token_ids"]);metrics=mask_prediction_metrics(logits,out["residual_mask"]);code_logits=code_prior(out["base_token_ids"],packed.positions,packed.ids,packed.attention_mask);nll=masked_code_nll(code_logits,packed.ids,packed.attention_mask);loss=metrics["mask_bce"]+nll
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(mask_prior.parameters())+list(code_prior.parameters()),cfg["train"]["grad_clip"]);opt.step();last={**metrics,"residual_nll":nll,"total":loss,"step":step}
 output=Path(cfg["train"]["output_dir"]);output.mkdir(parents=True,exist_ok=True);torch.save({"mask_prior":mask_prior.state_dict(),"residual_prior":code_prior.state_dict(),"config":cfg},output/"priors.pt");write_json(output/"prior_metrics.json",last);print("completed",step+1,"steps")
if __name__=="__main__":main()
