#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from pafr_vq.experiment import load_config,device_from_config,make_loader,make_tokenizer,write_json
from pafr_vq.utils import save_checkpoint
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--max-steps",type=int);args=p.parse_args();cfg=load_config(args.config);steps=args.max_steps or cfg["train"]["max_steps"];device=device_from_config(cfg["device"]);torch.manual_seed(cfg["seed"]);model=make_tokenizer(cfg,device);opt=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=cfg["train"]["learning_rate"],weight_decay=cfg["train"].get("weight_decay",0.));loader=make_loader(cfg);last={}
 for step,images in zip(range(steps),loader):
  out=model(images.to(device),active_ratio=cfg["selector"]["active_ratio"],budget=cfg["selector"]["budget"]);loss=out["losses"]["total"]
  if not torch.isfinite(loss):raise FloatingPointError(f"non-finite loss at step {step}")
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["train"]["grad_clip"]);opt.step();last={**out["losses"],**out["metrics"],"step":step}
 output=Path(cfg["train"]["output_dir"]);save_checkpoint(output/"residual_tokenizer.pt",model,opt,steps,cfg);write_json(output/"train_metrics.json",last);print("completed",steps,"steps")
if __name__=="__main__":main()
