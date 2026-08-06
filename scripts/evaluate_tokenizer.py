#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from pafr_vq.experiment import load_config,device_from_config,make_loader,make_tokenizer,write_json
from pafr_vq.metrics import reconstruction_metrics,frequency_metrics,codebook_metrics
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--max-batches",type=int,default=2);args=p.parse_args();cfg=load_config(args.config);device=device_from_config(cfg["device"]);model=make_tokenizer(cfg,device).eval();summary={};count=0
 with torch.no_grad():
  for images in make_loader(cfg):
   out=model(images.to(device),active_ratio=cfg["selector"]["active_ratio"]);values={**reconstruction_metrics(images.to(device),out["reconstruction"]),**frequency_metrics(images.to(device),out["reconstruction"]),**codebook_metrics(out["residual_token_ids"],model.quantizer.codebook_size,model.quantizer.null_code_id),**out["metrics"]}
   for key,value in values.items():summary[key]=summary.get(key,0.)+float(value)
   count+=1
   if count>=args.max_batches:break
 summary={key:value/count for key,value in summary.items()};write_json(Path(cfg["train"]["output_dir"])/"evaluation.json",summary);print(summary)
if __name__=="__main__":main()
