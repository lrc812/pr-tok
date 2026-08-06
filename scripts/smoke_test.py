#!/usr/bin/env python
"""No-network CPU smoke test for PAFR-VQ forward/backward/checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from pafr_vq.experiment import load_config, device_from_config, make_loader, make_tokenizer, write_json
from pafr_vq.utils import save_checkpoint, load_checkpoint

def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--config");parser.add_argument("--device",default="cpu");parser.add_argument("--steps",type=int,default=3);parser.add_argument("--output-dir",default="outputs/pafr_smoke");args=parser.parse_args()
    config=load_config(args.config);config["device"]=args.device;torch.manual_seed(config["seed"]);device=device_from_config(config["device"]);model=make_tokenizer(config,device);optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=config["train"]["learning_rate"])
    loader=make_loader(config);last={}
    for step,images in zip(range(args.steps),loader):
        images=images.to(device);output=model(images,active_ratio=config["selector"]["active_ratio"],budget=config["selector"]["budget"]);loss=output["losses"]["total"]
        if not torch.isfinite(loss):raise FloatingPointError("non-finite smoke-test loss")
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config["train"]["grad_clip"]);optimizer.step();last={**output["metrics"],**output["losses"],"shape":list(output["reconstruction"].shape),"step":step}
    path=Path(args.output_dir)/"smoke.pt";save_checkpoint(path,model,optimizer,args.steps,config);clone=make_tokenizer(config,device);load_checkpoint(path,clone)
    with torch.no_grad():clone(next(iter(loader)).to(device),active_ratio=config["selector"]["active_ratio"])
    write_json(Path(args.output_dir)/"metrics.json",last);print({key:(float(value.detach()) if isinstance(value,torch.Tensor) and value.numel()==1 else value) for key,value in last.items()})
if __name__=="__main__":main()
