#!/usr/bin/env python
"""Compare oracle allocation with a MaskPrior-predicted fixed-budget mask."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from pafr_vq.experiment import load_config, device_from_config, make_loader, make_tokenizer, write_json
from pafr_vq.models.mask_prior import MaskPrior
from pafr_vq.losses.predictability import mask_prediction_metrics

def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--budget",type=int);args=parser.parse_args()
    config=load_config(args.config);device=device_from_config(config["device"]);tokenizer=make_tokenizer(config,device).eval();images=next(iter(make_loader(config))).to(device)
    with torch.no_grad():
        oracle=tokenizer(images,active_ratio=config["selector"]["active_ratio"],budget=args.budget)
        budget=int(oracle["residual_mask"][0].sum());prior=MaskPrior(tokenizer.base.codebook_size,**config["prior"]).to(device).eval();logits=prior(oracle["base_token_ids"]);predicted=prior.infer_mask(oracle["base_token_ids"],budget=budget)
    metrics=mask_prediction_metrics(logits,oracle["residual_mask"]);metrics["predicted_active_ratio"]=predicted.float().mean();metrics["oracle_active_ratio"]=oracle["residual_mask"].float().mean()
    write_json(Path(config["train"]["output_dir"])/"mask_sample_metrics.json",metrics);print({key:float(value) for key,value in metrics.items()})
if __name__=="__main__":main()
