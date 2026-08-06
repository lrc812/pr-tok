#!/usr/bin/env python
"""Save input/base/score/mask/delta/final/error panels for a PAFR-VQ batch."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from pafr_vq.experiment import load_config, device_from_config, make_loader, make_tokenizer

def rgb(tensor: torch.Tensor) -> np.ndarray:
    value=tensor.detach().cpu().clamp(-1,1).add(1).mul(127.5).byte().permute(1,2,0).numpy()
    return value

def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--output",default=None);args=parser.parse_args()
    config=load_config(args.config);device=device_from_config(config["device"]);images=next(iter(make_loader(config))).to(device);model=make_tokenizer(config,device).eval()
    with torch.no_grad():out=model(images,active_ratio=config["selector"]["active_ratio"])
    image,base,final=images[0],out["base_reconstruction"][0],out["reconstruction"][0];score=out["residual_scores"][0];mask=out["residual_mask"][0].float();delta=out["delta_reconstruction"][0].abs();error=(image-final).abs()
    heat=lambda x: (x-x.min()).div((x.max()-x.min()).clamp_min(1e-6)).mul(255).byte().cpu().numpy()
    score=Image.fromarray(heat(score)).resize((image.shape[-1],image.shape[-2]),Image.Resampling.NEAREST).convert("RGB");mask=Image.fromarray((mask.cpu().numpy()*255).astype(np.uint8)).resize((image.shape[-1],image.shape[-2]),Image.Resampling.NEAREST).convert("RGB")
    panels=[Image.fromarray(rgb(item)) for item in (image,base)]+[score,mask]+[Image.fromarray(rgb(delta*2-1)),Image.fromarray(rgb(final)),Image.fromarray(rgb(error*2-1))]
    canvas=Image.new("RGB",(image.shape[-1]*len(panels),image.shape[-2]));[canvas.paste(panel,(index*image.shape[-1],0)) for index,panel in enumerate(panels)]
    output=Path(args.output or Path(config["train"]["output_dir"])/"pafr_panels.png");output.parent.mkdir(parents=True,exist_ok=True);canvas.save(output);print(output)
if __name__=="__main__":main()
