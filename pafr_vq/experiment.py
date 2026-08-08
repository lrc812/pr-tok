from __future__ import annotations
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from pafr_vq import TinyBaseTokenizer, PAFRTokenizer
from pafr_vq.data import SyntheticImages, ImageFolderDataset

DEFAULT={"seed":42,"device":"auto","image_size":32,"data":{"type":"synthetic","root":None,"batch_size":4,"num_workers":0},"base_tokenizer":{"type":"tiny","checkpoint":None,"freeze":True,"compression_factor":4},"scorer":{"type":"haar_dwt","pixel_weight":1.,"sobel_weight":1.,"dwt_weight":1.},"selector":{"active_ratio":.25,"budget":None},"residual":{"dim":16,"codebook_size":32,"commitment_weight":.25,"soft_temperature":1.},"loss":{"l1":1.,"frequency":1.,"gradient":.5,"vq":1.},"prior":{"model_dim":64,"depth":2,"heads":4,"dropout":.0},"train":{"max_steps":10,"learning_rate":1e-3,"grad_clip":1.,"output_dir":"outputs/pafr_vq"}}
def _merge(a:dict,b:dict)->dict:
    out=dict(a)
    for key,value in b.items():out[key]=_merge(a.get(key,{}),value) if isinstance(value,dict) else value
    return out
def load_config(path:str|None)->dict:
    if path is None:return DEFAULT
    try:
        import yaml
    except ImportError as exc:raise RuntimeError("PyYAML is required for YAML configs; install project dependencies") from exc
    with open(path) as handle:return _merge(DEFAULT,yaml.safe_load(handle) or {})
def device_from_config(value:str)->torch.device:return torch.device("cuda" if value=="auto" and torch.cuda.is_available() else ("cpu" if value=="auto" else value))
def make_loader(config:dict)->DataLoader:
    data=config["data"];dataset=SyntheticImages(32,config["image_size"],config["seed"]) if data["type"]=="synthetic" else ImageFolderDataset(data["root"],config["image_size"])
    return DataLoader(dataset,batch_size=data["batch_size"],shuffle=True,num_workers=data["num_workers"])
def make_tokenizer(config:dict,device:torch.device)->PAFRTokenizer:
    base_cfg,residual,scorer=config["base_tokenizer"],config["residual"],config["scorer"]
    if base_cfg["type"] != "tiny": raise RuntimeError("Use VQGANAdapter.from_checkpoint in a custom experiment; bundled scripts intentionally default to tiny/no-network mode.")
    base=TinyBaseTokenizer(token_dim=residual["dim"],codebook_size=residual["codebook_size"],compression_factor=base_cfg["compression_factor"])
    from pafr_vq.allocation import ResidualScorer
    return PAFRTokenizer(base,residual["dim"],residual["codebook_size"],ResidualScorer(scorer["type"],scorer["pixel_weight"],scorer["sobel_weight"],scorer["dwt_weight"]),freeze_base=base_cfg["freeze"],loss_weights=config["loss"]).to(device)
def write_json(path:str|Path,payload:dict)->None:
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps({k:(float(v.detach()) if isinstance(v,torch.Tensor) and v.numel()==1 else v) for k,v in payload.items()},indent=2))
