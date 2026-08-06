import os
import glob
import numpy as np
import torch
from omegaconf import OmegaConf

from main import instantiate_from_config


def _find_project_yaml(log_dir: str) -> str:
    cfg_dir = os.path.join(log_dir, "configs")
    if not os.path.isdir(cfg_dir):
        raise FileNotFoundError(f"configs 目录不存在: {cfg_dir}")

    ymls = sorted(glob.glob(os.path.join(cfg_dir, "*project.yaml")))
    if len(ymls) == 0:
        ymls = sorted(glob.glob(os.path.join(cfg_dir, "*project*.yaml")))
    if len(ymls) == 0:
        raise FileNotFoundError(f"在 {cfg_dir} 中找不到 *project.yaml 配置文件")

    return ymls[-1]


def _find_ckpt(log_dir: str) -> str:
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"checkpoints 目录不存在: {ckpt_dir}")

    last = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.isfile(last):
        return last

    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"在 {ckpt_dir} 中找不到任何 .ckpt 文件")
    return ckpts[-1]


def load_model_from_logdir(log_dir: str, device: str = "cuda"):
    project_yaml = _find_project_yaml(log_dir)
    ckpt_path = _find_ckpt(log_dir)

    cfg = OmegaConf.load(project_yaml)
    model = instantiate_from_config(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)

    print(f"[load] config: {project_yaml}")
    print(f"[load] ckpt:   {ckpt_path}")
    if len(missing) > 0:
        print(f"[load] missing keys (show up to 20): {missing[:20]}")
    if len(unexpected) > 0:
        print(f"[load] unexpected keys (show up to 20): {unexpected[:20]}")

    model.eval().to(device)
    return model, cfg