import argparse, os, sys, glob, math, time
import torch
import yaml
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from main import instantiate_from_config, DataModuleFromConfig
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import trange
from taming.models.vqgan import VQModel, GumbelVQ
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config
def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  if is_gumbel:
    model = GumbelVQ(**config.model.params)
  else:
    model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()
def g_data(model, device='cuda',config=None,outputfile=None,captionsfile=None):
    model =model.to(device)
    model.eval()
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    codebook_size = config['model']['params']['n_embed']  # 获取 codebook 大小
    print(f"Codebook size: {codebook_size}")
    usage_counts = torch.zeros(codebook_size, dtype=torch.long, device=device)  # 在 GPU 上初始化计数器
    with torch.no_grad():
        with open(outputfile, 'w') as out_f, open(captionsfile, 'w') as cap_f:
            for batch in data.train_dataloader(): 
                image = batch['image']
                images = image.to(device)
                if len(images.shape) == 4 and images.shape[-1] == 3:  # (batch, height, width, channels)
                    images = images.permute(0, 3, 1, 2)  # 转换为 (batch, channels, height, width)           
                # 通过编码器得到 z
                ## shape : 8 256 256 3
                _, _, info = model.encode(images)
                indices = info[-1]  # 获取量化后的索引 (min_encoding_indices)
                # print(f"indices dtype: {indices.dtype}, shape: {indices.shape}")
                # 如果 indices 是 3D (batch, h, w)，扁平化
                # print(f"indices shape: {indices.shape}") 2048?
                print(indices.shape)
                if indices.dim() == 3:
                    indices = indices.view(-1)
                       
            

if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_med = load_config("/home/disk1/lihaoran/vq-gan/taming-transformers/logs/2025-11-17T09-20-40_vqgan_openimage/configs/2025-11-17T09-20-40-project.yaml", display=False)
    model = load_vqgan(config_med, ckpt_path="/home/disk1/lihaoran/vq-gan/taming-transformers/logs/2025-11-17T09-20-40_vqgan_openimage/checkpoints/epoch=000007.ckpt").to(DEVICE)
    usage_counts = compute_codebook_usage(model, device=DEVICE, config=config_med)
   