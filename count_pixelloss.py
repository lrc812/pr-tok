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
def preprocess(img, target_image_size=256, map_dalle=True):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return img
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
def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)
  print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  xrec = model.decode(z)
  return xrec

def pixel_loss(model, device='cuda',config=None):
    model =model.to(device)
    model.eval()
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    codebook_size = config['model']['params']['n_embed']  # 获取 codebook 大小
    print(f"Codebook size: {codebook_size}")
    loss = []
    with torch.no_grad():
        for batch in data.val_dataloader():  # 假设数据集返回 (image, label)
            image = batch['image']
            images = image.to(device)
            # print("input image shape",images.shape)
            if len(images.shape) == 4 and images.shape[-1] == 3:  # (batch, height, width, channels)
                images = images.permute(0, 3, 1, 2)  # 转换为 (batch, channels, height, width)           
            xrec, qloss = model(images)
            # print("reconstruct image shape",xrec.shape)
            rec_loss = torch.abs(images.contiguous() - xrec.contiguous())
            rec_loss = torch.mean(rec_loss.clone())
            loss.append(rec_loss.item())
    avg_loss = np.mean(loss)
    print(avg_loss)
    plt.figure(figsize=(10, 6))
    plt.plot(loss, marker='o', label='Pixel Loss per Batch')
    plt.axhline(avg_loss, color='r', linestyle='--', label=f'Average Loss: {avg_loss:.4f}')
    plt.title('Pixel Loss Visualization')
    plt.xlabel('Batch Index')
    plt.ylabel('Pixel Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('pixel_loss_val_visualization.png')  # 保存图像
    plt.show()  # 显示图像
    return loss

if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_med = load_config("/home/disk1/lihaoran/vq-gan/taming-transformers/logs/2025-11-14T05-48-50_custom_vqgan/configs/2025-11-14T05-48-50-project.yaml", display=False)
    model = load_vqgan(config_med, ckpt_path="/home/disk1/lihaoran/vq-gan/taming-transformers/logs/2025-11-14T05-48-50_custom_vqgan/checkpoints/epoch=000049.ckpt").to(DEVICE)
    pixel_loss(model, device=DEVICE, config=config_med)

  