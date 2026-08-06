import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
import logging
from taming.models.vqgan import VQModel, GumbelVQ
# 确保 custom_collate 正确导入（用于处理字典类型batch）
from taming.data.utils import custom_collate
import warnings
# 导入你定义的模型（如果主函数和模型不在同一文件，需调整导入路径）
from new_translation import VQGANTextTranslator 
from torchvision import transforms
import albumentations
import argparse
from main import get_obj_from_str, get_parser, nondefault_trainer_args, instantiate_from_config, SetupCallback
def preprocess_image(image_path):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    rescaler = albumentations.SmallestMaxSize(max_size=256)
    cropper = albumentations.CenterCrop(256, 256)
    preprocessor = albumentations.Compose([rescaler, cropper])
    image = preprocessor(image=image)["image"]
    image = (image/127.5 - 1.0).astype(np.float32)
    return image
def load_vqgan(config_path, ckpt_path=None):
    config=OmegaConf.load(config_path)
    model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()
def load_trans(config_path, ckpt_path=None):
    config=OmegaConf.load(config_path)
    model = VQGANTextTranslator(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()
def get_image_transform(image_size=256):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 将像素值归一化到 [-1, 1]
    ])
def generate_image_from_indices(indices, vqgan_model):
    vqgan_model.eval()
    with torch.no_grad():
        quantized_vectors = vqgan_model.quantize.embed_code(indices)
        generated_image = vqgan_model.decode(quantized_vectors)
    return generated_image
@torch.no_grad()
def reconstruct(text, trans_model, vqgan_model):
    if text is None:
        return None
    indice = trans_model.generate_indices(text, 256)
    print("Generated Indices Shape:", len(indice))

    indices_tensor = torch.tensor(indice, dtype=torch.long)
    # 调整形状为 (16, 16)
    indices_tensor = indices_tensor.view(16, 16)
    print("count nonzero",indices_tensor.count_nonzero())
    print("Reshaped Indices Tensor Shape:", indices_tensor.shape)
    generated_image = generate_image_from_indices(indices_tensor, vqgan_model)
    print("generated image shape",generated_image.shape)
    generated_image=torch.clamp(generated_image, -1.0, 1.0)
    generated_image=(generated_image + 1.0) / 2.0
    if(len(generated_image.shape)==4):
        generated_image=generated_image.squeeze(0)
    pil_image = Image.fromarray((generated_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    pil_image.save(f"generated_image.png")
    return generated_image
@torch.no_grad()
def test_reconstruct(image_path,vqgan_model):
    # indice = trans_model.generate_indices(text, 256)
    # print("Generated Indices Shape:", len(indice))
    image = Image.open(image_path).convert("RGB")
    image_transform = get_image_transform()
    image_tensor = image_transform(image).unsqueeze(0)# [1, 3, 256, 256]
        # 使用 VQ-GAN 编码图像
    with torch.no_grad():
        _, _, info = vqgan_model.encode(image_tensor)
        # print(info[-1].shape)
        indice = info[-1].view(-1).cpu().numpy().tolist()
    indices_tensor = torch.tensor(indice, dtype=torch.long)
    # 调整形状为 (16, 16)
    indices_tensor = indices_tensor.view(16, 16)
    print("Reshaped Indices Tensor Shape:", indices_tensor.shape)
    generated_image = generate_image_from_indices(indices_tensor, vqgan_model)
    print("generated image shape",generated_image.shape)
    generated_image=torch.clamp(generated_image, -1.0, 1.0)
    generated_image=(generated_image + 1.0) / 2.0
    if(len(generated_image.shape)==4):
        generated_image=generated_image.squeeze(0)
    pil_image = Image.fromarray((generated_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    pil_image.save(f"test_construct_image.png")
    return generated_image
@torch.no_grad()
def test_vqgan(image_path,vqgan_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image=preprocess_image(image_path)
    # print(type(image)) #ndarray
    # print("preprocess image shape",image.shape) # (256, 256, 3)
    vqgan_model = vqgan_model.to(device)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    print("image tensor shape",image_tensor.shape) # [1, 3, 256, 256]
    reconstruct_image,_=vqgan_model(image_tensor)
    print("reconstruct image shape",reconstruct_image.shape)
    reconstruct_image=torch.clamp(reconstruct_image, -1.0, 1.0)
    reconstruct_image=(reconstruct_image + 1.0) / 2.0
    im=Image.open(image_path)
    im = np.array(im).astype(np.uint8)
    print("original image tensor shape",im.shape)
    pil_image = Image.fromarray((reconstruct_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    if pil_image.mode != "RGB":
        print("Converting image to grayscale")
        pil_image = pil_image.convert("L")
    pil_image.save(f"test_reconstruct_image.png")

    return reconstruct_image
if __name__ == "__main__":
   
   parser = argparse.ArgumentParser(description="Test VQGAN on a single image.")
   parser.add_argument("--image_path", type=str, default="/home/disk1/lihaoran/vq-gan/taming-transformers/data/roco/1/rocov2/valid_images/valid/ROCOv2_2023_valid_000014.jpg", help="Path to the input image.")
   args = parser.parse_args()
   vqgan_config_path = "/home/disk1/lihaoran/vq-gan/taming-transformers/logs/2025-11-14T05-48-50_custom_vqgan/configs/2025-11-14T05-48-50-project.yaml"
   vq_gan_ckpt_path = "/home/disk1/lihaoran/vq-gan/taming-transformers/logs/2025-11-14T05-48-50_custom_vqgan/checkpoints/last.ckpt"
   trans_config_path= "/home/disk1/lihaoran/vq-gan/taming-transformers/logs_translator/2025-12-03T16-54-09_train_translator/configs/2025-12-03T16-54-09-project.yaml"
   trans_ckpt_path= "/home/disk1/lihaoran/vq-gan/taming-transformers/logs_translator/2025-12-03T16-54-09_train_translator/checkpoints/last.ckpt"
   vggan_model=load_vqgan(vqgan_config_path, vq_gan_ckpt_path)
   print(vggan_model.quantize.n_e)

   trans_model=load_trans(trans_config_path, trans_ckpt_path)
   text=""
   reconstruct(text,trans_model,vggan_model)
   test_reconstruct(args.image_path,vggan_model)
   test_vqgan(args.image_path,vggan_model)