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
from taming.data.utils import custom_collate
import warnings
import subprocess
import pandas as pd
from tqdm import tqdm
if __name__ == "__main__":
    csv_path = "/home/disk1/lihaoran/vq-gan/taming-transformers/data/roco/1/rocov2/train_captions.csv"
    data = pd.read_csv(csv_path)
    image_dir="/home/disk1/lihaoran/vq-gan/taming-transformers/data/roco/1/rocov2/train_images/train"
    num_gray,num_color=0,0
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        image_id = row["ID"]
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        image=Image.open(image_path)
        # 判断是否为灰度图
        if image.mode == "L":
            num_gray+=1
            image_array = np.array(image)
            assert len(image_array.shape) == 2
        else:
            num_color+=1
    print(f"Gray images: {num_gray}, Color images: {num_color}")
    print("Finished!")