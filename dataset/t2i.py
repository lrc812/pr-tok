import os
import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from PIL import Image 


class Text2ImgDatasetImg(Dataset):
    def __init__(self, lst_dir, face_lst_dir, transform):
        img_path_list = []
        valid_file_path = []
        # collect valid jsonl
        for lst_name in sorted(os.listdir(lst_dir)):
            if not lst_name.endswith('.jsonl'):
                continue
            file_path = os.path.join(lst_dir, lst_name)
            valid_file_path.append(file_path)
        
        # collect valid jsonl for face
        if face_lst_dir is not None:
            for lst_name in sorted(os.listdir(face_lst_dir)):
                if not lst_name.endswith('_face.jsonl'):
                    continue
                file_path = os.path.join(face_lst_dir, lst_name)
                valid_file_path.append(file_path)            
        
        for file_path in valid_file_path:
            with open(file_path, 'r') as file:
                for line_idx, line in enumerate(file):
                    data = json.loads(line)
                    img_path = data['image_path']
                    code_dir = file_path.split('/')[-1].split('.')[0]
                    img_path_list.append((img_path, code_dir, line_idx))
        self.img_path_list = img_path_list
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path, code_dir, code_name = self.img_path_list[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, code_name 


class Text2ImgDataset(Dataset):
    def __init__(self, args, transform):
        # 读取 CSV 文件
        data = pd.read_csv(args.data_path)
        img_path_list = []
        for idx, row in data.iterrows():
            img_path = row['ID']  # 图像路径
            caption = row['Caption']  # 文本描述
            img_path_list.append((img_path, caption, idx))  # 保存行号作为索引
        self.img_path_list = img_path_list
        self.transform = transform

        self.t5_feat_path = args.t5_feat_path
        self.image_size = args.image_size
        self.t5_feature_max_len = 120
        self.t5_feature_dim = 1024
        self.max_seq_length = self.t5_feature_max_len + (self.image_size // args.downsample_size) ** 2
        self.downsample_size = args.downsample_size

    def __len__(self):
        return len(self.img_path_list)

    def dummy_data(self):
        img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim), dtype=torch.float32)
        attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).unsqueeze(0)
        valid = torch.zeros((self.image_size // self.downsample_size) ** 2, dtype=torch.float32)  # 修改为与 seq_len 一致的形状
        return img, t5_feat_padding, attn_mask, valid

    def __getitem__(self, index):
        img_path, caption, row_idx = self.img_path_list[index]
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"[ERROR] Image file not found: {img_path}")
            return self.dummy_data()

        if min(img.size) < self.image_size:
            return self.dummy_data()

        if self.transform is not None:
            img = self.transform(img)

        # 加载 T5 特征
        t5_file = os.path.join(self.t5_feat_path, f"{row_idx}.npy")
        if not os.path.isfile(t5_file):
            print(f"[ERROR] T5 feature file not found: {t5_file}")
            return self.dummy_data()

        try:
            t5_feat = torch.from_numpy(np.load(t5_file))
            t5_feat_len = t5_feat.shape[1]
            feat_len = min(self.t5_feature_max_len, t5_feat_len)
            t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
            t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len]
            emb_mask = torch.zeros((self.t5_feature_max_len,))
            emb_mask[-feat_len:] = 1
            attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length))
            T = self.t5_feature_max_len
            attn_mask[:, :T] = attn_mask[:, :T] * emb_mask.unsqueeze(0)
            eye_matrix = torch.eye(self.max_seq_length, self.max_seq_length)
            attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
            attn_mask = attn_mask.unsqueeze(0).to(torch.bool)
            valid = torch.ones((self.image_size // self.downsample_size) ** 2, dtype=torch.float32)  # 修改为与 seq_len 一致的形状
        except Exception as e:
            print(f"[ERROR] Failed to load T5 feature file: {t5_file}, Error: {e}")
            return self.dummy_data()

        # 将 caption 转换为张量，并固定长度
        caption_tensor = torch.zeros(1024, dtype=torch.long)
        caption_encoded = torch.tensor([ord(c) for c in caption[:1024]], dtype=torch.long)
        caption_tensor[:len(caption_encoded)] = caption_encoded

        return img, t5_feat_padding, attn_mask, valid


class Text2ImgDatasetCode(Dataset):
    def __init__(self, args):
        pass




def build_t2i_image(args, transform):
    return Text2ImgDatasetImg(args.data_path, args.data_face_path, transform)

def build_t2i(args, transform):
    return Text2ImgDataset(args, transform)

def build_t2i_code(args):
    return Text2ImgDatasetCode(args)