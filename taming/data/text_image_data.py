import torch
import bisect
import numpy as np
import albumentations
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms
from transformers import AutoModel, AutoTokenizer, AutoConfig

class TextImageDataset(Dataset):
    def __init__(self, csv_path,size =256):
        self.data = pd.read_csv(csv_path)
        self.size = size
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.random_crop = False
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"GPT-2 pad_token set to eos_token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs
    def __len__(self):
        return len(self.data)
    
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
    
    def __getitem__(self, idx):
        caption = str(self.data.iloc[idx]['Caption'])
        # 如果caption过长截断
        # words = caption.split()  # 按空格分割成单词列表
        # if len(words) > 110:
        #     caption =' '.join(words[:80])  # 截取前 max_words 个单词并拼接
        text_inputs = self.tokenizer(
            caption,
            padding= 'max_length',
            truncation=True,
            max_length= 120,
            return_tensors='pt'
        )
        input_ids = text_inputs['input_ids'].squeeze(0)
        # print(f"Tokenized length: {len(input_ids)}")  # 调试 token 数量
        return {
            'Caption' : input_ids.long(),
            'image' :   self.preprocess_image(self.data.iloc[idx]['ID'])
        }
    