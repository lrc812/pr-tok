import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
from taming.data.text_image_data import TextImageDataset
class TextImage(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        example = self.data[i]
        return example
    
class TextImageTrain(TextImage):
    def __init__(self, train_csv,size):
        super().__init__()
        self.data = TextImageDataset(csv_path=train_csv,
                                     size=size
                                     )
class TextImageVal(TextImage):
    def __init__(self, val_csv, size):
        super().__init__()
        self.data = TextImageDataset(csv_path=val_csv,
                                     size = size)


