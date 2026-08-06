import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
from text_indice_data import TextImageDataset
class Textindex(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        example = self.data[i]
        return example
    
class TextindexTrain(Textindex):
    def __init__(self, train_csv, text_model_name, max_seq_length, max_image_length):
        super().__init__()
        self.data = TextImageDataset(csv_path=train_csv,
                                     text_model_name=text_model_name,
                                     max_seq_length=max_seq_length,
                                     max_image_tokens=max_image_length)
class TextindexVal(Textindex):
    def __init__(self, val_csv, text_model_name, max_seq_length, max_image_length):
        super().__init__()
        self.data = TextImageDataset(csv_path=val_csv,
                                     text_model_name=text_model_name,
                                     max_seq_length=max_seq_length,
                                     max_image_tokens=max_image_length)


