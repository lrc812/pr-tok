from __future__ import annotations
from pathlib import Path
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np

class ImageFolderDataset(Dataset[Tensor]):
    def __init__(self, root: str, image_size: int=256) -> None:
        self.paths=[path for path in sorted(Path(root).rglob("*")) if path.suffix.lower() in {".jpg",".jpeg",".png",".webp"}]
        if not self.paths: raise FileNotFoundError(f"No image files found under {root}")
        self.image_size=image_size
    def __len__(self)->int:return len(self.paths)
    def __getitem__(self,index:int)->Tensor:
        image=Image.open(self.paths[index]).convert("RGB").resize((self.image_size,self.image_size),Image.Resampling.BICUBIC)
        return torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float().div(127.5).sub(1.)
