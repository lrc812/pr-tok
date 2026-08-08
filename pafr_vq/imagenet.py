"""ImageNet data helpers for PAFR tokenizer and class-conditional priors."""
from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.augmentation import center_crop_arr, random_crop_arr
from evaluation.vae.standalone_vqgan import load_standalone_vqgan
from pafr_vq.allocation import BudgetSelector, ResidualScorer
from pafr_vq.models.base_adapter import VQGANAdapter
from pafr_vq.models.tokenizer import PAFRTokenizer
from pafr_vq.rocov2 import (
    atomic_torch_save,
    load_residual_state,
    residual_state_dict,
)


class ImageNetListDataset(Dataset):
    """Read the existing ImageNet lists with stable WordNet-ID class labels."""

    def __init__(
        self,
        list_file: str | Path,
        image_size: int = 256,
        train: bool = False,
        return_label: bool = False,
        horizontal_flip: bool = False,
    ) -> None:
        self.list_file = Path(list_file)
        self.paths = [
            Path(line.strip())
            for line in self.list_file.read_text().splitlines()
            if line.strip()
        ]
        if not self.paths:
            raise ValueError(f"Image list is empty: {self.list_file}")
        if not self.paths[0].is_file():
            raise FileNotFoundError(f"First image does not exist: {self.paths[0]}")
        class_names = sorted({path.parent.name for path in self.paths})
        if len(class_names) != 1000:
            raise ValueError(f"Expected 1000 ImageNet classes, found {len(class_names)}")
        self.class_to_idx = {name: index for index, name in enumerate(class_names)}
        self.labels = [self.class_to_idx[path.parent.name] for path in self.paths]
        crop = random_crop_arr if train else center_crop_arr
        operations = [transforms.Lambda(lambda image: crop(image, image_size))]
        if horizontal_flip:
            operations.append(transforms.RandomHorizontalFlip())
        operations.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        self.transform = transforms.Compose(operations)
        self.return_label = return_label

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        try:
            image = Image.open(self.paths[index]).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load {self.paths[index]}") from exc
        image = self.transform(image)
        return (image, self.labels[index]) if self.return_label else image


def build_imagenet_pafr(
    base_config: str, base_checkpoint: str, device: torch.device | str,
    scorer_type: str = "hybrid", residual_dim: int = 64,
    residual_codebook_size: int = 1024, commitment_weight: float = 0.25,
) -> PAFRTokenizer:
    """Safely load the original CompVis checkpoint and freeze it under PAFR."""
    vqgan, _, _ = load_standalone_vqgan(Path(base_config), Path(base_checkpoint), device)
    adapter = VQGANAdapter(vqgan)
    scorer = ResidualScorer(scorer_type, pixel_weight=1.0, sobel_weight=1.0, dwt_weight=1.0)
    model = PAFRTokenizer(adapter, residual_dim=residual_dim, residual_codebook_size=residual_codebook_size, scorer=scorer, selector=BudgetSelector(deterministic=True), freeze_base=True)
    model.quantizer.commitment_weight = float(commitment_weight)
    return model.to(device)


__all__ = [
    "ImageNetListDataset",
    "atomic_torch_save",
    "build_imagenet_pafr",
    "load_residual_state",
    "residual_state_dict",
]
