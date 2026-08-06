"""Shared ROCOv2 helpers for real PAFR-VQ training and evaluation."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.augmentation import center_crop_arr
from pafr_vq.allocation import BudgetSelector, ResidualScorer
from pafr_vq.models.base_adapter import VQGANAdapter
from pafr_vq.models.tokenizer import PAFRTokenizer


class RocoImageListDataset(Dataset):
    def __init__(self, list_file: str | Path, image_size: int = 256) -> None:
        self.list_file = Path(list_file)
        self.paths = [Path(line.strip()) for line in self.list_file.read_text().splitlines() if line.strip()]
        if not self.paths:
            raise ValueError(f"Image list is empty: {self.list_file}")
        if not self.paths[0].is_file():
            raise FileNotFoundError(f"First image does not exist: {self.paths[0]}")
        self.transform = transforms.Compose([
            transforms.Lambda(lambda image: center_crop_arr(image, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        try:
            image = Image.open(self.paths[index]).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load {self.paths[index]}") from exc
        return self.transform(image)


def build_rocov2_pafr(
    base_config: str,
    base_checkpoint: str,
    device: torch.device | str,
    scorer_type: str = "hybrid",
    residual_dim: int = 64,
    residual_codebook_size: int = 1024,
    commitment_weight: float = 0.25,
) -> PAFRTokenizer:
    adapter = VQGANAdapter.from_checkpoint(base_config, base_checkpoint, str(device))
    # PAFR uses the VQ encoder/codebook/decoder, not the old discriminator or LARP prior.
    for unused in ("loss", "transformer"):
        if hasattr(adapter.vqgan, unused):
            setattr(adapter.vqgan, unused, None)
    scorer = ResidualScorer(scorer_type, pixel_weight=1.0, sobel_weight=1.0, dwt_weight=1.0)
    model = PAFRTokenizer(
        adapter,
        residual_dim=residual_dim,
        residual_codebook_size=residual_codebook_size,
        scorer=scorer,
        selector=BudgetSelector(deterministic=True),
        freeze_base=True,
    )
    model.quantizer.commitment_weight = float(commitment_weight)
    return model.to(device)


def residual_state_dict(model: PAFRTokenizer) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "residual_encoder": model.residual_encoder.state_dict(),
        "quantizer": model.quantizer.state_dict(),
        "residual_decoder": model.residual_decoder.state_dict(),
    }


def load_residual_state(model: PAFRTokenizer, checkpoint_path: str | Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.residual_encoder.load_state_dict(checkpoint["residual_encoder"])
    model.quantizer.load_state_dict(checkpoint["quantizer"])
    model.residual_decoder.load_state_dict(checkpoint["residual_decoder"])
    return checkpoint


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
