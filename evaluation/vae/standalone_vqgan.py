"""Shared helpers for evaluating standalone taming-transformers checkpoints."""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import albumentations as A
import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf.nodes import AnyNode
from PIL import Image
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import Dataset

from main import instantiate_from_config


DEFAULT_MODEL_DIR = PROJECT_ROOT / "pretrained_models" / "vqgan_f16_16384"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "imagenet" / "val"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ImageNetCenterCropDataset(Dataset):
    """Match taming.data.base.ImagePaths(size, random_crop=False)."""

    def __init__(self, files: list[Path], image_size: int, return_path=False):
        self.files = files
        self.return_path = return_path
        self.transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=image_size),
                A.CenterCrop(height=image_size, width=image_size),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        try:
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            array = self.transform(image=array)["image"]
        except Exception as error:
            raise RuntimeError(f"Failed to preprocess image: {path}") from error
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
        tensor = tensor.float().div_(127.5).sub_(1.0)
        return (tensor, str(path)) if self.return_path else tensor


def collect_images(image_dir: Path):
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def select_images(files: list[Path], count: int, seed: int):
    if count <= 0 or count >= len(files):
        return files
    indices = sorted(random.Random(seed).sample(range(len(files)), count))
    return [files[index] for index in indices]


def safe_load_checkpoint(checkpoint: Path):
    """Load tensors while allowlisting only known benign PL/OmegaConf containers."""

    allowed_globals = {
        "omegaconf.dictconfig.DictConfig": DictConfig,
        "omegaconf.nodes.AnyNode": AnyNode,
        "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint": ModelCheckpoint,
        "builtins.list": list,
        "omegaconf.base.ContainerMetadata": ContainerMetadata,
        "omegaconf.listconfig.ListConfig": ListConfig,
        "collections.defaultdict": defaultdict,
        "builtins.int": int,
        "builtins.dict": dict,
        "omegaconf.base.Metadata": Metadata,
    }
    referenced = set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint))
    unsupported = sorted(referenced.difference(allowed_globals))
    if unsupported:
        raise RuntimeError(
            "Checkpoint references globals outside the safe allowlist: "
            + ", ".join(unsupported)
        )
    with torch.serialization.safe_globals(
        [allowed_globals[name] for name in sorted(referenced)]
    ):
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(data).__name__}")
    return data


def load_standalone_vqgan(config_path: Path, checkpoint: Path, device):
    config = OmegaConf.load(config_path)
    model_config = OmegaConf.create(OmegaConf.to_container(config.model, resolve=True))
    # Training-only discriminator and LPIPS are unnecessary for reconstruction.
    model_config.params.lossconfig = {"target": "torch.nn.Identity"}
    model_config.params.pop("ckpt_path", None)
    model = instantiate_from_config(model_config)

    checkpoint_data = safe_load_checkpoint(checkpoint)
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing_core = [key for key in missing if not key.startswith("loss.")]
    unexpected_core = [key for key in unexpected if not key.startswith("loss.")]
    if missing_core or unexpected_core:
        raise RuntimeError(
            "Model/checkpoint mismatch outside omitted training loss: "
            f"missing={missing_core[:20]}, unexpected={unexpected_core[:20]}"
        )
    model.eval().requires_grad_(False).to(device)
    metadata = {
        "epoch": checkpoint_data.get("epoch"),
        "global_step": checkpoint_data.get("global_step"),
        "missing_keys": list(missing),
        "unexpected_training_loss_keys": list(unexpected),
    }
    return model, config, metadata


def resolve_device(value: str):
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def autocast_context(device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 inference requires CUDA")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)
