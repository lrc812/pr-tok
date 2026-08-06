from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class RocoText2ImageDataset(Dataset):
    """ROCO images paired with precomputed T5 caption features."""

    def __init__(
        self,
        captions_csv,
        image_root,
        t5_feat_root,
        transform,
        image_size=256,
        downsample_size=16,
        cls_token_num=120,
        caption_dim=1024,
    ):
        self.captions_csv = Path(captions_csv)
        self.image_root = Path(image_root)
        self.t5_feat_root = Path(t5_feat_root)
        self.transform = transform
        self.image_size = int(image_size)
        self.code_len = (self.image_size // int(downsample_size)) ** 2
        self.cls_token_num = int(cls_token_num)
        self.caption_dim = int(caption_dim)

        if not self.captions_csv.is_file():
            raise FileNotFoundError(f"ROCO captions CSV not found: {self.captions_csv}")
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"ROCO image root not found: {self.image_root}")
        if not self.t5_feat_root.is_dir():
            raise FileNotFoundError(f"ROCO T5 feature root not found: {self.t5_feat_root}")

        frame = pd.read_csv(self.captions_csv)
        missing_columns = {"ID", "Caption"} - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{self.captions_csv} is missing columns: {sorted(missing_columns)}"
            )
        self.rows = frame[["ID", "Caption"]].to_dict("records")
        if not self.rows:
            raise ValueError(f"ROCO captions CSV is empty: {self.captions_csv}")

        sequence_length = self.cls_token_num + self.code_len
        self.base_causal_mask = torch.tril(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool)
        )

        # Fail before DDP starts if paths or feature dimensions are inconsistent.
        image_path = self._resolve_image_path(self.rows[0]["ID"])
        feature_path = self._feature_path(0)
        if not image_path.is_file():
            raise FileNotFoundError(f"First ROCO image not found: {image_path}")
        if not feature_path.is_file():
            raise FileNotFoundError(f"First ROCO T5 feature not found: {feature_path}")
        self._load_t5_feature(feature_path)

    def __len__(self):
        return len(self.rows)

    def _resolve_image_path(self, csv_image_id):
        raw = Path(str(csv_image_id))
        if raw.is_file():
            return raw

        basename = raw.name
        if not Path(basename).suffix:
            basename = f"{basename}.jpg"
        return self.image_root / basename

    def _feature_path(self, index):
        return self.t5_feat_root / f"{index}.npy"

    def _load_t5_feature(self, path):
        array = np.load(path, allow_pickle=False)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3 or array.shape[0] != 1:
            raise ValueError(
                f"Expected T5 feature shape [1, length, dim], got {array.shape} at {path}"
            )
        if array.shape[-1] != self.caption_dim:
            raise ValueError(
                f"Expected T5 feature dim {self.caption_dim}, got {array.shape[-1]} at {path}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"T5 feature contains NaN/Inf: {path}")
        return torch.from_numpy(array).float()

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = self._resolve_image_path(row["ID"])
        feature_path = self._feature_path(index)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load ROCO image: {image_path}") from exc
        if self.transform is not None:
            image = self.transform(image)

        try:
            t5_feature = self._load_t5_feature(feature_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load ROCO T5 feature: {feature_path}") from exc

        feature_length = min(self.cls_token_num, t5_feature.shape[1])
        caption = torch.zeros(
            self.cls_token_num, self.caption_dim, dtype=torch.float32
        )
        caption[-feature_length:] = t5_feature[0, :feature_length]

        caption_valid = torch.zeros(self.cls_token_num, dtype=torch.bool)
        caption_valid[-feature_length:] = True
        attention_mask = self.base_causal_mask.clone()
        attention_mask[:, : self.cls_token_num] &= caption_valid.unsqueeze(0)
        attention_mask.fill_diagonal_(True)

        valid_targets = torch.ones(self.code_len, dtype=torch.float32)
        return image, caption, attention_mask, valid_targets


def roco_t2i_collate(batch):
    images, captions, attention_masks, valid_targets = zip(*batch)
    return (
        torch.stack(images),
        torch.stack(captions),
        torch.stack(attention_masks),
        torch.stack(valid_targets),
    )
