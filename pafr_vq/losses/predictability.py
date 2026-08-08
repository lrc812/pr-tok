from __future__ import annotations
import torch
from torch import Tensor
import torch.nn.functional as F


def mask_prediction_metrics(logits: Tensor, target: Tensor, threshold: float = 0.0) -> dict[str, Tensor]:
    pred, target = logits > threshold, target.bool()
    tp = (pred & target).sum().float(); fp = (pred & ~target).sum().float(); fn = (~pred & target).sum().float()
    eps = torch.finfo(torch.float32).eps
    precision, recall = tp / (tp + fp + eps), tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {"mask_bce": F.binary_cross_entropy_with_logits(logits, target.float()), "mask_precision": precision,
            "mask_recall": recall, "mask_f1": f1, "mask_iou": tp / (tp + fp + fn + eps)}


def masked_code_nll(logits: Tensor, ids: Tensor, attention_mask: Tensor) -> Tensor:
    if logits.numel() == 0 or not attention_mask.any():
        return logits.new_zeros(())
    loss = F.cross_entropy(logits.transpose(1, 2), ids.long(), reduction="none")
    return (loss * attention_mask.float()).sum() / attention_mask.sum().clamp_min(1)
