"""Adapters keep PAFR-VQ independent of a particular first-stage tokenizer."""
from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from torch import Tensor, nn


class BaseTokenizerAdapter(nn.Module, ABC):
    codebook_size: int
    token_dim: int
    @abstractmethod
    def encode(self, images: Tensor) -> dict[str, Tensor]: ...
    @abstractmethod
    def decode(self, token_ids: Tensor | None = None, quantized_features: Tensor | None = None) -> Tensor: ...
    def reconstruct(self, images: Tensor) -> dict[str, Tensor]:
        output = self.encode(images)
        output["reconstruction"] = self.decode(output["token_ids"], output["quantized_features"])
        return output
    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters(): parameter.requires_grad_(False)


class TinyBaseTokenizer(BaseTokenizerAdapter):
    """Self-contained tiny fallback for CPU tests, not a pretrained VQGAN."""
    def __init__(self, in_channels: int = 3, token_dim: int = 32, codebook_size: int = 64, compression_factor: int = 4) -> None:
        super().__init__()
        if compression_factor not in {2, 4, 8, 16}: raise ValueError("compression_factor must be 2, 4, 8, or 16")
        self.codebook_size, self.token_dim, self.compression_factor = codebook_size, token_dim, compression_factor
        stages = compression_factor.bit_length() - 1
        encoder: list[nn.Module] = []; channels = in_channels
        for _ in range(stages): encoder += [nn.Conv2d(channels, token_dim, 4, 2, 1), nn.SiLU()]; channels = token_dim
        self.encoder = nn.Sequential(*encoder)
        self.codebook = nn.Embedding(codebook_size, token_dim); nn.init.normal_(self.codebook.weight, std=token_dim ** -0.5)
        decoder: list[nn.Module] = []
        for index in range(stages):
            out = in_channels if index == stages - 1 else token_dim
            decoder += [nn.ConvTranspose2d(channels, out, 4, 2, 1)]
            if out != in_channels: decoder += [nn.SiLU()]
            channels = out
        self.decoder = nn.Sequential(*decoder)
    def encode(self, images: Tensor) -> dict[str, Tensor]:
        features = self.encoder(images); flat = features.permute(0, 2, 3, 1).reshape(-1, self.token_dim)
        distances = flat.square().sum(1, keepdim=True) + self.codebook.weight.square().sum(1) - 2 * flat @ self.codebook.weight.t()
        ids = distances.argmin(-1).view(images.shape[0], *features.shape[-2:])
        quantized = self.codebook(ids).permute(0, 3, 1, 2).contiguous()
        return {"token_ids": ids, "quantized_features": quantized, "encoder_features": features}
    def decode(self, token_ids: Tensor | None = None, quantized_features: Tensor | None = None) -> Tensor:
        if quantized_features is None:
            if token_ids is None: raise ValueError("provide token_ids or quantized_features")
            quantized_features = self.codebook(token_ids.long()).permute(0, 3, 1, 2).contiguous()
        return self.decoder(quantized_features)


class VQGANAdapter(BaseTokenizerAdapter):
    """Adapter for ``taming.models.vqgan.VQModel`` and compatible checkpoints."""
    def __init__(self, vqgan: nn.Module) -> None:
        super().__init__()
        if not all(hasattr(vqgan, name) for name in ("encoder", "quant_conv", "quantize", "decode")):
            raise TypeError("vqgan must expose encoder, quant_conv, quantize, and decode")
        self.vqgan = vqgan; quantizer = vqgan.quantize
        self.codebook_size = int(getattr(quantizer, "n_e", getattr(quantizer, "n_embed", 0)))
        self.token_dim = int(getattr(quantizer, "e_dim", getattr(quantizer, "embedding_dim", 0)))
        if not self.codebook_size or not self.token_dim: raise ValueError("Could not infer VQGAN dimensions")
    @classmethod
    def from_checkpoint(cls, config_path: str, checkpoint_path: str, device: str = "cpu") -> "VQGANAdapter":
        from omegaconf import OmegaConf
        from main import instantiate_from_config
        model = instantiate_from_config(OmegaConf.load(config_path).model)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
        return cls(model.to(device))
    def encode(self, images: Tensor) -> dict[str, Tensor]:
        encoder_features = self.vqgan.quant_conv(self.vqgan.encoder(images))
        quantized, _, info = self.vqgan.quantize(encoder_features); ids = info[2]
        if ids.ndim == 1: ids = ids.view(images.shape[0], *quantized.shape[-2:])
        return {"token_ids": ids.long(), "quantized_features": quantized, "encoder_features": encoder_features}
    def decode(self, token_ids: Tensor | None = None, quantized_features: Tensor | None = None) -> Tensor:
        if quantized_features is None:
            if token_ids is None: raise ValueError("provide token_ids or quantized_features")
            quantized_features = self.vqgan.quantize.embed_code(token_ids)
        return self.vqgan.decode(quantized_features)
