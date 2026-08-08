"""Path proxy for direct ``python scripts/<name>.py`` execution.

The actual package lives at repository root. Python otherwise exposes only the
``scripts`` directory when executing a script by path.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[2] / "pafr_vq")]

from .models.base_adapter import BaseTokenizerAdapter, TinyBaseTokenizer, VQGANAdapter
from .models.tokenizer import PAFRTokenizer
from .models.mask_prior import MaskPrior
from .models.residual_prior import ResidualARPrior

__all__ = ["BaseTokenizerAdapter", "TinyBaseTokenizer", "VQGANAdapter", "PAFRTokenizer", "MaskPrior", "ResidualARPrior"]
