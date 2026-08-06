from __future__ import annotations
from pathlib import Path
import torch
from torch import nn

def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer | None, step: int, config: dict | None = None) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict() if optimizer else None,"step":int(step),"config":config or {},"torch_rng":torch.get_rng_state()},path)

def load_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None, map_location: str = "cpu") -> dict:
    state=torch.load(path,map_location=map_location,weights_only=True); model.load_state_dict(state["model"])
    if optimizer and state.get("optimizer") is not None: optimizer.load_state_dict(state["optimizer"])
    return state
