#!/usr/bin/env python
"""Train the sparse PAFR residual tokenizer on ROCOv2 with a frozen VQGAN."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from pafr_vq.rocov2 import RocoImageListDataset, atomic_torch_save, build_rocov2_pafr, residual_state_dict


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1)); local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    return rank, world, local, torch.device("cuda", local)


def reduce_mean(value: torch.Tensor, world: int) -> torch.Tensor:
    value = value.detach().float()
    if world > 1:
        dist.all_reduce(value)
        value /= world
    return value


@torch.no_grad()
def evaluate(model, loader, device, active_ratio, max_batches, dtype, world):
    model.eval(); totals = torch.zeros(4, device=device); batches = 0
    for index, images in enumerate(loader):
        if max_batches and index >= max_batches: break
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype):
            output = model(images, active_ratio=active_ratio)
        totals += torch.stack([output["losses"]["total"], output["losses"]["l1"], output["losses"]["frequency"], output["metrics"]["active_code_utilization"]]).float()
        batches += 1
    totals = totals / max(batches, 1)
    if world > 1: dist.all_reduce(totals); totals /= world
    model.train()
    return {"val_loss": float(totals[0]), "val_l1": float(totals[1]), "val_frequency": float(totals[2]), "val_utilization": float(totals[3])}


def main(args) -> None:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for ROCOv2 training")
    rank, world, local, device = setup_distributed(); torch.manual_seed(args.seed + rank)
    output_dir = Path(args.output_dir); checkpoint_dir = output_dir / "checkpoints"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    if world > 1: dist.barrier()
    model = build_rocov2_pafr(args.base_config, args.base_checkpoint, device, args.scorer, args.residual_dim, args.residual_codebook_size, args.commitment_weight)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    start_epoch = 0; step = 0; best = float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.residual_encoder.load_state_dict(state["residual_encoder"]); model.quantizer.load_state_dict(state["quantizer"]); model.residual_decoder.load_state_dict(state["residual_decoder"])
        optimizer.load_state_dict(state["optimizer"]); start_epoch = int(state["epoch"]) + 1; step = int(state["step"]); best = float(state.get("best_val_loss", best))
    train_set = RocoImageListDataset(args.train_list, args.image_size); val_set = RocoImageListDataset(args.val_list, args.image_size)
    train_sampler = DistributedSampler(train_set, world, rank, shuffle=True, seed=args.seed, drop_last=True); val_sampler = DistributedSampler(val_set, world, rank, shuffle=False, drop_last=False)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_set, sampler=train_sampler, drop_last=True, **kwargs); val_loader = DataLoader(val_set, sampler=val_sampler, drop_last=False, **kwargs)
    wrapped = DDP(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    writer = SummaryWriter(output_dir / "tensorboard") if rank == 0 else None; dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    if rank == 0: print(f"PAFR ROCOv2: train={len(train_set)} val={len(val_set)} world={world} batch/GPU={args.batch_size} scorer={args.scorer} ratio={args.active_ratio}", flush=True)
    stop = False; last_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch); wrapped.train()
        for images in train_loader:
            images = images.to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype): output = wrapped(images, active_ratio=args.active_ratio); loss = output["losses"]["total"]
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward(); grad = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            if not torch.isfinite(grad): raise FloatingPointError(f"non-finite gradient at step {step}")
            optimizer.step(); step += 1
            if step % args.log_every == 0:
                mean_loss = reduce_mean(loss, world)
                if rank == 0:
                    elapsed = max(time.time() - last_time, 1e-6); print(f"step={step:07d} epoch={epoch:03d} loss={float(mean_loss):.5f} grad={float(grad):.4f} steps/s={args.log_every/elapsed:.3f}", flush=True)
                    if writer: writer.add_scalar("train/loss", float(mean_loss), step); writer.add_scalar("train/grad_norm", float(grad), step); writer.flush()
                last_time = time.time()
            if args.max_steps and step >= args.max_steps: stop = True; break
        metrics = evaluate(wrapped, val_loader, device, args.active_ratio, args.val_max_batches, dtype, world); is_best = metrics["val_loss"] < best; best = min(best, metrics["val_loss"])
        if rank == 0:
            print(f"validation epoch={epoch} step={step} {metrics} best={best:.6f}", flush=True)
            payload = {**residual_state_dict(model), "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step, "best_val_loss": best, "metrics": metrics, "args": vars(args)}
            atomic_torch_save(payload, checkpoint_dir / "last.pt")
            if is_best: atomic_torch_save(payload, checkpoint_dir / "best.pt")
            if writer:
                for key, value in metrics.items(): writer.add_scalar(f"validation/{key}", value, step)
                writer.flush()
        if world > 1: dist.barrier()
        if stop: break
    if writer: writer.close()
    if rank == 0: print("PAFR tokenizer training finished", flush=True)
    if world > 1: dist.destroy_process_group()


def parser():
    p=argparse.ArgumentParser();p.add_argument("--train-list",required=True);p.add_argument("--val-list",required=True);p.add_argument("--base-config",required=True);p.add_argument("--base-checkpoint",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--resume")
    p.add_argument("--image-size",type=int,default=256);p.add_argument("--scorer",choices=("random","pixel","sobel","haar_dwt","hybrid"),default="hybrid");p.add_argument("--active-ratio",type=float,default=.25);p.add_argument("--residual-dim",type=int,default=64);p.add_argument("--residual-codebook-size",type=int,default=1024);p.add_argument("--commitment-weight",type=float,default=.25)
    p.add_argument("--epochs",type=int,default=25);p.add_argument("--max-steps",type=int,default=0);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--num-workers",type=int,default=8);p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--weight-decay",type=float,default=.01);p.add_argument("--grad-clip",type=float,default=1.);p.add_argument("--precision",choices=("bf16","fp16"),default="bf16");p.add_argument("--seed",type=int,default=23);p.add_argument("--log-every",type=int,default=20);p.add_argument("--val-max-batches",type=int,default=100);return p
if __name__=="__main__": main(parser().parse_args())
