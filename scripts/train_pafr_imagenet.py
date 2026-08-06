#!/usr/bin/env python3
"""Train sparse PAFR residual branches over the frozen original ImageNet VQGAN."""
from __future__ import annotations

import argparse
import json
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

from pafr_vq.imagenet import ImageNetListDataset, atomic_torch_save, build_imagenet_pafr, residual_state_dict


def setup_distributed():
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1)); local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    return rank, world, local, torch.device("cuda", local)


def reduce_mean(value, world):
    value = value.detach().float()
    if world > 1:
        dist.all_reduce(value); value /= world
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
    totals /= max(batches, 1)
    if world > 1:
        dist.all_reduce(totals); totals /= world
    model.train()
    return {"val_loss": float(totals[0]), "val_l1": float(totals[1]), "val_frequency": float(totals[2]), "val_utilization": float(totals[3])}


def main(args):
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    rank, world, local, device = setup_distributed(); torch.manual_seed(args.seed + rank)
    output_dir = Path(args.output_dir); checkpoint_dir = output_dir / "checkpoints"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    if world > 1: dist.barrier()
    model = build_imagenet_pafr(args.base_config, args.base_checkpoint, device, args.scorer, args.residual_dim, args.residual_codebook_size, args.commitment_weight)
    if model.base.codebook_size != args.expected_vocab_size:
        raise ValueError(f"Expected original VQGAN vocabulary {args.expected_vocab_size}, found {model.base.codebook_size}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .95))
    start_epoch = 0; step = 0; best = float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.residual_encoder.load_state_dict(state["residual_encoder"]); model.quantizer.load_state_dict(state["quantizer"]); model.residual_decoder.load_state_dict(state["residual_decoder"])
        optimizer.load_state_dict(state["optimizer"]); start_epoch = int(state["epoch"]) + 1; step = int(state["step"]); best = float(state.get("best_val_loss", best))
    train_set = ImageNetListDataset(args.train_list, args.image_size, train=True)
    val_set = ImageNetListDataset(args.val_list, args.image_size, train=False)
    train_sampler = DistributedSampler(train_set, world, rank, shuffle=True, seed=args.seed, drop_last=True)
    val_sampler = DistributedSampler(val_set, world, rank, shuffle=False, drop_last=False)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_set, sampler=train_sampler, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, sampler=val_sampler, drop_last=False, **loader_args)
    wrapped = DDP(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    writer = SummaryWriter(output_dir / "tensorboard") if rank == 0 else None
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    if rank == 0:
        print(f"PAFR ImageNet/original-VQGAN: train={len(train_set)} val={len(val_set)} world={world} batch/GPU={args.batch_size} base_vocab={model.base.codebook_size} ratio={args.active_ratio}", flush=True)
    stop = False; last_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch); wrapped.train()
        for images in train_loader:
            images = images.to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                output = wrapped(images, active_ratio=args.active_ratio); loss = output["losses"]["total"]
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward(); grad = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            if not torch.isfinite(grad): raise FloatingPointError(f"non-finite gradient at step {step}")
            optimizer.step(); step += 1
            if step % args.log_every == 0:
                mean_loss = reduce_mean(loss, world)
                if rank == 0:
                    elapsed = max(time.time() - last_time, 1e-6)
                    print(f"step={step:07d} epoch={epoch:03d} loss={float(mean_loss):.5f} grad={float(grad):.4f} steps/s={args.log_every/elapsed:.3f}", flush=True)
                    if writer: writer.add_scalar("train/loss", float(mean_loss), step); writer.add_scalar("train/grad_norm", float(grad), step); writer.flush()
                last_time = time.time()
            if args.max_steps and step >= args.max_steps: stop = True; break
        metrics = evaluate(wrapped, val_loader, device, args.active_ratio, args.val_max_batches, dtype, world)
        is_best = metrics["val_loss"] < best; best = min(best, metrics["val_loss"])
        if rank == 0:
            print(f"validation epoch={epoch} step={step} {metrics} best={best:.6f}", flush=True)
            payload = {**residual_state_dict(model), "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step, "best_val_loss": best, "metrics": metrics, "args": vars(args), "base_codebook_size": model.base.codebook_size}
            atomic_torch_save(payload, checkpoint_dir / "last.pt")
            if is_best: atomic_torch_save(payload, checkpoint_dir / "best.pt")
        if world > 1: dist.barrier()
        if stop: break
    if writer: writer.close()
    if rank == 0: print("PAFR ImageNet tokenizer training finished", flush=True)
    if world > 1: dist.destroy_process_group()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", required=True); parser.add_argument("--val-list", required=True); parser.add_argument("--base-config", required=True); parser.add_argument("--base-checkpoint", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--resume")
    parser.add_argument("--image-size", type=int, default=256); parser.add_argument("--expected-vocab-size", type=int, default=16384); parser.add_argument("--scorer", choices=("random", "pixel", "sobel", "haar_dwt", "hybrid"), default="hybrid"); parser.add_argument("--active-ratio", type=float, default=.25); parser.add_argument("--residual-dim", type=int, default=64); parser.add_argument("--residual-codebook-size", type=int, default=1024); parser.add_argument("--commitment-weight", type=float, default=.25)
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--max-steps", type=int, default=0); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--num-workers", type=int, default=8); parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--weight-decay", type=float, default=.01); parser.add_argument("--grad-clip", type=float, default=1.); parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16"); parser.add_argument("--seed", type=int, default=23); parser.add_argument("--log-every", type=int, default=20); parser.add_argument("--val-max-batches", type=int, default=100)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
