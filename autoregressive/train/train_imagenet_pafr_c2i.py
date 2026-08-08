#!/usr/bin/env python3
"""Train text-free class-to-image LlamaGen on original VQGAN ImageNet tokens."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from autoregressive.models.gpt import GPT_models
from autoregressive.train.train_c2i import creat_optimizer
from autoregressive.train.train_roco_uncond import _atomic_save, _logger, _setup_distributed, _update_link
from evaluation.vae.standalone_vqgan import load_standalone_vqgan
from pafr_vq.models.base_adapter import VQGANAdapter
from pafr_vq.imagenet import ImageNetListDataset


def load_original_encoder(args, device, logger):
    vqgan, _, _ = load_standalone_vqgan(Path(args.base_config), Path(args.base_checkpoint), device)
    adapter = VQGANAdapter(vqgan)
    if adapter.codebook_size != args.vocab_size:
        raise ValueError(f"codebook mismatch: original VQGAN={adapter.codebook_size}, GPT={args.vocab_size}")
    for unused in ("decoder", "post_quant_conv", "loss"):
        if hasattr(adapter.vqgan, unused): setattr(adapter.vqgan, unused, None)
    adapter.freeze(); torch.cuda.empty_cache()
    logger.info(f"Loaded frozen original CompVis VQGAN encoder; vocabulary={adapter.codebook_size}")
    return adapter


def save_checkpoint(model, optimizer, scaler, args, checkpoint_dir, epoch, step, best, metrics, is_best):
    path = Path(checkpoint_dir) / f"{step:07d}.pt"
    payload = {
        "model": model.module.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
        "epoch": epoch, "steps": step, "best_val_loss": best, "metrics": metrics, "args": vars(args),
        "conditioning": {"type": "imagenet_class_id", "num_classes": args.num_classes, "text_features": False},
    }
    _atomic_save(payload, path); _update_link(path, Path(checkpoint_dir) / "last.pt")
    if is_best: _update_link(path, Path(checkpoint_dir) / "best.pt")
    protected = {path.name}
    for name in ("last.pt", "best.pt"):
        link = Path(checkpoint_dir) / name
        if link.is_symlink(): protected.add(os.readlink(link))
    candidates = sorted(candidate for candidate in Path(checkpoint_dir).glob("*.pt") if candidate.stem.isdigit())
    if args.keep_last_checkpoints > 0:
        for old in candidates[:-args.keep_last_checkpoints]:
            if old.name not in protected: old.unlink()
    return path


@torch.no_grad()
def evaluate(model, encoder, loader, device, dtype, precision, max_batches):
    model.eval(); totals = torch.zeros(3, dtype=torch.float64, device=device)
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches and batch_index >= max_batches: break
        images = images.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype, enabled=precision != "none"):
            tokens = encoder.encode(images)["token_ids"].reshape(images.shape[0], -1)
            logits, loss = model(cond_idx=labels, idx=tokens[:, :-1], targets=tokens)
        count = tokens.numel(); totals[0] += loss.double() * count; totals[1] += (logits.argmax(-1) == tokens).sum().double(); totals[2] += count
    dist.all_reduce(totals); count = max(totals[2].item(), 1); loss = totals[0].item() / count; model.train()
    return {"val_loss": loss, "val_perplexity": math.exp(min(loss, 20)), "val_token_accuracy": totals[1].item() / count}


def main(args):
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    if not Path(args.pafr_checkpoint).is_file(): raise FileNotFoundError(args.pafr_checkpoint)
    rank, world, local_rank, device = _setup_distributed()
    if args.global_batch_size % world: raise ValueError("global batch size must be divisible by world size")
    torch.manual_seed(args.seed * world + rank)
    output_dir = Path(args.results_dir) / args.run_name; checkpoint_dir = output_dir / "checkpoints"
    if rank == 0:
        if output_dir.exists() and any(output_dir.iterdir()) and not args.resume: raise FileExistsError(f"non-empty run directory: {output_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True); (output_dir / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    dist.barrier(); logger = _logger(output_dir, rank); writer = SummaryWriter(output_dir / "tensorboard") if rank == 0 else None
    code_length = (args.image_size // args.downsample_size) ** 2
    model = GPT_models[args.gpt_model](vocab_size=args.vocab_size, block_size=code_length, num_classes=args.num_classes, cls_token_num=1, model_type="c2i", class_dropout_prob=args.class_dropout_prob, resid_dropout_p=args.dropout, ffn_dropout_p=args.dropout, token_dropout_p=args.token_dropout, drop_path_rate=args.drop_path).to(device)
    logger.info(f"ImageNet text-free LlamaGen {args.gpt_model}: {sum(parameter.numel() for parameter in model.parameters()):,} parameters")
    logger.info(f"Conditioning: ImageNet class ID (1000 classes), no captions or T5 features; visual vocabulary={args.vocab_size}")
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger); scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    start_epoch = 0; step = 0; best = float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler"): scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1; step = int(state["steps"]); best = float(state.get("best_val_loss", best))
    encoder = load_original_encoder(args, device, logger)
    train_set = ImageNetListDataset(args.train_list, args.image_size, train=True, return_label=True, horizontal_flip=True)
    val_set = ImageNetListDataset(args.val_list, args.image_size, train=False, return_label=True)
    train_sampler = DistributedSampler(train_set, world, rank, shuffle=True, seed=args.seed, drop_last=True); val_sampler = DistributedSampler(val_set, world, rank, shuffle=False, drop_last=False)
    loader_args = dict(batch_size=args.global_batch_size // world, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_set, sampler=train_sampler, drop_last=True, **loader_args); val_loader = DataLoader(val_set, sampler=val_sampler, drop_last=False, **loader_args)
    logger.info(f"ImageNet train/val={len(train_set):,}/{len(val_set):,}; global batch={args.global_batch_size}; token length={code_length}")
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False); dtype = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    stop = False; log_loss = 0.; log_accuracy = 0.; log_count = 0; last_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch); model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=args.precision != "none"):
                tokens = encoder.encode(images)["token_ids"].reshape(images.shape[0], -1)
            if tokens.shape[1] != code_length: raise RuntimeError(f"expected {code_length} tokens, got {tokens.shape[1]}")
            if tokens.max() >= args.vocab_size: raise RuntimeError(f"token {tokens.max().item()} exceeds vocabulary {args.vocab_size}")
            with torch.autocast("cuda", dtype=dtype, enabled=args.precision != "none"):
                logits, loss = model(cond_idx=labels, idx=tokens[:, :-1], targets=tokens)
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at step {step}")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(grad): raise FloatingPointError(f"non-finite gradient at step {step}")
            scaler.step(optimizer); scaler.update(); step += 1
            log_loss += loss.item(); log_accuracy += (logits.argmax(-1) == tokens).float().mean().item(); log_count += 1
            if step % args.log_every == 0:
                values = torch.tensor([log_loss, log_accuracy, log_count], dtype=torch.float64, device=device); dist.all_reduce(values); denominator = max(values[2].item(), 1); elapsed = max(time.time() - last_time, 1e-6)
                if rank == 0:
                    logger.info(f"step={step:07d} epoch={epoch:03d} loss={values[0].item()/denominator:.5f} acc={values[1].item()/denominator:.4f} grad={grad.item():.4f} steps/s={args.log_every/elapsed:.3f}")
                    if writer: writer.add_scalar("train/loss", values[0].item()/denominator, step); writer.add_scalar("train/token_accuracy", values[1].item()/denominator, step); writer.flush()
                log_loss = log_accuracy = 0.; log_count = 0; last_time = time.time()
            if args.max_steps and step >= args.max_steps: stop = True; break
        metrics = evaluate(model, encoder, val_loader, device, dtype, args.precision, args.val_max_batches); is_best = metrics["val_loss"] < best; best = min(best, metrics["val_loss"])
        if rank == 0:
            checkpoint = save_checkpoint(model, optimizer, scaler, args, checkpoint_dir, epoch, step, best, metrics, is_best); logger.info(f"validation step={step}: {metrics}; checkpoint={checkpoint}")
        dist.barrier()
        if stop: break
    if writer: writer.close()
    if rank == 0: logger.info("ImageNet class-to-image LlamaGen training finished")
    dist.destroy_process_group()


def build_parser():
    parser = argparse.ArgumentParser(); parser.add_argument("--train-list", required=True); parser.add_argument("--val-list", required=True); parser.add_argument("--base-config", required=True); parser.add_argument("--base-checkpoint", required=True); parser.add_argument("--pafr-checkpoint", required=True); parser.add_argument("--results-dir", required=True); parser.add_argument("--run-name", required=True); parser.add_argument("--resume")
    parser.add_argument("--gpt-model", choices=list(GPT_models), default="GPT-L"); parser.add_argument("--image-size", type=int, default=256); parser.add_argument("--downsample-size", type=int, default=16); parser.add_argument("--vocab-size", type=int, default=16384); parser.add_argument("--num-classes", type=int, default=1000); parser.add_argument("--class-dropout-prob", type=float, default=.1)
    parser.add_argument("--epochs", type=int, default=60); parser.add_argument("--max-steps", type=int, default=0); parser.add_argument("--global-batch-size", type=int, default=64); parser.add_argument("--num-workers", type=int, default=8); parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--weight-decay", type=float, default=.05); parser.add_argument("--beta1", type=float, default=.9); parser.add_argument("--beta2", type=float, default=.95); parser.add_argument("--grad-clip", type=float, default=1.); parser.add_argument("--dropout", type=float, default=.1); parser.add_argument("--token-dropout", type=float, default=.1); parser.add_argument("--drop-path", type=float, default=0.); parser.add_argument("--precision", choices=("none", "bf16", "fp16"), default="bf16"); parser.add_argument("--seed", type=int, default=23); parser.add_argument("--log-every", type=int, default=20); parser.add_argument("--val-max-batches", type=int, default=100); parser.add_argument("--keep-last-checkpoints", type=int, default=3)
    return parser


if __name__ == "__main__": main(build_parser().parse_args())
