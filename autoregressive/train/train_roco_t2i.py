"""Train a ROCO text-to-image-token translator with the LlamaGen GPT."""

import argparse
import contextlib
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from autoregressive.models.gpt import GPT_models
from autoregressive.train.train_c2i import creat_optimizer
from dataset.augmentation import center_crop_arr
from dataset.roco_t2i import RocoText2ImageDataset, roco_t2i_collate
from evaluation.calc_entropy import load_model_from_logdir
from utils.distributed import init_distributed_mode


def _atomic_torch_save(payload, path):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _create_training_logger(experiment_dir, rank):
    logger = logging.getLogger(f"roco_t2i.rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    if rank != 0:
        logger.addHandler(logging.NullHandler())
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        Path(experiment_dir) / "log.txt", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def _checkpoint_payload(
    model,
    optimizer,
    scaler,
    args,
    epoch,
    batch_in_epoch,
    steps,
    metrics=None,
    best_val_loss=float("inf"),
):
    return {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "steps": int(steps),
        "args": vars(args),
        "metrics": dict(metrics or {}),
        "best_val_loss": float(best_val_loss),
    }


def _save_checkpoint(
    model,
    optimizer,
    scaler,
    args,
    checkpoint_dir,
    epoch,
    batch_in_epoch,
    steps,
    logger,
    metrics=None,
    best_val_loss=float("inf"),
    is_best=False,
):
    checkpoint_path = Path(checkpoint_dir) / f"{steps:07d}.pt"
    payload = _checkpoint_payload(
        model,
        optimizer,
        scaler,
        args,
        epoch,
        batch_in_epoch,
        steps,
        metrics=metrics,
        best_val_loss=best_val_loss,
    )
    _atomic_torch_save(payload, checkpoint_path)

    last_path = Path(checkpoint_dir) / "last.pt"
    temporary_link = Path(checkpoint_dir) / ".last.pt.tmp"
    if temporary_link.exists() or temporary_link.is_symlink():
        temporary_link.unlink()
    temporary_link.symlink_to(checkpoint_path.name)
    os.replace(temporary_link, last_path)

    if is_best:
        best_path = Path(checkpoint_dir) / "best.pt"
        temporary_best_link = Path(checkpoint_dir) / ".best.pt.tmp"
        if temporary_best_link.exists() or temporary_best_link.is_symlink():
            temporary_best_link.unlink()
        temporary_best_link.symlink_to(checkpoint_path.name)
        os.replace(temporary_best_link, best_path)

    protected_names = {checkpoint_path.name}
    best_path = Path(checkpoint_dir) / "best.pt"
    if best_path.is_symlink():
        protected_names.add(os.readlink(best_path))
    step_checkpoints = sorted(
        path
        for path in Path(checkpoint_dir).glob("*.pt")
        if path.stem.isdigit()
    )
    if args.keep_last_checkpoints > 0:
        for old_path in step_checkpoints[: -args.keep_last_checkpoints]:
            if old_path.name not in protected_names:
                old_path.unlink()

    best_text = ", best.pt updated" if is_best else ""
    logger.info(
        f"Saved checkpoint to {checkpoint_path} (last.pt updated{best_text}); "
        f"metrics={dict(metrics or {})}"
    )


def _load_frozen_vq_encoder(vq_log_dir, device, expected_vocab_size, logger):
    vq_model, _ = load_model_from_logdir(vq_log_dir, device=device)
    actual_vocab_size = int(vq_model.quantize.n_e)
    if actual_vocab_size != expected_vocab_size:
        raise ValueError(
            "Visual vocabulary mismatch: "
            f"LlamaGen vocab_size={expected_vocab_size}, "
            f"VQ-GAN codebook_size={actual_vocab_size}."
        )

    # Only encoder, quant_conv and quantize are needed for image tokenization.
    for unused_name in ("decoder", "post_quant_conv", "loss", "transformer"):
        if hasattr(vq_model, unused_name):
            setattr(vq_model, unused_name, None)
    vq_model.requires_grad_(False)
    vq_model.eval()
    torch.cuda.empty_cache()
    logger.info(
        f"Loaded frozen VQ-GAN encoder from {vq_log_dir}; "
        f"visual vocabulary={actual_vocab_size}"
    )
    return vq_model


def _reduce_training_metrics(loss_sum, accuracy_sum, count, device):
    metrics = torch.tensor(
        [loss_sum, accuracy_sum, count], dtype=torch.float64, device=device
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    denominator = max(metrics[2].item(), 1.0)
    return metrics[0].item() / denominator, metrics[1].item() / denominator


@torch.no_grad()
def _evaluate(
    model,
    vq_model,
    loader,
    device,
    precision_dtype,
    mixed_precision,
    max_batches=0,
):
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)

    for batch_index, (images, captions, attention_mask, valid) in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=precision_dtype,
            enabled=mixed_precision != "none",
        ):
            _, _, info = vq_model.encode(images)
            image_tokens = info[-1].reshape(images.shape[0], -1).long()
            logits, loss = model(
                cond_idx=captions,
                idx=image_tokens[:, :-1],
                targets=image_tokens,
                mask=attention_mask[:, None, :-1, :-1],
                valid=valid,
            )

        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                f"Non-finite validation loss at batch={batch_index}"
            )
        valid_bool = valid.bool()
        token_count = valid.sum(dtype=torch.float64)
        correct_count = (logits.argmax(dim=-1) == image_tokens)[valid_bool].sum()
        totals[0] += loss.double() * token_count
        totals[1] += correct_count.double()
        totals[2] += token_count

    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    token_count = max(totals[2].item(), 1.0)
    val_loss = totals[0].item() / token_count
    metrics = {
        "val_loss": val_loss,
        "val_perplexity": math.exp(min(val_loss, 20.0)),
        "val_token_accuracy": totals[1].item() / token_count,
    }
    model.train()
    return metrics


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("ROCO LlamaGen training requires CUDA.")
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if args.global_batch_size < 1:
        raise ValueError("global_batch_size must be at least 1.")
    if args.global_batch_size % int(os.environ.get("WORLD_SIZE", "1")) != 0:
        raise ValueError("global_batch_size must be divisible by world size.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1.")
    if args.val_every < 1:
        raise ValueError("val_every must be at least 1.")
    if args.keep_last_checkpoints < 0:
        raise ValueError("keep_last_checkpoints cannot be negative.")

    init_distributed_mode(args)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)
    torch.manual_seed(args.global_seed * world_size + rank)

    experiment_dir = Path(args.results_dir) / args.run_name
    checkpoint_dir = experiment_dir / "checkpoints"
    if rank == 0:
        if experiment_dir.exists() and any(experiment_dir.iterdir()) and not args.resume:
            raise FileExistsError(
                f"Experiment directory already exists and is not empty: {experiment_dir}. "
                "Use a new --run-name or pass --resume."
            )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    logger = _create_training_logger(experiment_dir, rank)
    writer = SummaryWriter(experiment_dir / "tensorboard") if rank == 0 else None
    if rank == 0:
        with open(experiment_dir / "args.json", "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    logger.info(
        f"Starting ROCO translator rank={rank}, world_size={world_size}, device={device}"
    )

    latent_size = args.image_size // args.downsample_size
    code_length = latent_size**2
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=code_length,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        caption_dim=args.caption_dim,
        model_type="t2i",
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,
        drop_path_rate=args.drop_path_rate,
    ).to(device)
    logger.info(f"LlamaGen {args.gpt_model} parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = creat_optimizer(
        model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == "fp16")

    start_epoch = 0
    resume_batch = 0
    train_steps = 0
    best_val_loss = float("inf")
    last_validation_metrics = {}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0))
        resume_batch = int(checkpoint.get("batch_in_epoch", -1)) + 1
        train_steps = int(checkpoint.get("steps", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        last_validation_metrics = dict(checkpoint.get("metrics", {}))
        logger.info(
            f"Resuming {args.resume}: epoch={start_epoch}, "
            f"next_batch={resume_batch}, steps={train_steps}"
        )
        del checkpoint

    vq_model = _load_frozen_vq_encoder(
        args.vq_log_dir, device, args.vocab_size, logger
    )

    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda image: center_crop_arr(image, args.image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                inplace=True,
            ),
        ]
    )
    full_dataset = RocoText2ImageDataset(
        captions_csv=args.data_path,
        image_root=args.image_root,
        t5_feat_root=args.t5_feat_path,
        transform=transform,
        image_size=args.image_size,
        downsample_size=args.downsample_size,
        cls_token_num=args.cls_token_num,
        caption_dim=args.caption_dim,
    )
    if not 0 < args.val_size < len(full_dataset):
        raise ValueError(
            f"val_size must be in [1, {len(full_dataset) - 1}], got {args.val_size}."
        )
    split_generator = torch.Generator().manual_seed(args.validation_seed)
    permutation = torch.randperm(
        len(full_dataset), generator=split_generator
    ).tolist()
    validation_indices = permutation[: args.val_size]
    training_indices = permutation[args.val_size :]
    train_dataset = Subset(full_dataset, training_indices)
    val_dataset = Subset(full_dataset, validation_indices)

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=args.validation_seed,
        drop_last=False,
    )
    per_gpu_batch_size = args.global_batch_size // world_size
    loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": per_gpu_batch_size,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
        "collate_fn": roco_t2i_collate,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)
    val_loader_kwargs = dict(loader_kwargs)
    val_loader_kwargs.update(
        dataset=val_dataset,
        sampler=val_sampler,
        drop_last=False,
    )
    val_loader = DataLoader(**val_loader_kwargs)
    logger.info(
        f"Dataset train/val={len(train_dataset):,}/{len(val_dataset):,}, "
        f"train/val batches={len(loader):,}/{len(val_loader):,}, "
        f"batch/GPU={per_gpu_batch_size}, global_batch={args.global_batch_size}, "
        f"code_length={code_length}"
    )

    model = DDP(model, device_ids=[args.gpu])
    model.train()
    optimizer.zero_grad(set_to_none=True)

    precision_dtype = {
        "none": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.mixed_precision]
    running_loss = 0.0
    running_accuracy = 0.0
    running_microbatches = 0
    optimizer_steps_since_log = 0
    last_log_time = time.time()
    stop_training = False
    last_saved_step = -1
    epoch = start_epoch
    batch_index = resume_batch - 1

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}")
        epoch_resume_batch = resume_batch if epoch == start_epoch else 0

        for batch_index, (images, captions, attention_mask, valid) in enumerate(loader):
            if batch_index < epoch_resume_batch:
                continue

            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            with torch.no_grad(), torch.autocast(
                device_type="cuda",
                dtype=precision_dtype,
                enabled=args.mixed_precision != "none",
            ):
                _, _, info = vq_model.encode(images)
                image_tokens = info[-1].reshape(images.shape[0], -1).long()

            if image_tokens.shape[1] != code_length:
                raise RuntimeError(
                    f"Expected {code_length} image tokens, got {image_tokens.shape[1]}"
                )
            if image_tokens.min().item() < 0 or image_tokens.max().item() >= args.vocab_size:
                raise RuntimeError(
                    f"VQ token outside [0, {args.vocab_size - 1}]: "
                    f"min={image_tokens.min().item()}, max={image_tokens.max().item()}"
                )

            accumulation_index = batch_index - epoch_resume_batch
            should_step = (
                (accumulation_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            sync_context = (
                contextlib.nullcontext() if should_step else model.no_sync()
            )
            with sync_context:
                with torch.autocast(
                    device_type="cuda",
                    dtype=precision_dtype,
                    enabled=args.mixed_precision != "none",
                ):
                    logits, raw_loss = model(
                        cond_idx=captions,
                        idx=image_tokens[:, :-1],
                        targets=image_tokens,
                        mask=attention_mask[:, None, :-1, :-1],
                        valid=valid,
                    )
                    loss = raw_loss / args.gradient_accumulation_steps

                if not torch.isfinite(raw_loss).item():
                    raise FloatingPointError(
                        f"Non-finite translator loss at epoch={epoch}, batch={batch_index}"
                    )
                scaler.scale(loss).backward()

            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                valid_bool = valid.bool()
                accuracy = (
                    (predictions == image_tokens)[valid_bool].float().mean().item()
                )
            running_loss += raw_loss.item()
            running_accuracy += accuracy
            running_microbatches += 1

            if not should_step:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            if not torch.isfinite(grad_norm).item():
                raise FloatingPointError(
                    f"Non-finite gradient norm at epoch={epoch}, batch={batch_index}"
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            train_steps += 1
            optimizer_steps_since_log += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize(device)
                mean_loss, mean_accuracy = _reduce_training_metrics(
                    running_loss, running_accuracy, running_microbatches, device
                )
                peak_memory = torch.tensor(
                    [
                        torch.cuda.max_memory_allocated(device) / (1024**3),
                        torch.cuda.max_memory_reserved(device) / (1024**3),
                    ],
                    dtype=torch.float64,
                    device=device,
                )
                dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
                elapsed = max(time.time() - last_log_time, 1e-6)
                steps_per_second = optimizer_steps_since_log / elapsed
                logger.info(
                    f"(step={train_steps:07d}, epoch={epoch:03d}) "
                    f"loss={mean_loss:.4f}, token_acc={mean_accuracy:.4f}, "
                    f"grad_norm={grad_norm.item():.4f}, steps/s={steps_per_second:.3f}, "
                    f"peak_mem={peak_memory[0].item():.2f}/{peak_memory[1].item():.2f} GiB"
                )
                if writer is not None:
                    writer.add_scalar("train/loss", mean_loss, train_steps)
                    writer.add_scalar("train/token_accuracy", mean_accuracy, train_steps)
                    writer.add_scalar("train/grad_norm", grad_norm.item(), train_steps)
                    writer.add_scalar(
                        "train/learning_rate",
                        optimizer.param_groups[0]["lr"],
                        train_steps,
                    )
                    writer.add_scalar(
                        "train/peak_memory_allocated_gib",
                        peak_memory[0].item(),
                        train_steps,
                    )
                    writer.add_scalar("train/epoch", epoch, train_steps)
                    writer.flush()
                running_loss = 0.0
                running_accuracy = 0.0
                running_microbatches = 0
                optimizer_steps_since_log = 0
                last_log_time = time.time()
                torch.cuda.reset_peak_memory_stats(device)

            if (
                not args.no_save
                and args.ckpt_every > 0
                and train_steps % args.ckpt_every == 0
            ):
                if rank == 0:
                    _save_checkpoint(
                        model,
                        optimizer,
                        scaler,
                        args,
                        checkpoint_dir,
                        epoch,
                        batch_index,
                        train_steps,
                        logger,
                        metrics=last_validation_metrics,
                        best_val_loss=best_val_loss,
                    )
                    last_saved_step = train_steps
                dist.barrier()

            if args.max_steps > 0 and train_steps >= args.max_steps:
                stop_training = True
                break

        if stop_training or (epoch + 1) % args.val_every == 0:
            val_sampler.set_epoch(epoch)
            validation_metrics = _evaluate(
                model=model,
                vq_model=vq_model,
                loader=val_loader,
                device=device,
                precision_dtype=precision_dtype,
                mixed_precision=args.mixed_precision,
                max_batches=args.max_val_batches,
            )
            last_validation_metrics = validation_metrics
            is_best = validation_metrics["val_loss"] < best_val_loss
            if is_best:
                best_val_loss = validation_metrics["val_loss"]

            logger.info(
                f"(validation epoch={epoch:03d}, step={train_steps:07d}) "
                f"loss={validation_metrics['val_loss']:.4f}, "
                f"perplexity={validation_metrics['val_perplexity']:.3f}, "
                f"token_acc={validation_metrics['val_token_accuracy']:.4f}, "
                f"best_val_loss={best_val_loss:.4f}"
            )
            if writer is not None:
                for metric_name, metric_value in validation_metrics.items():
                    writer.add_scalar(
                        f"validation/{metric_name[4:]}",
                        metric_value,
                        train_steps,
                    )
                writer.add_scalar(
                    "validation/best_loss", best_val_loss, train_steps
                )
                writer.flush()

            if not args.no_save:
                if rank == 0:
                    _save_checkpoint(
                        model,
                        optimizer,
                        scaler,
                        args,
                        checkpoint_dir,
                        epoch,
                        batch_index,
                        train_steps,
                        logger,
                        metrics=validation_metrics,
                        best_val_loss=best_val_loss,
                        is_best=is_best,
                    )
                    last_saved_step = train_steps
                dist.barrier()

        resume_batch = 0
        if stop_training:
            break

    if not args.no_save and rank == 0 and train_steps != last_saved_step:
        _save_checkpoint(
            model,
            optimizer,
            scaler,
            args,
            checkpoint_dir,
            epoch,
            batch_index,
            train_steps,
            logger,
            metrics=last_validation_metrics,
            best_val_loss=best_val_loss,
        )
    dist.barrier()
    if writer is not None:
        writer.close()
    logger.info("ROCO LlamaGen translator training finished.")
    dist.destroy_process_group()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--t5-feat-path", required=True)
    parser.add_argument("--vq-log-dir", required=True)
    parser.add_argument("--results-dir", default="results_roco_t2i")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--val-size", type=int, default=2048)
    parser.add_argument("--validation-seed", type=int, default=20260729)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--keep-last-checkpoints", type=int, default=3)

    parser.add_argument("--gpt-model", choices=list(GPT_models), default="GPT-L")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--downsample-size", type=int, default=16)
    parser.add_argument("--cls-token-num", type=int, default=120)
    parser.add_argument("--caption-dim", type=int, default=1024)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--token-dropout-p", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--global-batch-size", type=int, default=192)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--global-seed", type=int, default=23)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run without writing model checkpoints (useful for smoke tests).",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=("none", "fp16", "bf16"),
        default="bf16",
    )
    return parser


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main(build_parser().parse_args())
