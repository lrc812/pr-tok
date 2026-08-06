"""Generate ROCO images with a trained text-to-image LlamaGen checkpoint.

Supports one GPU or independent multi-GPU sharding under ``torchrun``.
Existing images are skipped, so interrupted jobs resume safely.
"""

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from autoregressive.models.generate import generate
from autoregressive.models.gpt import GPT_models
from evaluation.calc_entropy import load_model_from_logdir
from language.t5 import T5Embedder
DATA_ROOT = PROJECT_ROOT / "data" / "roco" / "1" / "rocov2"
DEFAULT_VQ_LOG_DIR = PROJECT_ROOT / "logs" / "2026-07-29T23-40-25_my_vqgan_experiment"
DEFAULT_GPT_CKPT = PROJECT_ROOT / "results_roco_t2i" / "my_vqgan_experiment" / "checkpoints" / "best.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "t2i" / "images_roco_valid_best"
DEFAULT_T5_MODEL = PROJECT_ROOT / "pretrained_models" / "flan-t5-large-local"


def distributed_context():
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def load_training_args(checkpoint_path, explicit_path=None):
    if explicit_path:
        args_path = Path(explicit_path).expanduser().resolve()
    else:
        args_path = Path(checkpoint_path).expanduser().resolve().parent.parent / "args.json"
    if not args_path.is_file():
        raise FileNotFoundError(
            f"Training args not found: {args_path}; pass --train-args explicitly."
        )
    with args_path.open("r", encoding="utf-8") as handle:
        return json.load(handle), args_path


def read_prompts(csv_path, empty_caption):
    records = []
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Prompt CSV has no header: {csv_path}")
        caption_column = next(
            (name for name in ("Caption", "Prompt", "caption", "prompt") if name in reader.fieldnames),
            None,
        )
        id_column = next(
            (name for name in ("ID", "id", "ImageID", "image_id") if name in reader.fieldnames),
            None,
        )
        if caption_column is None:
            raise ValueError(f"CSV needs Caption or Prompt; got {reader.fieldnames}")
        for index, row in enumerate(reader):
            caption = (row.get(caption_column) or "").strip()
            if not caption:
                caption = empty_caption
                print(
                    f"Warning: empty caption at CSV row {index + 2}; using {caption!r}.",
                    flush=True,
                )
            image_id = (row.get(id_column) or f"image_{index:06d}").strip()
            records.append((index, image_id, caption))
    if not records:
        raise ValueError(f"No prompts found in {csv_path}")
    return records


def safe_output_name(image_id, index):
    basename = Path(str(image_id)).stem
    basename = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    return f"{basename or f'image_{index:06d}'}.png"


def left_pad_caption_embeddings(caption_embs, emb_masks, max_length):
    """Match RocoText2ImageDataset's right-aligned T5 features."""
    batch_size, _, feature_dim = caption_embs.shape
    padded_embs = caption_embs.new_zeros((batch_size, max_length, feature_dim))
    padded_masks = emb_masks.new_zeros((batch_size, max_length))
    for index in range(batch_size):
        valid_length = min(int(emb_masks[index].sum().item()), max_length)
        if valid_length:
            padded_embs[index, -valid_length:] = caption_embs[index, :valid_length]
            padded_masks[index, -valid_length:] = 1
    return padded_embs, padded_masks


def extract_model_state(checkpoint):
    state = None
    for state_key in ("model", "module", "state_dict"):
        if checkpoint.get(state_key) is not None:
            state = checkpoint[state_key]
            break
    if state is None:
        raise KeyError("Checkpoint has no model/module/state_dict weights.")
    cleaned = {}
    for key, value in state.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    return cleaned


def build_gpt(train_args, checkpoint_path, device, precision):
    image_size = int(train_args["image_size"])
    downsample_size = int(train_args["downsample_size"])
    latent_size = image_size // downsample_size
    model_name = train_args["gpt_model"]
    if model_name not in GPT_models:
        raise ValueError(f"Unknown GPT model {model_name}")

    model = GPT_models[model_name](
        vocab_size=int(train_args["vocab_size"]),
        block_size=latent_size**2,
        num_classes=int(train_args.get("num_classes", 1000)),
        cls_token_num=int(train_args["cls_token_num"]),
        caption_dim=int(train_args["caption_dim"]),
        model_type="t2i",
        resid_dropout_p=float(train_args.get("dropout_p", 0.1)),
        ffn_dropout_p=float(train_args.get("dropout_p", 0.1)),
        token_dropout_p=float(train_args.get("token_dropout_p", 0.1)),
        drop_path_rate=float(train_args.get("drop_path_rate", 0.0)),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    model.load_state_dict(extract_model_state(checkpoint), strict=True)
    del checkpoint
    gc.collect()
    return model.to(device=device, dtype=precision).eval(), latent_size


def main(args):
    rank, world_size, local_rank = distributed_context()
    if not torch.cuda.is_available():
        raise RuntimeError("ROCO generation requires CUDA.")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    precision = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.precision]
    checkpoint_path = Path(args.gpt_ckpt).expanduser().resolve()
    vq_log_dir = Path(args.vq_log_dir).expanduser().resolve()
    prompts_csv = Path(args.prompts_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for required_file in (checkpoint_path, prompts_csv):
        if not required_file.is_file():
            raise FileNotFoundError(required_file)
    if not vq_log_dir.is_dir():
        raise FileNotFoundError(vq_log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_args, train_args_path = load_training_args(checkpoint_path, args.train_args)
    records = read_prompts(prompts_csv, args.empty_caption)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    assigned = records[rank::world_size]
    pending = []
    skipped = 0
    for record in assigned:
        output_path = output_dir / safe_output_name(record[1], record[0])
        if output_path.is_file() and not args.overwrite:
            skipped += 1
        else:
            pending.append(record)

    print(
        f"[rank {rank}/{world_size}] device={device}, prompts={len(records)}, "
        f"assigned={len(assigned)}, pending={len(pending)}, skipped={skipped}, "
        f"output={output_dir}",
        flush=True,
    )
    if not pending:
        return

    if rank == 0:
        run_config = {
            "prompts_csv": str(prompts_csv),
            "vq_log_dir": str(vq_log_dir),
            "gpt_ckpt": str(checkpoint_path),
            "train_args": str(train_args_path),
            "num_prompts": len(records),
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
            "precision": args.precision,
            "cfg_scale": args.cfg_scale,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed,
        }
        with (output_dir / "generation_config.json").open("w", encoding="utf-8") as handle:
            json.dump(run_config, handle, ensure_ascii=False, indent=2)

    print(f"[rank {rank}] Loading VQ-GAN from {vq_log_dir}", flush=True)
    vq_model = load_model_from_logdir(str(vq_log_dir), device=device)[0].eval()
    print(f"[rank {rank}] Loading GPT from {checkpoint_path}", flush=True)
    gpt_model, latent_size = build_gpt(train_args, checkpoint_path, device, precision)
    print(f"[rank {rank}] Loading T5 {args.t5_model_type}", flush=True)
    t5_model = T5Embedder(
        device=device,
        local_cache=False,
        cache_dir=None,
        dir_or_name=args.t5_model_type,
        use_text_preprocessing=False,
        torch_dtype=precision,
        model_max_length=int(train_args["cls_token_num"]),
    )

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    generated = 0
    start_time = time.time()
    progress = tqdm(
        range(0, len(pending), args.batch_size),
        desc=f"rank {rank}",
        disable=args.no_progress,
        dynamic_ncols=True,
    )
    for batch_start in progress:
        batch = pending[batch_start : batch_start + args.batch_size]
        caption_embs, emb_masks = t5_model.get_text_embeddings(
            [record[2] for record in batch]
        )
        caption_embs, emb_masks = left_pad_caption_embeddings(
            caption_embs, emb_masks, int(train_args["cls_token_num"])
        )
        c_indices = caption_embs * emb_masks[:, :, None]
        index_sample = generate(
            gpt_model,
            c_indices,
            latent_size**2,
            emb_masks,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_logits=not args.greedy,
        )
        samples = vq_model.decode_code(
            index_sample.reshape(-1, latent_size, latent_size)
        )
        for record, sample in zip(batch, samples):
            output_path = output_dir / safe_output_name(record[1], record[0])
            save_image(sample, output_path, normalize=True, value_range=(-1, 1))
            generated += 1
        progress.set_postfix(generated=generated)

    elapsed = time.time() - start_time
    print(
        f"[rank {rank}] Finished: generated={generated}, skipped={skipped}, "
        f"elapsed={elapsed:.1f}s, images/s={generated / max(elapsed, 1e-6):.3f}",
        flush=True,
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-csv", default=str(DATA_ROOT / "valid_captions.csv"))
    parser.add_argument("--vq-log-dir", default=str(DEFAULT_VQ_LOG_DIR))
    parser.add_argument("--gpt-ckpt", default=str(DEFAULT_GPT_CKPT))
    parser.add_argument("--train-args", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--t5-model-type", default=str(DEFAULT_T5_MODEL))
    parser.add_argument("--empty-caption", default="A medical image.")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
