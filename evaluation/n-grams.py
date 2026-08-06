import os
import glob
import math
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from omegaconf import OmegaConf

# 使用无显示环境也可保存图片
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import instantiate_from_config


def _find_project_yaml(log_dir: str) -> str:
    cfg_dir = os.path.join(log_dir, "configs")
    if not os.path.isdir(cfg_dir):
        raise FileNotFoundError(f"configs 目录不存在: {cfg_dir}")

    ymls = sorted(glob.glob(os.path.join(cfg_dir, "*project.yaml")))
    if len(ymls) == 0:
        ymls = sorted(glob.glob(os.path.join(cfg_dir, "*project*.yaml")))
    if len(ymls) == 0:
        raise FileNotFoundError(f"在 {cfg_dir} 中找不到 *project.yaml 配置文件")

    return ymls[-1]


def _find_ckpt(log_dir: str) -> str:
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"checkpoints 目录不存在: {ckpt_dir}")

    last = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.isfile(last):
        return last

    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"在 {ckpt_dir} 中找不到任何 .ckpt 文件")
    return ckpts[-1]


def load_model_from_logdir(log_dir: str, device: str = "cuda"):
    project_yaml = _find_project_yaml(log_dir)
    ckpt_path = _find_ckpt(log_dir)

    cfg = OmegaConf.load(project_yaml)
    model = instantiate_from_config(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)

    print(f"[load] config: {project_yaml}")
    print(f"[load] ckpt:   {ckpt_path}")
    if len(missing) > 0:
        print(f"[load] missing keys (show up to 20): {missing[:20]}")
    if len(unexpected) > 0:
        print(f"[load] unexpected keys (show up to 20): {unexpected[:20]}")

    model.eval().to(device)
    return model, cfg


@torch.no_grad()
def _extract_indices_from_batch(model, batch, device: str = "cuda") -> torch.Tensor:
    """
    返回 indices（量化 code 的索引），通常形状是 (B,H,W) 或 (B,HW)。
    """
    x = batch[model.image_key]
    if len(x.shape) == 3:
        x = x[..., None]
    x = x.permute(0, 3, 1, 2).contiguous().float().to(device)

    _, _, info = model.encode(x)
    indices = info[-1]
    return indices.detach().to("cpu")


def _flatten_to_sequences(indices: torch.Tensor) -> torch.Tensor:
    """
    把 indices 转成 (B, L) 的 token 序列（按行优先展开）。
    """
    if indices.dim() == 1:
        return indices.view(1, -1)
    if indices.dim() == 2:
        return indices
    b = indices.shape[0]
    return indices.view(b, -1)


def _vocab_size_from_model(model) -> int:
    return int(model.codebook_size) if hasattr(model, "codebook_size") else int(model.quantize.n_e)


def _entropy_from_counts_numpy(counts: np.ndarray, base: float = 2.0, eps: float = 1e-12):
    """
    counts: 1D array of non-negative counts
    return: (H, perplexity, nonzero, total)
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return 0.0, 0.0, 0, 0.0

    p = counts / total
    p = p[p > 0]
    if p.size == 0:
        return 0.0, 0.0, 0, total

    if base is None:
        H = -float(np.sum(p * np.log(p + eps)))
        ppl = float(np.exp(H))
    else:
        H = -float(np.sum(p * (np.log(p + eps) / np.log(base))))
        ppl = float(base ** H)

    nonzero = int((counts > 0).sum())
    return H, ppl, nonzero, total


def _save_entropy_report(
    *,
    counts: np.ndarray,
    out_path: str,
    n: int,
    vocab_size: int,
    base: float = 2.0,
):
    K = int(vocab_size**n)
    H, ppl, nonzero, total = _entropy_from_counts_numpy(counts, base=base)
    H_norm = H / math.log(K, base) if K > 1 else 0.0

    lines = [
        f"n={n}",
        f"vocab_size={vocab_size}",
        f"num_possible_ngrams(K)={K}",
        f"total_ngrams_counted={int(total)}",
        f"nonzero_ngrams={nonzero}",
        f"density(nonzero/K)={nonzero / max(K,1):.10f}",
        f"Entropy H (base={base})={H:.8f}",
        f"Perplexity (base={base})={ppl:.8f}",
        f"Normalized entropy H/log_base(K)={H_norm:.8f}",
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[save] entropy -> {out_path}")


def _save_basic_stats(counts: np.ndarray, out_path: str, n: int, vocab_size: int, topk: int = 20):
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    nonzero = int((counts > 0).sum())
    K = int(vocab_size**n)

    lines = [
        f"n={n}",
        f"vocab_size={vocab_size}",
        f"num_possible_ngrams(K)={K}",
        f"total_ngrams_counted={total}",
        f"nonzero_ngrams={nonzero}",
        f"zero_ngrams={K - nonzero}",
        f"density(nonzero/K)={nonzero / max(K, 1):.10f}",
    ]
    if counts.size > 0:
        lines += [
            f"max_count={int(counts.max())}",
            f"min_count={int(counts.min())}",
            f"mean_count={float(counts.mean()):.6f}",
            f"median_count={float(np.median(counts)):.6f}",
        ]

    if counts.size > 0 and counts.size <= 20_000_000:
        top_idx = np.argsort(-counts)[:topk]
        lines += ["", f"top{topk}_hashed_ngram_id_and_count:"]
        lines += [f"{int(i)}\t{int(counts[int(i)])}" for i in top_idx]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[save] stats -> {out_path}")


def _save_histogram_fixed_binwidth(
    values: np.ndarray,
    out_path: str,
    title: str,
    *,
    bin_width: float = 250.0,
    xlim: Optional[Tuple[float, float]] = None,
    log_y: bool = True,
    xlabel: str = "Frequency",
    ylabel: str = "Number of unique n-grams",
):
    """
    直方图关键：用固定 bin_width 控制“间隔”。
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    values = values[values >= 0]
    if values.size == 0:
        values = np.array([0.0], dtype=np.float64)

    if xlim is None:
        xmin, xmax = 0.0, float(np.max(values))
        xmax = max(xmax, bin_width)
    else:
        xmin, xmax = float(xlim[0]), float(xlim[1])

    edges = np.arange(xmin, xmax + bin_width, bin_width)
    if edges.size < 2:
        edges = np.array([xmin, xmin + bin_width], dtype=np.float64)

    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=edges, color="tab:blue", alpha=0.85, edgecolor="black", linewidth=0.4)
    if xlim is not None:
        plt.xlim(xlim)
    if log_y:
        plt.yscale("log")
    plt.title(f"{title} (bin_width={bin_width:g})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[save] hist -> {out_path}")


def _count_2grams_dense(seqs: torch.Tensor, vocab_size: int) -> torch.Tensor:
    base = int(vocab_size)
    out_size = base * base
    total = torch.zeros(out_size, dtype=torch.long)

    for i in range(seqs.shape[0]):
        s = seqs[i].to(torch.long)
        L = int(s.numel())
        if L < 2:
            continue
        a = s[0 : L - 1]
        b = s[1 : L]
        h = (a * base + b).to(torch.long)
        total += torch.bincount(h, minlength=out_size).to(torch.long)

    return total


def _count_3grams_sparse_dict(seqs: torch.Tensor, vocab_size: int) -> Dict[int, int]:
    base = int(vocab_size)
    merged: Dict[int, int] = {}
    for i in range(seqs.shape[0]):
        s = seqs[i].to(torch.long)
        L = int(s.numel())
        if L < 3:
            continue
        a = s[0 : L - 2]
        b = s[1 : L - 1]
        c = s[2 : L]
        h = (a * (base**2) + b * base + c).to(torch.long).numpy()
        for key in h:
            k = int(key)
            merged[k] = merged.get(k, 0) + 1
    return merged


@torch.no_grad()
def compute_ngram_usage_counts(
    model,
    dataloader,
    n: int,
    device: str = "cuda",
    max_batches=None,
):
    vocab_size = _vocab_size_from_model(model)

    if n == 2:
        out_size = vocab_size * vocab_size
        usage_counts = torch.zeros(out_size, dtype=torch.long, device="cpu")

        for bi, batch in enumerate(dataloader):
            if max_batches is not None and bi >= max_batches:
                break

            indices = _extract_indices_from_batch(model, batch, device=device)
            seqs = _flatten_to_sequences(indices)

            usage_counts += _count_2grams_dense(seqs, vocab_size=vocab_size).to("cpu")

            if (bi + 1) % 50 == 0:
                print(f"[2-gram] processed batches: {bi+1}")

        return usage_counts, vocab_size

    if n == 3:
        merged: Dict[int, int] = {}
        for bi, batch in enumerate(dataloader):
            if max_batches is not None and bi >= max_batches:
                break
            indices = _extract_indices_from_batch(model, batch, device=device)
            seqs = _flatten_to_sequences(indices)

            d = _count_3grams_sparse_dict(seqs, vocab_size=vocab_size)
            for k, v in d.items():
                merged[k] = merged.get(k, 0) + v

            if (bi + 1) % 50 == 0:
                print(f"[3-gram] processed batches: {bi+1}, unique_3grams={len(merged)}")

        return merged, vocab_size

    raise ValueError("Only n=2 or n=3 supported.")


def analyze_ngrams_from_logdir(
    log_dir: str,
    datamodule=None,
    device: str = "cuda",
    split: str = "validation",
    max_batches=None,
    out_subdir: str = "codebook_ngrams",
    base: float = 2.0,
    log_y: bool = True,
    xlim_max: Optional[int] = 30000,
    bin_width: float = 250.0,
    plot_nonzero_only: bool = True,
):
    model, cfg = load_model_from_logdir(log_dir, device=device)

    if datamodule is None:
        raise ValueError("请传入 datamodule（与训练一致）。例如：datamodule = instantiate_from_config(cfg.data)")

    datamodule.prepare_data()
    datamodule.setup()

    if split == "train":
        dl = datamodule.train_dataloader()
    elif split in ("validation", "val"):
        dl = datamodule.val_dataloader()
    elif split == "test":
        dl = datamodule.test_dataloader()
    else:
        raise ValueError(f"Unknown split: {split}")

    save_dir = os.path.join(log_dir, out_subdir)
    os.makedirs(save_dir, exist_ok=True)

    # 2-gram
    counts2_t, vocab_size = compute_ngram_usage_counts(model, dl, n=2, device=device, max_batches=max_batches)
    counts2 = counts2_t.detach().cpu().numpy().astype(np.int64)

    np.save(os.path.join(save_dir, "2gram_counts.npy"), counts2)
    _save_basic_stats(counts2, out_path=os.path.join(save_dir, "2gram_stats.txt"), n=2, vocab_size=vocab_size)
    _save_entropy_report(counts=counts2, out_path=os.path.join(save_dir, "2gram_entropy.txt"), n=2, vocab_size=vocab_size, base=base)

    values2 = counts2[counts2 > 0] if plot_nonzero_only else counts2
    _save_histogram_fixed_binwidth(
        values2,
        out_path=os.path.join(save_dir, "2gram_hist.png"),
        title="2-gram frequency histogram",
        bin_width=bin_width,
        xlim=(0.0, float(xlim_max)) if xlim_max is not None else None,
        log_y=log_y,
        xlabel="Frequency of 2-grams",
        ylabel="Number of unique 2-grams",
    )

    # 3-gram
    counts3_dict, vocab_size = compute_ngram_usage_counts(model, dl, n=3, device=device, max_batches=max_batches)
    keys3 = np.fromiter(counts3_dict.keys(), dtype=np.int64) if len(counts3_dict) > 0 else np.zeros((0,), dtype=np.int64)
    vals3 = np.fromiter(counts3_dict.values(), dtype=np.int64) if len(counts3_dict) > 0 else np.zeros((0,), dtype=np.int64)

    np.save(os.path.join(save_dir, "3gram_keys.npy"), keys3)
    np.save(os.path.join(save_dir, "3gram_values.npy"), vals3)

    _save_basic_stats(vals3, out_path=os.path.join(save_dir, "3gram_stats.txt"), n=3, vocab_size=vocab_size)
    _save_entropy_report(counts=vals3, out_path=os.path.join(save_dir, "3gram_entropy.txt"), n=3, vocab_size=vocab_size, base=base)

    values3 = vals3[vals3 > 0] if plot_nonzero_only else vals3
    _save_histogram_fixed_binwidth(
        values3 if values3.size > 0 else np.array([0], dtype=np.int64),
        out_path=os.path.join(save_dir, "3gram_hist.png"),
        title="3-gram frequency histogram",
        bin_width=bin_width,
        xlim=(0.0, float(xlim_max)) if xlim_max is not None else None,
        log_y=log_y,
        xlabel="Frequency of 3-grams",
        ylabel="Number of unique 3-grams",
    )

    print(f"[save] n-gram outputs -> {save_dir}")
    return save_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "val", "test"])
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_subdir", type=str, default="codebook_ngrams")
    parser.add_argument("--base", type=float, default=2.0, help="熵的对数底；2=bits")

    parser.add_argument("--bin_width", type=float, default=250.0, help="直方图固定 bin 宽度（间隔），如 100/250/500")
    parser.add_argument("--xlim_max", type=int, default=30000, help="x 轴最大值（例如 30000）")
    parser.add_argument("--log_y", action="store_true", help="y 轴使用 log scale")
    parser.add_argument("--include_zero", action="store_true", help="把 count=0 也画进直方图（默认不画）")

    args = parser.parse_args()

    project_yaml = _find_project_yaml(args.logdir)
    cfg = OmegaConf.load(project_yaml)
    datamodule = instantiate_from_config(cfg.data)

    analyze_ngrams_from_logdir(
        args.logdir,
        datamodule=datamodule,
        device=args.device,
        split=args.split,
        max_batches=args.max_batches,
        out_subdir=args.out_subdir,
        base=args.base,
        log_y=args.log_y,
        xlim_max=args.xlim_max,
        bin_width=args.bin_width,
        plot_nonzero_only=(not args.include_zero),
    )