import os
import glob
import numpy as np
import torch
from omegaconf import OmegaConf

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
def compute_codebook_usage_counts(
    model,
    dataloader,
    device="cuda",
    max_batches=None,
):
    n_embed = int(model.codebook_size) if hasattr(model, "codebook_size") else int(model.quantize.n_e)
    usage_counts = torch.zeros(n_embed, dtype=torch.long, device="cpu")

    for bi, batch in enumerate(dataloader):
        if max_batches is not None and bi >= max_batches:
            break

        x = batch[model.image_key]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).contiguous().float().to(device)

        _, _, info = model.encode(x)
        indices = info[-1]
        indices = indices.reshape(-1).detach().to("cpu")

        usage_counts += torch.bincount(indices, minlength=n_embed)

        if (bi + 1) % 50 == 0:
            print(f"[count] processed batches: {bi+1}")

    return usage_counts


def entropy_from_counts(counts, base: float = 2.0, eps: float = 1e-12):
    """
    counts: shape [K] 的非负整数频次
    返回:
      H: Shannon 熵 (单位由 base 决定；base=2 -> bits)
      perplexity: exp(H) (自然底) / 2^H (若 base=2 则为 2^H)
      p: 概率分布
    """
    if isinstance(counts, torch.Tensor):
        counts = counts.detach().cpu().numpy()
    counts = np.asarray(counts, dtype=np.float64)

    total = float(np.sum(counts))
    if total <= 0:
        raise ValueError("counts 总和为 0，无法计算熵。")

    p = counts / total
    mask = p > 0
    p_nz = p[mask]

    # H = - sum p log(p)
    if base is None:  # natural log
        H = -float(np.sum(p_nz * np.log(p_nz + eps)))
        perplexity = float(np.exp(H))
    else:
        H = -float(np.sum(p_nz * (np.log(p_nz + eps) / np.log(base))))
        # perplexity 与 base 对应：base=2 时 perplexity = 2^H
        perplexity = float(base ** H)

    return H, perplexity, p


def print_entropy_report(counts, savedir=None, base: float = 2.0):
    H, ppl, p = entropy_from_counts(counts, base=base)
    nonzero = int(np.sum(p > 0))
    K = int(p.size)

    lines = [
        f"Codebook size (K): {K}",
        f"Nonzero codes: {nonzero} / {K}",
        f"Entropy H (base={base}): {H:.6f}",
        f"Perplexity (base={base}): {ppl:.6f}",
        f"Normalized entropy H/log(K): {H / (np.log(K) / np.log(base)):.6f}",
    ]

    for line in lines:
        print(line)

    if savedir:
        os.makedirs(savedir, exist_ok=True)
        out_path = os.path.join(savedir, "entropy.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[save] entropy report -> {out_path}")


def analyze_codebook_entropy_from_logdir(
    log_dir: str,
    datamodule=None,
    device="cuda",
    split="validation",
    max_batches=None,
    out_subdir="codebook_entropy",
    base: float = 2.0,
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

    counts = compute_codebook_usage_counts(model, dl, device=device, max_batches=max_batches)

    save_dir = os.path.join(log_dir, out_subdir)
    os.makedirs(save_dir, exist_ok=True)

    print_entropy_report(counts, savedir=save_dir, base=base)
    return counts, save_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "val", "test"])
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_subdir", type=str, default="codebook_entropy")
    parser.add_argument("--base", type=float, default=2.0, help="熵的对数底；2=bits，np.e=nat（可用 2 或 2.718...）")
    args = parser.parse_args()

    project_yaml = _find_project_yaml(args.logdir)
    cfg = OmegaConf.load(project_yaml)
    datamodule = instantiate_from_config(cfg.data)

    analyze_codebook_entropy_from_logdir(
        args.logdir,
        datamodule=datamodule,
        device=args.device,
        split=args.split,
        max_batches=args.max_batches,
        out_subdir=args.out_subdir,
        base=args.base,
    )