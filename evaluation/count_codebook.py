import os
import glob
import re
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
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

    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

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
def reconstruct_with_vqgan(x, model):
    z, _, [_, _, indices] = model.encode(x)
    print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
    xrec = model.decode(z)
    return xrec


def print_codebook_statistics(usage_counts, savedir=None):
    if isinstance(usage_counts, torch.Tensor):
        usage_counts = usage_counts.detach().cpu().numpy()

    usage_counts = np.asarray(usage_counts).astype(np.int64)

    lines = [
        f"Max Frequency: {int(np.max(usage_counts))}",
        f"Min Frequency: {int(np.min(usage_counts))}",
        f"Mean Frequency: {float(np.mean(usage_counts)):.2f}",
        f"Median Frequency: {float(np.median(usage_counts)):.2f}",
        f"Standard Deviation: {float(np.std(usage_counts)):.2f}",
        f"Nonzero Codes: {int(np.sum(usage_counts > 0))} / {usage_counts.size}",
        f"Zero Codes: {int(np.sum(usage_counts == 0))} / {usage_counts.size}",
        f"Total Tokens Counted: {int(np.sum(usage_counts))}",
    ]

    for line in lines:
        print(line)

    if savedir:
        os.makedirs(savedir, exist_ok=True)
        out_path = os.path.join(savedir, "statistic.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[save] statistics -> {out_path}")


def visualize_codebook_usage_heatmap(usage_counts, title="Codebook Usage Heatmap", save_dir=None):
    if isinstance(usage_counts, torch.Tensor):
        usage_counts = usage_counts.detach().cpu().numpy()

    usage_counts = np.asarray(usage_counts).astype(np.float32)

    side_length = int(np.sqrt(len(usage_counts)))
    if side_length ** 2 != len(usage_counts):
        raise ValueError("Codebook size is not a perfect square, cannot reshape into heatmap.")
    usage_counts_2d = usage_counts.reshape(side_length, side_length)

    plt.figure(figsize=(8, 8))
    plt.imshow(usage_counts_2d, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Usage Frequency")
    plt.title(title, fontsize=14)
    plt.xlabel("X")
    plt.ylabel("Y")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "heatmap.jpg")
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"[save] heatmap -> {save_path}")
    else:
        plt.show()


def visualize_codebook_usage_histogram(
    usage_counts,
    title="Codebook Usage Histogram",
    save_dir=None,
    bins=50,
    log_y=True,
):
    if isinstance(usage_counts, torch.Tensor):
        usage_counts = usage_counts.detach().cpu().numpy()

    usage_counts = np.asarray(usage_counts).astype(np.int64)

    plt.figure(figsize=(10, 5))
    plt.hist(usage_counts, bins=bins, color="tab:blue", alpha=0.85)
    if log_y:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("Usage Count per Code")
    plt.ylabel("Number of Codes")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "histogram.jpg")
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"[save] histogram -> {save_path}")
    else:
        plt.show()


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


def analyze_codebook_from_logdir(
    log_dir: str,
    datamodule=None,
    device="cuda",
    split="validation",
    max_batches=None,
    out_subdir="codebook_analysis",
    hist_bins=80,
):
    model, cfg = load_model_from_logdir(log_dir, device=device)

    if datamodule is None:
        raise ValueError(
            "请传入 datamodule（与训练一致）。例如：datamodule = instantiate_from_config(cfg.data)"
        )

    datamodule.prepare_data()
    datamodule.setup()

    if split == "train":
        dl = datamodule.train_dataloader()
    elif split == "validation" or split == "val":
        dl = datamodule.val_dataloader()
    elif split == "test":
        dl = datamodule.test_dataloader()
    else:
        raise ValueError(f"Unknown split: {split}")

    usage_counts = compute_codebook_usage_counts(model, dl, device=device, max_batches=max_batches)

    save_dir = os.path.join(log_dir, out_subdir)
    os.makedirs(save_dir, exist_ok=True)

    print_codebook_statistics(usage_counts, savedir=save_dir)
    visualize_codebook_usage_heatmap(usage_counts, title="Codebook Usage Heatmap", save_dir=save_dir)
    visualize_codebook_usage_histogram(
        usage_counts, title="Codebook Usage Histogram", save_dir=save_dir, bins=hist_bins, log_y=True
    )
    return usage_counts, save_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "val", "test"])
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_subdir", type=str, default="codebook_analysis")
    parser.add_argument("--hist_bins", type=int, default=80)
    args = parser.parse_args()

    project_yaml = _find_project_yaml(args.logdir)
    cfg = OmegaConf.load(project_yaml)
    datamodule = instantiate_from_config(cfg.data)

    analyze_codebook_from_logdir(
        args.logdir,
        datamodule=datamodule,
        device=args.device,
        split=args.split,
        max_batches=args.max_batches,
        out_subdir=args.out_subdir,
        hist_bins=args.hist_bins,
    )