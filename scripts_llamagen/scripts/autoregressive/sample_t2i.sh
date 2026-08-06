# !/bin/bash
set -x

CUDA_VISIBLE_DEVICES=0 torchrun \
--nnodes=1 --nproc_per_node=1 --node_rank=0 \
--master_port=12346 \
autoregressive/sample/sample_t2i.py \
--vq-log-dir /home/disk1/lihaoran/vq-gan/taming-transformers/logs/2026-01-16T14-12-20_vqgan_with_larp \
--gpt-ckpt /home/disk1/lihaoran/vq-gan/taming-transformers/cloud_results/2026-03-10-15-48-10/001-GPT-L/checkpoints/0010000.pt \
"$@"
