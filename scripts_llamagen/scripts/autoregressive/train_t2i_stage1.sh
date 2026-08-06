# !/bin/bash
set -x

CUDA_VISIBLE_DEVICES=0,1,2,3,5 torchrun \
  --nnodes=1 --nproc_per_node=5 --node_rank=0 \
  --master_port=12337 \
  autoregressive/train/train_t2i.py \
  --vq-log-dir /home/disk1/lihaoran/vq-gan/taming-transformers/logs/2026-01-16T14-12-20_vqgan_with_larp \
  --data-path /home/disk1/lihaoran/vq-gan/taming-transformers/data/roco/1/rocov2/train_captions_modified.csv \
  --t5-feat-path /home/disk1/lihaoran/vq-gan/taming-transformers/data/roco/1/rocov2/train_images/train \
  --cloud-save-path /home/disk1/lihaoran/vq-gan/taming-transformers/cloud_results \
  --dataset t2i \
  --image-size 256 \
  --global-batch-size 120\
  "$@"
