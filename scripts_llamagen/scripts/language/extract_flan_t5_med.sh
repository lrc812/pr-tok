# !/bin/bash
set -x

CUDA_VISIBLE_DEVICES=0,1,2,3,5 torchrun \
  --nnodes=1 --nproc_per_node=5 --node_rank=0 \
  --master_port=12337 \
  language/extract_t5_feature.py \
  --data-path /home/disk1/lihaoran/vq-gan/taming-transformers/data/roco/1/rocov2/train_captions_modified.csv \
  --t5-path /home/disk1/lihaoran/vq-gan/taming-transformers/t5_feat \
  --trunc-caption \
  "$@"
