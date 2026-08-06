#!/usr/bin/env bash
set -Eeuo pipefail

# Train the ROCO LlamaGen text-to-image generator with the LARP tokenizer
# produced by the 2026-08-04 lr50 + metrics run.
#
# Common overrides:
#   GPUS=0,1 GLOBAL_BATCH_SIZE=96 ./scripts/train_roco_larp_lr50_generator.sh
#   RESUME=results_roco_t2i/<run>/checkpoints/last.pt \
#     ./scripts/train_roco_larp_lr50_generator.sh

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export VQ_LOG_DIR="${VQ_LOG_DIR:-${PROJECT_DIR}/logs/2026-08-04T21-08-29_roco_larp_lr50_metrics}"
export RUN_NAME="${RUN_NAME:-roco_llamagen_gpt_l_larp_lr50_20260804}"
export MASTER_PORT="${MASTER_PORT:-29542}"

exec "${PROJECT_DIR}/scripts/train_roco_llamagen_t2i.sh" "$@"
