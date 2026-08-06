#!/usr/bin/env bash
set -Eeuo pipefail

# Train the LARP-constrained VQ-GAN on ROCO.
#
# Defaults to four-GPU DDP with 12 samples per GPU (global batch size 48).
#
# Common overrides:
#   BATCH_SIZE=10 MAX_EPOCHS=100 ./scripts/train_roco_larp.sh
#   GPUS=0,1, ./scripts/train_roco_larp.sh
#   FAST_DEV_RUN=1 ./scripts/train_roco_larp.sh
#   RESUME=logs/<run>/checkpoints/last.ckpt ./scripts/train_roco_larp.sh
#   DATA_ROOT=/data/roco/1/rocov2 ./scripts/train_roco_larp.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/vqgan_with_larp_open.yaml}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/roco/1/rocov2}"
TRAIN_LIST="${TRAIN_LIST:-${DATA_ROOT}/medical_train.txt}"
VAL_LIST="${VAL_LIST:-${DATA_ROOT}/medical_test.txt}"

BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-25}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
PRECISION="${PRECISION:-bf16}"
GPUS="${GPUS:-0,1,2,3,}"
SEED="${SEED:-23}"
RUN_NAME="${RUN_NAME:-roco_larp_vqgan}"
RESUME="${RESUME:-}"
FAST_DEV_RUN="${FAST_DEV_RUN:-0}"
SYNC_ENV="${SYNC_ENV:-1}"

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

[[ -f "${CONFIG}" ]] || die "config not found: ${CONFIG}"
[[ -s "${TRAIN_LIST}" ]] || die "training list not found or empty: ${TRAIN_LIST}"
[[ -s "${VAL_LIST}" ]] || die "validation list not found or empty: ${VAL_LIST}"

IFS= read -r FIRST_TRAIN_IMAGE < "${TRAIN_LIST}"
FIRST_TRAIN_IMAGE="${FIRST_TRAIN_IMAGE%$'\r'}"
[[ -f "${FIRST_TRAIN_IMAGE}" ]] || die \
    "first training image does not exist: ${FIRST_TRAIN_IMAGE}. Regenerate the list or set DATA_ROOT/TRAIN_LIST."

GPU_LIST="${GPUS%,}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
GPU_COUNT="${#GPU_ARRAY[@]}"
(( GPU_COUNT > 0 )) || die "GPUS must contain at least one GPU index."
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * GPU_COUNT * ACCUMULATE_GRAD_BATCHES))

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [[ -x "/data1/lhran/.local/bin/uv" ]]; then
    UV_BIN="/data1/lhran/.local/bin/uv"
else
    die "uv was not found. Install uv first or add it to PATH."
fi

cd "${PROJECT_ROOT}"

if [[ "${SYNC_ENV}" == "1" ]]; then
    "${UV_BIN}" sync --frozen
fi

TRAIN_ARGS=(
    --base "${CONFIG}"
    --train True
    --no-test True
    --gpus "${GPUS}"
    --precision "${PRECISION}"
    --max_epochs "${MAX_EPOCHS}"
    --accumulate_grad_batches "${ACCUMULATE_GRAD_BATCHES}"
    --seed "${SEED}"
    "data.params.batch_size=${BATCH_SIZE}"
    "data.params.num_workers=${NUM_WORKERS}"
    "data.params.train.params.training_images_list_file=${TRAIN_LIST}"
    "data.params.validation.params.test_images_list_file=${VAL_LIST}"
    "data.params.test.params.test_images_list_file=${VAL_LIST}"
)

if [[ -n "${RESUME}" ]]; then
    [[ -e "${RESUME}" ]] || die "resume path does not exist: ${RESUME}"
    TRAIN_ARGS+=(--resume "${RESUME}")
else
    TRAIN_ARGS+=(--name "${RUN_NAME}")
fi

if [[ "${FAST_DEV_RUN}" == "1" ]]; then
    TRAIN_ARGS+=(--fast_dev_run True)
fi

printf 'ROCO data:       %s\n' "${DATA_ROOT}"
printf 'Train samples:   %s\n' "$(wc -l < "${TRAIN_LIST}")"
printf 'Val samples:     %s\n' "$(wc -l < "${VAL_LIST}")"
printf 'Batch/GPU:       %s\n' "${BATCH_SIZE}"
printf 'GPUs:            %s (%s GPUs)\n' "${GPUS}" "${GPU_COUNT}"
printf 'Global batch:    %s\n' "${GLOBAL_BATCH_SIZE}"
printf 'Precision:       %s\n' "${PRECISION}"
printf 'Max epochs:      %s\n' "${MAX_EPOCHS}"

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
# This host's four RTX 5090 cards have no NVLink and span two NUMA nodes.
# Avoid NCCL probing unavailable P2P/InfiniBand paths; callers may override either.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

exec "${UV_BIN}" run --frozen python main.py "${TRAIN_ARGS[@]}"
