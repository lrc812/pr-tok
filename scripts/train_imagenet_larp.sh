#!/usr/bin/env bash
set -Eeuo pipefail

# Train the same LARP-constrained VQ-GAN used for ROCO on local ImageNet-1K.
# Missing path manifests are generated once; image files are never copied.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/imagenet_larp_vqgan.yaml}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/imagenet}"
TRAIN_LIST="${TRAIN_LIST:-${DATA_ROOT}/imagenet_train.txt}"
VAL_LIST="${VAL_LIST:-${DATA_ROOT}/imagenet_val.txt}"

BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
PRECISION="${PRECISION:-bf16}"
GPUS="${GPUS:-0,1,2,3,}"
SEED="${SEED:-23}"
RUN_NAME="${RUN_NAME:-imagenet_larp_vqgan}"
RESUME="${RESUME:-}"
FAST_DEV_RUN="${FAST_DEV_RUN:-0}"
SYNC_ENV="${SYNC_ENV:-1}"

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

[[ -f "${CONFIG}" ]] || die "config not found: ${CONFIG}"
[[ -d "${DATA_ROOT}/train" ]] || die "training directory not found: ${DATA_ROOT}/train"
[[ -d "${DATA_ROOT}/val" ]] || die "validation directory not found: ${DATA_ROOT}/val"

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

if [[ ! -s "${TRAIN_LIST}" || ! -s "${VAL_LIST}" ]]; then
    "${UV_BIN}" run --frozen python scripts/prepare_imagenet_vqvae.py \
        --data-root "${DATA_ROOT}"
fi

IFS= read -r FIRST_TRAIN_IMAGE < "${TRAIN_LIST}"
FIRST_TRAIN_IMAGE="${FIRST_TRAIN_IMAGE%$'\r'}"
[[ -f "${FIRST_TRAIN_IMAGE}" ]] || die \
    "first training image does not exist: ${FIRST_TRAIN_IMAGE}"

IFS= read -r FIRST_VAL_IMAGE < "${VAL_LIST}"
FIRST_VAL_IMAGE="${FIRST_VAL_IMAGE%$'\r'}"
[[ -f "${FIRST_VAL_IMAGE}" ]] || die \
    "first validation image does not exist: ${FIRST_VAL_IMAGE}"

GPU_LIST="${GPUS%,}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
GPU_COUNT="${#GPU_ARRAY[@]}"
(( GPU_COUNT > 0 )) || die "GPUS must contain at least one GPU index."
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * GPU_COUNT * ACCUMULATE_GRAD_BATCHES))

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

printf 'ImageNet data:   %s\n' "${DATA_ROOT}"
printf 'Train samples:   %s\n' "$(wc -l < "${TRAIN_LIST}")"
printf 'Val samples:     %s\n' "$(wc -l < "${VAL_LIST}")"
printf 'Batch/GPU:       %s\n' "${BATCH_SIZE}"
printf 'GPUs:            %s (%s GPUs)\n' "${GPUS}" "${GPU_COUNT}"
printf 'Global batch:    %s\n' "${GLOBAL_BATCH_SIZE}"
printf 'Precision:       %s\n' "${PRECISION}"
printf 'Max epochs:      %s\n' "${MAX_EPOCHS}"

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

exec "${UV_BIN}" run --frozen python main.py "${TRAIN_ARGS[@]}"
