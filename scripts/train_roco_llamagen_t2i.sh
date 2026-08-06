#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data/roco/1/rocov2}"

GPUS="${GPUS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29541}"
RUN_NAME="${RUN_NAME:-roco_llamagen_gpt_l}"
GPT_MODEL="${GPT_MODEL:-GPT-L}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-192}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
EPOCHS="${EPOCHS:-60}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
CKPT_EVERY="${CKPT_EVERY:-5000}"
LOG_EVERY="${LOG_EVERY:-20}"
VAL_SIZE="${VAL_SIZE:-2048}"
VAL_EVERY="${VAL_EVERY:-1}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-3}"

DATA_PATH="${DATA_PATH:-${DATA_ROOT}/train_captions_modified.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-${DATA_ROOT}/train_images/train}"
T5_FEAT_PATH="${T5_FEAT_PATH:-${DATA_ROOT}/train_images/train}"
VQ_LOG_DIR="${VQ_LOG_DIR:-/data1/lhran/proj/vq-gan/taming-transformers/logs/2026-07-29T23-40-25_my_vqgan_experiment}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/results_roco_t2i}"
RESUME="${RESUME:-}"
UV_BIN="${UV_BIN:-uv}"
SYNC_ENV="${SYNC_ENV:-1}"

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NPROC_PER_NODE="${#GPU_LIST[@]}"

if (( NPROC_PER_NODE < 1 )); then
    echo "GPUS cannot be empty." >&2
    exit 2
fi
if (( GLOBAL_BATCH_SIZE % NPROC_PER_NODE != 0 )); then
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by ${NPROC_PER_NODE} GPUs." >&2
    exit 2
fi
for required_path in "${DATA_PATH}" "${IMAGE_ROOT}" "${T5_FEAT_PATH}" "${VQ_LOG_DIR}/configs"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 2
    fi
done
if [[ ! -f "${VQ_LOG_DIR}/checkpoints/last.ckpt" ]]; then
    echo "VQ-GAN checkpoint not found: ${VQ_LOG_DIR}/checkpoints/last.ckpt" >&2
    exit 2
fi

cd "${PROJECT_DIR}"
if [[ "${SYNC_ENV}" == "1" ]]; then
    "${UV_BIN}" sync --frozen
fi

RESUME_ARGS=()
if [[ -n "${RESUME}" ]]; then
    if [[ ! -f "${RESUME}" ]]; then
        echo "Resume checkpoint not found: ${RESUME}" >&2
        exit 2
    fi
    RESUME_ARGS=(--resume "${RESUME}")
fi

export CUDA_VISIBLE_DEVICES="${GPUS}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

echo "Starting ROCO LlamaGen training:"
echo "  GPUs=${GPUS} (${NPROC_PER_NODE} processes)"
echo "  model=${GPT_MODEL}, global_batch=${GLOBAL_BATCH_SIZE}, batch/GPU=$((GLOBAL_BATCH_SIZE / NPROC_PER_NODE))"
echo "  run=${RESULTS_DIR}/${RUN_NAME}"

exec "${UV_BIN}" run --frozen torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    --module \
    autoregressive.train.train_roco_t2i \
    --data-path "${DATA_PATH}" \
    --image-root "${IMAGE_ROOT}" \
    --t5-feat-path "${T5_FEAT_PATH}" \
    --vq-log-dir "${VQ_LOG_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --run-name "${RUN_NAME}" \
    --gpt-model "${GPT_MODEL}" \
    --vocab-size 1024 \
    --image-size 256 \
    --downsample-size 16 \
    --cls-token-num 120 \
    --caption-dim 1024 \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --epochs "${EPOCHS}" \
    --lr "${LEARNING_RATE}" \
    --num-workers "${NUM_WORKERS}" \
    --mixed-precision "${MIXED_PRECISION}" \
    --ckpt-every "${CKPT_EVERY}" \
    --log-every "${LOG_EVERY}" \
    --val-size "${VAL_SIZE}" \
    --val-every "${VAL_EVERY}" \
    --max-val-batches "${MAX_VAL_BATCHES}" \
    --keep-last-checkpoints "${KEEP_LAST_CHECKPOINTS}" \
    "${RESUME_ARGS[@]}" \
    "$@"
