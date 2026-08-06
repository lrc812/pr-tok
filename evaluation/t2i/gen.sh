#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data/roco/1/rocov2}"

GPUS="${GPUS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29542}"
PROMPTS_CSV="${PROMPTS_CSV:-${DATA_ROOT}/valid_captions.csv}"
VQ_LOG_DIR="${VQ_LOG_DIR:-${PROJECT_DIR}/logs/2026-07-29T23-40-25_my_vqgan_experiment}"
GPT_CKPT="${GPT_CKPT:-${PROJECT_DIR}/results_roco_t2i/my_vqgan_experiment/checkpoints/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/evaluation/t2i/images_roco_valid_best}"
T5_MODEL_TYPE="${T5_MODEL_TYPE:-${PROJECT_DIR}/pretrained_models/flan-t5-large-local}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PRECISION="${PRECISION:-bf16}"
CFG_SCALE="${CFG_SCALE:-7.5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:-1000}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-23}"
UV_BIN="${UV_BIN:-uv}"

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NPROC_PER_NODE="${#GPU_LIST[@]}"
if (( NPROC_PER_NODE < 1 )); then
    echo "GPUS cannot be empty." >&2
    exit 2
fi
for required_path in \
    "${PROMPTS_CSV}" \
    "${VQ_LOG_DIR}/checkpoints/last.ckpt" \
    "${GPT_CKPT}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 2
    fi
done

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Starting ROCO valid generation:"
echo "  GPUs=${GPUS} (${NPROC_PER_NODE} processes)"
echo "  prompts=${PROMPTS_CSV}"
echo "  GPT=${GPT_CKPT}"
echo "  output=${OUTPUT_DIR}"
echo "  batch/GPU=${BATCH_SIZE}"

exec "${UV_BIN}" run --frozen torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    evaluation/t2i/generation.py \
    --prompts-csv "${PROMPTS_CSV}" \
    --vq-log-dir "${VQ_LOG_DIR}" \
    --gpt-ckpt "${GPT_CKPT}" \
    --output-dir "${OUTPUT_DIR}" \
    --t5-model-type "${T5_MODEL_TYPE}" \
    --batch-size "${BATCH_SIZE}" \
    --precision "${PRECISION}" \
    --cfg-scale "${CFG_SCALE}" \
    --temperature "${TEMPERATURE}" \
    --top-k "${TOP_K}" \
    --top-p "${TOP_P}" \
    --seed "${SEED}" \
    "$@"
