#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
IMAGE_DIR="${IMAGE_DIR:-${PROJECT_ROOT}/data/roco/1/rocov2/valid_images/valid}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/2026-07-29T23-40-25_my_vqgan_experiment}"
TEST_RESULTS_DIR="${TEST_RESULTS_DIR:-${PROJECT_ROOT}/evaluation/t2i/test_results}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
GPU="${GPU:-0}"
PRECISION="${PRECISION:-fp32}"
RECON_BATCH_SIZE="${RECON_BATCH_SIZE:-16}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-50}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DIMS="${DIMS:-2048}"
ANALYSIS_TOP_K="${ANALYSIS_TOP_K:-10}"
PAIRED_METRICS="${PAIRED_METRICS:-1}"
SEED="${SEED:-23}"
MIN_SPLIT_RATIO="${MIN_SPLIT_RATIO:-0.4}"
MAX_SPLIT_RATIO="${MAX_SPLIT_RATIO:-0.6}"
MAX_IMAGES="${MAX_IMAGES:-0}"
TEMP_DIR="${TEMP_DIR:-}"
RECURSIVE="${RECURSIVE:-0}"
ALLOW_TF32="${ALLOW_TF32:-0}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-0}"

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

[[ -x "${PYTHON}" ]] || die "Python not executable: ${PYTHON}"
[[ -d "${IMAGE_DIR}" ]] || die "Image directory not found: ${IMAGE_DIR}"
[[ -d "${LOG_DIR}" ]] || die "Tokenizer log directory not found: ${LOG_DIR}"

ARGS=(
    "${PROJECT_ROOT}/evaluation/t2i/reconstruction_fid.py"
    --image-dir "${IMAGE_DIR}"
    --log-dir "${LOG_DIR}"
    --device cuda:0
    --precision "${PRECISION}"
    --reconstruction-batch-size "${RECON_BATCH_SIZE}"
    --fid-batch-size "${FID_BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --dims "${DIMS}"
    --analysis-top-k "${ANALYSIS_TOP_K}"
    --seed "${SEED}"
    --min-split-ratio "${MIN_SPLIT_RATIO}"
    --max-split-ratio "${MAX_SPLIT_RATIO}"
    --max-images "${MAX_IMAGES}"
    --output-dir "${TEST_RESULTS_DIR}"
)

[[ -z "${OUTPUT_JSON}" ]] || ARGS+=(--output-json "${OUTPUT_JSON}")
[[ -z "${TEMP_DIR}" ]] || ARGS+=(--temp-dir "${TEMP_DIR}")
[[ "${RECURSIVE}" == "1" ]] && ARGS+=(--recursive)
[[ "${ALLOW_TF32}" == "1" ]] && ARGS+=(--allow-tf32)
[[ "${CUDNN_BENCHMARK}" == "1" ]] && ARGS+=(--cudnn-benchmark)
[[ "${PAIRED_METRICS}" == "0" ]] && ARGS+=(--skip-paired-metrics)

printf 'Reconstruction cross-FID configuration:\n'
printf '  image_dir=%s\n' "${IMAGE_DIR}"
printf '  log_dir=%s\n' "${LOG_DIR}"
printf '  gpu=%s precision=%s recon_batch=%s fid_batch=%s workers=%s\n' \
    "${GPU}" "${PRECISION}" "${RECON_BATCH_SIZE}" "${FID_BATCH_SIZE}" "${NUM_WORKERS}"
printf '  split_ratio=[%s, %s] seed=%s dims=%s analysis_top_k=%s\n' \
    "${MIN_SPLIT_RATIO}" "${MAX_SPLIT_RATIO}" "${SEED}" "${DIMS}" \
    "${ANALYSIS_TOP_K}"
printf '  paired_metrics=%s (PSNR/SSIM/LPIPS)\n' "${PAIRED_METRICS}"
if [[ -n "${OUTPUT_JSON}" ]]; then
    printf '  output_json=%s\n' "${OUTPUT_JSON}"
else
    printf '  output_dir=%s (timestamped filename)\n' "${TEST_RESULTS_DIR}"
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
exec "${PYTHON}" "${ARGS[@]}" "$@"
