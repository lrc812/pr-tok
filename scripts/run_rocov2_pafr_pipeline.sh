#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/data1/lhran/proj/vq-gan/taming-transformers"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
TORCHRUN="$PROJECT_ROOT/.venv/bin/torchrun"
TRAIN_LIST="$PROJECT_ROOT/data/roco/1/rocov2/medical_train.txt"
TEST_LIST="$PROJECT_ROOT/data/roco/1/rocov2/medical_test.txt"
BASE_LOG="$PROJECT_ROOT/logs/2026-08-05T17-06-10_rocov2_larp_l2norm_20260805"
BASE_CONFIG="$BASE_LOG/configs/2026-08-05T17-06-10-project.yaml"
BASE_CHECKPOINT="$BASE_LOG/checkpoints/last.ckpt"

TOKENIZER_DIR="$PROJECT_ROOT/results_pafr_vq/rocov2_pafr_hybrid_r25_20260806"
PAFR_CHECKPOINT="$TOKENIZER_DIR/checkpoints/best.pt"
FID_DIR="$PROJECT_ROOT/evaluation/vae/fid_results"
GENERATOR_ROOT="$PROJECT_ROOT/results_roco_uncond"
GENERATOR_RUN="rocov2_llamagen_gpt_l_pafr_base_20260806"
PIPELINE_LOG_DIR="$PROJECT_ROOT/pipeline_logs"
PIPELINE_LOG="$PIPELINE_LOG_DIR/rocov2_pafr_pipeline_20260806.log"

mkdir -p "$PIPELINE_LOG_DIR" "$FID_DIR" "$GENERATOR_ROOT"
exec > >(tee -a "$PIPELINE_LOG") 2>&1
cd "$PROJECT_ROOT"

for required in "$PYTHON" "$TORCHRUN" "$TRAIN_LIST" "$TEST_LIST" "$BASE_CONFIG" "$BASE_CHECKPOINT"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required path: $required"
        exit 1
    fi
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
echo "[$(date --iso-8601=seconds)] Pipeline started on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo "[$(date --iso-8601=seconds)] Stage 1/3: train ROCOv2 PAFR tokenizer"
"$TORCHRUN" --standalone --nproc_per_node=4 \
    scripts/train_pafr_rocov2.py \
    --train-list "$TRAIN_LIST" \
    --val-list "$TEST_LIST" \
    --base-config "$BASE_CONFIG" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --output-dir "$TOKENIZER_DIR" \
    --scorer hybrid \
    --active-ratio 0.25 \
    --residual-dim 64 \
    --residual-codebook-size 1024 \
    --epochs 25 \
    --batch-size 8 \
    --num-workers 8 \
    --precision bf16 \
    --val-max-batches 100

if [[ ! -f "$PAFR_CHECKPOINT" ]]; then
    echo "Tokenizer finished without expected checkpoint: $PAFR_CHECKPOINT"
    exit 1
fi

echo "[$(date --iso-8601=seconds)] Stage 2/3: evaluate full ROCOv2 reconstruction rFID"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" evaluation/vae/rocov2_pafr_reconstruction_fid.py \
    --test-list "$TEST_LIST" \
    --base-config "$BASE_CONFIG" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --pafr-checkpoint "$PAFR_CHECKPOINT" \
    --output-dir "$FID_DIR" \
    --active-ratio 0.25 \
    --scorer hybrid \
    --batch-size 12 \
    --num-workers 8 \
    --precision bf16 \
    --dims 2048 \
    --allow-tf32

echo "[$(date --iso-8601=seconds)] Stage 3/3: train text-free ROCOv2 LlamaGen"
"$TORCHRUN" --standalone --nproc_per_node=4 \
    --module autoregressive.train.train_roco_uncond \
    --train-list "$TRAIN_LIST" \
    --val-list "$TEST_LIST" \
    --base-config "$BASE_CONFIG" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --pafr-checkpoint "$PAFR_CHECKPOINT" \
    --results-dir "$GENERATOR_ROOT" \
    --run-name "$GENERATOR_RUN" \
    --gpt-model GPT-L \
    --global-batch-size 64 \
    --epochs 60 \
    --num-workers 8 \
    --precision bf16 \
    --val-max-batches 100

echo "[$(date --iso-8601=seconds)] All three stages completed successfully"
