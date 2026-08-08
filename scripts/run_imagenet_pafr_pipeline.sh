#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/data1/lhran/proj/vq-gan/taming-transformers"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
TORCHRUN="$PROJECT_ROOT/.venv/bin/torchrun"
TRAIN_LIST="$PROJECT_ROOT/data/imagenet/imagenet_train.txt"
VAL_LIST="$PROJECT_ROOT/data/imagenet/imagenet_val.txt"
BASE_DIR="$PROJECT_ROOT/pretrained_models/vqgan_f16_16384"
BASE_CONFIG="$BASE_DIR/config.yaml"
BASE_CHECKPOINT="$BASE_DIR/model.ckpt"
EXPECTED_CHECKPOINT_SHA256="ef7accb930de61689bc697b578933b230ef710a212706e985210a1d0947f7456"

TOKENIZER_DIR="$PROJECT_ROOT/results_pafr_vq/imagenet_original_vqgan_f16_16384_pafr_hybrid_r25_20260807"
PAFR_CHECKPOINT="$TOKENIZER_DIR/checkpoints/best.pt"
FID_DIR="$PROJECT_ROOT/evaluation/vae/fid_results"
GENERATOR_ROOT="$PROJECT_ROOT/results_imagenet_c2i"
GENERATOR_RUN="imagenet_llamagen_gpt_l_original_vqgan_pafr_20260807"
PIPELINE_LOG_DIR="$PROJECT_ROOT/pipeline_logs"
PIPELINE_LOG="$PIPELINE_LOG_DIR/imagenet_original_vqgan_pafr_pipeline_20260807.log"

mkdir -p "$PIPELINE_LOG_DIR" "$FID_DIR" "$GENERATOR_ROOT"
exec > >(tee -a "$PIPELINE_LOG") 2>&1
cd "$PROJECT_ROOT"

for required in "$PYTHON" "$TORCHRUN" "$TRAIN_LIST" "$VAL_LIST" "$BASE_CONFIG" "$BASE_CHECKPOINT"; do
    if [[ ! -e "$required" ]]; then echo "Missing required path: $required"; exit 1; fi
done
actual_sha256="$(sha256sum "$BASE_CHECKPOINT" | cut -d ' ' -f 1)"
if [[ "$actual_sha256" != "$EXPECTED_CHECKPOINT_SHA256" ]]; then
    echo "Original VQGAN checkpoint checksum mismatch: $actual_sha256"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NO_ALBUMENTATIONS_UPDATE=1
echo "[$(date --iso-8601=seconds)] ImageNet pipeline started; base=$BASE_DIR; sha256=$actual_sha256"

echo "[$(date --iso-8601=seconds)] Stage 1/3: train ImageNet PAFR over original VQGAN f16/16384"
"$TORCHRUN" --standalone --nproc_per_node=4 scripts/train_pafr_imagenet.py \
    --train-list "$TRAIN_LIST" --val-list "$VAL_LIST" \
    --base-config "$BASE_CONFIG" --base-checkpoint "$BASE_CHECKPOINT" \
    --output-dir "$TOKENIZER_DIR" --expected-vocab-size 16384 \
    --scorer hybrid --active-ratio 0.25 --residual-dim 64 --residual-codebook-size 1024 \
    --epochs 25 --batch-size 8 --num-workers 8 --precision bf16 --val-max-batches 100

if [[ ! -f "$PAFR_CHECKPOINT" ]]; then echo "Missing PAFR checkpoint: $PAFR_CHECKPOINT"; exit 1; fi

echo "[$(date --iso-8601=seconds)] Stage 2/3: evaluate all 50,000 ImageNet validation reconstructions"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" evaluation/vae/imagenet_pafr_reconstruction_fid.py \
    --val-list "$VAL_LIST" --base-config "$BASE_CONFIG" --base-checkpoint "$BASE_CHECKPOINT" \
    --pafr-checkpoint "$PAFR_CHECKPOINT" --output-dir "$FID_DIR" \
    --active-ratio 0.25 --scorer hybrid --batch-size 12 --num-workers 8 --precision bf16 --dims 2048

echo "[$(date --iso-8601=seconds)] Stage 3/3: train text-free ImageNet class-to-image LlamaGen"
"$TORCHRUN" --standalone --nproc_per_node=4 --module autoregressive.train.train_imagenet_pafr_c2i \
    --train-list "$TRAIN_LIST" --val-list "$VAL_LIST" \
    --base-config "$BASE_CONFIG" --base-checkpoint "$BASE_CHECKPOINT" --pafr-checkpoint "$PAFR_CHECKPOINT" \
    --results-dir "$GENERATOR_ROOT" --run-name "$GENERATOR_RUN" \
    --gpt-model GPT-L --vocab-size 16384 --num-classes 1000 --class-dropout-prob 0.1 \
    --global-batch-size 64 --epochs 60 --num-workers 8 --precision bf16 --val-max-batches 100

echo "[$(date --iso-8601=seconds)] ImageNet pipeline completed successfully"
