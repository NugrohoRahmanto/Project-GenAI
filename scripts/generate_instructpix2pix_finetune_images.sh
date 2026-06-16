#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE_IMAGES="${SOURCE_IMAGES:-50}"
GENERATION_MODELS="${GENERATION_MODELS:-flux_fill_nf4 sdxl_inpainting}"
GENERATION_RESOLUTION="${GENERATION_RESOLUTION:-1024}"
GENERATION_STEPS="${GENERATION_STEPS:-30}"
GENERATION_GUIDANCE_SCALE="${GENERATION_GUIDANCE_SCALE:-7.5}"
GENERATION_STRENGTH="${GENERATION_STRENGTH:-0.95}"
SEED="${SEED:-100}"
FLUX_FILL_DIR="${FLUX_FILL_DIR:-models/flux-fill-nf4}"
FLUX_BASE_MODEL="${FLUX_BASE_MODEL:-black-forest-labs/FLUX.1-dev}"
SDXL_INPAINT_CKPT="${SDXL_INPAINT_CKPT:-models/sdxl-inpainting/sd_xl_base_1.0_inpainting_0.1.safetensors}"
ROOT_LOG="${ROOT_LOG:-output.log}"

PROJECT_ROOT="$(pwd)"
RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_NAME="${RUN_STAMP}_train_pix2pix_dataset_img${SOURCE_IMAGES}"
RUN_DIR="outputs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
NOTEBOOK_DIR="${RUN_DIR}/notebook"
RUNNER_LOG="${LOG_DIR}/runner.log"
EXECUTED_NOTEBOOK="${PROJECT_ROOT}/${NOTEBOOK_DIR}/${RUN_NAME}.ipynb"

mkdir -p "$LOG_DIR" "$NOTEBOOK_DIR"

log_msg() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$message" >> "$ROOT_LOG"
}

: > "$ROOT_LOG"
log_msg "Starting InstructPix2Pix fine-tune image generation"
log_msg "RUN_NAME=$RUN_NAME"
log_msg "SOURCE_IMAGES=$SOURCE_IMAGES"
log_msg "GENERATION_MODELS=$GENERATION_MODELS"
log_msg "GENERATION_RESOLUTION=$GENERATION_RESOLUTION"
log_msg "GENERATION_STEPS=$GENERATION_STEPS"
log_msg "GENERATION_GUIDANCE_SCALE=$GENERATION_GUIDANCE_SCALE"
log_msg "GENERATION_STRENGTH=$GENERATION_STRENGTH"
log_msg "SEED=$SEED"
log_msg "FLUX_FILL_DIR=$FLUX_FILL_DIR"
log_msg "FLUX_BASE_MODEL=$FLUX_BASE_MODEL"
log_msg "SDXL_INPAINT_CKPT=$SDXL_INPAINT_CKPT"
log_msg "Run directory: $RUN_DIR"
log_msg "Runner log: $RUNNER_LOG"

(
  export GADANG_RUN_STAMP="$RUN_STAMP"
  export GADANG_GENERATOR_RUN_NAME="$RUN_NAME"
  export GADANG_NOTEBOOK_WRITES_ROOT_LOG=0
  export GADANG_GENERATE_SOURCE_IMAGES="$SOURCE_IMAGES"
  export GADANG_GENERATION_MODELS="$GENERATION_MODELS"
  export GADANG_GENERATION_RESOLUTION="$GENERATION_RESOLUTION"
  export GADANG_GENERATION_STEPS="$GENERATION_STEPS"
  export GADANG_GENERATION_GUIDANCE_SCALE="$GENERATION_GUIDANCE_SCALE"
  export GADANG_GENERATION_STRENGTH="$GENERATION_STRENGTH"
  export GADANG_SEED="$SEED"
  export GADANG_FLUX_FILL_DIR="$PROJECT_ROOT/$FLUX_FILL_DIR"
  export GADANG_FLUX_BASE_MODEL="$FLUX_BASE_MODEL"
  export GADANG_SDXL_INPAINT_CKPT="$PROJECT_ROOT/$SDXL_INPAINT_CKPT"

  uv run jupyter nbconvert \
    --to notebook \
    --execute notebook/generate_instructpix2pix_finetune_images.ipynb \
    --output "$EXECUTED_NOTEBOOK" \
    --ExecutePreprocessor.timeout=-1
) > >(tee "$RUNNER_LOG" >> "$ROOT_LOG") 2> >(tee -a "$RUNNER_LOG" >> "$ROOT_LOG")

log_msg "Finished InstructPix2Pix fine-tune image generation"
log_msg "Executed notebook: $EXECUTED_NOTEBOOK"
