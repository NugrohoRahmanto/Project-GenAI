#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE_COUNTS="${IMAGE_COUNTS:-10 25 50}"
LEARNING_RATES="${LEARNING_RATES:-5e-5 1e-4 2e-4}"
LORA_RANKS="${LORA_RANKS:-4 8 16}"
STABLE_DIFFUSION_CKPT="${STABLE_DIFFUSION_CKPT:-models/v1-5-pruned-emaonly.ckpt}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-200}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
RUN_METRICS="${RUN_METRICS:-1}"
EVAL_IMAGES_PER_CLASS="${EVAL_IMAGES_PER_CLASS:-2}"
EVAL_PROMPT_COUNT="${EVAL_PROMPT_COUNT:-3}"
EVAL_NUM_INFERENCE_STEPS="${EVAL_NUM_INFERENCE_STEPS:-30}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

ROOT_LOG="${ROOT_LOG:-output.log}"
PROJECT_ROOT="$(pwd)"

log_msg() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$message" >> "$ROOT_LOG"
}

slug_lr() {
  local lr="$1"
  printf '%s' "$lr" | sed 's/\./p/g; s/-/m/g; s/[^A-Za-z0-9_]/_/g'
}

find_existing_run_dir() {
  local image_count="$1"
  local lr_slug="$2"
  local lora_rank="$3"
  local pattern="outputs/*_img${image_count}_lr${lr_slug}_rank${lora_rank}_steps${MAX_TRAIN_STEPS}"
  compgen -G "$pattern" | sort | tail -n 1 || true
}

log_msg "Starting Gadang LoRA grid"
log_msg "IMAGE_COUNTS=$IMAGE_COUNTS"
log_msg "LEARNING_RATES=$LEARNING_RATES"
log_msg "LORA_RANKS=$LORA_RANKS"
log_msg "STABLE_DIFFUSION_CKPT=$STABLE_DIFFUSION_CKPT"
log_msg "MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS"
log_msg "CHECKPOINT_EVERY=$CHECKPOINT_EVERY"
log_msg "RUN_EVALUATION=$RUN_EVALUATION"
log_msg "RUN_METRICS=$RUN_METRICS"
log_msg "EVAL_IMAGES_PER_CLASS=$EVAL_IMAGES_PER_CLASS"
log_msg "EVAL_PROMPT_COUNT=$EVAL_PROMPT_COUNT"
log_msg "Root log: $ROOT_LOG"

for image_count in $IMAGE_COUNTS; do
  for learning_rate in $LEARNING_RATES; do
    for lora_rank in $LORA_RANKS; do
      lr_slug="$(slug_lr "$learning_rate")"
      existing_run_dir="$(find_existing_run_dir "$image_count" "$lr_slug" "$lora_rank")"
      if [[ -n "$existing_run_dir" ]]; then
        run_dir="$existing_run_dir"
        run_name="$(basename "$run_dir")"
        run_stamp="${run_name%%_img*}"
        resume_mode=1
      else
        run_stamp="$(date '+%Y%m%d_%H%M%S')"
        run_name="${run_stamp}_img${image_count}_lr${lr_slug}_rank${lora_rank}_steps${MAX_TRAIN_STEPS}"
        run_dir="outputs/${run_name}"
        resume_mode=0
      fi
      adapter_path="${run_dir}/checkpoints/lora_adapter/adapter_model.safetensors"
      run_complete_path="${run_dir}/run_complete.json"
      artifact_log_path="${run_dir}/artifact_log.json"
      mkdir -p "${run_dir}/logs" "${run_dir}/notebook"
      executed_notebook="${PROJECT_ROOT}/${run_dir}/notebook/${run_name}.ipynb"
      log_path="${run_dir}/logs/runner.log"

      if [[ "$SKIP_EXISTING" == "1" && -f "$adapter_path" && ( -f "$run_complete_path" || -f "$artifact_log_path" ) ]]; then
        log_msg "Skipping completed run: $run_name"
        continue
      fi

      : > "$ROOT_LOG"
      log_msg "Running LoRA run: $run_name"
      log_msg "Resume mode: $resume_mode"
      log_msg "Run log: $log_path"
      log_msg "Run directory: $run_dir"
      (
        export GADANG_RUN_NAME="$run_name"
        export GADANG_RUN_STAMP="$run_stamp"
        export GADANG_NOTEBOOK_WRITES_ROOT_LOG=0
        export GADANG_CONCEPT_IMAGES="$image_count"
        export GADANG_LEARNING_RATE="$learning_rate"
        export GADANG_LORA_RANK="$lora_rank"
        export GADANG_LORA_ALPHA="$lora_rank"
        export GADANG_STABLE_DIFFUSION_CKPT="$PROJECT_ROOT/$STABLE_DIFFUSION_CKPT"
        export GADANG_MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS"
        export GADANG_CHECKPOINT_EVERY="$CHECKPOINT_EVERY"
        export GADANG_GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS"
        export GADANG_TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE"
        export GADANG_RUN_EVALUATION="$RUN_EVALUATION"
        export GADANG_RUN_METRICS="$RUN_METRICS"
        export GADANG_EVAL_IMAGES_PER_CLASS="$EVAL_IMAGES_PER_CLASS"
        export GADANG_EVAL_PROMPT_COUNT="$EVAL_PROMPT_COUNT"
        export GADANG_EVAL_NUM_INFERENCE_STEPS="$EVAL_NUM_INFERENCE_STEPS"

        uv run jupyter nbconvert \
          --to notebook \
          --execute notebook/lora_concept_grid.ipynb \
          --output "${executed_notebook}" \
          --ExecutePreprocessor.timeout=-1
      ) > >(tee "$log_path" >> "$ROOT_LOG") 2> >(tee -a "$log_path" >> "$ROOT_LOG")

      log_msg "Finished LoRA run: $run_name"
      log_msg "Executed notebook: $executed_notebook"
    done
  done
done

log_msg "All requested Gadang LoRA grid runs finished."
