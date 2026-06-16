#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE_COUNTS="${IMAGE_COUNTS:-10 25 50}"
LEARNING_RATES="${LEARNING_RATES:-5e-6 1e-5 2e-5}"
DREAMBOOTH_SCOPES="${DREAMBOOTH_SCOPES:-4 8 16}"
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
EVAL_IMAGE_GUIDANCE_SCALE="${EVAL_IMAGE_GUIDANCE_SCALE:-1.5}"
EVAL_STRENGTH="${EVAL_STRENGTH:-0.75}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

ROOT_LOG="${ROOT_LOG:-output.log}"
PROJECT_ROOT="$(pwd)"

log_msg() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$message" >> "$ROOT_LOG"
}

slug_value() {
  local value="$1"
  printf '%s' "$value" | sed 's/\./p/g; s/-/m/g; s/[^A-Za-z0-9_]/_/g'
}

log_msg "Starting Gadang DreamBooth grid"
log_msg "IMAGE_COUNTS=$IMAGE_COUNTS"
log_msg "LEARNING_RATES=$LEARNING_RATES"
log_msg "DREAMBOOTH_SCOPES=$DREAMBOOTH_SCOPES"
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
    for scope in $DREAMBOOTH_SCOPES; do
      lr_slug="$(slug_value "$learning_rate")"
      scope_slug="$(slug_value "$scope")"
      run_stamp="$(date '+%Y%m%d_%H%M%S')"
      run_name="${run_stamp}_img${image_count}_lr${lr_slug}_scope${scope_slug}_steps${MAX_TRAIN_STEPS}"
      run_dir="outputs/${run_name}"
      pipeline_path="${run_dir}/checkpoints/pipeline/model_index.json"
      mkdir -p "${run_dir}/logs" "${run_dir}/notebook"
      executed_notebook="${PROJECT_ROOT}/${run_dir}/notebook/${run_name}.ipynb"
      log_path="${run_dir}/logs/runner.log"

      if [[ "$SKIP_EXISTING" == "1" && -f "$pipeline_path" ]]; then
        log_msg "Skipping existing run: $run_name"
        continue
      fi

      : > "$ROOT_LOG"
      log_msg "Running DreamBooth run: $run_name"
      log_msg "Run log: $log_path"
      log_msg "Run directory: $run_dir"
      (
        export GADANG_RUN_NAME="$run_name"
        export GADANG_RUN_STAMP="$run_stamp"
        export GADANG_NOTEBOOK_WRITES_ROOT_LOG=0
        export GADANG_CONCEPT_IMAGES="$image_count"
        export GADANG_LEARNING_RATE="$learning_rate"
        export GADANG_DREAMBOOTH_SCOPE="$scope"
        export GADANG_LORA_RANK="$scope"
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
        export GADANG_EVAL_IMAGE_GUIDANCE_SCALE="$EVAL_IMAGE_GUIDANCE_SCALE"
        export GADANG_EVAL_STRENGTH="$EVAL_STRENGTH"

        uv run jupyter nbconvert \
          --to notebook \
          --execute notebook/dreambooth_grid.ipynb \
          --output "${executed_notebook}" \
          --ExecutePreprocessor.timeout=-1
      ) > >(tee "$log_path" >> "$ROOT_LOG") 2> >(tee -a "$log_path" >> "$ROOT_LOG")

      log_msg "Finished DreamBooth run: $run_name"
      log_msg "Executed notebook: $executed_notebook"
    done
  done
done

log_msg "All requested Gadang DreamBooth grid runs finished."
