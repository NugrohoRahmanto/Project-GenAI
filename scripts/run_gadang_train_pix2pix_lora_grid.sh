#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PAIR_COUNTS="${PAIR_COUNTS:-50}"
LEARNING_RATES="${LEARNING_RATES:-5e-5 1e-4 2e-4}"
LORA_RANKS="${LORA_RANKS:-4 8 16}"
INSTRUCT_PIX2PIX_CKPT="${INSTRUCT_PIX2PIX_CKPT:-models/instruct-pix2pix-00-22000.ckpt}"
TRAIN_PIX2PIX_METADATA="${TRAIN_PIX2PIX_METADATA:-}"
TRAIN_PIX2PIX_GENERATORS="${TRAIN_PIX2PIX_GENERATORS:-all}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-200}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
RUN_METRICS="${RUN_METRICS:-1}"
EVAL_PAIR_COUNT="${EVAL_PAIR_COUNT:-4}"
EVAL_NUM_INFERENCE_STEPS="${EVAL_NUM_INFERENCE_STEPS:-30}"
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

latest_metadata() {
  find outputs -path '*/generated_instructpix2pix_dataset/metadata.jsonl' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}'
}

abs_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s/%s\n' "$PROJECT_ROOT" "$value"
  fi
}

if [[ -z "$TRAIN_PIX2PIX_METADATA" ]]; then
  TRAIN_PIX2PIX_METADATA="$(latest_metadata || true)"
fi

if [[ -z "$TRAIN_PIX2PIX_METADATA" ]]; then
  log_msg "No generated train_pix2pix metadata found. Run scripts/generate_instructpix2pix_finetune_images.sh first or set TRAIN_PIX2PIX_METADATA."
  exit 1
fi

TRAIN_PIX2PIX_METADATA_ABS="$(abs_path "$TRAIN_PIX2PIX_METADATA")"
INSTRUCT_PIX2PIX_CKPT_ABS="$(abs_path "$INSTRUCT_PIX2PIX_CKPT")"

log_msg "Starting Gadang train_pix2pix LoRA grid"
log_msg "PAIR_COUNTS=$PAIR_COUNTS"
log_msg "LEARNING_RATES=$LEARNING_RATES"
log_msg "LORA_RANKS=$LORA_RANKS"
log_msg "TRAIN_PIX2PIX_METADATA=$TRAIN_PIX2PIX_METADATA_ABS"
log_msg "TRAIN_PIX2PIX_GENERATORS=$TRAIN_PIX2PIX_GENERATORS"
log_msg "INSTRUCT_PIX2PIX_CKPT=$INSTRUCT_PIX2PIX_CKPT"
log_msg "MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS"
log_msg "CHECKPOINT_EVERY=$CHECKPOINT_EVERY"
log_msg "Root log: $ROOT_LOG"

for pair_count in $PAIR_COUNTS; do
  for learning_rate in $LEARNING_RATES; do
    for lora_rank in $LORA_RANKS; do
      lr_slug="$(slug_value "$learning_rate")"
      run_stamp="$(date '+%Y%m%d_%H%M%S')"
      run_name="${run_stamp}_train_pix2pix_lora_pairs${pair_count}_lr${lr_slug}_rank${lora_rank}_steps${MAX_TRAIN_STEPS}"
      run_dir="outputs/${run_name}"
      adapter_path="${run_dir}/checkpoints/lora_adapter/adapter_model.safetensors"
      mkdir -p "${run_dir}/logs" "${run_dir}/notebook"
      executed_notebook="${PROJECT_ROOT}/${run_dir}/notebook/${run_name}.ipynb"
      log_path="${run_dir}/logs/runner.log"

      if [[ "$SKIP_EXISTING" == "1" && -f "$adapter_path" ]]; then
        log_msg "Skipping existing run: $run_name"
        continue
      fi

      : > "$ROOT_LOG"
      log_msg "Running train_pix2pix LoRA run: $run_name"
      log_msg "Run log: $log_path"
      log_msg "Run directory: $run_dir"
      (
        export GADANG_RUN_NAME="$run_name"
        export GADANG_RUN_STAMP="$run_stamp"
        export GADANG_NOTEBOOK_WRITES_ROOT_LOG=0
        export GADANG_TRAIN_PIX2PIX_PAIRS="$pair_count"
        export GADANG_TRAIN_PIX2PIX_METADATA="$TRAIN_PIX2PIX_METADATA_ABS"
        export GADANG_TRAIN_PIX2PIX_GENERATORS="$TRAIN_PIX2PIX_GENERATORS"
        export GADANG_LEARNING_RATE="$learning_rate"
        export GADANG_LORA_RANK="$lora_rank"
        export GADANG_LORA_ALPHA="$lora_rank"
        export GADANG_INSTRUCT_PIX2PIX_CKPT="$INSTRUCT_PIX2PIX_CKPT_ABS"
        export GADANG_MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS"
        export GADANG_CHECKPOINT_EVERY="$CHECKPOINT_EVERY"
        export GADANG_GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS"
        export GADANG_TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE"
        export GADANG_RUN_EVALUATION="$RUN_EVALUATION"
        export GADANG_RUN_METRICS="$RUN_METRICS"
        export GADANG_EVAL_PAIR_COUNT="$EVAL_PAIR_COUNT"
        export GADANG_EVAL_NUM_INFERENCE_STEPS="$EVAL_NUM_INFERENCE_STEPS"

        uv run jupyter nbconvert \
          --to notebook \
          --execute notebook/instructpix2pix_lora_grid.ipynb \
          --output "${executed_notebook}" \
          --ExecutePreprocessor.timeout=-1
      ) > >(tee "$log_path" >> "$ROOT_LOG") 2> >(tee -a "$log_path" >> "$ROOT_LOG")

      log_msg "Finished train_pix2pix LoRA run: $run_name"
      log_msg "Executed notebook: $executed_notebook"
    done
  done
done

log_msg "All requested Gadang train_pix2pix LoRA grid runs finished."
