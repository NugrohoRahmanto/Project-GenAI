#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE_COUNTS="${IMAGE_COUNTS:-10 25 50}"
TILE_SIZE="${TILE_SIZE:-192}"
LABEL_HEIGHT="${LABEL_HEIGHT:-42}"
ROW_LABEL_WIDTH="${ROW_LABEL_WIDTH:-220}"
MAX_EVAL_IMAGES="${MAX_EVAL_IMAGES:-8}"
RESULTS_DIR="${RESULTS_DIR:-results/gadang_evaluation_grids}"

uv run python scripts/generate_gadang_eval_grids.py \
  --image-counts "$IMAGE_COUNTS" \
  --tile-size "$TILE_SIZE" \
  --label-height "$LABEL_HEIGHT" \
  --row-label-width "$ROW_LABEL_WIDTH" \
  --max-eval-images "$MAX_EVAL_IMAGES" \
  --results-dir "$RESULTS_DIR"
