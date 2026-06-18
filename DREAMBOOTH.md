# DreamBooth-Style Gadang Fine-Tuning

This document keeps the DreamBooth-style workflow separate from the main LoRA concept README.

The DreamBooth experiment trains a heavier Stable Diffusion image-to-image pipeline on Rumah Gadang images. It is useful as a comparison point, but it is not the main pipeline of this repository.

Main files:

```text
notebook/dreambooth_grid.ipynb
scripts/run_gadang_dreambooth_grid.sh
```

## Method

The DreamBooth-style workflow:

```text
Rumah Gadang images
        -> Stable Diffusion 1.5 DreamBooth-style fine-tuning
        -> saved fine-tuned pipeline / UNet
        -> Stable Diffusion img2img evaluation
```

Unlike the LoRA concept pipeline, this workflow does not only save a small LoRA adapter. It saves larger fine-tuned pipeline artifacts.

## Setup

Install dependencies:

```bash
uv sync
```

Prepare the local Stable Diffusion 1.5 checkpoint:

```text
models/v1-5-pruned-emaonly.ckpt
```

Dataset folders:

```text
traditional_houses/gadang/
traditional_houses/joglo/
traditional_houses/honai/
traditional_houses/panjang/
traditional_houses/tongkonan/
```

## Smoke Test

Run a tiny DreamBooth-style smoke test:

```bash
IMAGE_COUNTS="10" LEARNING_RATES="1e-5" DREAMBOOTH_SCOPES="8" \
MAX_TRAIN_STEPS=1 CHECKPOINT_EVERY=1 \
RUN_EVALUATION=1 RUN_METRICS=0 \
EVAL_IMAGES_PER_CLASS=1 EVAL_PROMPT_COUNT=2 EVAL_NUM_INFERENCE_STEPS=5 \
scripts/run_gadang_dreambooth_grid.sh
```

Watch the root log:

```bash
tail -f output.log
```

## Full DreamBooth Grid

Default grid:

```text
IMAGE_COUNTS="10 25 50"
LEARNING_RATES="5e-6 1e-5 2e-5"
DREAMBOOTH_SCOPES="4 8 16"
STABLE_DIFFUSION_CKPT="models/v1-5-pruned-emaonly.ckpt"
MAX_TRAIN_STEPS=2000
CHECKPOINT_EVERY=200
```

Run:

```bash
nohup scripts/run_gadang_dreambooth_grid.sh >/dev/null 2>&1 &
```

## Runtime Options

```bash
IMAGE_COUNTS="10 25 50"
LEARNING_RATES="5e-6 1e-5 2e-5"
DREAMBOOTH_SCOPES="4 8 16"
STABLE_DIFFUSION_CKPT="models/v1-5-pruned-emaonly.ckpt"
MAX_TRAIN_STEPS=2000
CHECKPOINT_EVERY=200
GRADIENT_ACCUMULATION_STEPS=4
TRAIN_BATCH_SIZE=1
RUN_EVALUATION=1
RUN_METRICS=1
EVAL_IMAGES_PER_CLASS=2
EVAL_PROMPT_COUNT=3
EVAL_NUM_INFERENCE_STEPS=30
EVAL_IMAGE_GUIDANCE_SCALE=1.5
EVAL_STRENGTH=0.75
SKIP_EXISTING=1
ROOT_LOG=output.log
```

## Checkpoints

DreamBooth-style checkpoints are larger than LoRA adapter checkpoints:

```text
outputs/<run_name>/checkpoints/step_0200/pipeline/
outputs/<run_name>/checkpoints/step_0200/unet/
...
outputs/<run_name>/checkpoints/final/pipeline/
outputs/<run_name>/checkpoints/final/unet/
```

Compatibility paths for the final model:

```text
outputs/<run_name>/checkpoints/pipeline/
outputs/<run_name>/checkpoints/unet/
```

Checkpoint metadata:

```text
outputs/<run_name>/checkpoints/checkpoints_manifest.json
```

## Evaluation

DreamBooth evaluation uses Stable Diffusion img2img, not InstructPix2Pix.

Evaluation output:

```text
outputs/<run_name>/evaluation/checkpoints/<checkpoint_label>/generated_images/
outputs/<run_name>/evaluation/checkpoints/<checkpoint_label>/grids/
outputs/<run_name>/evaluation/eval_manifest.json
outputs/<run_name>/evaluation/evaluation_records.json
```

Grid columns:

```text
input | original/base img2img | dreambooth output
```

## Metrics

When `RUN_METRICS=1`, the notebook computes:

- FID against held-out real Gadang images
- CLIP text-image similarity if CLIP is locally available
- simple image diversity score

Important metric fields:

```text
fid_original_vs_real_gadang
fid_dreambooth_vs_real_gadang
clip_text_image_original_mean
clip_text_image_dreambooth_mean
```

Lower FID is better. Higher CLIP text-image similarity is better.
