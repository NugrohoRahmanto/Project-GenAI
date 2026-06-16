# Gadang Identity Fine-Tuning

Project ini menguji apakah model diffusion bisa mempelajari konsep visual **Rumah Gadang** melalui keyword khusus `gadang`, lalu memakai konsep itu saat inference/editing.

Eksperimen utama membandingkan dua metode:

- **LoRA Concept Fine-Tuning**
  - Notebook: `notebook/lora_concept_grid.ipynb`
  - Runner: `scripts/run_gadang_lora_grid.sh`
  - Training hanya menyimpan adapter LoRA.
  - Evaluasi memakai InstructPix2Pix image editing.

- **DreamBooth-Style Fine-Tuning**
  - Notebook: `notebook/dreambooth_grid.ipynb`
  - Runner: `scripts/run_gadang_dreambooth_grid.sh`
  - Training menyimpan pipeline/UNet fine-tuned.
  - Evaluasi memakai Stable Diffusion img2img.
  - Trainable UNet disimpan dalam `float32` saat mixed precision aktif agar tidak memicu error `Attempting to unscale FP16 gradients`.

Kedua workflow memakai seed, split, jumlah data, file evaluasi, dan prompt evaluasi yang sama agar perbandingan LoRA vs DreamBooth lebih fair.

## Project Layout

```text
notebook/
  lora_concept_grid.ipynb
  dreambooth_grid.ipynb
  generate_instructpix2pix_finetune_images.ipynb
  instructpix2pix_lora_grid.ipynb
  instructpix2pix_dreambooth_grid.ipynb

scripts/
  run_gadang_lora_grid.sh
  run_gadang_dreambooth_grid.sh
  generate_instructpix2pix_finetune_images.sh
  run_gadang_train_pix2pix_lora_grid.sh
  run_gadang_train_pix2pix_dreambooth_grid.sh
  summarize_gadang_experiments.py

traditional_houses/
  gadang/
  joglo/
  honai/
  panjang/
  tongkonan/

outputs/
  yyyymmdd_hhmmss_<scheme>/
    concept_dataset/
    checkpoints/
    evaluation/
    metrics/
    logs/
    notebook/

results/
  gadang_experiment_summary/
```

## Setup

Install dependency:

```bash
uv sync
```

Workflow utama sekarang memakai checkpoint lokal, sehingga tidak perlu login Hugging Face untuk training/evaluasi LoRA dan DreamBooth Stable Diffusion.

Pastikan file berikut tersedia:

```text
models/v1-5-pruned-emaonly.ckpt
models/instruct-pix2pix-00-22000.ckpt
models/flux-fill-nf4/
models/sdxl-inpainting/sd_xl_base_1.0_inpainting_0.1.safetensors
```

## Dataset

Dataset utama diletakkan di:

```text
traditional_houses/gadang/
```

Folder evaluasi pembanding:

```text
traditional_houses/joglo/
traditional_houses/honai/
traditional_houses/panjang/
traditional_houses/tongkonan/
```

Untuk evaluasi Tongkonan, pipeline memprioritaskan file:

```text
traditional_houses/tongkonan/tongkonan (5).png
```

Semua gambar akan di-resize dengan center-crop, bukan diberi border putih.

## Smoke Test

Gunakan smoke test 1 step untuk memastikan training, checkpoint, logging, dan evaluasi bisa jalan.

### LoRA

```bash
IMAGE_COUNTS="10" LEARNING_RATES="1e-4" LORA_RANKS="8" \
MAX_TRAIN_STEPS=1 CHECKPOINT_EVERY=1 \
RUN_EVALUATION=1 RUN_METRICS=0 \
EVAL_IMAGES_PER_CLASS=1 EVAL_PROMPT_COUNT=2 EVAL_NUM_INFERENCE_STEPS=5 \
scripts/run_gadang_lora_grid.sh
```

### DreamBooth

```bash
IMAGE_COUNTS="10" LEARNING_RATES="1e-5" DREAMBOOTH_SCOPES="8" \
MAX_TRAIN_STEPS=1 CHECKPOINT_EVERY=1 \
RUN_EVALUATION=1 RUN_METRICS=0 \
EVAL_IMAGES_PER_CLASS=1 EVAL_PROMPT_COUNT=2 EVAL_NUM_INFERENCE_STEPS=5 \
scripts/run_gadang_dreambooth_grid.sh
```

## Generate InstructPix2Pix Fine-Tuning Pairs

Sebelum fine-tune InstructPix2Pix, kamu bisa membuat pasangan data editing dari gambar rumah tradisional asli. Generator ini memilih 50 gambar source secara deterministik dari kelas non-Gadang, membuat mask edit, lalu menghasilkan target edit dengan dua model:

```text
flux_fill_nf4
sdxl_inpainting
```

Command utama:

```bash
nohup scripts/generate_instructpix2pix_finetune_images.sh >/dev/null 2>&1 &
```

Pantau dengan:

```bash
tail -f output.log
```

Konfigurasi yang bisa diubah:

```bash
SOURCE_IMAGES=50
GENERATION_MODELS="flux_fill_nf4 sdxl_inpainting"
GENERATION_RESOLUTION=1024
GENERATION_STEPS=30
GENERATION_GUIDANCE_SCALE=7.5
GENERATION_STRENGTH=0.95
FLUX_FILL_DIR="models/flux-fill-nf4"
SDXL_INPAINT_CKPT="models/sdxl-inpainting/sd_xl_base_1.0_inpainting_0.1.safetensors"
```

Output:

```text
outputs/<timestamp>_train_pix2pix_dataset_img50/
  generated_instructpix2pix_dataset/
    input_images/
    masks/
    edited_images/
      flux_fill_nf4/
      sdxl_inpainting/
    comparison_grids/
    source_manifest.json
    metadata.jsonl
    generation_manifest.json
```

`metadata.jsonl` berisi pasangan `input_image`, `mask_image`, `edited_image`, dan `edit_prompt` untuk fine-tune InstructPix2Pix.

## Train Pix2Pix From Generated Pairs

Setelah dataset generator menghasilkan `generated_instructpix2pix_dataset/metadata.jsonl`, dua runner berikut bisa langsung fine-tune InstructPix2Pix memakai pasangan:

```text
input_image + edit_prompt -> edited_image
```

Smoke test LoRA:

```bash
PAIR_COUNTS="50" LEARNING_RATES="1e-4" LORA_RANKS="8" \
MAX_TRAIN_STEPS=1 CHECKPOINT_EVERY=1 \
RUN_EVALUATION=1 RUN_METRICS=0 EVAL_PAIR_COUNT=2 EVAL_NUM_INFERENCE_STEPS=5 \
scripts/run_gadang_train_pix2pix_lora_grid.sh
```

Smoke test DreamBooth-style:

```bash
PAIR_COUNTS="50" LEARNING_RATES="1e-5" DREAMBOOTH_SCOPES="8" \
MAX_TRAIN_STEPS=1 CHECKPOINT_EVERY=1 \
RUN_EVALUATION=1 RUN_METRICS=0 EVAL_PAIR_COUNT=2 EVAL_NUM_INFERENCE_STEPS=5 \
scripts/run_gadang_train_pix2pix_dreambooth_grid.sh
```

Jika ingin menunjuk metadata tertentu:

```bash
TRAIN_PIX2PIX_METADATA="outputs/<run>/generated_instructpix2pix_dataset/metadata.jsonl" \
scripts/run_gadang_train_pix2pix_lora_grid.sh
```

Filter pasangan berdasarkan model generator:

```bash
TRAIN_PIX2PIX_GENERATORS="sdxl_inpainting"
TRAIN_PIX2PIX_GENERATORS="flux_fill_nf4 sdxl_inpainting"
```

Output run train-pix2pix:

```text
outputs/<timestamp>_train_pix2pix_lora_pairs50_<scheme>/
outputs/<timestamp>_train_pix2pix_dreambooth_pairs50_<scheme>/
```

## Full Grid Runs

LoRA default grid:

```text
IMAGE_COUNTS="10 25 50"
LEARNING_RATES="5e-5 1e-4 2e-4"
LORA_RANKS="4 8 16"
STABLE_DIFFUSION_CKPT="models/v1-5-pruned-emaonly.ckpt"
MAX_TRAIN_STEPS=2000
CHECKPOINT_EVERY=200
```

Jalankan LoRA:

```bash
nohup scripts/run_gadang_lora_grid.sh >/dev/null 2>&1 &
```

DreamBooth default grid:

```text
IMAGE_COUNTS="10 25 50"
LEARNING_RATES="5e-6 1e-5 2e-5"
DREAMBOOTH_SCOPES="4 8 16"
STABLE_DIFFUSION_CKPT="models/v1-5-pruned-emaonly.ckpt"
MAX_TRAIN_STEPS=2000
CHECKPOINT_EVERY=200
```

Jalankan DreamBooth:

```bash
nohup scripts/run_gadang_dreambooth_grid.sh >/dev/null 2>&1 &
```

Jalankan LoRA train-pix2pix full grid:

```bash
nohup scripts/run_gadang_train_pix2pix_lora_grid.sh >/dev/null 2>&1 &
```

Jalankan DreamBooth train-pix2pix full grid:

```bash
nohup scripts/run_gadang_train_pix2pix_dreambooth_grid.sh >/dev/null 2>&1 &
```

## Runtime Options

Opsi umum untuk kedua runner:

```bash
IMAGE_COUNTS="10 25 50"
LEARNING_RATES="5e-5 1e-4 2e-4"
STABLE_DIFFUSION_CKPT="models/v1-5-pruned-emaonly.ckpt"
INSTRUCT_PIX2PIX_CKPT="models/instruct-pix2pix-00-22000.ckpt"
MAX_TRAIN_STEPS=2000
CHECKPOINT_EVERY=200
GRADIENT_ACCUMULATION_STEPS=4
TRAIN_BATCH_SIZE=1
RUN_EVALUATION=1
RUN_METRICS=1
EVAL_IMAGES_PER_CLASS=2
EVAL_PROMPT_COUNT=3
EVAL_NUM_INFERENCE_STEPS=30
SKIP_EXISTING=1
ROOT_LOG=output.log
```

Opsi khusus LoRA:

```bash
LORA_RANKS="4 8 16"
```

Opsi khusus DreamBooth:

```bash
DREAMBOOTH_SCOPES="4 8 16"
EVAL_IMAGE_GUIDANCE_SCALE=1.5
EVAL_STRENGTH=0.75
```

## Logging

Setiap run menulis log ke root dan folder run:

```text
output.log
outputs/<run_name>/logs/runner.log
outputs/<run_name>/logs/run_output.log
```

`output.log` di root akan ditimpa setiap run baru. Pantau dengan:

```bash
tail -f output.log
```

Nama run selalu memakai timestamp:

```text
outputs/yyyymmdd_hhmmss_img10_lr1em4_rank8_steps2000/
outputs/yyyymmdd_hhmmss_img10_lr1em5_scope8_steps2000/
```

## Checkpoints

Training dikontrol dengan optimizer step, bukan epoch. Default:

```text
STABLE_DIFFUSION_CKPT="models/v1-5-pruned-emaonly.ckpt"
MAX_TRAIN_STEPS=2000
CHECKPOINT_EVERY=200
```

LoRA dan DreamBooth memuat Stable Diffusion dari checkpoint lokal tersebut dengan `from_single_file`. Jika file tidak ada, notebook akan berhenti dengan pesan error yang eksplisit.

LoRA evaluation juga memuat InstructPix2Pix dari checkpoint lokal:

```text
INSTRUCT_PIX2PIX_CKPT="models/instruct-pix2pix-00-22000.ckpt"
```

LoRA checkpoint:

```text
outputs/<run_name>/checkpoints/step_0200/lora_adapter/
outputs/<run_name>/checkpoints/step_0400/lora_adapter/
...
outputs/<run_name>/checkpoints/final/lora_adapter/
```

LoRA juga menyimpan path kompatibilitas untuk checkpoint final:

```text
outputs/<run_name>/checkpoints/lora_adapter/
```

DreamBooth checkpoint:

```text
outputs/<run_name>/checkpoints/step_0200/pipeline/
outputs/<run_name>/checkpoints/step_0200/unet/
...
outputs/<run_name>/checkpoints/final/pipeline/
outputs/<run_name>/checkpoints/final/unet/
```

DreamBooth juga menyimpan path kompatibilitas untuk checkpoint final:

```text
outputs/<run_name>/checkpoints/pipeline/
outputs/<run_name>/checkpoints/unet/
```

Daftar checkpoint disimpan di:

```text
outputs/<run_name>/checkpoints/checkpoints_manifest.json
```

## Evaluation Output

Jika `RUN_EVALUATION=1`, evaluasi dibuat untuk setiap checkpoint yang tersimpan.

Output per checkpoint:

```text
outputs/<run_name>/evaluation/checkpoints/<checkpoint_label>/generated_images/
outputs/<run_name>/evaluation/checkpoints/<checkpoint_label>/grids/
```

Manifest dan record evaluasi:

```text
outputs/<run_name>/evaluation/eval_manifest.json
outputs/<run_name>/evaluation/evaluation_records.json
```

Grid per checkpoint berisi kolom:

```text
input | original/base model | fine-tuned checkpoint
```

LoRA memakai InstructPix2Pix sebagai base edit pipeline. DreamBooth memakai Stable Diffusion img2img. Keduanya memakai file input dan prompt evaluasi yang sama.

## Run Artifacts

Setiap run menghasilkan struktur berikut:

```text
outputs/<run_name>/run_config.json
outputs/<run_name>/artifact_log.json
outputs/<run_name>/concept_dataset/metadata.jsonl
outputs/<run_name>/concept_dataset/split_manifest.json
outputs/<run_name>/checkpoints/checkpoints_manifest.json
outputs/<run_name>/evaluation/eval_manifest.json
outputs/<run_name>/evaluation/evaluation_records.json
outputs/<run_name>/evaluation/checkpoints/
outputs/<run_name>/metrics/metrics_summary.json
outputs/<run_name>/metrics/training_loss.csv
outputs/<run_name>/logs/runner.log
outputs/<run_name>/logs/run_output.log
outputs/<run_name>/notebook/<run_name>.ipynb
```

## Metrics

Notebook menghitung metrik berikut jika `RUN_METRICS=1`:

- FID terhadap held-out real Gadang images
- CLIP text-image similarity jika CLIP berhasil dimuat
- image diversity score sederhana

Kolom metrik LoRA:

```text
fid_original_vs_real_gadang
fid_fine_tuned_vs_real_gadang
clip_text_image_original_mean
clip_text_image_fine_tuned_mean
```

Kolom metrik DreamBooth:

```text
fid_original_vs_real_gadang
fid_dreambooth_vs_real_gadang
clip_text_image_original_mean
clip_text_image_dreambooth_mean
```

Interpretasi umum:

```text
FID lebih rendah lebih baik.
CLIP text-image similarity lebih tinggi lebih baik.
```

## Summarize Results

Setelah beberapa run selesai:

```bash
uv run python scripts/summarize_gadang_experiments.py
```

Output summary:

```text
results/gadang_experiment_summary/gadang_lora_vs_dreambooth_metrics.csv
```

## Fair Comparison Rules

- LoRA dan DreamBooth memakai seed yang sama.
- Split train/val/test Gadang dibuat deterministik.
- Jumlah concept image dikontrol oleh `IMAGE_COUNTS`.
- Jumlah step dikontrol oleh `MAX_TRAIN_STEPS`.
- Evaluasi memakai jumlah gambar yang sama dari `EVAL_IMAGES_PER_CLASS`.
- Evaluasi memakai jumlah prompt yang sama dari `EVAL_PROMPT_COUNT`.
- Tongkonan eval memprioritaskan `tongkonan (5).png`.
- Semua output run disimpan di satu folder `outputs/<run_name>/`.
