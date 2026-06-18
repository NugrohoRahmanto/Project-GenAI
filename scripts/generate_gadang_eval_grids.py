from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RUN_RE = re.compile(
    r"^(?P<stamp>\d{8}_\d{6})_img(?P<images>\d+)_lr(?P<lr>.+)_rank(?P<rank>\d+)_steps(?P<steps>\d+)$"
)
CHECKPOINT_LABELS = [f"step_{step:04d}" for step in range(200, 2001, 200)]
PROMPT_INDICES = [1, 2, 3]
VARIANT_SUFFIX = {
    "input": "input",
    "original": "original",
    "fine_tuned": "fine_tuned",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparison grids from Gadang LoRA evaluation images."
    )
    parser.add_argument("--outputs-dir", default="outputs", type=Path)
    parser.add_argument("--results-dir", default="results/gadang_evaluation_grids", type=Path)
    parser.add_argument("--image-counts", default="10 25 50")
    parser.add_argument("--tile-size", default=192, type=int)
    parser.add_argument("--label-height", default=42, type=int)
    parser.add_argument("--row-label-width", default=220, type=int)
    parser.add_argument("--max-eval-images", default=8, type=int)
    return parser.parse_args()


def run_sort_key(run_dir: Path) -> tuple[int, str, int, str]:
    match = RUN_RE.match(run_dir.name)
    if not match:
        return (999, "", 999, run_dir.name)
    return (int(match.group("images")), match.group("lr"), int(match.group("rank")), run_dir.name)


def discover_runs(outputs_dir: Path, image_count: int) -> list[Path]:
    runs = []
    for run_dir in outputs_dir.glob(f"*_img{image_count}_lr*_rank*_steps*"):
        if (run_dir / "evaluation" / "checkpoints").exists() and RUN_RE.match(run_dir.name):
            runs.append(run_dir)
    return sorted(runs, key=run_sort_key)


def scheme_label(run_dir: Path) -> str:
    match = RUN_RE.match(run_dir.name)
    if not match:
        return run_dir.name
    return f"lr={match.group('lr')} rank={match.group('rank')}"


def discover_eval_image_ids(run_dirs: list[Path]) -> list[str]:
    ids = set()
    for run_dir in run_dirs:
        generated = run_dir / "evaluation" / "checkpoints" / CHECKPOINT_LABELS[-1] / "generated_images"
        if not generated.exists():
            generated = run_dir / "evaluation" / "checkpoints" / "final" / "generated_images"
        if generated.exists():
            ids.update(path.name for path in generated.iterdir() if path.is_dir())
    return sorted(ids)


def image_path(run_dir: Path, checkpoint: str, image_id: str, prompt_index: int, variant: str) -> Path:
    return (
        run_dir
        / "evaluation"
        / "checkpoints"
        / checkpoint
        / "generated_images"
        / image_id
        / f"prompt{prompt_index:02d}_{VARIANT_SUFFIX[variant]}.png"
    )


def load_tile(path: Path, tile_size: int) -> Image.Image:
    tile = Image.new("RGB", (tile_size, tile_size), "white")
    if not path.exists():
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, tile_size - 1, tile_size - 1), outline=(220, 220, 220))
        draw.text((12, tile_size // 2 - 8), "missing", fill=(160, 0, 0))
        return tile
    image = Image.open(path).convert("RGB")
    image = image.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    return image


def draw_centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill=(20, 20, 20)) -> None:
    x0, y0, x1, y1 = box
    lines = []
    words = text.split()
    current = ""
    max_width = x1 - x0 - 10
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    line_height = font.size + 2 if hasattr(font, "size") else 14
    total_height = len(lines) * line_height
    y = y0 + max((y1 - y0 - total_height) // 2, 0)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = x0 + max((x1 - x0 - (bbox[2] - bbox[0])) // 2, 0)
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height


def make_grid(
    run_dirs: list[Path],
    image_id: str,
    out_path: Path,
    tile_size: int,
    label_height: int,
    row_label_width: int,
) -> dict:
    columns = ["Input", "Original Instruct"] + [label.replace("step_", "") for label in CHECKPOINT_LABELS]
    rows = [(run_dir, prompt_index) for run_dir in run_dirs for prompt_index in PROMPT_INDICES]
    width = row_label_width + len(columns) * tile_size
    height = label_height + len(rows) * (tile_size + label_height)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        header_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        header_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.rectangle((0, 0, width - 1, height - 1), outline=(210, 210, 210))
    draw_centered_text(draw, (0, 0, row_label_width, label_height), image_id, header_font)
    for col_index, col_name in enumerate(columns):
        x = row_label_width + col_index * tile_size
        draw.rectangle((x, 0, x + tile_size, label_height), fill=(245, 245, 245), outline=(220, 220, 220))
        draw_centered_text(draw, (x, 0, x + tile_size, label_height), col_name, header_font)

    missing = 0
    for row_index, (run_dir, prompt_index) in enumerate(rows):
        y0 = label_height + row_index * (tile_size + label_height)
        label_y1 = y0 + tile_size + label_height
        draw.rectangle((0, y0, row_label_width, label_y1), fill=(248, 248, 248), outline=(220, 220, 220))
        draw_centered_text(
            draw,
            (0, y0, row_label_width, label_y1),
            f"{scheme_label(run_dir)}\nprompt {prompt_index}",
            small_font,
        )

        base_checkpoint = CHECKPOINT_LABELS[0]
        input_path = image_path(run_dir, base_checkpoint, image_id, prompt_index, "input")
        original_path = image_path(run_dir, base_checkpoint, image_id, prompt_index, "original")
        paths = [input_path, original_path] + [
            image_path(run_dir, checkpoint, image_id, prompt_index, "fine_tuned")
            for checkpoint in CHECKPOINT_LABELS
        ]
        for col_index, path in enumerate(paths):
            x = row_label_width + col_index * tile_size
            label_y = y0
            tile_y = y0 + label_height
            if not path.exists():
                missing += 1
            draw.rectangle((x, label_y, x + tile_size, tile_y), fill=(255, 255, 255), outline=(230, 230, 230))
            if col_index >= 2:
                draw_centered_text(draw, (x, label_y, x + tile_size, tile_y), CHECKPOINT_LABELS[col_index - 2].replace("step_", ""), small_font)
            elif col_index == 0:
                draw_centered_text(draw, (x, label_y, x + tile_size, tile_y), "input", small_font)
            else:
                draw_centered_text(draw, (x, label_y, x + tile_size, tile_y), "original", small_font)
            tile = load_tile(path, tile_size)
            canvas.paste(tile, (x, tile_y))
            draw.rectangle((x, tile_y, x + tile_size, tile_y + tile_size), outline=(230, 230, 230))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return {
        "image_id": image_id,
        "output_path": str(out_path),
        "rows": len(rows),
        "columns": len(columns),
        "missing_cells": missing,
    }


def main() -> None:
    args = parse_args()
    image_counts = [int(item) for item in args.image_counts.split()]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for image_count in image_counts:
        run_dirs = discover_runs(args.outputs_dir, image_count)
        if not run_dirs:
            print(f"[img{image_count}] no runs found")
            continue
        image_ids = discover_eval_image_ids(run_dirs)[: args.max_eval_images]
        print(f"[img{image_count}] runs={len(run_dirs)} eval_images={len(image_ids)}")
        group_dir = args.results_dir / f"img{image_count}"
        for image_id in image_ids:
            out_path = group_dir / f"{image_id}_grid.png"
            row = make_grid(
                run_dirs=run_dirs,
                image_id=image_id,
                out_path=out_path,
                tile_size=args.tile_size,
                label_height=args.label_height,
                row_label_width=args.row_label_width,
            )
            row.update(
                {
                    "image_count": image_count,
                    "run_count": len(run_dirs),
                    "run_names": "|".join(run_dir.name for run_dir in run_dirs),
                }
            )
            manifest_rows.append(row)
            print(f"  saved {out_path}")

    manifest_path = args.results_dir / "grid_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_count",
            "image_id",
            "output_path",
            "rows",
            "columns",
            "missing_cells",
            "run_count",
            "run_names",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
