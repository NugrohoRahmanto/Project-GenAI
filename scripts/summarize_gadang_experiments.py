from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOTS = [
    ("all_runs", PROJECT_ROOT / "outputs"),
]
SUMMARY_DIR = PROJECT_ROOT / "results" / "gadang_experiment_summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for method, root in OUTPUT_ROOTS:
        if not root.exists():
            continue
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if not (run_dir / "run_config.json").exists():
                continue
            config = read_json(run_dir / "run_config.json")
            metrics = read_json(run_dir / "metrics" / "metrics_summary.json")
            if not metrics:
                metrics = read_json(run_dir / "metrics_summary.json")
            artifact_log = read_json(run_dir / "artifact_log.json")
            row = {
                "method": metrics.get("method", config.get("method", method)),
                "run_name": run_dir.name,
                "max_concept_images": config.get("max_concept_images"),
                "max_train_steps": config.get("max_train_steps"),
                "learning_rate": config.get("learning_rate"),
                "lora_rank": config.get("lora_rank"),
                "lora_alpha": config.get("lora_alpha"),
                "dreambooth_scope_label": config.get("dreambooth_scope_label"),
                "trainable_mode": config.get("trainable_mode"),
                "trainable_params": config.get("trainable_params"),
                "total_params": config.get("total_params"),
                "metrics_skipped": metrics.get("metrics_skipped", False),
                "fid_original_vs_real_gadang": metrics.get("fid_original_vs_real_gadang"),
                "fid_fine_tuned_vs_real_gadang": metrics.get("fid_fine_tuned_vs_real_gadang"),
                "fid_dreambooth_vs_real_gadang": metrics.get("fid_dreambooth_vs_real_gadang"),
                "clip_text_image_original_mean": metrics.get("clip_text_image_original_mean"),
                "clip_text_image_fine_tuned_mean": metrics.get("clip_text_image_fine_tuned_mean"),
                "clip_text_image_dreambooth_mean": metrics.get("clip_text_image_dreambooth_mean"),
                "diversity_original": metrics.get("diversity_original"),
                "diversity_fine_tuned": metrics.get("diversity_fine_tuned"),
                "diversity_dreambooth": metrics.get("diversity_dreambooth"),
                "output_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                "artifact_log": str((run_dir / "artifact_log.json").relative_to(PROJECT_ROOT)) if artifact_log else None,
            }
            if row["method"] == "lora_concept":
                row["fid_method_vs_real_gadang"] = row["fid_fine_tuned_vs_real_gadang"]
                row["clip_text_image_method_mean"] = row["clip_text_image_fine_tuned_mean"]
                row["diversity_method"] = row["diversity_fine_tuned"]
            elif row["method"] == "dreambooth":
                row["fid_method_vs_real_gadang"] = row["fid_dreambooth_vs_real_gadang"]
                row["clip_text_image_method_mean"] = row["clip_text_image_dreambooth_mean"]
                row["diversity_method"] = row["diversity_dreambooth"]
            rows.append(row)
    return rows


def main() -> None:
    rows = collect_rows()
    df = pd.DataFrame(rows)
    out_path = SUMMARY_DIR / "gadang_lora_vs_dreambooth_metrics.csv"
    if not df.empty:
        sort_cols = [col for col in ["method", "max_concept_images", "learning_rate", "lora_rank", "dreambooth_scope_label"] if col in df.columns]
        df = df.sort_values(sort_cols, na_position="last")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    if df.empty:
        print("No run metrics found yet.")
    else:
        display_cols = [
            "method",
            "run_name",
            "max_concept_images",
            "learning_rate",
            "lora_rank",
            "dreambooth_scope_label",
            "fid_method_vs_real_gadang",
            "clip_text_image_method_mean",
            "diversity_method",
        ]
        print(df[[col for col in display_cols if col in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
