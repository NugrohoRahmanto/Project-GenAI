from __future__ import annotations

import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results" / "gadang_lora_progress"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS_DIR / "gadang_lora_progress.csv"

RUN_RE = re.compile(
    r"^(?P<stamp>\d{8}_\d{6})_img(?P<images>\d+)_lr(?P<lr>.+)_rank(?P<rank>\d+)_steps(?P<target_steps>\d+)$"
)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def checkpoint_info(run_dir: Path, target_steps: int) -> tuple[int, str, bool, int]:
    manifest_path = run_dir / "checkpoints" / "checkpoints_manifest.json"
    manifest = read_json(manifest_path)
    checkpoints = list(manifest.get("checkpoints", []))
    checkpoint_root = run_dir / "checkpoints"
    for step_dir in sorted(checkpoint_root.glob("step_*/lora_adapter")):
        match = re.search(r"step_(\d+)", step_dir.parent.name)
        if match:
            step = int(match.group(1))
            checkpoints.append({"step": step, "label": f"step_{step:04d}", "adapter_path": str(step_dir)})
    final_dir = checkpoint_root / "final" / "lora_adapter"
    if final_dir.exists():
        checkpoints.append({"step": target_steps, "label": "final", "adapter_path": str(final_dir)})
    valid = []
    final_exists = (run_dir / "checkpoints" / "lora_adapter" / "adapter_model.safetensors").exists()
    seen = set()
    for item in checkpoints:
        adapter_path = Path(item.get("adapter_path", ""))
        if not adapter_path.is_absolute():
            adapter_path = PROJECT_ROOT / adapter_path
        key = (int(item.get("step", 0)), str(item.get("label", "")), str(adapter_path))
        if key in seen:
            continue
        seen.add(key)
        if (adapter_path / "adapter_model.safetensors").exists():
            valid.append(item)
            if item.get("label") == "final":
                final_exists = True
    if not valid:
        return 0, "", final_exists, 0
    valid = sorted(valid, key=lambda x: (int(x.get("step", 0)), x.get("label") == "final"))
    latest = valid[-1]
    latest_non_final = [item for item in valid if item.get("label") != "final"]
    resume_latest = (latest_non_final or valid)[-1]
    return int(resume_latest.get("step", 0)), str(resume_latest.get("label", "")), final_exists, len(valid)


def main() -> None:
    rows = []
    for run_dir in sorted(OUTPUTS_DIR.glob("*")):
        if not run_dir.is_dir():
            continue
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        config = read_json(run_dir / "run_config.json")
        artifact_log_exists = (run_dir / "artifact_log.json").exists()
        run_complete_exists = (run_dir / "run_complete.json").exists()
        target_steps = int(match.group("target_steps"))
        latest_step, latest_label, final_exists, checkpoint_count = checkpoint_info(run_dir, target_steps)
        training_complete = final_exists and latest_step >= target_steps
        run_complete = training_complete and (artifact_log_exists or run_complete_exists)
        status = "complete" if run_complete else "partial"
        if training_complete and not run_complete:
            status = "training_complete_pending_finalization"
        if checkpoint_count == 0 and not final_exists:
            status = "no_checkpoint"
        rows.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                "image_count": int(match.group("images")),
                "learning_rate_slug": match.group("lr"),
                "lora_rank": int(match.group("rank")),
                "target_steps": target_steps,
                "latest_checkpoint_step": latest_step,
                "latest_checkpoint_label": latest_label,
                "checkpoint_count": checkpoint_count,
                "final_adapter_exists": final_exists,
                "artifact_log_exists": artifact_log_exists,
                "run_complete_exists": run_complete_exists,
                "training_complete": training_complete,
                "run_complete": run_complete,
                "status": status,
                "actual_optimizer_steps": config.get("actual_optimizer_steps", ""),
                "resumed_from_step": config.get("resumed_from_step", ""),
            }
        )

    rows = sorted(rows, key=lambda r: (r["image_count"], r["learning_rate_slug"], r["lora_rank"], r["run_name"]))
    fieldnames = [
        "run_name",
        "run_dir",
        "image_count",
        "learning_rate_slug",
        "lora_rank",
        "target_steps",
        "latest_checkpoint_step",
        "latest_checkpoint_label",
        "checkpoint_count",
        "final_adapter_exists",
        "artifact_log_exists",
        "run_complete_exists",
        "training_complete",
        "run_complete",
        "status",
        "actual_optimizer_steps",
        "resumed_from_step",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved LoRA progress audit: {OUT_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
