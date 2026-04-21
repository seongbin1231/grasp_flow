"""
Hyperparameter sweep v1 — 18 configs × 15 epochs ≈ 6h.

Grid:
  block      ∈ {film, adaln_zero}
  lr         ∈ {1e-4, 3e-4, 1e-3}
  n_blocks   ∈ {4, 6, 8}

Fixed:
  batch_size=16, hidden=512, cond_dropout=0.2, aux_mode_weight=0.1,
  EMA=0.999, warmup=0.1, seed=42, epochs=15

Each run writes:
  runs/yolograsp_v2/sweep_v1/<name>/metrics.jsonl
  runs/yolograsp_v2/sweep_v1/<name>/checkpoints/best.pt
After all runs:
  runs/yolograsp_v2/sweep_v1/summary.json
  runs/yolograsp_v2/sweep_v1/ranking.md   (sorted by best val_flow)

Usage:
  # Foreground (you watch it):
  /home/robotics/anaconda3/bin/python scripts/sweep_v1.py

  # Background (preferred for 6h run):
  nohup /home/robotics/anaconda3/bin/python scripts/sweep_v1.py \
        > runs/yolograsp_v2/sweep_v1/sweep.log 2>&1 &
  echo $! > runs/yolograsp_v2/sweep_v1/sweep.pid
"""
from __future__ import annotations

import json
import subprocess
import time
from itertools import product
from pathlib import Path

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
PYTHON = "/home/robotics/anaconda3/bin/python"
TRAIN_SCRIPT = ROOT / "scripts/train_flow.py"
SWEEP_DIR = ROOT / "runs/yolograsp_v2/sweep_v1"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 15
BATCH = 16
SEED = 42

GRID = {
    "block": ["film", "adaln_zero"],
    "lr": [1e-4, 3e-4, 1e-3],
    "n_blocks": [4, 6, 8],
}


def config_name(cfg):
    return f"{cfg['block']}_lr{cfg['lr']:.0e}_nb{cfg['n_blocks']}"


def build_configs():
    keys = list(GRID.keys())
    cfgs = []
    for values in product(*[GRID[k] for k in keys]):
        cfg = dict(zip(keys, values))
        cfg["name"] = config_name(cfg)
        cfgs.append(cfg)
    return cfgs


def run_one(cfg):
    run_dir = SWEEP_DIR / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(TRAIN_SCRIPT),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH),
        "--workers", "4",
        "--block", cfg["block"],
        "--lr", f"{cfg['lr']:g}",
        "--n_blocks", str(cfg["n_blocks"]),
        "--seed", str(SEED),
        "--run", str(run_dir),
    ]
    t0 = time.time()
    log_path = run_dir / "run.log"
    with open(log_path, "w") as f:
        ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    ok = ret.returncode == 0
    return ok, dt


def parse_metrics(run_dir: Path):
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return None
    recs = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if not recs:
        return None
    best = min(recs, key=lambda r: r["val_flow_loss"])
    last = recs[-1]
    return {
        "best_val_flow": best["val_flow_loss"],
        "best_epoch": best["epoch"],
        "last_val_flow": last["val_flow_loss"],
        "last_train_flow": last["train_flow_loss_mean"],
        "last_mode_acc": last.get("val_mode_acc"),
        "n_epochs": len(recs),
    }


def write_ranking(summary, path: Path):
    """Sort by best_val_flow ascending (lower=better)."""
    valid = [s for s in summary if s.get("best_val_flow") is not None]
    valid.sort(key=lambda s: s["best_val_flow"])
    lines = ["# Sweep v1 ranking\n"]
    lines.append("| rank | name | block | lr | n_blocks | best_val | ep | last_train | mode_acc | time |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, s in enumerate(valid, 1):
        lines.append(
            f"| {i} | {s['name']} | {s['block']} | {s['lr']:g} | {s['n_blocks']} | "
            f"**{s['best_val_flow']:.4f}** | {s['best_epoch']} | "
            f"{s['last_train_flow']:.4f} | {s['last_mode_acc']:.3f} | "
            f"{s['time_sec']/60:.1f} min |"
        )
    failed = [s for s in summary if s.get("best_val_flow") is None]
    if failed:
        lines.append("\n## Failed")
        for s in failed:
            lines.append(f"- {s['name']}")
    path.write_text("\n".join(lines) + "\n")


def main():
    configs = build_configs()
    print(f"[sweep] total configs = {len(configs)}")
    for c in configs:
        print(f"  - {c['name']}")
    print(f"[sweep] expected total time ≈ 6h (est 18~23 min/run)")
    print()

    summary = []
    t_start = time.time()

    for i, cfg in enumerate(configs, 1):
        elapsed = (time.time() - t_start) / 60
        print(f"\n=== [{i}/{len(configs)}] {cfg['name']} (elapsed={elapsed:.1f}min) ===")
        ok, dt = run_one(cfg)
        metrics = parse_metrics(SWEEP_DIR / cfg["name"])
        entry = {**cfg, "time_sec": dt, "ok": ok}
        if metrics:
            entry.update(metrics)
        summary.append(entry)

        # write interim summary after each run (crash-safe)
        (SWEEP_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
        write_ranking(summary, SWEEP_DIR / "ranking.md")

        if metrics:
            print(f"  ✓ best val={metrics['best_val_flow']:.4f} @ ep{metrics['best_epoch']}"
                  f"  ({dt/60:.1f} min)")
        else:
            print(f"  ✗ FAILED or no metrics ({dt/60:.1f} min)")

    total_min = (time.time() - t_start) / 60
    print(f"\n[sweep] done. total={total_min:.1f} min")
    print(f"[sweep] summary: {SWEEP_DIR / 'summary.json'}")
    print(f"[sweep] ranking: {SWEEP_DIR / 'ranking.md'}")


if __name__ == "__main__":
    main()
