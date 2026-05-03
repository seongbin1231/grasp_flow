"""Wandb sweep wrapper for v9 PC-only.

Maps `variant` arg to FlowGraspNetPC config (xattn ON/OFF + hidden/n_blocks)
and invokes train_flow.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
PYTHON = "/home/robotics/anaconda3/bin/python"

# variant → (xattn flag, hidden, n_blocks)
VARIANTS = {
    "A_xattn_on_768_8":    ("--xattn", 768, 8),
    "B_xattn_off_768_8":   ("",        768, 8),
    "C_xattn_off_512_6":   ("",        512, 6),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.04)
    ap.add_argument("--ema_decay", type=float, default=0.9998)
    ap.add_argument("--cond_dropout", type=float, default=0.2)
    ap.add_argument("--rot_loss_weight", type=float, default=2.0)
    ap.add_argument("--marker_boost", type=float, default=1.5)
    ap.add_argument("--spam_boost", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    xattn_flag, hidden, n_blocks = VARIANTS[args.variant]
    run_dir = ROOT / "runs/yolograsp_v2/repr_compare/zhou_9d_pc_sweep" / args.variant

    cmd = [
        PYTHON, str(ROOT / "scripts/train_flow.py"),
        "--run", str(run_dir),
        "--rot_repr", "zhou6d",
        "--use_pc_only",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--warmup_frac", str(args.warmup_frac),
        "--ema_decay", str(args.ema_decay),
        "--cond_dropout", str(args.cond_dropout),
        "--block", "adaln_zero",
        "--n_blocks", str(n_blocks),
        "--hidden", str(hidden),
        "--symmetry_loss",
        "--rot_loss_weight", str(args.rot_loss_weight),
        "--marker_boost", str(args.marker_boost),
        "--spam_boost", str(args.spam_boost),
        "--seed", str(args.seed),
        "--wandb",
        "--wandb_project", "yolograsp-v2",
    ]
    if xattn_flag:
        cmd.append(xattn_flag)

    print(f"[sweep_pc_v9] variant={args.variant}  hidden={hidden}  n_blocks={n_blocks}  "
          f"xattn={'ON' if xattn_flag else 'OFF'}")
    print(f"[sweep_pc_v9] CMD: {' '.join(cmd)}")
    sys.stdout.flush()

    rc = subprocess.call(cmd)
    sys.exit(rc)


if __name__ == "__main__":
    main()
