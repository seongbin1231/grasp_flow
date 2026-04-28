"""Wandb sweep wrapper for v8 ablation.

variant 인자만 받아서 train_flow.py 의 적절한 플래그 조합으로 dispatch.
wandb sweep agent 가 이 스크립트를 호출 → train_flow.py 가 wandb.init 으로 sweep 정보 픽업.

사용:
  wandb sweep scripts/sweep_v8.yaml      # sweep 등록 → sweep_id 받음
  wandb agent <sweep_id>                  # agent 실행 (각 variant 1회씩)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
PY = "/home/robotics/anaconda3/bin/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["baseline", "multiscale_only", "xattn_only", "full"],
                    help="Architecture variant for ablation")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_frac", type=float, default=0.06)
    ap.add_argument("--scale_dropout", type=float, default=0.5)
    ap.add_argument("--cond_dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_dir = ROOT / f"runs/yolograsp_v2/v8_sweep/{args.variant}"
    cmd = [
        PY, str(ROOT / "scripts/train_flow.py"),
        "--run", str(run_dir),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--warmup_frac", str(args.warmup_frac),
        "--ema_decay", "0.9998",
        "--cond_dropout", str(args.cond_dropout),
        "--block", "adaln_zero",
        "--n_blocks", "12",
        "--hidden", "1024",
        "--symmetry_loss",
        "--rot_loss_weight", "2.0",
        "--marker_boost", "1.5",
        "--spam_boost", "2.5",
        "--seed", str(args.seed),
        "--wandb",
        "--wandb_project", "yolograsp-v2",
    ]

    # variant 매핑: A1 (xattn) / A2 (multiscale)
    if args.variant in ("multiscale_only", "full"):
        cmd += ["--multiscale_local", "96,192,384",
                "--scale_dropout", str(args.scale_dropout)]
    if args.variant in ("xattn_only", "full"):
        cmd += ["--xattn"]
    # baseline 은 두 옵션 모두 OFF (= v7 와 동등 + weight_decay/warmup 차이만)

    print(f"[sweep_v8] variant = {args.variant}")
    print(f"[sweep_v8] cmd:")
    print("  " + " \\\n    ".join(cmd))

    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
