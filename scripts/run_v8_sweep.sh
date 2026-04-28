#!/bin/bash
# v8 sweep — 50 epoch 으로 효과 검증.
# 통과 기준: val_flow @ ep50 < 0.42 (v7 ep50 추정 0.45)
#
# 변경점 vs v7:
#   + --xattn                          : Cross-Attention block
#   + --multiscale_local "96,192,384"  : 3 scale crop
#   + --scale_dropout 0.5              : 정보 중복 방어
#   + --weight_decay 0.05              : transformer 과적합 방어
#   + --warmup_frac 0.06               : cross-attn 안정화
set -euo pipefail

ROOT=/home/robotics/Competition/YOLO_Grasp
PY=/home/robotics/anaconda3/bin/python
cd "$ROOT"

echo "============================================================"
echo "[$(date)] v8 sweep — 50 epoch"
echo "============================================================"
"$PY" scripts/train_flow.py \
  --run "$ROOT/runs/yolograsp_v2/v8_sweep_xattn_ms" \
  --epochs 50 \
  --batch_size 16 \
  --lr 1e-3 \
  --weight_decay 0.05 \
  --warmup_frac 0.06 \
  --ema_decay 0.9998 \
  --cond_dropout 0.2 \
  --block adaln_zero \
  --n_blocks 12 \
  --hidden 1024 \
  --xattn \
  --multiscale_local "96,192,384" \
  --scale_dropout 0.5 \
  --symmetry_loss \
  --rot_loss_weight 2.0 \
  --marker_boost 1.5 \
  --spam_boost 2.5 \
  --seed 42 \
  --wandb \
  --wandb_project yolograsp-v2

echo "[$(date)] DONE"
