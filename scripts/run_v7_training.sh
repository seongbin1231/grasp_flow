#!/bin/bash
# v7_v4policy_big 학습 풀 파이프라인.
#   데이터 재빌드 (idempotent) → Flow Matching 학습 (hidden 1024, n_blocks 12, 250ep)
# 백그라운드 nohup 실행 권장: nohup bash scripts/run_v7_training.sh > runs/v7_train.log 2>&1 &
set -euo pipefail

ROOT=/home/robotics/Competition/YOLO_Grasp
PY=/home/robotics/anaconda3/bin/python
cd "$ROOT"

echo "============================================================"
echo "[$(date)] Step 1/2 — build_grasp_v2.py (dataset 재빌드)"
echo "============================================================"
"$PY" scripts/build_grasp_v2.py

echo ""
echo "============================================================"
echo "[$(date)] Step 2/2 — train_flow.py (v7_v4policy_big)"
echo "============================================================"
"$PY" scripts/train_flow.py \
  --run "$ROOT/runs/yolograsp_v2/v7_v4policy_big" \
  --epochs 250 \
  --batch_size 16 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --warmup_frac 0.04 \
  --ema_decay 0.9998 \
  --cond_dropout 0.2 \
  --block adaln_zero \
  --n_blocks 12 \
  --hidden 1024 \
  --symmetry_loss \
  --rot_loss_weight 2.0 \
  --marker_boost 1.5 \
  --spam_boost 2.5 \
  --seed 42 \
  --wandb \
  --wandb_project yolograsp-v2

echo ""
echo "============================================================"
echo "[$(date)] ALL DONE"
echo "============================================================"
