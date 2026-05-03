#!/bin/bash
# zhou_9d_full_250ep 학습 풀 파이프라인.
#   9D Zhou 6D rotation 표현 (zhou_9d 50ep best val_flow=0.2786 의 250ep 확장판)
#   데이터 재빌드 idempotent → Flow Matching 학습 (hidden 768, n_blocks 8, 250ep)
#   v7 v4policy_big (8D, val 0.3676) 대체 후보.
#
# 실행:
#   nohup bash scripts/run_zhou9d_full_250ep.sh > runs/zhou9d_train.log 2>&1 &
#
# 예상 시간: ~5h (zhou_9d 50ep ~1h 기반, 약 5배)
set -euo pipefail

ROOT=/home/robotics/Competition/YOLO_Grasp
PY=/home/robotics/anaconda3/bin/python
cd "$ROOT"

echo "============================================================"
echo "[$(date)] Step 1/2 — build_grasp_v2.py (dataset 재빌드, idempotent)"
echo "============================================================"
"$PY" scripts/build_grasp_v2.py

echo ""
echo "============================================================"
echo "[$(date)] Step 2/2 — train_flow.py (zhou_9d_full_250ep)"
echo "============================================================"
"$PY" scripts/train_flow.py \
  --run "$ROOT/runs/yolograsp_v2/zhou_9d_full_250ep" \
  --rot_repr zhou6d \
  --epochs 250 \
  --batch_size 16 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --warmup_frac 0.04 \
  --ema_decay 0.9998 \
  --cond_dropout 0.2 \
  --block adaln_zero \
  --n_blocks 8 \
  --hidden 768 \
  --symmetry_loss \
  --rot_loss_weight 2.0 \
  --marker_boost 1.5 \
  --spam_boost 2.5 \
  --seed 42 \
  --wandb \
  --wandb_project yolograsp-v2

echo ""
echo "============================================================"
echo "[$(date)] ALL DONE — best.pt at runs/yolograsp_v2/zhou_9d_full_250ep/"
echo "============================================================"
