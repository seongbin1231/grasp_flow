#!/bin/bash
# v9 PC-only full 250ep — Trial B 셋팅 (xattn OFF, h768/nb8) 의 풀 학습.
#   sweep B (30ep best 0.3176) → continue (60ep best 0.3078) → 같은 config 250ep from scratch.
#   zhou_9d depth 250ep (val 0.2419) 와 직접 비교용.
#
# 실행:
#   nohup bash scripts/run_pc_full_250ep.sh > runs/pc_full_250ep.log 2>&1 &
set -euo pipefail

ROOT=/home/robotics/Competition/YOLO_Grasp
PY=/home/robotics/anaconda3/bin/python
cd "$ROOT"

echo "============================================================"
echo "[$(date)] zhou_9d_pc_only_full_250ep — PC-B config from scratch"
echo "============================================================"

"$PY" scripts/train_flow.py \
    --run "$ROOT/runs/yolograsp_v2/zhou_9d_pc_only_full_250ep" \
    --rot_repr zhou6d \
    --use_pc_only \
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
echo "[$(date)] DONE — best.pt at runs/.../zhou_9d_pc_only_full_250ep/"
echo "============================================================"
