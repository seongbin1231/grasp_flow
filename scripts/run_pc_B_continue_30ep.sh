#!/bin/bash
# v9 PC-only Trial B (xattn OFF, h768/nb8, ~14.8M) warm-start continue 30ep.
#   Sweep best (val_flow=0.3176 @ ep28/30) → fresh cosine schedule + 30ep.
#
# 실행:
#   nohup bash scripts/run_pc_B_continue_30ep.sh > runs/pc_B_continue.log 2>&1 &
set -euo pipefail

ROOT=/home/robotics/Competition/YOLO_Grasp
PY=/home/robotics/anaconda3/bin/python
cd "$ROOT"

PRETRAINED="$ROOT/runs/yolograsp_v2/repr_compare/zhou_9d_pc_sweep/B_xattn_off_768_8/adaln_zero_lr0.001_nb8_h768/checkpoints/best.pt"

if [ ! -f "$PRETRAINED" ]; then
    echo "[ERR] pretrained ckpt not found: $PRETRAINED"
    exit 1
fi

echo "============================================================"
echo "[$(date)] PC-B continue 30ep — warm-start from sweep best"
echo "  pretrained: $PRETRAINED"
echo "============================================================"

"$PY" scripts/train_flow.py \
    --run "$ROOT/runs/yolograsp_v2/repr_compare/zhou_9d_pc_sweep/B_continue_30ep" \
    --rot_repr zhou6d \
    --use_pc_only \
    --epochs 30 \
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
    --wandb_project yolograsp-v2 \
    --pretrained "$PRETRAINED"

echo ""
echo "============================================================"
echo "[$(date)] DONE — best.pt at runs/.../B_continue_30ep/"
echo "============================================================"
