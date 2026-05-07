"""SKELETON — GraspNet-1B offsets tensor flatten.

graspnetAPI offsets shape (검증됨, graspnet.py:615-636):
  (N_points, 300, 12, 4, 3)
  - dim 1 (300): view templates (Fibonacci sphere, generate_views(300))
  - dim 2 (12):  num_angles
  - dim 3 (4):   num_depths
  - dim 4 (3):   (angle, depth, width)

Friction labels: 6 thresholds [0.2, 0.4, 0.6, 0.8, 1.0, 1.2], TOP_K=50
점수 규칙 (eval_utils.py:377-382 + graspnet_eval.py:194):
  friction <= threshold AND friction > 0 만 valid
  friction = -1 = collision/invalid
  lower friction = better grasp

Filter (학습용):
  mask = (fric > 0) & (fric <= 0.4) & (~coll)  # 가장 엄격한 0.4 만 채택

TODO:
  1. for each (scene, ann, obj):
       load offsets, fric, coll
       apply mask
       output (N_kept, 7) = [pos(3), R(3x3 flat 9 또는 quat 4)]
  2. obj_to_cam (다른 스크립트) 와 결합
"""
from __future__ import annotations
import argparse
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graspnet_root", type=Path, required=True)
    ap.add_argument("--out_h5", type=Path, required=True)
    ap.add_argument("--fric_thresh", type=float, default=0.4)
    return ap.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError("TODO: implement after graspnetAPI install")


if __name__ == "__main__":
    main()
