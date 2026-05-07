"""SKELETON — meta['poses'] (3,4,n) → 4×4 promote → object frame grasp 를 camera frame 으로 변환.

⚠️ Critical fact-check (검증됨):
  meta['poses'] shape = (3, 4, n)  NOT (4, 4, n)
  → [0, 0, 0, 1] 직접 append 필수

  assert poses.shape[0] == 3 and poses.shape[1] == 4

각 객체 i 의 grasp 변환:
  T_w_obj = poses[:, :, i]  # (3, 4)
  T_w_obj_4x4 = np.vstack([T_w_obj, [0,0,0,1]])
  T_cam = T_w_cam_inv @ T_w_obj_4x4 @ T_obj_grasp
"""
from __future__ import annotations
import argparse
from pathlib import Path


def promote_pose_3x4_to_4x4(pose_3x4):
    """(3,4) → (4,4). assertion 포함."""
    assert pose_3x4.shape == (3, 4), f"unexpected shape {pose_3x4.shape}"
    import numpy as np
    return np.vstack([pose_3x4, np.array([[0, 0, 0, 1.0]])])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graspnet_root", type=Path, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError("TODO: integrate with flatten_grasps.py output")


if __name__ == "__main__":
    main()
