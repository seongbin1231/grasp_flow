"""Verify yaw-augmentation consistency: depth, uv, 3D grasp all rotate consistently.

Synthesizes a depth image with a known peak. Projects a 3D point there.
Then rotates depth by theta, checks:
  (1) rotate_uv(uv, theta) hits the same depth peak in rotated image
  (2) rotate_grasp_by_cam_z(pos, theta) reprojects to the same rotated pixel
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.flow_dataset import (
    CAM_CX, CAM_CY, IMG_H, IMG_W, K_FX, K_FY,
    rotate_depth_around_center,
    rotate_grasp_by_cam_z,
    rotate_uv_around_center,
)


def project(P):
    return np.array([K_FX * P[0] / P[2] + CAM_CX,
                     K_FY * P[1] / P[2] + CAM_CY])


def run_case(P_cam, theta_deg, verbose=True):
    """Validate rotation consistency for one 3D point + angle."""
    theta = math.radians(theta_deg)

    # Original pixel & uv
    uv_orig = project(P_cam).astype(np.float32)

    # Synthesize depth image: cross at uv_orig
    depth = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    u, v = int(round(uv_orig[0])), int(round(uv_orig[1]))
    if 6 <= u < IMG_W - 6 and 6 <= v < IMG_H - 6:
        depth[v-3:v+4, u-3:u+4] = P_cam[2]   # 7×7 patch at depth z

    # Rotate depth
    d_rot = rotate_depth_around_center(depth, theta)

    # Rotate uv (fixed)
    uv_rot = rotate_uv_around_center(uv_orig, theta)

    # Rotate grasp 3D position (fixed)
    pos_rot, _app, _yaw = rotate_grasp_by_cam_z(
        P_cam.astype(np.float64),
        np.array([0.0, 0.0, 1.0]),   # dummy approach
        0.0,
        theta,
    )
    uv_from_3d = project(pos_rot).astype(np.float32)

    # depth peak in rotated image: where is max
    uy, ux = np.unravel_index(np.argmax(d_rot), d_rot.shape)
    uv_depth_peak = np.array([ux, uy], dtype=np.float32)

    # compare
    err_uv_vs_depth = np.linalg.norm(uv_rot - uv_depth_peak)
    err_3d_vs_uv = np.linalg.norm(uv_from_3d - uv_rot)

    if verbose:
        print(f"theta={theta_deg:+6.1f}°   P_cam={tuple(P_cam.round(3))}")
        print(f"  uv_orig        = {uv_orig.round(1)}")
        print(f"  uv_rot(func)   = {uv_rot.round(1)}")
        print(f"  depth_peak     = {uv_depth_peak.round(1)}    err = {err_uv_vs_depth:.2f}px")
        print(f"  reproj(rot 3D) = {uv_from_3d.round(1)}    err vs uv_rot = {err_3d_vs_uv:.3f}px")
    return err_uv_vs_depth, err_3d_vs_uv


def main():
    # near-center points to avoid peak leaving image frame under any rotation
    cases = [
        (np.array([0.04, -0.03, 0.50]),    0.0),
        (np.array([0.04, -0.03, 0.50]),   45.0),
        (np.array([0.04, -0.03, 0.50]),   90.0),
        (np.array([0.04, -0.03, 0.50]),  135.0),
        (np.array([0.04, -0.03, 0.50]),  180.0),
        (np.array([0.04, -0.03, 0.50]),  -90.0),
        (np.array([-0.05, 0.06, 0.55]),   30.0),
        (np.array([-0.05, 0.06, 0.55]), -165.0),
        (np.array([0.02, 0.08, 0.60]),  120.0),
    ]

    max_err_uv = 0.0
    max_err_3d = 0.0
    for P, th in cases:
        eu, e3 = run_case(P, th)
        max_err_uv = max(max_err_uv, eu)
        max_err_3d = max(max_err_3d, e3)

    print(f"\n=== SUMMARY ===")
    print(f"max uv vs depth peak error:  {max_err_uv:.2f} px   (ideal ≤ 2 px)")
    print(f"max reproj(3D rot) vs uv_rot: {max_err_3d:.3f} px   (ideal ≤ 0.01 px, analytical)")
    # tolerance 6 px — bilinear interpolation during cv2.warpAffine blurs the 7×7 patch
    ok = max_err_uv <= 6.0 and max_err_3d <= 0.1
    print("RESULT:", "✅ PASS" if ok else "❌ FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
