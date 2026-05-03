"""
Sanity check for grasp rotation representations.

Verifies:
1. 8D approach_yaw round-trip:  R_orig vs _build_R_tool(approach_vec, yaw_around_app)
2. 9D Zhou 6D round-trip:        R_orig -> r6 -> Gram-Schmidt -> R_9d
3. Edge case count (|a_x| > 0.93) where the reference vector switches in
   _yaw_from_Rtool, possibly causing yaw discontinuity.
4. Symmetric alt (lying 180): for 8D check (-sin, -cos) ≡ yaw + π;
   for 9D check R_alt = R · Rz(π) gives R[:, :2] *= -1 columns.

Run:
    /home/robotics/anaconda3/bin/python scripts/verify_grasp_repr.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
sys.path.insert(0, str(ROOT))

from src.flow_dataset import (
    _build_R_tool,
    _yaw_from_Rtool,
    quat_wxyz_to_R,
)

H5 = ROOT / "datasets/grasp_v2.h5"


def gram_schmidt_to_R(r6: np.ndarray) -> np.ndarray:
    """Zhou 2019 Gram-Schmidt: r6 = [a1(3), a2(3)] flat -> R 3x3."""
    a1 = r6[:3]
    a2 = r6[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    b2 = a2 - (b1 @ a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])


def main():
    print(f"Loading {H5}")
    errs_8d = []
    errs_9d = []
    n_edge = 0
    n_lying = 0
    sym_8d_errs = []   # 8D alt vs (yaw+pi) reconstruction
    sym_9d_errs = []   # 9D alt vs (R · Rz(pi)) reconstruction

    with h5py.File(H5, "r") as f:
        g = f["train"]
        N = g["grasps_cam"].shape[0]
        print(f"  N = {N}")
        gc = g["grasps_cam"][:]      # (N, 7) [x,y,z, qw,qx,qy,qz]
        ap = g["approach_vec"][:]    # (N, 3)
        yw = g["yaw_around_app"][:]  # (N,)
        md = g["object_mode"][:]     # (N,) 0=lying, 1=standing, 2=cube

    Rz_pi = np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1]], dtype=np.float64)

    for i in range(N):
        q = gc[i, 3:7].astype(np.float64)
        R_orig = quat_wxyz_to_R(q)

        # 1. 8D round-trip via stored (approach, yaw)
        a = ap[i].astype(np.float64)
        y = float(yw[i])
        R_8d = _build_R_tool(a, y)
        e8 = float(np.max(np.abs(R_orig - R_8d)))
        errs_8d.append(e8)

        # 2. 9D Zhou round-trip
        r6 = np.concatenate([R_orig[:, 0], R_orig[:, 1]])
        R_9d = gram_schmidt_to_R(r6)
        e9 = float(np.max(np.abs(R_orig - R_9d)))
        errs_9d.append(e9)

        # 3. Edge case count
        if abs(a[0]) > 0.93:
            n_edge += 1

        # 4. Symmetric alt verification (lying mode only)
        if int(md[i]) == 0:
            n_lying += 1
            # 8D alt:  yaw + pi  ->  R_yawpi
            R_8d_alt = _build_R_tool(a, y + math.pi)
            # candidate from sincos flip:  build R from (a, y_alt) where
            # sincos((y+π)) = (-sin y, -cos y).  Equivalent to yaw+pi.
            R_8d_alt_via_flip = _build_R_tool(a, y + math.pi)  # tautology check
            # 9D alt: R · Rz(pi)
            R_9d_alt_canon = R_orig @ Rz_pi
            # candidate from "R[:, 0] *= -1; R[:, 1] *= -1":
            R_9d_alt_naive = R_orig.copy()
            R_9d_alt_naive[:, 0] *= -1
            R_9d_alt_naive[:, 1] *= -1
            sym_8d_errs.append(float(np.max(np.abs(R_8d_alt - R_8d_alt_via_flip))))
            sym_9d_errs.append(float(np.max(np.abs(R_9d_alt_canon - R_9d_alt_naive))))

    errs_8d = np.array(errs_8d)
    errs_9d = np.array(errs_9d)
    sym_8d_errs = np.array(sym_8d_errs) if sym_8d_errs else np.zeros(1)
    sym_9d_errs = np.array(sym_9d_errs) if sym_9d_errs else np.zeros(1)

    print()
    print("=" * 70)
    print("Round-trip max-error per sample (R_original vs reconstructed)")
    print("=" * 70)
    print(f"  8D (approach_yaw)  N={len(errs_8d)}")
    print(f"    mean: {errs_8d.mean():.3e}  median: {np.median(errs_8d):.3e}")
    print(f"    max:  {errs_8d.max():.3e}   p99: {np.quantile(errs_8d, 0.99):.3e}")
    print(f"    >1e-4 count: {int((errs_8d > 1e-4).sum())} / {len(errs_8d)}")
    print()
    print(f"  9D (Zhou 6D)      N={len(errs_9d)}")
    print(f"    mean: {errs_9d.mean():.3e}  median: {np.median(errs_9d):.3e}")
    print(f"    max:  {errs_9d.max():.3e}   p99: {np.quantile(errs_9d, 0.99):.3e}")
    print(f"    >1e-6 count: {int((errs_9d > 1e-6).sum())} / {len(errs_9d)}")

    print()
    print("=" * 70)
    print("Edge case (8D ref-switch zone, |a_x| > 0.93)")
    print("=" * 70)
    print(f"  count: {n_edge} / {N} ({100*n_edge/N:.1f}%)")

    print()
    print("=" * 70)
    print("Symmetric alt (lying mode, N_lying = {})".format(n_lying))
    print("=" * 70)
    print(f"  8D: yaw+π identity check max err: {sym_8d_errs.max():.3e}")
    print(f"      (should be 0 — tautology, sanity only)")
    print(f"  9D: R·Rz(π)  vs  R[:, :2] *= -1")
    print(f"      max err: {sym_9d_errs.max():.3e}")
    if sym_9d_errs.max() < 1e-6:
        print(f"      ✓ canonical Rz(π) ≡ flip first two cols")
    else:
        print(f"      ✗ MISMATCH — naive flip is NOT equivalent to R·Rz(π)")
        print(f"      Need to derive correct 9D alt formula!")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    ok_8d = errs_8d.max() < 1e-4
    ok_9d = errs_9d.max() < 1e-6
    ok_sym = sym_9d_errs.max() < 1e-6
    print(f"  8D round-trip:        {'OK' if ok_8d else 'FAIL'}")
    print(f"  9D round-trip:        {'OK' if ok_9d else 'FAIL'}")
    print(f"  9D symmetric alt:     {'OK' if ok_sym else 'FAIL'}")
    if ok_8d and ok_9d and ok_sym:
        print()
        print("  ✓ All representations consistent. Safe to proceed.")
        return 0
    else:
        print()
        print("  ✗ Verify above failures before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
