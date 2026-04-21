"""
Visualize yaw-augmentation consistency.

Pick a standing bottle/can object (16 grasps). Apply 6 different augmentation
thetas. For each theta, show:
  - rotated depth (grayscale)
  - rotated uv (magenta ×)
  - all 16 grasps projected to rotated image (color by group)

If augmentation is correct, grasps visually follow the rotated object.
Output: scripts/_aug_viz/augmentation_check.png
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.flow_dataset import (
    CAM_CX, CAM_CY, IMG_H, IMG_W, K_FX, K_FY,
    _build_R_tool,
    rotate_depth_around_center,
    rotate_grasp_by_cam_z,
    rotate_uv_around_center,
)

H5_PATH = ROOT / "datasets/grasp_v2.h5"
OUT_DIR = ROOT / "scripts/_aug_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRIPPER_HALF = 0.0425
FINGER_LEN = 0.040

GROUP_COLORS = {0: "#e41a1c", 1: "#ff7f00", 2: "#377eb8", 3: "#984ea3"}
GROUP_NAMES = {0: "top-down", 1: "side-cap", 2: "lying", 3: "cube"}


def project(P):
    return np.array([K_FX * P[0] / P[2] + CAM_CX,
                     K_FY * P[1] / P[2] + CAM_CY])


def gripper_wireframe_px(pos, approach, yaw):
    """Return dict with projected 'U' polyline and TCP pixel."""
    R = _build_R_tool(approach, yaw)
    binormal = R[:, 1]
    a = R[:, 2]
    tip1 = pos + binormal * GRIPPER_HALF
    tip2 = pos - binormal * GRIPPER_HALF
    base1 = tip1 - a * FINGER_LEN
    base2 = tip2 - a * FINGER_LEN
    pts3 = np.stack([tip1, base1, base2, tip2])
    pts2 = np.array([project(P) for P in pts3])
    tcp2 = project(pos)
    return pts2, tcp2


def pick_target_object(f, preferred_mode=1):
    """Find a standing object (mode=1) with 16 grasps. Prefer larger fitness."""
    g = f["train"]
    obj_ref = g["object_ref"][:]
    mode = g["object_mode"][:]
    fit = g["fitness"][:]

    best = None
    # iterate unique objects
    for oid in np.unique(obj_ref):
        idxs = np.where(obj_ref == oid)[0]
        if len(idxs) != 16:   # want a standing with 16 grasps
            continue
        if mode[idxs[0]] != preferred_mode:
            continue
        mean_fit = float(fit[idxs].mean())
        if best is None or mean_fit > best[1]:
            best = (oid, mean_fit, idxs)
    return best


def load_object_rows(f, idxs):
    g = f["train"]
    depth_ref = int(g["depth_ref"][idxs[0]])
    depth = g["depths"][depth_ref][:]
    uv = g["uvs"][idxs[0]]
    grasps = g["grasps_cam"][idxs]
    approach = g["approach_vec"][idxs]
    yaw = g["yaw_around_app"][idxs]
    group = g["grasp_group"][idxs]
    sample_ref = g["sample_ref"][idxs[0]].decode() if isinstance(g["sample_ref"][idxs[0]], bytes) else str(g["sample_ref"][idxs[0]])
    return depth, uv, grasps, approach, yaw, group, sample_ref


def draw_panel(ax, depth, uv, grasps3_list, title):
    # depth
    d_show = np.clip(depth, 0.1, 1.5)
    ax.imshow(d_show, cmap="gray_r", vmin=0.3, vmax=1.2, origin="upper")

    # uv
    ax.scatter([uv[0]], [uv[1]], s=140, marker="x", color="magenta",
               linewidths=2.5, zorder=6)

    # draw each grasp as U-shape + TCP + approach arrow
    for pos, approach, yaw, group in grasps3_list:
        pts2, tcp2 = gripper_wireframe_px(pos, approach, yaw)
        color = GROUP_COLORS[int(group)]
        # U-shape
        ax.plot(pts2[:, 0], pts2[:, 1], "-", color=color, lw=1.7, alpha=0.85, zorder=4)
        # TCP
        ax.scatter([tcp2[0]], [tcp2[1]], s=14, color=color, zorder=5,
                   edgecolors="white", linewidths=0.6)
        # approach arrow: from TCP, backward along -approach (toward camera side)
        # show where the arm comes from
        arrow_end_3d = pos - approach * 0.05
        arrow_end = project(arrow_end_3d)
        ax.plot([tcp2[0], arrow_end[0]], [tcp2[1], arrow_end[1]],
                ":", color=color, lw=1.0, alpha=0.6)

    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def build_grasps3_list(grasps, approach, yaw, group, theta):
    """Apply rotation to every grasp and return list of tuples."""
    out = []
    for i in range(len(grasps)):
        pos = grasps[i, :3].astype(np.float64)
        app = approach[i].astype(np.float64)
        y = float(yaw[i])
        if theta != 0.0:
            pos, app, y = rotate_grasp_by_cam_z(pos, app, y, theta)
        out.append((pos, app, y, int(group[i])))
    return out


def main():
    thetas_deg = [0, 30, 90, 135, -60, -150]   # 6 panels

    with h5py.File(H5_PATH, "r") as f:
        # --- Pick a standing can/bottle for first row ---
        st_pick = pick_target_object(f, preferred_mode=1)
        # --- Pick a lying case (mode=0) for second row ---
        ly_pick = pick_target_object(f, preferred_mode=0)
        # Lying bottle/can has 8 grasps (marker/spam has 6). Our picker requires 16;
        # relax for lying:
        g = f["train"]
        obj_ref = g["object_ref"][:]
        mode = g["object_mode"][:]
        fit = g["fitness"][:]
        cls_ = g["object_class"][:]
        best_ly = None
        for oid in np.unique(obj_ref):
            idxs = np.where(obj_ref == oid)[0]
            if mode[idxs[0]] != 0 or cls_[idxs[0]] not in (0, 1):   # lying bottle/can
                continue
            if len(idxs) != 8:
                continue
            mf = float(fit[idxs].mean())
            if best_ly is None or mf > best_ly[1]:
                best_ly = (oid, mf, idxs)

        assert st_pick is not None, "no standing 16-grasp object"
        assert best_ly is not None, "no lying 8-grasp object"

        rows_st = load_object_rows(f, st_pick[2])
        rows_ly = load_object_rows(f, best_ly[2])

    for label, rows, title_cls in [("standing", rows_st, "standing bottle/can"),
                                    ("lying", rows_ly, "lying bottle/can")]:
        depth, uv, grasps, approach, yaw, group, sample_ref = rows
        fig, axes = plt.subplots(2, 3, figsize=(17, 9))
        axes = axes.flatten()
        for ax, theta_deg in zip(axes, thetas_deg):
            theta = math.radians(theta_deg)
            if theta_deg == 0:
                d_rot = depth.copy()
                uv_rot = uv.copy()
            else:
                d_rot = rotate_depth_around_center(depth, theta)
                uv_rot = rotate_uv_around_center(uv.astype(np.float32), theta)
            grasps3 = build_grasps3_list(grasps, approach, yaw, group, theta)
            tt = f"θ = {theta_deg:+d}°   ({'original' if theta_deg == 0 else 'augmented'})"
            draw_panel(ax, d_rot, uv_rot, grasps3, tt)
        # legend
        handles = [plt.Line2D([0], [0], color=GROUP_COLORS[k], lw=2,
                              label=GROUP_NAMES[k])
                   for k in sorted(set(int(g) for g in group))]
        handles.append(plt.Line2D([0], [0], marker="x", color="magenta",
                                   linestyle="", markersize=10, label="uv (YOLO)"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"Augmentation check — {title_cls}  (sample={sample_ref}, n_grasps={len(grasps)})",
                     fontsize=13)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        out_path = OUT_DIR / f"aug_check_{label}.png"
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
