"""Sanity-check: depth + YOLO mask → per-object 3D point cloud.

Confirms that for detected objects the depth values inside the mask produce
a clean, object-shaped 3D point cloud in the camera frame — the exact input
the ICP stage will consume.

Picks 1~2 samples per scene, renders:
  - Left  : RGB + mask overlay
  - Right : 3D scatter (X,Y,Z in camera frame, meters)
Each object gets a distinct color. Saves to runs/pc_sanity/{sid}.png
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT_ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
DEF_H5 = PROJECT_ROOT / "img_dataset/yolo_cache_v3/detections.h5"
OUT_DIR = PROJECT_ROOT / "runs/pc_sanity_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FX, FY, CX, CY = 1109.0, 1109.0, 640.0, 360.0
H, W = 720, 1280

CUBE_CLASSES = {"cube_blue", "cube_green", "cube_p", "cube_red"}

PALETTE = [  # BGR for cv2 overlay and normalized RGB for matplotlib
    (96, 160, 255), (255, 180, 60), (220, 60, 180), (60, 220, 120),
    (80, 240, 240), (255, 80, 80), (140, 255, 140), (255, 255, 120),
]


def poly_to_mask(poly_flat: np.ndarray) -> np.ndarray:
    if poly_flat.size < 4:
        return np.zeros((H, W), dtype=bool)
    pts = poly_flat.reshape(-1, 2).astype(np.int32)
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def depth_mask_to_points(depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return (N,3) cam-frame points for pixels in mask with valid depth."""
    ys, xs = np.where(mask & (depth_m > 0.1) & (depth_m < 3.0) & np.isfinite(depth_m))
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    Z = depth_m[ys, xs].astype(np.float32)
    X = (xs.astype(np.float32) - CX) * Z / FX
    Y = (ys.astype(np.float32) - CY) * Z / FY
    return np.stack([X, Y, Z], axis=1)


def unified_name(raw: str) -> str:
    return "cube" if raw in CUBE_CLASSES else raw


def render_sample(sid: str, grp: h5py.Group, class_names: list[str]) -> None:
    rgb_path = PROJECT_ROOT / grp.attrs["rgb_path"]
    depth_path = PROJECT_ROOT / grp.attrs["depth_path"] if grp.attrs["depth_path"] else None
    if depth_path is None or not depth_path.exists():
        print(f"  skip {sid}: depth missing")
        return

    rgb_bgr = cv2.imread(str(rgb_path))
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    depth_m = depth_raw.astype(np.float32) / 1000.0
    depth_m[~np.isfinite(depth_m)] = 0

    classes = grp["classes"][:]
    confs = grp["confidences"][:]
    polys = grp["mask_poly"][:]

    fig = plt.figure(figsize=(16, 7))
    # Left: RGB + mask overlay
    ax_rgb = fig.add_subplot(1, 2, 1)
    overlay = rgb_bgr.copy()
    for k in range(len(classes)):
        mask = poly_to_mask(polys[k])
        color = PALETTE[k % len(PALETTE)]
        overlay[mask] = (0.55 * np.array(color) + 0.45 * overlay[mask]).astype(np.uint8)
    blend = cv2.addWeighted(overlay, 0.45, rgb_bgr, 0.55, 0)
    for k in range(len(classes)):
        mask = poly_to_mask(polys[k])
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        color = PALETTE[k % len(PALETTE)]
        y1, x1 = ys.min(), xs.min()
        label = f"{unified_name(class_names[classes[k]])} {confs[k]:.2f}"
        cv2.rectangle(blend, (x1, y1 - 22), (x1 + 12 * len(label), y1), color, -1)
        cv2.putText(blend, label, (x1 + 2, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    ax_rgb.imshow(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
    ax_rgb.set_title(f"{sid}  — RGB + masks ({len(classes)} objects)")
    ax_rgb.axis("off")

    # Right: 3D scatter, camera frame
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    any_pts = False
    total_pts = 0
    z_min_all, z_max_all = 10.0, 0.0
    for k in range(len(classes)):
        mask = poly_to_mask(polys[k])
        pts = depth_mask_to_points(depth_m, mask)
        n = pts.shape[0]
        total_pts += n
        if n == 0:
            continue
        any_pts = True
        color = np.array(PALETTE[k % len(PALETTE)][::-1], dtype=np.float32) / 255.0  # BGR→RGB
        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.6, c=[color], alpha=0.6,
                     label=f"{unified_name(class_names[classes[k]])} ({n} pts)")
        z_min_all = min(z_min_all, pts[:, 2].min())
        z_max_all = max(z_max_all, pts[:, 2].max())
    ax3d.set_xlabel("X_cam (m)")
    ax3d.set_ylabel("Y_cam (m)")
    ax3d.set_zlabel("Z_cam (m) — depth")
    ax3d.set_title(f"Camera-frame point cloud (total {total_pts} pts)")
    if any_pts:
        ax3d.legend(loc="upper right", fontsize=7)
        # Top-down-ish view — camera looks along +Z
        ax3d.view_init(elev=-60, azim=-90)
        # equal-ish aspect
        ax3d.set_box_aspect((1, 0.56, 0.7))
    fig.suptitle(f"Scene={grp.attrs['scene']}  Z range=[{z_min_all:.2f}, {z_max_all:.2f}] m",
                 fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / f"{sid}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {sid} → {out.relative_to(PROJECT_ROOT)}  ({total_pts} pts)")


def main():
    random.seed(7)
    if not DEF_H5.exists():
        print(f"[ERROR] {DEF_H5} missing — run batch_yolo_cache_v3.py first")
        return

    with h5py.File(DEF_H5, "r") as f:
        class_names = [s.decode() if isinstance(s, bytes) else s
                       for s in f.attrs["class_names"]]
        # Pick 2 detected samples per scene
        buckets: dict[str, list[str]] = {}
        for name in f.keys():
            if not name.startswith("sample_"):
                continue
            grp = f[name]
            if not grp.attrs.get("detected", False):
                continue
            scene = grp.attrs.get("scene")
            if isinstance(scene, bytes):
                scene = scene.decode()
            buckets.setdefault(scene, []).append(name)

        picks = []
        for scene in sorted(buckets.keys()):
            picks.extend(random.sample(buckets[scene], k=min(2, len(buckets[scene]))))
        print(f"[viz] {len(picks)} samples across {len(buckets)} scenes")

        for name in picks:
            sid = name.replace("sample_", "")
            render_sample(sid, f[name], class_names)

    print(f"\n[done] outputs: {OUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
