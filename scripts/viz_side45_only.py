"""can_standing 의 side-45 group 만 단독 시각화.

시각 대비를 높이기 위해 top-down / side-cap 은 숨기고 side-45° 8개만 그림.
"""
from __future__ import annotations

from pathlib import Path
import cv2
import h5py
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
GRASP_H5 = ROOT / "img_dataset/grasp_cache/grasps.h5"
POSES_H5 = ROOT / "img_dataset/icp_cache/poses.h5"
DET_H5 = ROOT / "img_dataset/yolo_cache_v3/detections.h5"
PLY_DIR = Path("/home/robotics/Competition/RoboCup_ARM/models/ply")
OUT_DIR = ROOT / "deploy/viz/gt_policy"

K_FX = K_FY = 1109.0
K_CX, K_CY = 640.0, 360.0
IMG_H, IMG_W = 720, 1280

GRIPPER_HALF = 0.0425
FINGER_LEN = 0.040
PALM_BACK = 0.025

SIDE_45_GID = 4
TARGET_MODE = "standing"
TARGET_CLASSES = ["bottle", "can"]


def poly_to_mask(poly):
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def depth_mask_to_pc(depth_m, mask):
    ys, xs = np.where(mask & (depth_m > 0.1) & (depth_m < 2.0))
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth_m[ys, xs]
    x = (xs - K_CX) * z / K_FX
    y = (ys - K_CY) * z / K_FY
    return np.stack([x, y, z], axis=1).astype(np.float32)


def quat_wxyz_to_R(q):
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ], dtype=np.float32)


def gripper_segs(pos, q):
    R = quat_wxyz_to_R(q)
    a = R[:, 2]
    b = R[:, 1]
    tip1 = pos + b * GRIPPER_HALF
    tip2 = pos - b * GRIPPER_HALF
    base1 = tip1 - a * FINGER_LEN
    base2 = tip2 - a * FINGER_LEN
    palm = pos - a * FINGER_LEN
    wrist = palm - a * PALM_BACK
    return [(wrist, palm), (base1, base2), (base1, tip1), (base2, tip2)]


def load_ply_points(name):
    pcd = o3d.io.read_point_cloud(str(PLY_DIR / name))
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.ptp(axis=0).max() > 1.0:
        pts *= 0.001
    pts = pts - pts.mean(axis=0)
    if len(pts) > 2000:
        idx = np.random.default_rng(0).choice(len(pts), 2000, replace=False)
        pts = pts[idx]
    return pts


def make_gripper_trace(grasps, color, width=4, name=""):
    xs, ys, zs = [], [], []
    for pose in grasps:
        pos = pose[:3]; q = pose[3:7]
        for a, b in gripper_segs(pos, q):
            xs += [a[0], b[0], None]
            ys += [a[1], b[1], None]
            zs += [a[2], b[2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                        line=dict(color=color, width=width),
                        name=name, showlegend=False)


def collect_samples(cls_name):
    rows = []
    with h5py.File(GRASP_H5, "r") as f, h5py.File(POSES_H5, "r") as p:
        for sid in f.keys():
            g_s = f[sid]; p_s = p[sid]
            for oname in g_s.keys():
                g_o = g_s[oname]; p_o = p_s[oname]
                if "excluded_reason" in g_o.attrs: continue
                if int(g_o.attrs.get("n_grasps", 0)) == 0: continue
                if g_o.attrs["class_name"] != cls_name: continue
                if g_o.attrs["mode"] != TARGET_MODE: continue
                rows.append({
                    "sid": sid,
                    "obj_idx": int(oname.split("_")[1]),
                    "depth_path": g_s.attrs["depth_path"],
                    "fitness": float(p_o["fitness"][()]),
                    "ply_file": p_o.attrs["ply_file"],
                    "pose7_obj": np.asarray(p_o["pose_cam"]),
                    "grasps_cam": np.asarray(g_o["grasps_cam"]),
                    "groups": np.asarray(g_o["grasp_group"]),
                })
    return rows


def render_class(cls_name, det_h5):
    rows = collect_samples(cls_name)
    if not rows:
        print(f"[warn] no samples for {cls_name}_{TARGET_MODE}")
        return None
    rows.sort(key=lambda d: d["fitness"])
    n = len(rows)
    picks = [rows[int(n * 0.9)], rows[n // 2], rows[int(n * 0.1)]]
    print(f"[info] {cls_name}_{TARGET_MODE} n={n}, picks fitness="
          f"{[round(p['fitness'],3) for p in picks]}")

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}] * 3],
        subplot_titles=[
            f"<b>{p['sid']} / obj_{p['obj_idx']}</b><br>"
            f"fit={p['fitness']:.2f}  side-45 only (8 azimuths)"
            for p in picks
        ],
        horizontal_spacing=0.02,
    )

    for i, s in enumerate(picks, 1):
            depth = cv2.imread(str(ROOT / s["depth_path"]), cv2.IMREAD_UNCHANGED)
            depth_m = depth.astype(np.float32) / 1000.0
            poly = np.asarray(det_h5[s["sid"]]["mask_poly"][s["obj_idx"]]).reshape(-1, 2)
            mask = poly_to_mask(poly)
            scene_pts = depth_mask_to_pc(depth_m, mask)
            if len(scene_pts) > 1500:
                idx = np.random.default_rng(0).choice(len(scene_pts), 1500, replace=False)
                scene_pts = scene_pts[idx]
            fig.add_trace(go.Scatter3d(
                x=scene_pts[:, 0], y=scene_pts[:, 1], z=scene_pts[:, 2],
                mode="markers", marker=dict(size=1.5, color="#4a90e2", opacity=0.4),
                showlegend=False), row=1, col=i)

            ply_pts = load_ply_points(s["ply_file"])
            pose = s["pose7_obj"]
            R = quat_wxyz_to_R(pose[3:7])
            tf = (ply_pts @ R.T) + pose[:3]
            fig.add_trace(go.Scatter3d(
                x=tf[:, 0], y=tf[:, 1], z=tf[:, 2],
                mode="markers", marker=dict(size=1.0, color="#d0021b", opacity=0.25),
                showlegend=False), row=1, col=i)

            # only side-45 grasps
            mask45 = (s["groups"] == SIDE_45_GID)
            fig.add_trace(make_gripper_trace(
                s["grasps_cam"][mask45], "#e74c3c", width=5), row=1, col=i)

            c = scene_pts.mean(axis=0)
            r = np.max(scene_pts.ptp(axis=0)) * 1.1 + 0.08
            fig.update_scenes(
                row=1, col=i,
                xaxis=dict(range=[c[0] - r, c[0] + r], title=""),
                yaxis=dict(range=[c[1] - r, c[1] + r], title=""),
                zaxis=dict(range=[c[2] - r, c[2] + r], title=""),
                aspectmode="cube",
                camera=dict(eye=dict(x=1.6, y=-1.6, z=-1.2),
                            up=dict(x=0, y=-1, z=0)),
            )

    fig.update_layout(
        title=f"<b>{cls_name} standing — side-45° 전용 시각화</b>  "
              f"(빨강 8 grasp, azimuth 0/45/90/.../315°)",
        height=720,
        margin=dict(l=10, r=10, t=90, b=10),
        font=dict(family="Helvetica,Arial,sans-serif", size=12),
    )
    out = OUT_DIR / f"{cls_name}_{TARGET_MODE}_side45only.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"[viz] {out}")
    return out


if __name__ == "__main__":
    with h5py.File(DET_H5, "r") as det_h5:
        for cls in TARGET_CLASSES:
            render_class(cls, det_h5)
