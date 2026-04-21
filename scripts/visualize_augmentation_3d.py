"""
3D interactive HTML visualization of yaw augmentation consistency.

For one standing object and one lying object:
  - 2×3 grid of 3D scenes, each a different augmentation theta
  - Scene point cloud (rotated), grasp U-brackets (rotated), uv-projected 3D marker
  - Interactive: pan/zoom/rotate in browser

Output:
  scripts/_aug_viz/aug_check_standing.html
  scripts/_aug_viz/aug_check_lying.html
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.flow_dataset import (
    CAM_CX, CAM_CY, IMG_H, IMG_W, K_FX, K_FY,
    _build_R_tool,
)

H5_PATH = ROOT / "datasets/grasp_v2.h5"
OUT_DIR = ROOT / "scripts/_aug_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRIPPER_HALF = 0.0425
FINGER_LEN = 0.040
PALM_BACK = 0.025
APPROACH_STEM = 0.050

GROUP_COLORS = {0: "#e41a1c", 1: "#ff7f00", 2: "#377eb8", 3: "#984ea3"}
GROUP_NAMES = {0: "top-down", 1: "side-cap", 2: "lying", 3: "cube"}

THETAS_DEG = [0, 30, 90, 135, -60, -150]


# ------- geometry helpers -------

def rot3d_z_consistent(P, theta):
    """R_z(-theta) @ P, matching cv2 image rotation by +theta (y-down CCW)."""
    c, s = math.cos(theta), math.sin(theta)
    Rz = np.array([[ c,  s, 0],
                   [-s,  c, 0],
                   [ 0,  0, 1]], dtype=np.float64)
    return (Rz @ P.T).T


def back_project(depth, stride=3):
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[ys, xs]
    valid = (z > 0.1) & (z < 2.0)
    z = z[valid]; xs = xs[valid]; ys = ys[valid]
    x = (xs - CAM_CX) * z / K_FX
    y = (ys - CAM_CY) * z / K_FY
    return np.stack([x, y, z], axis=1).astype(np.float32)


def uv_to_3d(depth, uv, win=5):
    u, v = int(round(uv[0])), int(round(uv[1]))
    u0 = max(0, u - win); u1 = min(IMG_W, u + win + 1)
    v0 = max(0, v - win); v1 = min(IMG_H, v + win + 1)
    patch = depth[v0:v1, u0:u1]
    valid = patch[(patch > 0.1) & (patch < 2.0)]
    if valid.size == 0:
        return None
    z = float(np.median(valid))
    return np.array([(uv[0] - CAM_CX) * z / K_FX,
                     (uv[1] - CAM_CY) * z / K_FY,
                     z], dtype=np.float64)


def gripper_3d_lines(pos, approach, yaw):
    """Return (u_poly, palm_stem, approach_stem, cone_tail, cone_dir, tcp).

    u_poly: (5,3) polyline tip1→base1→base2→tip2→None
    palm_stem: (3,3)  palm→wrist
    approach_stem: (3,3) app_start→wrist
    cone_tail, cone_dir: (3,), (3,) for cone arrowhead
    """
    R = _build_R_tool(approach, yaw)
    a = R[:, 2]
    b = R[:, 1]
    tip1 = pos + b * GRIPPER_HALF
    tip2 = pos - b * GRIPPER_HALF
    base1 = tip1 - a * FINGER_LEN
    base2 = tip2 - a * FINGER_LEN
    palm_center = pos - a * FINGER_LEN
    wrist = palm_center - a * PALM_BACK
    app_start = wrist - a * APPROACH_STEM
    u_poly = np.stack([tip1, base1, base2, tip2])
    palm_stem = np.stack([palm_center, wrist])
    app_stem = np.stack([app_start, wrist])
    return u_poly, palm_stem, app_stem, app_start, a, pos


# ------- data loaders -------

def pick_standing_16(f):
    g = f["train"]
    obj_ref = g["object_ref"][:]
    mode = g["object_mode"][:]
    fit = g["fitness"][:]
    best = None
    for oid in np.unique(obj_ref):
        idxs = np.where(obj_ref == oid)[0]
        if len(idxs) != 16 or mode[idxs[0]] != 1:
            continue
        mf = float(fit[idxs].mean())
        if best is None or mf > best[1]:
            best = (oid, mf, idxs)
    return best


def pick_lying_8_bottle_or_can(f):
    g = f["train"]
    obj_ref = g["object_ref"][:]
    mode = g["object_mode"][:]
    fit = g["fitness"][:]
    cls_ = g["object_class"][:]
    best = None
    for oid in np.unique(obj_ref):
        idxs = np.where(obj_ref == oid)[0]
        if len(idxs) != 8 or mode[idxs[0]] != 0 or cls_[idxs[0]] not in (0, 1):
            continue
        mf = float(fit[idxs].mean())
        if best is None or mf > best[1]:
            best = (oid, mf, idxs)
    return best


def load_rows(f, idxs):
    g = f["train"]
    dref = int(g["depth_ref"][idxs[0]])
    depth = g["depths"][dref][:]
    uv = g["uvs"][idxs[0]]
    pos = g["grasps_cam"][idxs][:, :3]
    approach = g["approach_vec"][idxs]
    yaw = g["yaw_around_app"][idxs]
    group = g["grasp_group"][idxs]
    sample_ref = g["sample_ref"][idxs[0]]
    if isinstance(sample_ref, bytes):
        sample_ref = sample_ref.decode()
    return depth, uv, pos, approach, yaw, group, str(sample_ref)


# ------- plotly panel builder -------

def add_scene_traces(fig, row, col, theta_rad, depth, uv, pos_arr, app_arr, yaw_arr, group_arr, showlegend_for=None):
    # 1) rotated point cloud
    pc = back_project(depth, stride=3)
    if len(pc) > 6000:
        sel = np.random.choice(len(pc), 6000, replace=False)
        pc = pc[sel]
    pc_rot = rot3d_z_consistent(pc, theta_rad)

    fig.add_trace(go.Scatter3d(
        x=pc_rot[:, 0], y=pc_rot[:, 1], z=pc_rot[:, 2],
        mode="markers",
        marker=dict(size=1.8, color=pc_rot[:, 2],
                    colorscale="Greys", opacity=0.55, showscale=False),
        name="scene PC", showlegend=(showlegend_for == "scene"),
    ), row=row, col=col)

    # 2) uv 3D marker
    uv3 = uv_to_3d(depth, uv)
    if uv3 is not None:
        uv3_rot = rot3d_z_consistent(uv3[None], theta_rad)[0]
        fig.add_trace(go.Scatter3d(
            x=[uv3_rot[0]], y=[uv3_rot[1]], z=[uv3_rot[2]],
            mode="markers",
            marker=dict(size=7, color="magenta", symbol="x", line=dict(color="white", width=1)),
            name="uv (YOLO)", showlegend=(showlegend_for == "scene"),
        ), row=row, col=col)

    # 3) per-group grasp wireframes
    from collections import defaultdict
    buckets = defaultdict(lambda: dict(ux=[], uy=[], uz=[], sx=[], sy=[], sz=[],
                                        ax=[], ay=[], az=[],
                                        cx=[], cy=[], cz=[], cu=[], cv=[], cw=[],
                                        tx=[], ty=[], tz=[]))
    n_grasps = len(pos_arr)
    for i in range(n_grasps):
        pos = pos_arr[i].astype(np.float64)
        app = app_arr[i].astype(np.float64) / np.linalg.norm(app_arr[i])
        y = float(yaw_arr[i])
        # rotate
        pos_r = rot3d_z_consistent(pos[None], theta_rad)[0]
        app_r = rot3d_z_consistent(app[None], theta_rad)[0]
        # rebuild R_tool and extract new yaw
        from src.flow_dataset import _yaw_from_Rtool
        R_tool = _build_R_tool(app, y)
        c, s = math.cos(theta_rad), math.sin(theta_rad)
        Rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float64)
        R_tool_new = Rz @ R_tool
        y_r = _yaw_from_Rtool(R_tool_new)

        u_poly, palm_stem, app_stem, cone_tail, cone_dir, tcp = gripper_3d_lines(pos_r, app_r, y_r)
        grp = int(group_arr[i])
        bk = buckets[grp]
        bk["ux"].extend([*u_poly[:, 0], None])
        bk["uy"].extend([*u_poly[:, 1], None])
        bk["uz"].extend([*u_poly[:, 2], None])
        bk["sx"].extend([*palm_stem[:, 0], None])
        bk["sy"].extend([*palm_stem[:, 1], None])
        bk["sz"].extend([*palm_stem[:, 2], None])
        bk["ax"].extend([*app_stem[:, 0], None])
        bk["ay"].extend([*app_stem[:, 1], None])
        bk["az"].extend([*app_stem[:, 2], None])
        bk["cx"].append(cone_tail[0]); bk["cy"].append(cone_tail[1]); bk["cz"].append(cone_tail[2])
        bk["cu"].append(cone_dir[0]);  bk["cv"].append(cone_dir[1]);  bk["cw"].append(cone_dir[2])
        bk["tx"].append(tcp[0]); bk["ty"].append(tcp[1]); bk["tz"].append(tcp[2])

    for grp, bk in buckets.items():
        color = GROUP_COLORS[grp]
        name = f"{GROUP_NAMES[grp]}"
        fig.add_trace(go.Scatter3d(
            x=bk["ux"], y=bk["uy"], z=bk["uz"],
            mode="lines", line=dict(color=color, width=5),
            name=name, legendgroup=name,
            showlegend=(showlegend_for == "scene"),
        ), row=row, col=col)
        fig.add_trace(go.Scatter3d(
            x=bk["sx"], y=bk["sy"], z=bk["sz"],
            mode="lines", line=dict(color=color, width=3),
            legendgroup=name, showlegend=False,
        ), row=row, col=col)
        fig.add_trace(go.Scatter3d(
            x=bk["ax"], y=bk["ay"], z=bk["az"],
            mode="lines", line=dict(color=color, width=2, dash="dot"),
            legendgroup=name, showlegend=False,
        ), row=row, col=col)
        fig.add_trace(go.Cone(
            x=bk["cx"], y=bk["cy"], z=bk["cz"],
            u=bk["cu"], v=bk["cv"], w=bk["cw"],
            sizemode="absolute", sizeref=0.012, anchor="tail",
            colorscale=[[0, color], [1, color]], showscale=False,
            hoverinfo="skip", legendgroup=name, showlegend=False,
        ), row=row, col=col)
        fig.add_trace(go.Scatter3d(
            x=bk["tx"], y=bk["ty"], z=bk["tz"],
            mode="markers",
            marker=dict(size=3, color=color, line=dict(color="white", width=0.5)),
            legendgroup=name, showlegend=False,
        ), row=row, col=col)


def make_html(label, title_cls, rows, out_path):
    depth, uv, pos_arr, app_arr, yaw_arr, group_arr, sample_ref = rows

    # 2×3 subplots, each is a 3D scene
    specs = [[{"type": "scene"} for _ in range(3)] for _ in range(2)]
    titles = [f"θ = {th:+d}°   ({'original' if th == 0 else 'augmented'})"
              for th in THETAS_DEG]
    fig = make_subplots(rows=2, cols=3, specs=specs, subplot_titles=titles,
                        horizontal_spacing=0.02, vertical_spacing=0.06)

    for i, th_deg in enumerate(THETAS_DEG):
        r, c = i // 3 + 1, i % 3 + 1
        theta = math.radians(th_deg)
        showlegend = "scene" if i == 0 else None
        add_scene_traces(fig, r, c, theta, depth, uv, pos_arr, app_arr, yaw_arr,
                          group_arr, showlegend_for=showlegend)

    # identical scene formatting for all 6
    scene_props = dict(
        aspectmode="data",
        xaxis=dict(title="X cam (m)", backgroundcolor="#fafafa",
                   gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(title="Y cam (m)", backgroundcolor="#fafafa",
                   gridcolor="rgba(0,0,0,0.08)"),
        zaxis=dict(title="Z cam (m, depth→)", backgroundcolor="#fafafa",
                   gridcolor="rgba(0,0,0,0.08)"),
        camera=dict(up=dict(x=0, y=-1, z=0),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.2, y=-1.2, z=-1.2)),
        dragmode="turntable",
    )
    for axn in ["scene", "scene2", "scene3", "scene4", "scene5", "scene6"]:
        fig.layout[axn].update(**scene_props)

    fig.update_layout(
        title=f"<b>Augmentation check — {title_cls}</b>   sample={sample_ref}   "
              f"n_grasps={len(pos_arr)}",
        width=1600, height=1000,
        template="plotly_white",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="gray",
                    borderwidth=1, itemsizing="constant"),
        margin=dict(l=10, r=10, t=80, b=10),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"wrote {out_path}  ({out_path.stat().st_size/1e6:.2f} MB)")


def main():
    np.random.seed(0)
    with h5py.File(H5_PATH, "r") as f:
        st_pick = pick_standing_16(f)
        ly_pick = pick_lying_8_bottle_or_can(f)
        assert st_pick is not None, "no standing 16-grasp object"
        assert ly_pick is not None, "no lying 8-grasp bottle/can"

        rows_st = load_rows(f, st_pick[2])
        rows_ly = load_rows(f, ly_pick[2])

    make_html("standing", "standing bottle/can (16 grasps)",
              rows_st, OUT_DIR / "aug_check_standing.html")
    make_html("lying", "lying bottle/can (8 grasps)",
              rows_ly, OUT_DIR / "aug_check_lying.html")


if __name__ == "__main__":
    main()
