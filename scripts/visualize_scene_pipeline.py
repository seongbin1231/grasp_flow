"""
Full pipeline 시각화 — 씬 하나에 **모든 stable object** 의 ICP + grasp 합성 결과를
한 화면에서 확인. 각 물체:
  - scene PC (클래스별 색)
  - ICP-aligned PLY (진한 초록 / 녹청)
  - 물체 좌표계 축
  - grasp U-bracket (정책: standing 16, lying 8/6, cube 2)

선정 기준: 다양한 mode·class 조합이 한 씬에 나타나는 샘플 N개.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import h5py
import numpy as np
import open3d as o3d
import plotly.graph_objects as go

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
POSES_H5 = ROOT / "img_dataset/icp_cache/poses.h5"
DET_H5 = ROOT / "img_dataset/yolo_cache_v3/detections.h5"
PLY_DIR = Path("/home/robotics/Competition/RoboCup_ARM/models/ply")
OUT_DIR = ROOT / "scripts/_scene_pipeline_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_FX = K_FY = 1109.0
K_CX, K_CY = 640.0, 360.0
IMG_W, IMG_H = 1280, 720

GRIPPER_HALF = 0.0425
FINGER_LEN = 0.040
PALM_BACK = 0.025
APPROACH_STEM_LEN = 0.050
SIDE_CAP_OFFSET_FROM_TIP_M = 0.015
SIDE_AZIMUTHS = np.linspace(0, 2 * np.pi, 8, endpoint=False)

CLASS_TO_PLY = {
    "bottle": "blueBottle.ply", "can": "greenCan.ply",
    "cube_blue": "cube.ply", "cube_green": "cube.ply",
    "cube_p": "cube.ply", "cube_red": "cube.ply",
    "marker": "marker.ply", "spam": "Simsort_SPAM.ply",
}

# 각 물체 grasp 그룹 색
GROUP_COLORS = {
    "top-down": "#e53935",      # 빨강
    "side-cap": "#fb8c00",      # 주황
    "lying": "#1e88e5",         # 파랑
    "cube": "#8e24aa",          # 보라
}

# 씬별 물체 scene-PC 색 팔레트 (물체별로 구분)
OBJECT_PC_PALETTE = [
    "#bbbbbb", "#a3b1c6", "#c5bfa8", "#a6c8a8",
    "#c9a8a8", "#a8c5c9", "#c9b5a8", "#bab0d0",
]


# ================== geom helpers ==================

def load_ply_points(name):
    pcd = o3d.io.read_point_cloud(str(PLY_DIR / name))
    pts = np.asarray(pcd.points)
    if pts.ptp(axis=0).max() > 1.0:
        pts = pts * 0.001
    return pts - pts.mean(axis=0)


def load_depth_meter(p):
    return cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0


def poly_to_mask(poly):
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def depth_mask_to_pc(depth_m, mask):
    ys, xs = np.where(mask & (depth_m > 0.1) & (depth_m < 2.0))
    z = depth_m[ys, xs]
    x = (xs - K_CX) * z / K_FX
    y = (ys - K_CY) * z / K_FY
    return np.stack([x, y, z], axis=1).astype(np.float32)


def build_gripper_frame(approach_vec, yaw):
    a = approach_vec / np.linalg.norm(approach_vec)
    ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.95 else np.array([0, 1.0, 0])
    b0 = ref - (ref @ a) * a; b0 /= np.linalg.norm(b0)
    n0 = np.cross(a, b0)
    b = b0 * np.cos(yaw) + n0 * np.sin(yaw); b /= np.linalg.norm(b)
    return b, np.cross(a, b)


def decide_mode(R, long_axis_idx, cls_name):
    if cls_name.startswith("cube"):
        return "cube"
    e = np.zeros(3); e[long_axis_idx] = 1.0
    return "standing" if abs((R @ e)[2]) > 0.7 else "lying"


def tilted_approach(tilt, az):
    ax = np.array([np.cos(az), np.sin(az), 0.0])
    v = np.array([0, 0, 1.0])
    c, s = np.cos(tilt), np.sin(tilt)
    return v * c + np.cross(ax, v) * s + ax * (ax @ v) * (1 - c)


def gen_grasps(R_icp, t_icp, extent, long_axis_idx, mode, cls_name):
    grasps: list[tuple] = []
    L = float(extent[long_axis_idx])
    e_long = np.zeros(3); e_long[long_axis_idx] = 1.0
    long_cam = R_icp @ e_long
    sign_cam = -1.0 if long_cam[2] > 0 else 1.0

    if mode == "standing":
        top_off = sign_cam * (L * 0.5) * e_long
        p_top = (t_icp + R_icp @ top_off).copy(); p_top[2] += 0.003
        for yaw in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            grasps.append((p_top, np.array([0, 0, 1.0]), float(yaw), "top-down"))
        side_off = sign_cam * (L * 0.5 - SIDE_CAP_OFFSET_FROM_TIP_M) * e_long
        p_side = (t_icp + R_icp @ side_off).copy()
        for az in SIDE_AZIMUTHS:
            grasps.append((p_side, np.array([np.cos(az), np.sin(az), 0.0]),
                           0.0, "side-cap"))
    elif mode == "lying":
        N = 3 if cls_name in ("marker", "spam") else 4
        ss = np.linspace(-0.35 * L, 0.35 * L, N)
        short = np.array([float(extent[i]) for i in range(3) if i != long_axis_idx])
        short_r = short.mean() / 2
        lxy = long_cam[:2]; lxy /= (np.linalg.norm(lxy) + 1e-9)
        base_yaw = np.arctan2(lxy[1], lxy[0]) + np.pi / 2
        app = np.array([0, 0, 1.0])
        for s in ss:
            off = np.zeros(3); off[long_axis_idx] = s
            p = t_icp + R_icp @ off
            p = np.array([p[0], p[1], p[2] - short_r])
            for sym in (0.0, np.pi):
                grasps.append((p, app, float(base_yaw + sym), "lying"))
    elif mode == "cube":
        cam_z_proj = R_icp[2, :]
        vcol = int(np.argmax(np.abs(cam_z_proj)))
        edges = [i for i in range(3) if i != vcol]
        cube_half = float(extent[vcol]) / 2
        p = np.array([t_icp[0], t_icp[1], t_icp[2] - cube_half + 0.002])
        app = np.array([0, 0, 1.0])
        for ec in edges:
            ev = R_icp[:, ec]
            grasps.append((p, app, float(np.arctan2(ev[1], ev[0])), "cube"))
    return grasps


def add_gripper_traces(fig, pos, app, yaw, color, legend_group, show_legend):
    a = app / np.linalg.norm(app)
    b, _ = build_gripper_frame(app, yaw)
    tip1 = pos + b * GRIPPER_HALF
    tip2 = pos - b * GRIPPER_HALF
    base1 = tip1 - a * FINGER_LEN
    base2 = tip2 - a * FINGER_LEN
    palm_c = pos - a * FINGER_LEN
    wrist = palm_c - a * PALM_BACK
    start = wrist - a * APPROACH_STEM_LEN
    # U-bracket
    fig.add_trace(go.Scatter3d(
        x=[tip1[0], base1[0], base2[0], tip2[0]],
        y=[tip1[1], base1[1], base2[1], tip2[1]],
        z=[tip1[2], base1[2], base2[2], tip2[2]],
        mode="lines",
        line=dict(color=color, width=6),
        legendgroup=legend_group, showlegend=show_legend,
        name=legend_group, hoverinfo="skip",
    ))
    # body stem
    fig.add_trace(go.Scatter3d(
        x=[palm_c[0], wrist[0]], y=[palm_c[1], wrist[1]], z=[palm_c[2], wrist[2]],
        mode="lines", line=dict(color=color, width=4),
        legendgroup=legend_group, showlegend=False, hoverinfo="skip",
    ))
    # approach path dotted
    fig.add_trace(go.Scatter3d(
        x=[start[0], wrist[0]], y=[start[1], wrist[1]], z=[start[2], wrist[2]],
        mode="lines", line=dict(color=color, width=2, dash="dot"),
        legendgroup=legend_group, showlegend=False, hoverinfo="skip",
    ))


def add_object_frame(fig, R, t, length=0.04, legend_name="obj frame"):
    colors = ["#ff0033", "#33cc00", "#3366ff"]
    names = ["X", "Y", "Z"]
    for i in range(3):
        end = t + R[:, i] * length
        fig.add_trace(go.Scatter3d(
            x=[t[0], end[0]], y=[t[1], end[1]], z=[t[2], end[2]],
            mode="lines", line=dict(color=colors[i], width=4),
            legendgroup=legend_name, showlegend=False, hoverinfo="skip",
            name=f"{legend_name}.{names[i]}",
        ))


def scatter_pc(fig, pts, color, size, name, opacity, legendgroup, showlegend):
    if len(pts) > 3000:
        idx = np.random.choice(len(pts), 3000, replace=False); pts = pts[idx]
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(size=size, color=color, opacity=opacity),
        name=name, legendgroup=legendgroup, showlegend=showlegend,
        hoverinfo="skip",
    ))


# ================== per-scene rendering ==================

def render_scene(poses, dets, sid: str, out_dir: Path):
    if sid not in poses or sid not in dets:
        return 0
    g_pose = poses[sid]
    g_det = dets[sid]
    class_names = list(dets.attrs["class_names"])

    # collect all stable objects
    stable_objs = []
    for obj_name in g_pose.keys():
        g_obj = g_pose[obj_name]
        if not g_obj.attrs.get("stable_flag", False):
            continue
        stable_objs.append((obj_name, g_obj))
    if not stable_objs:
        return 0

    fig = go.Figure()
    ply_cache: dict[str, np.ndarray] = {}
    grasp_group_seen: set[str] = set()

    depth_path = ROOT / g_pose.attrs["depth_path"]
    depth_m = load_depth_meter(depth_path)

    total_grasps = 0
    summary_rows = []

    for i, (obj_name, g_obj) in enumerate(stable_objs):
        k = int(obj_name.split("_")[1])
        cls_name = g_obj.attrs["class_name"]
        poly = np.asarray(g_det["mask_poly"][k]).reshape(-1, 2)
        if poly.size < 6:
            continue
        mask = poly_to_mask(poly)
        scene_pts = depth_mask_to_pc(depth_m, mask)

        R = np.asarray(g_obj["R_cam"])
        t = np.asarray(g_obj["t_cam"])
        extent = np.asarray(g_obj["model_extent"])
        long_idx = int(np.asarray(g_obj["model_long_axis_idx"]))
        fit = float(np.asarray(g_obj["fitness"]))
        ply_file = g_obj.attrs["ply_file"]
        if ply_file not in ply_cache:
            ply_cache[ply_file] = load_ply_points(ply_file)
        aligned = (R @ ply_cache[ply_file].T).T + t

        pc_color = OBJECT_PC_PALETTE[i % len(OBJECT_PC_PALETTE)]
        obj_label = f"[{i}] {cls_name}"
        # scene PC
        scatter_pc(fig, scene_pts, pc_color, 1.8,
                   f"{obj_label} scene", 0.55,
                   legendgroup=obj_label, showlegend=True)
        # aligned PLY
        scatter_pc(fig, aligned, "#00c864", 2.2,
                   f"{obj_label} PLY", 0.85,
                   legendgroup=obj_label, showlegend=False)
        # t_cam marker
        fig.add_trace(go.Scatter3d(
            x=[t[0]], y=[t[1]], z=[t[2]],
            mode="markers",
            marker=dict(size=6, color="cyan", symbol="x"),
            legendgroup=obj_label, showlegend=False,
            name=f"{obj_label} t_cam",
            hovertext=f"{obj_label} fit={fit:.2f}", hoverinfo="text",
        ))
        # obj frame
        add_object_frame(fig, R, t, length=max(float(extent.max()) * 0.4, 0.03),
                         legend_name=obj_label)

        # grasps
        mode = decide_mode(R, long_idx, cls_name)
        grasps = gen_grasps(R, t, extent, long_idx, mode, cls_name)
        total_grasps += len(grasps)
        summary_rows.append((cls_name, mode, len(grasps), fit))
        for pos, app, yaw, group in grasps:
            color = GROUP_COLORS.get(group, "#ffd700")
            show = group not in grasp_group_seen
            grasp_group_seen.add(group)
            add_gripper_traces(fig, pos, app, yaw, color, group, show)

    # ----- layout -----
    # camera view
    all_pts_list = []
    for obj_name, g_obj in stable_objs:
        all_pts_list.append(np.asarray(g_obj["t_cam"])[None])
    all_pts = np.vstack(all_pts_list)
    cx, cy, cz = all_pts.mean(axis=0)

    title_items = " | ".join(
        f"{c}[{m}]={n}" for c, m, n, _ in summary_rows
    )
    fig.update_layout(
        title=(f"<b>{sid}</b>   objects={len(stable_objs)}   "
               f"total_grasps=<b>{total_grasps}</b><br>"
               f"<span style='font-size:0.85em;color:#555'>{title_items}</span>"),
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X cam (m)", backgroundcolor="#fafafa"),
            yaxis=dict(title="Y cam (m)", backgroundcolor="#fafafa"),
            zaxis=dict(title="Z cam (m, depth→)", backgroundcolor="#fafafa"),
            camera=dict(up=dict(x=0, y=-1, z=0),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.3, y=-1.2, z=-1.4)),
            dragmode="turntable",
        ),
        template="plotly_white",
        width=1400, height=950,
        margin=dict(l=10, r=10, t=80, b=10),
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.99,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="gray",
                    borderwidth=1, itemsizing="constant",
                    groupclick="toggleitem"),
    )

    out_path = out_dir / f"{sid}.html"
    fig.write_html(str(out_path))
    print(f"wrote {out_path}  (objects={len(stable_objs)}, grasps={total_grasps})")
    return total_grasps


def pick_diverse_scenes(poses, n=8):
    """각 씬에서 stable object 수 많고 mode 다양한 것 우선."""
    scored = []
    for sid in poses.keys():
        g = poses[sid]
        modes = set(); n_stable = 0
        classes = set()
        for obj_name in g.keys():
            g_obj = g[obj_name]
            if not g_obj.attrs.get("stable_flag", False):
                continue
            n_stable += 1
            cls = g_obj.attrs["class_name"]
            R = np.asarray(g_obj["R_cam"])
            long_idx = int(np.asarray(g_obj["model_long_axis_idx"]))
            modes.add(decide_mode(R, long_idx, cls))
            classes.add(cls)
        if n_stable == 0:
            continue
        # 점수: 다른 mode 수 × n_stable × class 다양성
        score = len(modes) * 10 + n_stable * 2 + len(classes)
        scored.append((score, sid, n_stable, tuple(modes)))
    scored.sort(reverse=True)
    # 씬별 scene prefix 다양화 (random1..6)
    picked: list[str] = []
    seen_scene = set()
    for _, sid, _, _ in scored:
        scene_prefix = sid.rsplit("_", 1)[0]
        if scene_prefix in seen_scene and len(picked) >= n // 2:
            continue
        seen_scene.add(scene_prefix)
        picked.append(sid)
        if len(picked) >= n:
            break
    # 부족하면 낮은 점수로 채움
    i = 0
    while len(picked) < n and i < len(scored):
        if scored[i][1] not in picked:
            picked.append(scored[i][1])
        i += 1
    return picked[:n]


def main():
    with h5py.File(POSES_H5, "r") as poses, h5py.File(DET_H5, "r") as dets:
        # 테스트용 — 다양한 mode/class 섞인 씬 3개만
        scenes = pick_diverse_scenes(poses, n=3)
        print(f"Selected scenes: {scenes}")
        for sid in scenes:
            render_scene(poses, dets, sid, OUT_DIR)


if __name__ == "__main__":
    main()
