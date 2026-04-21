"""
3D interactive visualization (plotly HTML) for each object mode-class combo.

각 케이스:
  - Scene PC (회색)
  - ICP-aligned PLY (초록)
  - 계획된 grasp 후보들 (gripper 선분 + approach arrow)
  - ICP t_cam (시안 +), cap center (노란 별)

Tilt 도입: standing cap은 pure top-down + 4 azimuth × 30° tilt로 reachability 대비.
Lying은 긴축 따라 N 위치 × 180° 대칭 × 3 tilt.
Cube는 2 yaw × 3 tilt.
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
OUT_DIR = ROOT / "scripts/_grasp_3d_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_FX = K_FY = 1109.0
K_CX, K_CY = 640.0, 360.0
IMG_W, IMG_H = 1280, 720

GRIPPER_HALF = 0.0425         # 최대 개폐폭 0.085/2
FINGER_LEN = 0.040            # finger 길이 (Robotiq 2F-85 ≈ 40mm)
PALM_BACK = 0.025             # palm 뒤 wrist/body 시각 길이
APPROACH_STEM_LEN = 0.050     # dotted 접근 경로 길이

TARGET_CASES = [
    ("standing", "bottle"),
    ("standing", "can"),
    ("lying", "bottle"),
    ("lying", "can"),
    ("lying", "marker"),
    ("lying", "spam"),
    ("cube", "cube_red"),
]

# Standing bottle/can 정책:
#   1) pure top-down 8 yaw
#   2) 측면(horizontal, cam Z축에 수직) 8방위에서 cap 잡기
# 그 외(lying, cube, standing marker/spam): pure top-down only
SIDE_AZIMUTHS = np.linspace(0, 2 * np.pi, 8, endpoint=False)
SIDE_CAP_OFFSET_FROM_TIP_M = 0.015   # cap 밴드 잡기 위해 tip에서 15mm 아래


def decide_mode(R, long_axis_idx, cls_name):
    if cls_name.startswith("cube"):
        return "cube"
    e = np.zeros(3); e[long_axis_idx] = 1.0
    return "standing" if abs((R @ e)[2]) > 0.7 else "lying"


def load_ply_points(name):
    pcd = o3d.io.read_point_cloud(str(PLY_DIR / name))
    pts = np.asarray(pcd.points)
    if pts.ptp(axis=0).max() > 1.0:
        pts = pts * 0.001
    return pts - pts.mean(axis=0)


def depth_mask_to_pc(depth_m, mask):
    ys, xs = np.where(mask & (depth_m > 0.1) & (depth_m < 2.0))
    z = depth_m[ys, xs]
    x = (xs - K_CX) * z / K_FX
    y = (ys - K_CY) * z / K_FY
    return np.stack([x, y, z], axis=1).astype(np.float32)


def poly_to_mask(poly):
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def tilted_approach(tilt, az):
    """Rotate cam +Z by `tilt` around horizontal axis at azimuth `az`."""
    ax = np.array([np.cos(az), np.sin(az), 0.0])  # rotation axis in cam XY
    v = np.array([0, 0, 1.0])                     # base approach (+Z)
    c, s = np.cos(tilt), np.sin(tilt)
    return v * c + np.cross(ax, v) * s + ax * (ax @ v) * (1 - c)


def build_gripper_frame(approach_vec, yaw_around_approach):
    """Return (binormal_unit, axis_unit) so binormal = open direction."""
    a = approach_vec / np.linalg.norm(approach_vec)
    ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.95 else np.array([0, 1.0, 0])
    b0 = ref - (ref @ a) * a; b0 /= np.linalg.norm(b0)
    n0 = np.cross(a, b0)
    b = b0 * np.cos(yaw_around_approach) + n0 * np.sin(yaw_around_approach)
    b /= np.linalg.norm(b)
    return b, np.cross(a, b)


def gen_grasps_v3(R_icp, t_icp, extent, long_axis_idx, mode, cls_name):
    """Generate grasps with tilted approaches.

    Returns list of (pos(3,), approach(3,), yaw_around_approach, group_str).
    """
    grasps: list[tuple] = []
    L = float(extent[long_axis_idx])

    e_long = np.zeros(3); e_long[long_axis_idx] = 1.0
    long_cam = R_icp @ e_long
    sign_toward_cam = -1.0 if long_cam[2] > 0 else 1.0

    if mode == "standing":
        # 세워진 bottle/can: **정확히 16개** = top-down 8 + 측면 horizontal 8 (두 그룹이 90° 직각)
        # (A) top-down: cap 상단에서 아래로 접근, yaw 8방향
        top_off_obj = sign_toward_cam * (L * 0.5) * e_long
        p_cap_top = (t_icp + R_icp @ top_off_obj).copy()
        p_cap_top[2] += 0.003
        app_top = np.array([0.0, 0.0, 1.0])     # cam +Z (위에서 아래로)
        for yaw in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            grasps.append((p_cap_top, app_top, float(yaw), "top-down"))

        # (B) side-cap: 수평 8방위 approach, cap 밴드 높이
        side_off_obj = sign_toward_cam * (L * 0.5 - SIDE_CAP_OFFSET_FROM_TIP_M) * e_long
        p_cap_side = (t_icp + R_icp @ side_off_obj).copy()
        for az in SIDE_AZIMUTHS:
            app_h = np.array([np.cos(az), np.sin(az), 0.0])   # cam Z 에 수직 = 90°
            grasps.append((p_cap_side, app_h, 0.0, "side-cap"))

    elif mode == "lying":
        N = 3 if cls_name in ("marker", "spam") else 4
        ss = np.linspace(-0.35 * L, 0.35 * L, N)
        short = np.array([float(extent[i]) for i in range(3) if i != long_axis_idx])
        short_r = short.mean() / 2
        long_xy = long_cam[:2]; long_xy /= np.linalg.norm(long_xy) + 1e-9
        long_az = np.arctan2(long_xy[1], long_xy[0])
        base_yaw = long_az + np.pi / 2
        app = tilted_approach(0, 0)   # pure top-down
        for s in ss:
            off = np.zeros(3); off[long_axis_idx] = s
            p = t_icp + R_icp @ off
            p = np.array([p[0], p[1], p[2] - short_r])
            for sym in (0.0, np.pi):   # 180° 대칭
                grasps.append((p, app, base_yaw + sym, "lying"))

    elif mode == "cube":
        # cube top face (카메라 쪽) 위에서 잡기. 중요: yaw는 **면 edge에 정렬**,
        # 대각선 금지.
        # R_icp의 3 column 중 카메라 Z에 가장 정렬된 column = cube의 "수직" 축.
        # 나머지 2 column이 top face의 변(edge) 방향.
        cam_z_proj = R_icp[2, :]
        vertical_col = int(np.argmax(np.abs(cam_z_proj)))
        edge_cols = [i for i in range(3) if i != vertical_col]
        # cube top surface z offset = half of vertical extent
        cube_half = float(extent[vertical_col]) / 2
        p = np.array([t_icp[0], t_icp[1], t_icp[2] - cube_half + 0.002])
        app = tilted_approach(0, 0)
        # edge direction in cam XY
        for ec in edge_cols:
            edge_cam = R_icp[:, ec]
            yaw = float(np.arctan2(edge_cam[1], edge_cam[0]))
            grasps.append((p, app, yaw, "cube"))
    return grasps


# -----------------------------------------------------------

def scatter_points(pts, color, size, name, opacity=0.7, showlegend=True):
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(size=size, color=color, opacity=opacity),
        name=name, showlegend=showlegend,
    )


GROUP_COLORS = {
    "top-down": "red",
    "side-cap": "orange",
    "lying": "blue",
    "cube": "magenta",
}


def add_grippers(fig, grasps):
    """
    각 grasp = 2F-85 그리퍼 wireframe:
      tip1 ─ base1 ━━ base2 ─ tip2       (U자: 좌핑거 + 팜 + 우핑거)
                  │
                  palm_center
                  │  (팜 뒤 body stem, PALM_BACK)
                  │
                  wrist
                  │  (approach stem, dotted, APPROACH_STEM_LEN)
                  │
                  (화살표 cone)
    """
    from collections import defaultdict
    buckets: dict[str, dict] = defaultdict(lambda: {
        # wireframe solid lines: finger1 + palm + finger2
        "ux": [], "uy": [], "uz": [],
        # palm-to-wrist body stem (solid)
        "bx": [], "by": [], "bz": [],
        # wrist backward dotted approach path
        "ax": [], "ay": [], "az": [],
        # cone for approach arrow (at wrist, pointing toward palm)
        "cx": [], "cy": [], "cz": [],
        "cu": [], "cv": [], "cw": [],
        # TCP marker
        "tx": [], "ty": [], "tz": [],
        "hover": [],
    })
    for pos, app, yaw, group in grasps:
        a = app / np.linalg.norm(app)
        b, _ = build_gripper_frame(app, yaw)
        tip1 = pos + b * GRIPPER_HALF
        tip2 = pos - b * GRIPPER_HALF
        base1 = tip1 - a * FINGER_LEN
        base2 = tip2 - a * FINGER_LEN
        palm_center = pos - a * FINGER_LEN
        wrist = palm_center - a * PALM_BACK
        app_start = wrist - a * APPROACH_STEM_LEN

        bk = buckets[group]
        # U-bracket: tip1 → base1 → base2 → tip2
        bk["ux"].extend([tip1[0], base1[0], base2[0], tip2[0], None])
        bk["uy"].extend([tip1[1], base1[1], base2[1], tip2[1], None])
        bk["uz"].extend([tip1[2], base1[2], base2[2], tip2[2], None])
        # palm → wrist body
        bk["bx"].extend([palm_center[0], wrist[0], None])
        bk["by"].extend([palm_center[1], wrist[1], None])
        bk["bz"].extend([palm_center[2], wrist[2], None])
        # wrist backward dotted path
        bk["ax"].extend([app_start[0], wrist[0], None])
        bk["ay"].extend([app_start[1], wrist[1], None])
        bk["az"].extend([app_start[2], wrist[2], None])
        # cone at wrist tip (pointing along approach, anchor=tail)
        bk["cx"].append(app_start[0]); bk["cy"].append(app_start[1]); bk["cz"].append(app_start[2])
        bk["cu"].append(a[0]);         bk["cv"].append(a[1]);         bk["cw"].append(a[2])
        # TCP marker
        bk["tx"].append(pos[0]); bk["ty"].append(pos[1]); bk["tz"].append(pos[2])
        bk["hover"].append(f"{group} yaw={np.degrees(yaw):.0f}°")

    for group, bk in buckets.items():
        color = GROUP_COLORS.get(group, "yellow")
        n = len(bk["hover"])
        # U-bracket — finger + palm + finger (solid, thick)
        fig.add_trace(go.Scatter3d(
            x=bk["ux"], y=bk["uy"], z=bk["uz"],
            mode="lines",
            line=dict(color=color, width=8),
            name=f"{group} ({n})",
            legendgroup=group, showlegend=True,
        ))
        # palm→wrist body stem (solid, slightly thinner)
        fig.add_trace(go.Scatter3d(
            x=bk["bx"], y=bk["by"], z=bk["bz"],
            mode="lines",
            line=dict(color=color, width=5),
            legendgroup=group, showlegend=False,
        ))
        # approach dotted path
        fig.add_trace(go.Scatter3d(
            x=bk["ax"], y=bk["ay"], z=bk["az"],
            mode="lines",
            line=dict(color=color, width=2, dash="dot"),
            legendgroup=group, showlegend=False,
        ))
        # cone arrowhead (at stem start, pointing in approach dir)
        fig.add_trace(go.Cone(
            x=bk["cx"], y=bk["cy"], z=bk["cz"],
            u=bk["cu"], v=bk["cv"], w=bk["cw"],
            sizemode="absolute", sizeref=0.012,
            anchor="tail",
            colorscale=[[0, color], [1, color]],
            showscale=False,
            legendgroup=group, showlegend=False, hoverinfo="skip",
        ))
        # TCP markers (small dots at grasp center)
        fig.add_trace(go.Scatter3d(
            x=bk["tx"], y=bk["ty"], z=bk["tz"],
            mode="markers",
            marker=dict(size=4, color=color,
                        line=dict(color="white", width=1)),
            legendgroup=group, showlegend=False,
            hovertext=bk["hover"], hoverinfo="text",
        ))


def add_object_frame(fig, R, t, length=0.05):
    """ICP rotation의 3축을 R(X)/G(Y)/B(Z)로 표시."""
    colors = ["#ff0033", "#33cc00", "#3366ff"]
    names = ["obj X", "obj Y", "obj Z"]
    for i in range(3):
        end = t + R[:, i] * length
        fig.add_trace(go.Scatter3d(
            x=[t[0], end[0]], y=[t[1], end[1]], z=[t[2], end[2]],
            mode="lines",
            line=dict(color=colors[i], width=6),
            name=names[i], legendgroup="obj_frame",
            showlegend=(i == 0),
        ))


def add_table_plane(fig, scene_pts, size=0.4):
    """scene PC의 Z max 근처로 테이블 평면 시각화."""
    if len(scene_pts) < 20:
        return
    z_table = float(np.percentile(scene_pts[:, 2], 95))
    cx = float(np.mean(scene_pts[:, 0]))
    cy = float(np.mean(scene_pts[:, 1]))
    xs = np.array([cx - size, cx + size, cx + size, cx - size, cx - size])
    ys = np.array([cy - size, cy - size, cy + size, cy + size, cy - size])
    zs = np.full_like(xs, z_table)
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color="rgba(120,120,120,0.5)", width=2),
        name="table plane",
        showlegend=True,
    ))


def render_case(poses, dets, sid, obj_name, key):
    mode, cls = key
    g_pose = poses[sid]
    g_det = dets[sid]
    g_obj = g_pose[obj_name]
    k = int(obj_name.split("_")[1])

    depth_path = ROOT / g_pose.attrs["depth_path"]
    depth_m = cv2.imread(str(depth_path),
                         cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    poly = np.asarray(g_det["mask_poly"][k]).reshape(-1, 2)
    mask = poly_to_mask(poly)
    scene_pts = depth_mask_to_pc(depth_m, mask)

    R = np.asarray(g_obj["R_cam"])
    t = np.asarray(g_obj["t_cam"])
    extent = np.asarray(g_obj["model_extent"])
    long_idx = int(np.asarray(g_obj["model_long_axis_idx"]))
    ply_file = g_obj.attrs["ply_file"]
    model_pts = load_ply_points(ply_file)
    aligned = (R @ model_pts.T).T + t

    grasps = gen_grasps_v3(R, t, extent, long_idx, mode, cls)

    fig = go.Figure()
    # Table plane (희미하게)
    add_table_plane(fig, scene_pts, size=0.15)
    # Scene PC
    pc = scene_pts
    if len(pc) > 6000:
        idx = np.random.choice(len(pc), 6000, replace=False); pc = pc[idx]
    fig.add_trace(go.Scatter3d(
        x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
        mode="markers",
        marker=dict(size=2.0, color=pc[:, 2], colorscale="Greys",
                    opacity=0.6,
                    colorbar=dict(title="z cam (m)", thickness=10, len=0.5)),
        name="scene PC",
    ))
    # Aligned PLY
    ap = aligned
    if len(ap) > 2000:
        idx = np.random.choice(len(ap), 2000, replace=False); ap = ap[idx]
    fig.add_trace(scatter_points(ap, "#00c864", 2.6, "ICP PLY", 0.9))
    # ICP t_cam marker
    fig.add_trace(go.Scatter3d(
        x=[t[0]], y=[t[1]], z=[t[2]],
        mode="markers",
        marker=dict(size=7, color="cyan", symbol="x"),
        name="ICP t_cam"))
    # 물체 좌표계 축
    add_object_frame(fig, R, t, length=max(float(extent.max()) * 0.5, 0.04))
    # Grasps
    add_grippers(fig, grasps)

    # bbox for view range
    all_pts = np.vstack([scene_pts, aligned, t.reshape(1, 3)])
    cx = float(all_pts[:, 0].mean())
    cy = float(all_pts[:, 1].mean())
    cz = float(all_pts[:, 2].mean())
    rng = float(np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0)) * 0.6)

    # title에 group별 count 분해
    from collections import Counter
    group_cnt = Counter(g[3] for g in grasps)
    group_str = " + ".join(f"{k} {v}" for k, v in sorted(group_cnt.items()))
    fig.update_layout(
        title=(f"<b>{mode} / {cls}</b>   sample={sid}   obj={obj_name}   "
               f"grasps=<b>{len(grasps)}</b> ({group_str})   "
               f"fit={float(np.asarray(g_obj['fitness'])):.2f}   "
               f"rmse={float(np.asarray(g_obj['inlier_rmse'])) * 1000:.1f}mm"),
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X cam (m)", backgroundcolor="#fafafa",
                       gridcolor="rgba(0,0,0,0.08)"),
            yaxis=dict(title="Y cam (m)", backgroundcolor="#fafafa",
                       gridcolor="rgba(0,0,0,0.08)"),
            zaxis=dict(title="Z cam (m, depth→)", backgroundcolor="#fafafa",
                       gridcolor="rgba(0,0,0,0.08)"),
            # isometric 카메라 — 축이 모두 명확히 보임
            camera=dict(up=dict(x=0, y=-1, z=0),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.2, y=-1.2, z=-1.2)),
            dragmode="turntable",
        ),
        template="plotly_white",
        width=1300, height=900,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.99,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="gray",
                    borderwidth=1, itemsizing="constant"),
        hovermode="closest",
    )
    out = OUT_DIR / f"{mode}__{cls}.html"
    fig.write_html(str(out))
    print(f"wrote {out}  (grasps={len(grasps)})")


def main():
    with h5py.File(POSES_H5, "r") as poses, h5py.File(DET_H5, "r") as dets:
        found: dict[tuple, tuple] = {}
        for sid in poses.keys():
            g = poses[sid]
            for obj_name in g.keys():
                g_obj = g[obj_name]
                if not g_obj.attrs.get("stable_flag", False):
                    continue
                cls = g_obj.attrs["class_name"]
                R = np.asarray(g_obj["R_cam"])
                long_idx = int(np.asarray(g_obj["model_long_axis_idx"]))
                mode = decide_mode(R, long_idx, cls)
                key = (mode, cls)
                if key not in found:
                    # prefer central-image detection (higher fit + lower rmse)
                    found[key] = (sid, obj_name,
                                  float(np.asarray(g_obj["fitness"])))
                else:
                    old_fit = found[key][2]
                    new_fit = float(np.asarray(g_obj["fitness"]))
                    if new_fit > old_fit:
                        found[key] = (sid, obj_name, new_fit)

        print("Representative samples:")
        for k, v in found.items():
            print(f"  {k}: {v[:2]}  fit={v[2]:.2f}")

        for key in TARGET_CASES:
            if key not in found:
                print(f"[miss] {key}")
                continue
            sid, obj_name, _ = found[key]
            render_case(poses, dets, sid, obj_name, key)


if __name__ == "__main__":
    main()
