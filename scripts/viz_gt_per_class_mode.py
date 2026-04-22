"""GT grasp viz per (class, mode). 각 case 별 3 샘플씩.

Scene PC (파랑) + 물체 PLY 정합 (빨강) + 모든 GT grasp 그리퍼 (초록, collision_ok).

출력: deploy/viz/gt_policy/{case}.html + index.html (탭 네비)
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict

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
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_FX = K_FY = 1109.0
K_CX, K_CY = 640.0, 360.0
IMG_H, IMG_W = 720, 1280

# 그리퍼 dimensions
GRIPPER_HALF = 0.0425
FINGER_LEN = 0.040
PALM_BACK = 0.025

GROUP_COLOR = {
    0: "#2ecc71",  # top-down 초록
    1: "#f39c12",  # side-cap 주황
    2: "#2ecc71",  # lying 초록
    3: "#2ecc71",  # cube 초록
}


# ---------- helpers ----------

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


_ply_cache: dict[str, np.ndarray] = {}


def load_ply_points(name):
    if name in _ply_cache:
        return _ply_cache[name]
    pcd = o3d.io.read_point_cloud(str(PLY_DIR / name))
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.ptp(axis=0).max() > 1.0:
        pts *= 0.001
    pts = pts - pts.mean(axis=0)
    if len(pts) > 2500:
        idx = np.random.default_rng(0).choice(len(pts), 2500, replace=False)
        pts = pts[idx]
    _ply_cache[name] = pts
    return pts


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


def gripper_lines(pos, q_wxyz):
    """그리퍼 wireframe: [wrist, palm, tip1/2, back1/2]. 6 세그먼트."""
    R = quat_wxyz_to_R(q_wxyz)
    a = R[:, 2]   # Tool Z = approach
    b = R[:, 1]   # Tool Y = open direction
    tip1 = pos + b * GRIPPER_HALF
    tip2 = pos - b * GRIPPER_HALF
    base1 = tip1 - a * FINGER_LEN
    base2 = tip2 - a * FINGER_LEN
    palm = pos - a * FINGER_LEN
    wrist = palm - a * PALM_BACK
    # 선분 리스트
    segs = [
        (wrist, palm),      # stem
        (base1, base2),     # bridge
        (base1, tip1),      # finger 1
        (base2, tip2),      # finger 2
    ]
    return segs


def make_gripper_trace(grasps_cam, groups, color_map, name="GT grasps"):
    """여러 grasp을 하나의 Scatter3d(mode=lines) 로 묶기."""
    xs, ys, zs = [], [], []
    colors_per_seg = []  # hack — we'll use uniform color then split per group
    # For simplicity: single color (most common group)
    for (pose, grp) in zip(grasps_cam, groups):
        pos = pose[:3]
        q = pose[3:7]
        segs = gripper_lines(pos, q)
        for a, b in segs:
            xs += [a[0], b[0], None]
            ys += [a[1], b[1], None]
            zs += [a[2], b[2], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color_map, width=3),
        name=name, showlegend=False,
    )


# ---------- collection ----------

def collect_cases():
    """Return: dict[(class, mode)] -> list of sample dicts."""
    per_case = defaultdict(list)
    with h5py.File(GRASP_H5, "r") as f, h5py.File(POSES_H5, "r") as p:
        for sid in f.keys():
            g_s = f[sid]
            p_s = p[sid]
            for oname in g_s.keys():
                g_o = g_s[oname]
                p_o = p_s[oname]
                if "excluded_reason" in g_o.attrs:
                    continue
                if int(g_o.attrs.get("n_grasps", 0)) == 0:
                    continue
                cls = g_o.attrs["class_name"]
                mode = g_o.attrs["mode"]
                per_case[(cls, mode)].append({
                    "sid": sid,
                    "obj_idx": int(oname.split("_")[1]),
                    "depth_path": g_s.attrs["depth_path"],
                    "fitness": float(p_o["fitness"][()]),
                    "ply_file": p_o.attrs["ply_file"],
                    "pose7_obj": np.asarray(p_o["pose_cam"]),
                    "grasps_cam": np.asarray(g_o["grasps_cam"]),
                    "groups": np.asarray(g_o["grasp_group"]),
                    "ok": np.asarray(g_o["collision_ok"]),
                    "n_grasps": int(g_o.attrs["n_grasps"]),
                })
    return per_case


def pick_samples(items, k=3):
    """fitness 기준 고/중/저 k개."""
    if len(items) <= k:
        return items
    s = sorted(items, key=lambda d: d["fitness"])
    n = len(s)
    picks = [s[int(n * 0.9)], s[n // 2], s[int(n * 0.1)]]
    # dedup
    seen = set(); out = []
    for p in picks:
        key = (p["sid"], p["obj_idx"])
        if key not in seen:
            seen.add(key); out.append(p)
    return out


# ---------- render ----------

def make_scene_trace(sample, det_h5):
    depth = cv2.imread(str(ROOT / sample["depth_path"]), cv2.IMREAD_UNCHANGED)
    depth_m = depth.astype(np.float32) / 1000.0
    g_det = det_h5[sample["sid"]]
    poly = np.asarray(g_det["mask_poly"][sample["obj_idx"]]).reshape(-1, 2)
    mask = poly_to_mask(poly)
    scene_pts = depth_mask_to_pc(depth_m, mask)
    if len(scene_pts) > 1500:
        idx = np.random.default_rng(0).choice(len(scene_pts), 1500, replace=False)
        scene_pts = scene_pts[idx]
    return go.Scatter3d(
        x=scene_pts[:, 0], y=scene_pts[:, 1], z=scene_pts[:, 2],
        mode="markers",
        marker=dict(size=1.5, color="#4a90e2", opacity=0.5),
        name="scene PC", showlegend=False,
    ), scene_pts


def make_model_trace(sample):
    pts_model = load_ply_points(sample["ply_file"])
    pose = sample["pose7_obj"]
    R = quat_wxyz_to_R(pose[3:7])
    t = pose[:3]
    pts = (pts_model @ R.T) + t
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(size=1.2, color="#d0021b", opacity=0.35),
        name="PLY", showlegend=False,
    )


def render_case(cls, mode, samples, det_h5):
    n = len(samples)
    titles = []
    for s in samples:
        ok_count = int(s["ok"].sum())
        titles.append(
            f"<b>{s['sid']} / obj_{s['obj_idx']}</b><br>"
            f"fit={s['fitness']:.2f}  grasps={s['n_grasps']} "
            f"(ok={ok_count})"
        )

    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type": "scene"}] * n],
        subplot_titles=titles,
        horizontal_spacing=0.02,
    )
    for i, s in enumerate(samples, 1):
        scene_tr, scene_pts = make_scene_trace(s, det_h5)
        model_tr = make_model_trace(s)
        fig.add_trace(scene_tr, row=1, col=i)
        fig.add_trace(model_tr, row=1, col=i)
        # Separate trace per group color
        groups = s["groups"]
        grasps = s["grasps_cam"]
        ok = s["ok"]
        for gid, color in [(0, "#2ecc71"),  # top-down = green
                            (1, "#f39c12"),  # side-cap (90°) = orange
                            (2, "#2ecc71"),  # lying = green
                            (3, "#2ecc71"),  # cube = green
                            (4, "#e74c3c")]: # side-45 = red
            mask = (groups == gid) & ok
            if mask.sum() == 0:
                continue
            g_trace = make_gripper_trace(grasps[mask], groups[mask], color)
            fig.add_trace(g_trace, row=1, col=i)

        if len(scene_pts) > 0:
            c = scene_pts.mean(axis=0)
            r = np.max(scene_pts.ptp(axis=0)) * 0.8 + 0.08
        else:
            c = s["pose7_obj"][:3]; r = 0.15
        fig.update_scenes(
            row=1, col=i,
            xaxis=dict(range=[c[0] - r, c[0] + r], title=""),
            yaxis=dict(range=[c[1] - r, c[1] + r], title=""),
            zaxis=dict(range=[c[2] - r, c[2] + r], title=""),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.4, y=-1.4, z=-1.4), up=dict(x=0, y=-1, z=0)),
        )

    title_case = f"{cls} ({mode})"
    fig.update_layout(
        title=f"<b>GT grasp 정책 — {title_case}</b>  "
              f"(초록=top-down/lying/cube, 빨강=side-45°, 주황=side-cap 90°, PLY=dark-red)",
        height=650,
        margin=dict(l=10, r=10, t=90, b=10),
        font=dict(family="Helvetica,Arial,sans-serif", size=11),
    )
    return fig


def make_index(cases, counts):
    total_objs = sum(counts[c]["objs"] for c in cases)
    total_grasps = sum(counts[c]["grasps"] for c in cases)

    # side-45 단독 viz 가 있으면 링크 추가 (bottle_standing / can_standing)
    def side45_link(cls, mode):
        if mode != "standing" or cls not in ("bottle", "can"):
            return ""
        p = OUT_DIR / f"{cls}_{mode}_side45only.html"
        if not p.exists():
            return ""
        return (f" &nbsp;<a href='{p.name}' "
                f"style='color:#e74c3c;font-size:12px'>[side-45 단독]</a>")

    rows = "\n".join(
        f"<tr><td><a href='{cls}_{mode}.html'>{cls} ({mode})</a>"
        f"{side45_link(cls, mode)}</td>"
        f"<td align='right'>{counts[(cls, mode)]['objs']}</td>"
        f"<td align='right'>{counts[(cls, mode)]['grasps_per_obj']}</td>"
        f"<td align='right'>{counts[(cls, mode)]['grasps']}</td></tr>"
        for (cls, mode) in cases
    )
    # ICP 정합 viz 링크 (있을 경우)
    icp_link = ""
    icp_index = OUT_DIR.parent / "icp" / "index.html"
    if icp_index.exists():
        icp_link = (f"<div class='sub'>🔗 관련: "
                    f"<a href='../icp/index.html'>ICP 정합 품질 viz</a></div>")

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>GT grasp 정책 — 클래스×모드</title>
<style>
body {{ font-family: Helvetica,Arial,sans-serif; margin:0; padding:20px; background:#f5f5f5; }}
h1 {{ margin:0 0 12px 0; }}
.sub {{ color:#666; font-size:13px; margin-bottom:10px; }}
table {{ border-collapse:collapse; background:white; border-radius:8px; overflow:hidden;
        box-shadow:0 2px 6px rgba(0,0,0,0.1); margin-bottom:20px; width:720px; }}
td,th {{ padding:10px 16px; border-bottom:1px solid #eee; }}
th {{ background:#4a90e2; color:white; text-align:left; }}
tr:hover {{ background:#f9f9f9; }}
a {{ color:#2c7cd2; text-decoration:none; font-weight:500; }}
a:hover {{ text-decoration:underline; }}
.info {{ background:white; padding:16px 20px; border-radius:8px;
        border-left:4px solid #4a90e2; width:720px; }}
.total {{ background:#e8f5e9; font-weight:bold; }}
</style></head><body>
<h1>GT grasp 정책 시각화</h1>
<div class="sub">policy_version=<b>6dof-v4</b>, 새 ICP 2485 stable → 정책 제외 후 {total_objs} object, 총 <b>{total_grasps:,}</b> grasp</div>
{icp_link}
<table>
  <tr><th>case</th><th>objects</th><th>/obj</th><th>total</th></tr>
  {rows}
  <tr class='total'><td>TOTAL</td><td align='right'>{total_objs}</td><td></td><td align='right'>{total_grasps:,}</td></tr>
</table>
<div class="info">
  <strong>색상 규약:</strong>
  <ul>
    <li><span style="color:#4a90e2">●</span> scene depth PC</li>
    <li><span style="color:#d0021b">●</span> PLY 모델 (ICP pose 로 변환됨)</li>
    <li><span style="color:#2ecc71">┤├</span> top-down / lying / cube grasp (초록)</li>
    <li><span style="color:#e74c3c">┤├</span> side-45° grasp (빨강, standing 중간 tilt)</li>
    <li><span style="color:#f39c12">┤├</span> side-cap 90° grasp (주황, standing 수평)</li>
  </ul>
  <strong>샘플 선정:</strong> 각 case fitness p90 / p50 / p10 (고/중/저 정합)<br>
  <strong>정책 제외 (v4):</strong> marker_standing × 7 / spam_standing × 4 / spam_lying 뚜껑 측면 × 149
</div>
</body></html>"""
    (OUT_DIR / "index.html").write_text(html)


# ---------- main ----------

def main():
    per_case = collect_cases()
    # counts
    counts = {}
    for k, items in per_case.items():
        n_objs = len(items)
        g_per = items[0]["n_grasps"]
        counts[k] = {"objs": n_objs, "grasps_per_obj": g_per,
                     "grasps": n_objs * g_per}

    print(f"{'class':12s} {'mode':10s} {'objs':>6s} {'g/obj':>6s} {'grasps':>8s}")
    for k in sorted(per_case.keys()):
        c = counts[k]
        print(f"{k[0]:12s} {k[1]:10s} {c['objs']:6d} "
              f"{c['grasps_per_obj']:6d} {c['grasps']:8d}")

    rendered = []
    with h5py.File(DET_H5, "r") as det_h5:
        for (cls, mode), items in sorted(per_case.items()):
            picks = pick_samples(items, 3)
            if not picks:
                continue
            fig = render_case(cls, mode, picks, det_h5)
            path = OUT_DIR / f"{cls}_{mode}.html"
            fig.write_html(str(path), include_plotlyjs="cdn")
            print(f"[viz] {path.name}  ({len(picks)} samples)")
            rendered.append((cls, mode))

    make_index(rendered, counts)
    print(f"\n[done] {len(rendered)} cases → {OUT_DIR}/index.html")


if __name__ == "__main__":
    main()
