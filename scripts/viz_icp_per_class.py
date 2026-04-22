"""ICP 결과 per-class 3-sample 시각화.

각 클래스에서 fitness 분포가 골고루 보이도록 3개 선택:
  - 고품질 (p90)
  - 중간 (p50)
  - gate 경계 (0.30~0.35)
scene mask 포인트(파랑) + 정합된 PLY 모델(빨강) 3D 오버레이.

출력: deploy/viz/icp/icp_{class}.html + index.html (tab 네비)
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
POSES_H5 = ROOT / "img_dataset/icp_cache/poses.h5"
DET_H5 = ROOT / "img_dataset/yolo_cache_v3/detections.h5"
PLY_DIR = Path("/home/robotics/Competition/RoboCup_ARM/models/ply")
OUT_DIR = ROOT / "deploy/viz/icp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_FX = K_FY = 1109.0
K_CX, K_CY = 640.0, 360.0
IMG_H, IMG_W = 720, 1280

CLASS_TO_PLY = {
    "bottle": "blueBottle.ply",
    "can": "greenCan.ply",
    "cube_blue": "cube.ply",
    "cube_green": "cube.ply",
    "cube_p": "cube.ply",
    "cube_red": "cube.ply",
    "marker": "marker.ply",
    "spam": "Simsort_SPAM.ply",
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
    # downsample if too many
    if len(pts) > 3000:
        idx = np.random.default_rng(0).choice(len(pts), 3000, replace=False)
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


# ---------- data collection ----------

def collect_per_class():
    """각 클래스별 stable 한 ICP 결과 리스트 수집."""
    per_cls: dict[str, list[dict]] = {}
    with h5py.File(POSES_H5, "r") as f:
        for sid in f.keys():
            g_s = f[sid]
            for obj_key in g_s.keys():
                g_o = g_s[obj_key]
                if "skipped_reason" in g_o.attrs:
                    continue
                if not g_o.attrs.get("stable_flag", False):
                    continue
                cls = g_o.attrs["class_name"]
                per_cls.setdefault(cls, []).append({
                    "sid": sid,
                    "obj_idx": int(obj_key.split("_")[1]),
                    "fitness": float(g_o["fitness"][()]),
                    "rmse": float(g_o["inlier_rmse"][()]),
                    "pose7": np.asarray(g_o["pose_cam"]),
                    "depth_path": g_s.attrs["depth_path"],
                    "class_name": cls,
                    "ply_file": g_o.attrs["ply_file"],
                    "best_init": g_o.attrs.get("best_init", "?"),
                    "mask_px": int(g_o["mask_px"][()]) if "mask_px" in g_o else -1,
                })
    return per_cls


def pick_3_samples(items):
    """fitness 분포에서 고/중/저 3개 샘플 선택."""
    if len(items) <= 3:
        return items
    items_sorted = sorted(items, key=lambda d: d["fitness"])
    n = len(items_sorted)
    # near-gate (낮은 쪽): gate 0.30 바로 위
    low = min(items_sorted, key=lambda d: abs(d["fitness"] - 0.32))
    # p50 중간
    mid = items_sorted[n // 2]
    # p90 고품질
    high = items_sorted[int(n * 0.9)]
    # dedup by sid+obj
    seen = set()
    picks = []
    for s in [high, mid, low]:
        key = (s["sid"], s["obj_idx"])
        if key not in seen:
            seen.add(key)
            picks.append(s)
    return picks


# ---------- 3D rendering ----------

def make_sample_scene(sample, det_h5):
    """한 샘플의 3D scene: scene pts (파랑) + PLY 변환 (빨강)."""
    # depth & mask
    depth = cv2.imread(str(ROOT / sample["depth_path"]), cv2.IMREAD_UNCHANGED)
    depth_m = depth.astype(np.float32) / 1000.0
    g_det = det_h5[sample["sid"]]
    poly = np.asarray(g_det["mask_poly"][sample["obj_idx"]]).reshape(-1, 2)
    mask = poly_to_mask(poly)
    scene_pts = depth_mask_to_pc(depth_m, mask)
    # subsample scene
    if len(scene_pts) > 2000:
        idx = np.random.default_rng(0).choice(len(scene_pts), 2000, replace=False)
        scene_pts = scene_pts[idx]

    # PLY transformed
    model_pts = load_ply_points(sample["ply_file"])
    pose = sample["pose7"]
    t = pose[:3]
    R = quat_wxyz_to_R(pose[3:7])
    model_tf = (model_pts @ R.T) + t

    trace_scene = go.Scatter3d(
        x=scene_pts[:, 0], y=scene_pts[:, 1], z=scene_pts[:, 2],
        mode="markers",
        marker=dict(size=1.8, color="#4a90e2", opacity=0.7),
        name="scene PC",
        showlegend=False,
    )
    trace_model = go.Scatter3d(
        x=model_tf[:, 0], y=model_tf[:, 1], z=model_tf[:, 2],
        mode="markers",
        marker=dict(size=1.5, color="#d0021b", opacity=0.55),
        name="ICP model",
        showlegend=False,
    )
    return [trace_scene, trace_model], scene_pts, model_tf


def render_class(cls_name, samples, det_h5):
    """클래스별 3-subplot HTML 생성."""
    n = len(samples)
    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type": "scene"}] * n],
        subplot_titles=[
            f"<b>{s['sid']} / obj_{s['obj_idx']}</b><br>"
            f"fit={s['fitness']:.3f}  rmse={s['rmse']*1000:.1f}mm  "
            f"init={s['best_init']}<br>mask_px={s['mask_px']}"
            for s in samples
        ],
        horizontal_spacing=0.02,
    )
    for i, s in enumerate(samples, 1):
        traces, scene_pts, _ = make_sample_scene(s, det_h5)
        for tr in traces:
            fig.add_trace(tr, row=1, col=i)
        # 축 등비
        if len(scene_pts):
            c = scene_pts.mean(axis=0)
            r = np.max(scene_pts.ptp(axis=0)) * 0.75 + 0.05
        else:
            c = np.zeros(3); r = 0.15
        fig.update_scenes(
            row=1, col=i,
            xaxis=dict(range=[c[0] - r, c[0] + r], title=""),
            yaxis=dict(range=[c[1] - r, c[1] + r], title=""),
            zaxis=dict(range=[c[2] - r, c[2] + r], title=""),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.4, y=-1.4, z=-1.4), up=dict(x=0, y=-1, z=0)),
        )

    fig.update_layout(
        title=f"<b>ICP 정합 결과 — class: {cls_name}</b>  "
              f"(파랑=scene depth PC, 빨강=PLY transformed by ICP pose)",
        height=650,
        margin=dict(l=10, r=10, t=90, b=10),
        font=dict(family="Helvetica, Arial, sans-serif", size=11),
    )
    return fig


def make_index(classes):
    """tab 네비 index.html."""
    tabs = "\n".join(
        f'<a href="icp_{c}.html" class="tab">{c}</a>' for c in classes)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>ICP 정합 시각화 — 클래스별 3샘플</title>
<style>
body {{ font-family: Helvetica,Arial,sans-serif; margin:0; padding:20px; background:#f5f5f5; }}
h1 {{ margin:0 0 12px 0; }}
.sub {{ color:#666; font-size:13px; margin-bottom:20px; }}
.tabs {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:20px; }}
.tab {{ padding:10px 18px; background:white; border:1px solid #ccc; border-radius:6px;
       text-decoration:none; color:#333; font-weight:500; }}
.tab:hover {{ background:#4a90e2; color:white; border-color:#4a90e2; }}
.info {{ background:white; padding:16px 20px; border-radius:8px;
        border-left:4px solid #4a90e2; }}
table {{ border-collapse:collapse; margin-top:10px; }}
td,th {{ padding:6px 12px; border:1px solid #ddd; text-align:left; }}
th {{ background:#f0f0f0; }}
</style></head><body>
<h1>ICP 정합 결과 시각화</h1>
<div class="sub">필터 업그레이드: mask_px p10 + fitness≥0.30 + rmse≤3mm 적용 후</div>
<div class="tabs">{tabs}</div>
<div class="info">
  <strong>시각화 규약:</strong>
  <ul>
    <li>각 클래스별 3개 샘플 (fitness p90 / p50 / gate 경계 ≈0.32)</li>
    <li>파랑(●): YOLO mask 기반 scene depth 포인트클라우드</li>
    <li>빨강(●): PLY 모델을 ICP pose 로 변환한 결과</li>
    <li>정합이 잘 되면 두 색이 거의 겹쳐 보임</li>
  </ul>
</div>
</body></html>"""
    (OUT_DIR / "index.html").write_text(html)


# ---------- main ----------

def main():
    print(f"[info] loading from {POSES_H5.name}")
    per_cls = collect_per_class()
    print(f"[info] classes: {list(per_cls.keys())}")
    for c, arr in per_cls.items():
        fits = np.array([d["fitness"] for d in arr])
        print(f"  {c:12s} n={len(arr):4d}  "
              f"fit p10={np.percentile(fits, 10):.3f}  "
              f"p50={np.percentile(fits, 50):.3f}  "
              f"p90={np.percentile(fits, 90):.3f}")

    with h5py.File(DET_H5, "r") as det_h5:
        classes_rendered = []
        for cls_name, items in sorted(per_cls.items()):
            samples = pick_3_samples(items)
            if not samples:
                continue
            fig = render_class(cls_name, samples, det_h5)
            out_path = OUT_DIR / f"icp_{cls_name}.html"
            fig.write_html(str(out_path), include_plotlyjs="cdn")
            print(f"[viz] {out_path.name}  (picks: "
                  f"{', '.join(f'fit={s["fitness"]:.2f}' for s in samples)})")
            classes_rendered.append(cls_name)

    make_index(classes_rendered)
    print(f"\n[done] {len(classes_rendered)} classes → {OUT_DIR}/index.html")


if __name__ == "__main__":
    main()
