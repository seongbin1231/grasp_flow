"""Fig 1 — 객체 모드별 GT 파지 자세 합성 (3-panel).

논문용 정적 PNG 생성. matplotlib 3D scatter 기반.
Standing bottle / Lying can / Cube 각 1샘플 × 전체 GT grasp 오버레이.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # cv2/Qt 충돌 회피 (반드시 cv2 import 이전)
import cv2
import h5py
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
GRASP_H5 = ROOT / "img_dataset/grasp_cache/grasps.h5"
POSES_H5 = ROOT / "img_dataset/icp_cache/poses.h5"
DET_H5 = ROOT / "img_dataset/yolo_cache_v3/detections.h5"
PLY_DIR = Path("/home/robotics/Competition/RoboCup_ARM/models/ply")
OUT = ROOT / "paper_figs"
OUT.mkdir(parents=True, exist_ok=True)

K_FX = K_FY = 1109.0
K_CX, K_CY = 640.0, 360.0
IMG_H, IMG_W = 720, 1280

GRIPPER_HALF = 0.0425 * 0.80
FINGER_LEN = 0.040 * 0.80
PALM_BACK = 0.025 * 0.80

# 선택 case (각 모드 대표) — 실제 존재하는 obj_idx 로 수정
# 사용자 요청: 서있는 캔, 누워있는 캔, 큐브
# (label, sid, obj_idx, cls, mode, elev, azim)
CASES = [
    ("Standing can (N=24)",
     "sample_random6_32", 0, "can", "standing", -15, 205),
    ("Lying can (N=24)",
     "sample_random6_11", 0, "can", "lying", -15, 205),
    ("Cube (N=2)",
     "sample_random6_30", 7, "cube_red", "cube", -15, 205),
]


def poly_to_mask(poly):
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def depth_mask_to_pc_rgb(depth_m, mask, rgb_bgr):
    ys, xs = np.where(mask & (depth_m > 0.1) & (depth_m < 2.0))
    if len(xs) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    z = depth_m[ys, xs]
    x = (xs - K_CX) * z / K_FX
    y = (ys - K_CY) * z / K_FY
    pc = np.stack([x, y, z], axis=1).astype(np.float32)
    bgr = rgb_bgr[ys, xs]
    rgb = bgr[:, ::-1].astype(np.float32) / 255.0
    return pc, rgb


def rgb_path_for_depth(depth_path: Path) -> Path:
    name = depth_path.name.replace("_depth_", "_")
    folder = depth_path.parent.name.replace("_dep", "")
    base = depth_path.parent.parent.parent / "captured_images"
    return base / folder / name


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-w*z),     2*(x*z+w*y)],
        [2*(x*y+w*z),     1 - 2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),     2*(y*z+w*x),     1 - 2*(x*x+y*y)],
    ], dtype=np.float32)


def gripper_segs(pos, q):
    R = quat_wxyz_to_R(q)
    a = R[:, 2]; b = R[:, 1]
    tip1 = pos + b * GRIPPER_HALF
    tip2 = pos - b * GRIPPER_HALF
    base1 = tip1 - a * FINGER_LEN
    base2 = tip2 - a * FINGER_LEN
    palm = pos - a * FINGER_LEN
    wrist = palm - a * PALM_BACK
    return [(wrist, palm), (base1, base2), (base1, tip1), (base2, tip2)]


_ply_cache = {}
def load_ply(name):
    if name in _ply_cache:
        return _ply_cache[name]
    pcd = o3d.io.read_point_cloud(str(PLY_DIR / name))
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.ptp(axis=0).max() > 1.0:
        pts *= 0.001
    pts -= pts.mean(axis=0)
    if len(pts) > 1500:
        pts = pts[np.random.default_rng(0).choice(len(pts), 1500, replace=False)]
    _ply_cache[name] = pts
    return pts


def render_panel(ax, sample_id, obj_idx, title, det_h5, grasps_h5, poses_h5,
                 elev=15, azim=-65):
    g_s = grasps_h5[sample_id]
    p_s = poses_h5[sample_id]
    g_o = g_s[f"object_{obj_idx}"]
    p_o = p_s[f"object_{obj_idx}"]

    depth_path = ROOT / g_s.attrs["depth_path"]
    depth_m = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    rgb_bgr = cv2.imread(str(rgb_path_for_depth(depth_path)), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(f"RGB not found for {depth_path}")
    poly = np.asarray(det_h5[sample_id]["mask_poly"][obj_idx]).reshape(-1, 2)
    mask = poly_to_mask(poly)
    scene_pts, scene_rgb = depth_mask_to_pc_rgb(depth_m, mask, rgb_bgr)
    if len(scene_pts) > 1500:
        sel = np.random.default_rng(0).choice(len(scene_pts), 1500, replace=False)
        scene_pts = scene_pts[sel]; scene_rgb = scene_rgb[sel]

    ply_pts = load_ply(p_o.attrs["ply_file"])
    pose = np.asarray(p_o["pose_cam"])
    R = quat_wxyz_to_R(pose[3:7])
    obj_pts = (ply_pts @ R.T) + pose[:3]

    # scene PC: 실제 RGB 색 (그리퍼 가리지 않도록 투명/작게)
    if len(scene_pts) > 0:
        ax.scatter(scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2],
                   s=2.5, c=scene_rgb, alpha=0.45, edgecolors='none')
    if len(obj_pts) > 0:
        ax.scatter(obj_pts[:, 0], obj_pts[:, 1], obj_pts[:, 2],
                   s=0.8, c='#bdbdbd', alpha=0.18, edgecolors='none')

    # All GT grasps
    grasps = np.asarray(g_o["grasps_cam"])
    groups = np.asarray(g_o["grasp_group"])

    # group color: top-down/lying/cube 진한 파랑, side-cap 주황, side-45 빨강
    color_map = {0: "#1976d2", 1: "#f39c12", 2: "#1976d2",
                 3: "#1976d2", 4: "#e74c3c"}
    for k, grp in enumerate(groups):
        col = color_map.get(int(grp), "#666")
        for a, b in gripper_segs(grasps[k, :3], grasps[k, 3:7]):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                    color=col, linewidth=1.5, alpha=1.0)

    if len(scene_pts) > 5:
        c = scene_pts.mean(axis=0)
        r = max(float(scene_pts.ptp(axis=0).max()), 0.10) * 0.65
    else:
        c = pose[:3]
        r = max(float(obj_pts.ptp(axis=0).max()) if len(obj_pts) else 0.10, 0.10) * 0.65
    ax.set_xlim(c[0]-r, c[0]+r)
    ax.set_ylim(c[1]-r, c[1]+r)
    ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim, vertical_axis='y')
    ax.invert_yaxis()   # cam +Y(아래) → 화면 아래로
    ax.set_title(title, fontsize=22, pad=2, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    # remove panes
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((0.85, 0.85, 0.85, 0.5))


def main():
    fig = plt.figure(figsize=(13, 4.5), dpi=150)
    with h5py.File(GRASP_H5, "r") as g, h5py.File(POSES_H5, "r") as p, \
         h5py.File(DET_H5, "r") as d:
        for i, (label, sid, idx, cls, mode, elev, azim) in enumerate(CASES, 1):
            ax = fig.add_subplot(1, 3, i, projection="3d")
            render_panel(ax, sid, idx, label, d, g, p, elev=elev, azim=azim)

    plt.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.0,
                        wspace=0.0)
    out_png = OUT / "fig1_gt_synthesis.png"
    out_pdf = OUT / "fig1_gt_synthesis.pdf"
    plt.savefig(out_png, dpi=220, bbox_inches='tight', pad_inches=0.0)
    plt.savefig(out_pdf, bbox_inches='tight', pad_inches=0.0)
    print(f"[fig1] {out_png}")
    print(f"[fig1] {out_pdf}")


if __name__ == "__main__":
    main()
