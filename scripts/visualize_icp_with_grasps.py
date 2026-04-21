"""
ICP pose + planned grasp GT 복합 시각화.

각 샘플별로:
  - 노란색: YOLO mask contour
  - 초록색 점: ICP-aligned PLY reprojection
  - 시안 +: ICP pose의 t_cam 투영 (물체 중심)
  - 모드별 gripper overlay:
      standing (빨강) : 물체 top, 8 yaw
      lying    (주황) : 긴축 N=4 (bottle/can) or N=3 (marker/spam) × 180° sym
      cube     (보라) : 중심, 2 yaw (0°, 90°)
  - 각 grasp = 두 손가락 pad (작은 원) + 연결선 (gripper open 방향)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
POSES_H5 = ROOT / "img_dataset/icp_cache/poses.h5"
DET_H5 = ROOT / "img_dataset/yolo_cache_v3/detections.h5"
PLY_DIR = Path("/home/robotics/Competition/RoboCup_ARM/models/ply")
OUT_DIR = ROOT / "scripts/_icp_grasp_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_FX = K_FY = 1109.0
K_CX, K_CY = 640.0, 360.0
IMG_W, IMG_H = 1280, 720

GRIPPER_HALF_W = 0.042      # m (최대 개폐폭 0.085m / 2)
GRIPPER_FINGER_LEN = 0.025  # finger 시각 길이 (m, 그리퍼 0.14m 중 일부)

MODE_COLOR = {
    "standing": (0, 0, 255),     # 빨강 (BGR)
    "lying": (0, 140, 255),      # 주황
    "cube": (255, 0, 255),       # 마젠타
}

VIZ_SAMPLES = [
    "sample_random1_1", "sample_random1_50",
    "sample_random2_5", "sample_random2_20",
    "sample_random3_10", "sample_random3_30",
    "sample_random4_1", "sample_random4_20",
    "sample_random5_10", "sample_random6_15",
]

CLASS_TO_PLY = {
    "bottle": "blueBottle.ply", "can": "greenCan.ply",
    "cube_blue": "cube.ply", "cube_green": "cube.ply",
    "cube_p": "cube.ply", "cube_red": "cube.ply",
    "marker": "marker.ply", "spam": "Simsort_SPAM.ply",
}


def load_ply_centered(p):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(p))
    pts = np.asarray(pcd.points)
    if pts.ptp(axis=0).max() > 1.0:
        pcd.scale(0.001, center=(0, 0, 0))
        pts = np.asarray(pcd.points)
    pcd.translate(-pts.mean(axis=0))
    return np.asarray(pcd.points)


def project(pts_cam):
    z = np.clip(pts_cam[:, 2], 1e-6, None)
    u = K_FX * pts_cam[:, 0] / z + K_CX
    v = K_FY * pts_cam[:, 1] / z + K_CY
    return np.stack([u, v], axis=1)


def decide_mode(R, extent, long_axis_idx, cls_name):
    """lying / standing / cube."""
    if cls_name.startswith("cube"):
        return "cube"
    e_long = np.zeros(3)
    e_long[long_axis_idx] = 1.0
    long_cam = R @ e_long
    cos_z = abs(long_cam[2])
    return "standing" if cos_z > 0.7 else "lying"


def gen_grasps(R, t, extent, long_axis_idx, mode, cls_name):
    """
    return: list of (pos_cam(3,), yaw(float))
    yaw = gripper open 방향의 camera-XY 방위각 (rad)
    """
    grasps = []
    L = extent[long_axis_idx]

    if mode == "standing":
        # 물체 중심 XY, z는 top 부근
        # height = extent[long_axis_idx] (긴축이 Z에 평행)
        # cam Z가 깊이 증가 방향이니, top = 카메라에 가까운 = z 작은 쪽
        height = L
        pos = np.array([t[0], t[1], t[2] - height * 0.5 + 0.003])
        for i in range(8):
            yaw = i * np.pi / 4
            grasps.append((pos, yaw))

    elif mode == "lying":
        long_cam = R @ np.eye(3)[long_axis_idx]
        long_xy = long_cam[:2]
        n = np.linalg.norm(long_xy)
        if n < 1e-6:
            return []
        alpha = np.arctan2(long_xy[1], long_xy[0])
        yaw_base = alpha + np.pi / 2  # 긴축에 수직으로 grip open

        # short-axis radius (긴축 제외한 두 extent의 평균/2)
        short = np.array([extent[i] for i in range(3) if i != long_axis_idx])
        short_r = float(short.mean()) / 2

        N = 3 if cls_name in ("marker", "spam") else 4
        # 긴축 따라 sample (끝 10% 제외)
        ss = np.linspace(-0.4 * L, 0.4 * L, N)
        for s in ss:
            off_obj = np.zeros(3)
            off_obj[long_axis_idx] = s
            pos = t + R @ off_obj
            pos = np.array([pos[0], pos[1], pos[2] - short_r])
            for sym in (0, np.pi):  # 180° 대칭
                grasps.append((pos, yaw_base + sym))

    elif mode == "cube":
        pos = np.array([t[0], t[1], t[2] - extent[2] * 0.5 + 0.002])
        for y in (0.0, np.pi / 2):
            grasps.append((pos, y))
    return grasps


def draw_grasp(img, pos_cam, yaw, color, thickness=2):
    """두 finger pad + 연결선."""
    half = GRIPPER_HALF_W
    dx = half * np.cos(yaw)
    dy = half * np.sin(yaw)
    p1 = np.array([pos_cam[0] + dx, pos_cam[1] + dy, pos_cam[2]])
    p2 = np.array([pos_cam[0] - dx, pos_cam[1] - dy, pos_cam[2]])
    uv1 = project(p1[None])[0]
    uv2 = project(p2[None])[0]
    uv_c = project(pos_cam[None])[0]
    if any(np.isnan(uv1)) or any(np.isnan(uv2)):
        return
    pt1 = (int(uv1[0]), int(uv1[1]))
    pt2 = (int(uv2[0]), int(uv2[1]))
    ptc = (int(uv_c[0]), int(uv_c[1]))
    cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
    cv2.circle(img, pt1, 4, color, -1, cv2.LINE_AA)
    cv2.circle(img, pt2, 4, color, -1, cv2.LINE_AA)
    cv2.circle(img, ptc, 3, (255, 255, 255), -1, cv2.LINE_AA)


def render_sample(poses, dets, sid, ply_cache):
    if sid not in poses or sid not in dets:
        return None
    g_pose = poses[sid]
    g_det = dets[sid]
    rgb = cv2.imread(str(ROOT / g_pose.attrs["rgb_path"]))
    if rgb is None:
        return None

    # polygon 원본 가져오기 (mask contour용)
    poly_ds = g_det["mask_poly"]
    bboxes = np.asarray(g_det["bboxes"])
    class_names = list(dets.attrs["class_names"])

    overlay = rgb.copy()

    for obj_name in g_pose.keys():
        g_obj = g_pose[obj_name]
        if not g_obj.attrs.get("stable_flag", False):
            continue
        k = int(obj_name.split("_")[1])
        cls_name = g_obj.attrs["class_name"]

        # mask contour
        poly = np.asarray(poly_ds[k]).reshape(-1, 2)
        if poly.size >= 6:
            pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)

        # ICP-aligned PLY reprojection
        R = np.asarray(g_obj["R_cam"])
        t = np.asarray(g_obj["t_cam"])
        extent = np.asarray(g_obj["model_extent"])
        long_axis_idx = int(np.asarray(g_obj["model_long_axis_idx"]))
        ply_file = g_obj.attrs["ply_file"]

        if ply_file not in ply_cache:
            ply_cache[ply_file] = load_ply_centered(PLY_DIR / ply_file)
        model_pts = ply_cache[ply_file]

        # subsample
        if len(model_pts) > 1500:
            idx = np.random.choice(len(model_pts), 1500, replace=False)
            model_pts_s = model_pts[idx]
        else:
            model_pts_s = model_pts
        aligned = (R @ model_pts_s.T).T + t
        uvs = project(aligned)
        for u, v in uvs:
            if 0 <= u < IMG_W and 0 <= v < IMG_H:
                cv2.circle(overlay, (int(u), int(v)), 1, (0, 255, 0), -1)

        # object center (t_cam) 투영
        uv_t = project(t[None])[0]
        if 0 <= uv_t[0] < IMG_W and 0 <= uv_t[1] < IMG_H:
            cv2.drawMarker(overlay, (int(uv_t[0]), int(uv_t[1])),
                           (255, 255, 0), cv2.MARKER_CROSS, 14, 2)

        # mode + grasps
        mode = decide_mode(R, extent, long_axis_idx, cls_name)
        color = MODE_COLOR[mode]
        grasps = gen_grasps(R, t, extent, long_axis_idx, mode, cls_name)
        for pos, yaw in grasps:
            draw_grasp(overlay, pos, yaw, color, thickness=2)

        # label
        fit = float(np.asarray(g_obj["fitness"]))
        rmse_mm = float(np.asarray(g_obj["inlier_rmse"])) * 1000
        bbox = bboxes[k]
        label = f"{cls_name}[{mode}] n={len(grasps)} fit={fit:.2f} rmse={rmse_mm:.1f}mm"
        text_org = (int(bbox[0]), max(int(bbox[1]) - 6, 20))
        cv2.putText(overlay, label, text_org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)

    # blend for slightly transparent PLY overlay
    final = cv2.addWeighted(overlay, 0.85, rgb, 0.15, 0)

    # legend bar
    legend = [
        ("mask", (0, 255, 255)),
        ("ICP PLY", (0, 255, 0)),
        ("obj center", (255, 255, 0)),
        ("grasp: standing", MODE_COLOR["standing"]),
        ("grasp: lying", MODE_COLOR["lying"]),
        ("grasp: cube", MODE_COLOR["cube"]),
    ]
    y = 30
    for txt, col in legend:
        cv2.putText(final, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, col, 2, cv2.LINE_AA)
        y += 28

    cv2.putText(final, sid, (IMG_W - 400, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return final


def main():
    ply_cache: dict[str, np.ndarray] = {}
    with h5py.File(POSES_H5, "r") as poses, h5py.File(DET_H5, "r") as dets:
        for sid in VIZ_SAMPLES:
            img = render_sample(poses, dets, sid, ply_cache)
            if img is None:
                print(f"skip {sid}")
                continue
            out = OUT_DIR / f"{sid}.jpg"
            cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
