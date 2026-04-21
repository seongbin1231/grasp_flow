"""
2D ICP overlay — RGB 위에 (ICP-aligned PLY reprojection) + YOLO mask contour.

각 샘플/객체별:
  - 왼쪽: RGB + mask contour
  - 오른쪽: RGB + mask contour + re-projected PLY points (초록 점)

v1/v2 두 버전 모두 렌더링해서 시각적으로 비교.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import open3d as o3d

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
DET_H5 = ROOT / "img_dataset/yolo_cache_v3/detections.h5"
PLY_DIR = Path("/home/robotics/Competition/RoboCup_ARM/models/ply")

K_FX = K_FY = 1109.0
K_CX, K_CY = 640.0, 360.0
IMG_W, IMG_H = 1280, 720

CLASS_TO_PLY = {
    "bottle": "blueBottle.ply", "can": "greenCan.ply",
    "cube_blue": "cube.ply", "cube_green": "cube.ply",
    "cube_p": "cube.ply", "cube_red": "cube.ply",
    "marker": "marker.ply", "spam": "Simsort_SPAM.ply",
}

BBOX_MARGIN_PX = 20

TEST_SAMPLES = [
    "sample_random1_1", "sample_random2_5", "sample_random3_10",
    "sample_random4_1", "sample_random4_20", "sample_random5_10",
    "sample_random6_15", "sample_random1_50",
]


def load_depth_meter(p: Path) -> np.ndarray:
    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return raw.astype(np.float32) / 1000.0


def poly_to_mask(poly: np.ndarray) -> np.ndarray:
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def bbox_inside(bbox: np.ndarray, margin: int) -> bool:
    x1, y1, x2, y2 = bbox
    return (x1 >= margin and y1 >= margin
            and x2 <= IMG_W - margin and y2 <= IMG_H - margin)


def depth_mask_to_pc(depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask & (depth_m > 0.1) & (depth_m < 2.0))
    z = depth_m[ys, xs]
    x = (xs - K_CX) * z / K_FX
    y = (ys - K_CY) * z / K_FY
    return np.stack([x, y, z], axis=1).astype(np.float32)


def load_ply_centered(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    if pts.ptp(axis=0).max() > 1.0:
        pcd.scale(0.001, center=(0, 0, 0))
        pts = np.asarray(pcd.points)
    pcd.translate(-pts.mean(axis=0))
    return pcd


# === v1 ICP (baseline) ===
def multiscale_icp_v1(source, target, init):
    voxels = [0.012, 0.006, 0.003]
    iters = [200, 200, 500]
    T = init.copy()
    res = None
    for v, it in zip(voxels, iters):
        s = source.voxel_down_sample(v)
        t = target.voxel_down_sample(v)
        res = o3d.pipelines.registration.registration_icp(
            s, t, v * 2.0, T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it))
        T = np.asarray(res.transformation).copy()
    return T, float(res.fitness), float(res.inlier_rmse)


def align_v1(scene_pts, model_pcd):
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_pts)
    init = np.eye(4)
    init[:3, 3] = scene_pts.mean(axis=0)
    return multiscale_icp_v1(model_pcd, scene_pcd, init)


# === v2 ICP with PCA init + outlier removal + fixed distance (simplified) ===
def pca_axes(pts):
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T)
    vals, vecs = np.linalg.eigh(cov)
    R = vecs[:, np.argsort(vals)[::-1]]
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return R


def svd_cleanup(R):
    U, _, Vt = np.linalg.svd(R)
    R2 = U @ Vt
    if np.linalg.det(R2) < 0:
        U[:, -1] *= -1
        R2 = U @ Vt
    return R2


def align_v2(scene_pts, model_pcd):
    """PCA 4-flip init + fixed distance multi-scale P2P + outlier removal."""
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_pts)
    scene_pcd, _ = scene_pcd.remove_statistical_outlier(4, 2.0)
    scene_clean = np.asarray(scene_pcd.points)

    model_pts = np.asarray(model_pcd.points)
    R_m = pca_axes(model_pts)
    R_s = pca_axes(scene_clean)
    R_base = svd_cleanup(R_s @ R_m.T)
    t_base = scene_clean.mean(axis=0) - R_base @ model_pts.mean(axis=0)

    flips = [np.diag([1, 1, 1]), np.diag([-1, -1, 1]),
             np.diag([-1, 1, -1]), np.diag([1, -1, -1])]

    best_T = None
    best_fit = -1.0
    best_rmse = 1.0
    for F in flips:
        R_init = svd_cleanup(R_base @ F)
        init = np.eye(4)
        init[:3, :3] = R_init
        init[:3, 3] = t_base
        T, fit, rmse = multiscale_icp_v1(model_pcd, scene_pcd, init)
        if fit > best_fit or (fit == best_fit and rmse < best_rmse):
            best_T, best_fit, best_rmse = T, fit, rmse
    return best_T, best_fit, best_rmse


def project_points(pts_cam: np.ndarray) -> np.ndarray:
    """(N,3) camera frame -> (N,2) pixel."""
    z = np.clip(pts_cam[:, 2], 1e-6, None)
    u = K_FX * pts_cam[:, 0] / z + K_CX
    v = K_FY * pts_cam[:, 1] / z + K_CY
    return np.stack([u, v], axis=1)


def draw_overlay(rgb: np.ndarray, mask: np.ndarray,
                 aligned_pts_cam: np.ndarray, label: str) -> np.ndarray:
    img = rgb.copy()
    # mask contour yellow
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 255), 2)
    # reprojected PLY green dots
    uvs = project_points(aligned_pts_cam)
    for u, v in uvs:
        if 0 <= u < IMG_W and 0 <= v < IMG_H:
            cv2.circle(img, (int(u), int(v)), 1, (0, 255, 0), -1)
    # label
    cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def process_sample(h5, sid: str, out_dir: Path):
    g = h5[sid]
    rgb = cv2.imread(str(ROOT / g.attrs["rgb_path"]))
    depth_m = load_depth_meter(ROOT / g.attrs["depth_path"])
    class_names = list(h5.attrs["class_names"])
    classes = np.asarray(g["classes"])
    confs = np.asarray(g["confidences"])
    bboxes = np.asarray(g["bboxes"])
    poly_ds = g["mask_poly"]

    panels: list[np.ndarray] = []
    for k in range(len(classes)):
        if float(confs[k]) < 0.5:
            continue
        cls_name = class_names[int(classes[k])]
        if not bbox_inside(bboxes[k], BBOX_MARGIN_PX):
            continue
        poly = np.asarray(poly_ds[k]).reshape(-1, 2)
        if poly.size < 6:
            continue
        mask = poly_to_mask(poly)
        scene_pts = depth_mask_to_pc(depth_m, mask)
        if len(scene_pts) < 100:
            continue
        ply_name = CLASS_TO_PLY.get(cls_name)
        if ply_name is None:
            continue
        model = load_ply_centered(PLY_DIR / ply_name)

        # crop around mask for readability
        ys, xs = np.where(mask)
        y0, y1 = max(ys.min() - 40, 0), min(ys.max() + 40, IMG_H)
        x0, x1 = max(xs.min() - 40, 0), min(xs.max() + 40, IMG_W)

        # v1
        T1, f1, r1 = align_v1(scene_pts, o3d.io.read_point_cloud(str(PLY_DIR / ply_name)))
        model_v1 = load_ply_centered(PLY_DIR / ply_name)
        model_v1.transform(T1)
        pts1 = np.asarray(model_v1.points)
        img1 = draw_overlay(rgb, mask, pts1,
                            f"v1 {cls_name} fit={f1:.2f} rmse={r1*1000:.1f}mm")

        # v2
        T2, f2, r2 = align_v2(scene_pts, load_ply_centered(PLY_DIR / ply_name))
        model_v2 = load_ply_centered(PLY_DIR / ply_name)
        model_v2.transform(T2)
        pts2 = np.asarray(model_v2.points)
        img2 = draw_overlay(rgb, mask, pts2,
                            f"v2 {cls_name} fit={f2:.2f} rmse={r2*1000:.1f}mm")

        c1 = img1[y0:y1, x0:x1]
        c2 = img2[y0:y1, x0:x1]
        panel = np.hstack([c1, c2])
        panels.append(panel)

    if not panels:
        return
    # stack panels vertically (each panel may have different width → pad)
    maxw = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        if p.shape[1] < maxw:
            pad = np.zeros((p.shape[0], maxw - p.shape[1], 3), dtype=np.uint8)
            p = np.hstack([p, pad])
        padded.append(p)
    final = np.vstack(padded)
    out_path = out_dir / f"{sid}_v1vs_v2.jpg"
    cv2.imwrite(str(out_path), final)
    print(f"wrote {out_path}  ({len(panels)} objects)")


def main():
    out_dir = ROOT / "scripts/_icp_compare_v1_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(DET_H5, "r") as h5:
        for sid in TEST_SAMPLES:
            if sid not in h5:
                print(f"skip missing {sid}")
                continue
            process_sample(h5, sid, out_dir)


if __name__ == "__main__":
    main()
