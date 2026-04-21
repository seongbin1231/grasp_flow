"""v1 vs v3 2D overlay — 노란 윤곽 = mask, 초록 점 = ICP-aligned PLY reprojection."""

from __future__ import annotations

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
BBOX_MARGIN_PX = 20

CLASS_TO_PLY = {
    "bottle": "blueBottle.ply", "can": "greenCan.ply",
    "cube_blue": "cube.ply", "cube_green": "cube.ply",
    "cube_p": "cube.ply", "cube_red": "cube.ply",
    "marker": "marker.ply", "spam": "Simsort_SPAM.ply",
}

TEST_SAMPLES = [
    "sample_random1_1", "sample_random2_5", "sample_random3_10",
    "sample_random4_1", "sample_random4_20", "sample_random5_10",
    "sample_random6_15", "sample_random1_50",
]

VOXELS = (0.012, 0.006, 0.003)
DISTS_V3 = (0.048, 0.020, 0.005)
DISTS_V1 = (0.024, 0.012, 0.006)  # v1: dist = voxel*2
ITERS_V3 = (200, 400, 1000)
ITERS_V1 = (200, 200, 500)


def load_depth(p): return cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0


def poly_to_mask(poly):
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def bbox_inside(b, m):
    return b[0] >= m and b[1] >= m and b[2] <= IMG_W - m and b[3] <= IMG_H - m


def depth_mask_to_pc(d, mask):
    ys, xs = np.where(mask & (d > 0.1) & (d < 2.0))
    z = d[ys, xs]
    return np.stack([(xs - K_CX) * z / K_FX, (ys - K_CY) * z / K_FY, z], axis=1).astype(np.float32)


def load_ply_centered(p):
    pcd = o3d.io.read_point_cloud(str(p))
    pts = np.asarray(pcd.points)
    if pts.ptp(axis=0).max() > 1.0:
        pcd.scale(0.001, center=(0, 0, 0))
        pts = np.asarray(pcd.points)
    pcd.translate(-pts.mean(axis=0))
    return pcd


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


def icp_stack(src, tgt, init, voxels, dists, iters):
    T = init.copy()
    fit, rmse = 0.0, 0.0
    for v, d, it in zip(voxels, dists, iters):
        s = src.voxel_down_sample(v)
        t = tgt.voxel_down_sample(v)
        res = o3d.pipelines.registration.registration_icp(
            s, t, d, T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it))
        T = np.asarray(res.transformation).copy()
        fit, rmse = float(res.fitness), float(res.inlier_rmse)
    return T, fit, rmse


def align_v1(scene_pts, model):
    scene = o3d.geometry.PointCloud()
    scene.points = o3d.utility.Vector3dVector(scene_pts)
    init = np.eye(4)
    init[:3, 3] = scene_pts.mean(axis=0)
    return icp_stack(model, scene, init, VOXELS, DISTS_V1, ITERS_V1)


def align_v3(scene_pts, model):
    scene = o3d.geometry.PointCloud()
    scene.points = o3d.utility.Vector3dVector(scene_pts)
    scene, _ = scene.remove_statistical_outlier(4, 2.0)
    scene_clean = np.asarray(scene.points)
    model_pts = np.asarray(model.points)

    R_m = pca_axes(model_pts)
    R_s = pca_axes(scene_clean)
    R_base = svd_cleanup(R_s @ R_m.T)
    sc = scene_clean.mean(axis=0)
    mc = model_pts.mean(axis=0)

    flips = [np.diag([1, 1, 1.0]), np.diag([-1, -1, 1.0]),
             np.diag([-1, 1, -1.0]), np.diag([1, -1, -1.0])]
    inits: list[tuple[str, np.ndarray]] = []
    for i, F in enumerate(flips):
        R = svd_cleanup(R_base @ F)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = sc - R @ mc
        inits.append((f"flip{i}", T))
    T_id = np.eye(4)
    T_id[:3, 3] = sc - mc
    inits.append(("identity", T_id))

    best = None
    for name, init in inits:
        T, fit, rmse = icp_stack(model, scene, init, VOXELS, DISTS_V3, ITERS_V3)
        if best is None or fit > best[2] + 1e-6 or (abs(fit - best[2]) < 1e-6 and rmse < best[3]):
            best = (name, T, fit, rmse)
    return best[1], best[2], best[3], best[0]


def proj(pts):
    z = np.clip(pts[:, 2], 1e-6, None)
    return np.stack([K_FX * pts[:, 0] / z + K_CX,
                     K_FY * pts[:, 1] / z + K_CY], axis=1)


def draw(rgb, mask, aligned_pts, label):
    img = rgb.copy()
    cts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cts, -1, (0, 255, 255), 2)
    uvs = proj(aligned_pts)
    for u, v in uvs:
        if 0 <= u < IMG_W and 0 <= v < IMG_H:
            cv2.circle(img, (int(u), int(v)), 1, (0, 255, 0), -1)
    cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def process(h5, sid, out_dir):
    g = h5[sid]
    rgb = cv2.imread(str(ROOT / g.attrs["rgb_path"]))
    depth = load_depth(ROOT / g.attrs["depth_path"])
    class_names = list(h5.attrs["class_names"])
    classes = np.asarray(g["classes"])
    confs = np.asarray(g["confidences"])
    bboxes = np.asarray(g["bboxes"])
    polys = g["mask_poly"]

    panels = []
    for k in range(len(classes)):
        if float(confs[k]) < 0.5:
            continue
        cls = class_names[int(classes[k])]
        if not bbox_inside(bboxes[k], BBOX_MARGIN_PX):
            continue
        poly = np.asarray(polys[k]).reshape(-1, 2)
        if poly.size < 6:
            continue
        mask = poly_to_mask(poly)
        scene_pts = depth_mask_to_pc(depth, mask)
        if len(scene_pts) < 100:
            continue
        ply = CLASS_TO_PLY.get(cls)
        if ply is None:
            continue

        ys, xs = np.where(mask)
        y0, y1 = max(ys.min() - 40, 0), min(ys.max() + 40, IMG_H)
        x0, x1 = max(xs.min() - 40, 0), min(xs.max() + 40, IMG_W)

        model_v1 = load_ply_centered(PLY_DIR / ply)
        T1, f1, r1 = align_v1(scene_pts, model_v1)
        m1 = load_ply_centered(PLY_DIR / ply)
        m1.transform(T1)
        img1 = draw(rgb, mask, np.asarray(m1.points),
                    f"v1 {cls} fit={f1:.2f} rmse={r1 * 1000:.1f}mm")

        model_v3 = load_ply_centered(PLY_DIR / ply)
        T3, f3, r3, best_init = align_v3(scene_pts, model_v3)
        m3 = load_ply_centered(PLY_DIR / ply)
        m3.transform(T3)
        img3 = draw(rgb, mask, np.asarray(m3.points),
                    f"v3[{best_init}] {cls} fit={f3:.2f} rmse={r3 * 1000:.1f}mm")

        panel = np.hstack([img1[y0:y1, x0:x1], img3[y0:y1, x0:x1]])
        panels.append(panel)

    if not panels:
        return
    maxw = max(p.shape[1] for p in panels)
    padded = [np.hstack([p, np.zeros((p.shape[0], maxw - p.shape[1], 3), dtype=np.uint8)])
              if p.shape[1] < maxw else p for p in panels]
    out = out_dir / f"{sid}_v1_vs_v3.jpg"
    cv2.imwrite(str(out), np.vstack(padded))
    print(f"wrote {out} ({len(panels)} objects)")


def main():
    out_dir = ROOT / "scripts/_icp_compare_v1_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(DET_H5, "r") as h5:
        for sid in TEST_SAMPLES:
            if sid in h5:
                process(h5, sid, out_dir)


if __name__ == "__main__":
    main()
