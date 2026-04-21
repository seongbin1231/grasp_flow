"""
ICP test v2 — MATLAB 기법 반영:
  - bbox margin filter (<20px 외곽 제외)
  - statistical outlier removal (pcdenoise 대응)
  - PCA-based init + 4-flip (최고 fitness 선택)
  - multi-scale: stage1 point-to-point, stage2/3 point-to-plane
  - 적응적 correspondence distance (이전 RMSE 기반)
  - stage2 재시도 (distance 2배)
  - SVD cleanup
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
import open3d as o3d

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
DET_H5 = ROOT / "img_dataset/yolo_cache_v3/detections.h5"
PLY_DIR = Path("/home/robotics/Competition/RoboCup_ARM/models/ply")
OUT_DIR = ROOT / "scripts/_icp_test_out_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_FX = K_FY = 1109.0
K_CX, K_CY = 640.0, 360.0
IMG_W, IMG_H = 1280, 720

BBOX_MARGIN_PX = 20  # 외곽 필터: 이 값보다 경계에 가까우면 제외

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
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth_m[ys, xs]
    x = (xs - K_CX) * z / K_FX
    y = (ys - K_CY) * z / K_FY
    return np.stack([x, y, z], axis=1).astype(np.float32)


def make_pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def denoise_pc(pcd: o3d.geometry.PointCloud,
               nb_neighbors: int = 4, std_ratio: float = 2.0
               ) -> o3d.geometry.PointCloud:
    """MATLAB pcdenoise 대응 — statistical outlier removal."""
    if len(pcd.points) < nb_neighbors + 1:
        return pcd
    clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return clean


def load_ply_centered(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    if pts.ptp(axis=0).max() > 1.0:  # mm -> m
        pcd.scale(0.001, center=(0, 0, 0))
        pts = np.asarray(pcd.points)
    pcd.translate(-pts.mean(axis=0))
    return pcd


def pca_axes(pts: np.ndarray) -> np.ndarray:
    """PCA → 고유벡터 (3,3), 열이 긴 축→짧은 축 순, 오른손 좌표계 보장."""
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T)
    vals, vecs = np.linalg.eigh(cov)  # 오름차순
    order = np.argsort(vals)[::-1]    # 내림차순
    R = vecs[:, order]
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return R


def svd_cleanup(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    R2 = U @ Vt
    if np.linalg.det(R2) < 0:
        U[:, -1] *= -1
        R2 = U @ Vt
    return R2


def build_init_candidates(model_pts: np.ndarray,
                          scene_pts: np.ndarray) -> list[np.ndarray]:
    """PCA 정렬 후 4-flip 대칭 후보 생성."""
    R_model = pca_axes(model_pts)
    R_scene = pca_axes(scene_pts)
    T_base = np.eye(4)
    T_base[:3, :3] = svd_cleanup(R_scene @ R_model.T)
    T_base[:3, 3] = scene_pts.mean(axis=0) - T_base[:3, :3] @ model_pts.mean(axis=0)

    flips = [
        np.diag([1, 1, 1]),
        np.diag([-1, -1, 1]),
        np.diag([-1, 1, -1]),
        np.diag([1, -1, -1]),
    ]
    out: list[np.ndarray] = []
    for F in flips:
        T = T_base.copy()
        R_flipped = T[:3, :3] @ F
        T[:3, :3] = svd_cleanup(R_flipped)
        out.append(T)
    return out


def run_icp(source: o3d.geometry.PointCloud,
            target: o3d.geometry.PointCloud,
            init: np.ndarray, dist: float, max_iter: int,
            use_plane: bool) -> tuple[np.ndarray, float, float]:
    if use_plane:
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    res = o3d.pipelines.registration.registration_icp(
        source, target, dist, init, est,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
    return np.asarray(res.transformation).copy(), float(res.fitness), float(res.inlier_rmse)


def multiscale_icp_adaptive(source: o3d.geometry.PointCloud,
                            target: o3d.geometry.PointCloud,
                            init: np.ndarray) -> dict:
    """Stage1 P2P coarse, Stage2/3 P2Plane with adaptive distance."""
    voxels = [0.012, 0.006, 0.003]
    iters = [200, 400, 1000]
    use_plane = [False, True, True]

    # Pre-downsample + normals for plane stages
    T = init.copy()
    best_fit = 0.0
    best_rmse = 0.05
    for i, (v, it, plane) in enumerate(zip(voxels, iters, use_plane)):
        s = source.voxel_down_sample(v)
        t = target.voxel_down_sample(v)
        if plane:
            s.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                radius=v * 2.5, max_nn=30))
            t.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                radius=v * 2.5, max_nn=30))

        # Adaptive correspondence distance: 이전 RMSE의 3배, 최소 v*2
        if i == 0:
            dist = v * 4.0
        else:
            dist = max(best_rmse * 3.0, v * 2.0)

        T, fit, rmse = run_icp(s, t, T, dist, it, plane)

        # Stage2 저 fitness 재시도 — dist 2배 완화
        if i == 1 and fit < 0.3:
            T2, fit2, rmse2 = run_icp(s, t, T, dist * 2.0, it, plane)
            if fit2 > fit:
                T, fit, rmse = T2, fit2, rmse2

        best_fit, best_rmse = fit, rmse

    T[:3, :3] = svd_cleanup(T[:3, :3])
    return {"transform": T, "fitness": best_fit, "inlier_rmse": best_rmse}


def align_object(scene_pts: np.ndarray,
                 model_pcd: o3d.geometry.PointCloud) -> dict:
    model_pts = np.asarray(model_pcd.points)
    scene_pcd = make_pcd(scene_pts)
    scene_pcd = denoise_pc(scene_pcd)

    inits = build_init_candidates(model_pts, np.asarray(scene_pcd.points))
    best: dict | None = None
    per_init: list[dict] = []
    for idx, init in enumerate(inits):
        res = multiscale_icp_adaptive(model_pcd, scene_pcd, init)
        per_init.append({"flip": idx, "fitness": res["fitness"],
                         "rmse": res["inlier_rmse"]})
        if best is None or res["fitness"] > best["fitness"]:
            best = {**res, "flip": idx}
    return {"best": best, "all_flips": per_init,
            "scene_n_after_denoise": int(len(scene_pcd.points))}


def test_sample(h5: h5py.File, sid: str) -> list[dict]:
    if sid not in h5:
        return [{"sample": sid, "error": "not_found"}]
    g = h5[sid]
    if not g.attrs.get("detected", False):
        return [{"sample": sid, "error": "not_detected"}]

    depth_m = load_depth_meter(ROOT / g.attrs["depth_path"])
    class_names = list(h5.attrs["class_names"])
    classes = np.asarray(g["classes"])
    confs = np.asarray(g["confidences"])
    bboxes = np.asarray(g["bboxes"])
    poly_ds = g["mask_poly"]

    results: list[dict] = []
    for k in range(len(classes)):
        cls_id = int(classes[k])
        cls_name = class_names[cls_id]
        conf = float(confs[k])
        if conf < 0.5:
            continue
        if not bbox_inside(bboxes[k], BBOX_MARGIN_PX):
            results.append({"sample": sid, "object": k, "class": cls_name,
                            "conf": conf, "skipped": "bbox_on_border",
                            "bbox": bboxes[k].tolist()})
            continue
        poly = np.asarray(poly_ds[k]).reshape(-1, 2)
        if poly.size < 6:
            continue
        mask = poly_to_mask(poly)
        scene_pts = depth_mask_to_pc(depth_m, mask)
        if len(scene_pts) < 100:
            results.append({"sample": sid, "object": k, "class": cls_name,
                            "skipped": "too_few_points",
                            "n_points": int(len(scene_pts))})
            continue
        ply_name = CLASS_TO_PLY.get(cls_name)
        if ply_name is None:
            continue
        model = load_ply_centered(PLY_DIR / ply_name)

        out = align_object(scene_pts, model)
        best = out["best"]
        results.append({
            "sample": sid, "object": k, "class": cls_name,
            "conf": conf,
            "n_scene": int(len(scene_pts)),
            "n_after_denoise": out["scene_n_after_denoise"],
            "best_flip": best["flip"],
            "fitness": best["fitness"],
            "inlier_rmse": best["inlier_rmse"],
            "translation": best["transform"][:3, 3].tolist(),
            "all_flip_fit": [f["fitness"] for f in out["all_flips"]],
        })
    return results


def main() -> int:
    all_results: list[dict] = []
    with h5py.File(DET_H5, "r") as h5:
        for sid in TEST_SAMPLES:
            print(f"\n=== {sid} ===", flush=True)
            rs = test_sample(h5, sid)
            for r in rs:
                print("  ", json.dumps(r, default=float))
                all_results.append(r)

    (OUT_DIR / "test_report.json").write_text(
        json.dumps(all_results, indent=2, default=float))

    gated = [r for r in all_results if "fitness" in r]
    skipped_border = sum(1 for r in all_results if r.get("skipped") == "bbox_on_border")
    print(f"\n--- summary ---")
    print(f"total candidates: {len(all_results)}")
    print(f"skipped on border: {skipped_border}")
    print(f"aligned objects: {len(gated)}")
    if gated:
        fits = [r["fitness"] for r in gated]
        rmses = [r["inlier_rmse"] for r in gated]
        print(f"fitness   mean={np.mean(fits):.3f}  min={min(fits):.3f}  max={max(fits):.3f}")
        print(f"rmse (mm) mean={np.mean(rmses)*1000:.2f}  max={max(rmses)*1000:.2f}")
        passes = sum(1 for r in gated
                     if r["fitness"] >= 0.02 and r["inlier_rmse"] <= 0.005)
        print(f"gate (fit>=0.02 & rmse<=0.005): {passes}/{len(gated)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
