"""
ICP test v3 — v2 실패 원인(spam 같은 납작/대칭 물체의 PCA 불안정) 보완.

개선:
  - Init 후보 = PCA 4-flip (4) + identity+centroid (1) = 총 5개
  - 고정 correspondence distance (MATLAB 값): 0.048 / 0.020 / 0.005 m
  - Multi-scale iter: 200 / 400 / 1000
  - 모두 point-to-point (partial↔full mismatch 위험 회피)
  - Scene statistical outlier removal
  - 외곽 필터(<20px)
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
OUT_DIR = ROOT / "scripts/_icp_test_out_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_FX = K_FY = 1109.0
K_CX, K_CY = 640.0, 360.0
IMG_W, IMG_H = 1280, 720

BBOX_MARGIN_PX = 20

VOXELS = (0.012, 0.006, 0.003)
DISTS = (0.048, 0.020, 0.005)   # MATLAB 고정값
ITERS = (200, 400, 1000)

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


def load_ply_centered(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    if pts.ptp(axis=0).max() > 1.0:
        pcd.scale(0.001, center=(0, 0, 0))
        pts = np.asarray(pcd.points)
    pcd.translate(-pts.mean(axis=0))
    return pcd


def pca_axes(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T)
    vals, vecs = np.linalg.eigh(cov)
    R = vecs[:, np.argsort(vals)[::-1]]
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


def build_inits(model_pts: np.ndarray,
                scene_pts: np.ndarray) -> list[np.ndarray]:
    """PCA 4-flip + identity+centroid = 5개."""
    scene_c = scene_pts.mean(axis=0)
    model_c = model_pts.mean(axis=0)

    # PCA base
    R_m = pca_axes(model_pts)
    R_s = pca_axes(scene_pts)
    R_base = svd_cleanup(R_s @ R_m.T)

    flips = [
        np.diag([1.0, 1.0, 1.0]),
        np.diag([-1.0, -1.0, 1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([1.0, -1.0, -1.0]),
    ]
    out: list[np.ndarray] = []
    for F in flips:
        R = svd_cleanup(R_base @ F)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = scene_c - R @ model_c
        out.append(T)

    # identity + centroid translation
    T_id = np.eye(4)
    T_id[:3, 3] = scene_c - model_c
    out.append(T_id)
    return out


def multiscale_icp(source: o3d.geometry.PointCloud,
                   target: o3d.geometry.PointCloud,
                   init: np.ndarray) -> tuple[np.ndarray, float, float]:
    T = init.copy()
    fit = 0.0
    rmse = 0.0
    for v, d, it in zip(VOXELS, DISTS, ITERS):
        s = source.voxel_down_sample(v)
        t = target.voxel_down_sample(v)
        res = o3d.pipelines.registration.registration_icp(
            s, t, d, T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it))
        T = np.asarray(res.transformation).copy()
        fit = float(res.fitness)
        rmse = float(res.inlier_rmse)
    T[:3, :3] = svd_cleanup(T[:3, :3])
    return T, fit, rmse


def align_object(scene_pts: np.ndarray,
                 model_pcd: o3d.geometry.PointCloud) -> dict:
    model_pts = np.asarray(model_pcd.points)
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_pts)
    scene_pcd, _ = scene_pcd.remove_statistical_outlier(4, 2.0)
    scene_clean = np.asarray(scene_pcd.points)

    inits = build_inits(model_pts, scene_clean)
    names = ["pca_flip0", "pca_flip1", "pca_flip2", "pca_flip3", "identity"]

    best: dict | None = None
    all_cand: list[dict] = []
    for name, init in zip(names, inits):
        T, fit, rmse = multiscale_icp(model_pcd, scene_pcd, init)
        cand = {"init": name, "fitness": fit, "inlier_rmse": rmse,
                "transform": T}
        all_cand.append({"init": name, "fitness": fit, "rmse": rmse})
        # prefer higher fitness AND lower rmse. fitness tie-break: smaller rmse.
        if best is None:
            best = cand
        elif (fit > best["fitness"] + 1e-6
              or (abs(fit - best["fitness"]) < 1e-6 and rmse < best["inlier_rmse"])):
            best = cand

    return {"best": best, "candidates": all_cand,
            "scene_n_after_denoise": int(len(scene_clean))}


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
                            "conf": conf, "skipped": "bbox_on_border"})
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
            "best_init": best["init"],
            "fitness": best["fitness"],
            "inlier_rmse": best["inlier_rmse"],
            "translation": best["transform"][:3, 3].tolist(),
            "candidates": out["candidates"],
        })
    return results


def main() -> int:
    all_results: list[dict] = []
    with h5py.File(DET_H5, "r") as h5:
        for sid in TEST_SAMPLES:
            print(f"\n=== {sid} ===", flush=True)
            rs = test_sample(h5, sid)
            for r in rs:
                short = {k: v for k, v in r.items()
                         if k not in ("translation", "candidates")}
                print("  ", json.dumps(short, default=float))
                all_results.append(r)

    (OUT_DIR / "test_report.json").write_text(
        json.dumps(all_results, indent=2, default=float))

    gated = [r for r in all_results if "fitness" in r]
    skipped_border = sum(1 for r in all_results if r.get("skipped") == "bbox_on_border")
    print(f"\n--- summary ---")
    print(f"skipped on border: {skipped_border}")
    print(f"aligned: {len(gated)}")
    if gated:
        fits = [r["fitness"] for r in gated]
        rmses = [r["inlier_rmse"] for r in gated]
        print(f"fitness   mean={np.mean(fits):.3f}  min={min(fits):.3f}  max={max(fits):.3f}")
        print(f"rmse (mm) mean={np.mean(rmses)*1000:.2f}  max={max(rmses)*1000:.2f}")
        passes = sum(1 for r in gated
                     if r["fitness"] >= 0.02 and r["inlier_rmse"] <= 0.005)
        print(f"gate (fit>=0.02 & rmse<=0.005): {passes}/{len(gated)}")
        # which init won?
        from collections import Counter
        init_pick = Counter(r["best_init"] for r in gated)
        print("best_init distribution:", dict(init_pick))
    return 0


if __name__ == "__main__":
    sys.exit(main())
