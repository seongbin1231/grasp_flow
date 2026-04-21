"""
Small-scale ICP test — 3 samples, all detected objects.

Goal: verify multi-scale ICP (open3d) pipeline produces reasonable
fitness/inlier_rmse before running on all 588 samples.

Inputs:
  - img_dataset/yolo_cache_v3/detections.h5 (YOLO result)
  - img_dataset/captured_images_depth/{scene}_dep/{scene}_depth_{idx}.png
  - RoboCup_ARM/models/ply/{name}.ply

Output:
  - scripts/_icp_test_out/test_report.json
  - scripts/_icp_test_out/{sample}_{object}.png (overlay)
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
OUT_DIR = ROOT / "scripts/_icp_test_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_FX = 1109.0
K_FY = 1109.0
K_CX = 640.0
K_CY = 360.0
IMG_W, IMG_H = 1280, 720

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

# Test samples (picked arbitrarily from different scenes for diversity)
TEST_SAMPLES = ["sample_random1_1", "sample_random2_5", "sample_random3_10"]


def load_depth_meter(depth_path: Path) -> np.ndarray:
    """uint16 PNG (mm) -> float32 (m)."""
    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"failed to read {depth_path}")
    if raw.dtype != np.uint16:
        raise RuntimeError(f"unexpected depth dtype {raw.dtype}")
    if raw.shape != (IMG_H, IMG_W):
        raise RuntimeError(f"unexpected depth shape {raw.shape}")
    return raw.astype(np.float32) / 1000.0


def poly_to_mask(poly: np.ndarray) -> np.ndarray:
    """poly: (N,2) float pixels -> (H,W) bool mask."""
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def depth_mask_to_pc(depth_m: np.ndarray, mask: np.ndarray,
                    z_min: float = 0.1, z_max: float = 2.0) -> np.ndarray:
    """Backproject masked depth pixels to camera-frame 3D points."""
    ys, xs = np.where(mask & (depth_m > z_min) & (depth_m < z_max))
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth_m[ys, xs]
    x = (xs - K_CX) * z / K_FX
    y = (ys - K_CY) * z / K_FY
    return np.stack([x, y, z], axis=1).astype(np.float32)


def load_ply_centered(path: Path) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise RuntimeError(f"empty PLY: {path}")
    # Convert mm -> m if model is in mm (heuristic: extent > 1.0m suggests mm)
    extent = pts.ptp(axis=0)
    if extent.max() > 1.0:
        pcd.scale(0.001, center=(0.0, 0.0, 0.0))
        pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    pcd.translate(-centroid)
    return pcd, pcd.get_axis_aligned_bounding_box().get_extent()


def multiscale_icp(source: o3d.geometry.PointCloud,
                   target: o3d.geometry.PointCloud,
                   init: np.ndarray,
                   voxels=(0.012, 0.006, 0.003),
                   max_iters=(200, 200, 500)) -> dict:
    """Coarse-to-fine point-to-point ICP."""
    T = init.copy()
    result = None
    for v, it in zip(voxels, max_iters):
        s = source.voxel_down_sample(v)
        t = target.voxel_down_sample(v)
        result = o3d.pipelines.registration.registration_icp(
            s, t, max_correspondence_distance=v * 2.0, init=T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it),
        )
        T = result.transformation
    return {
        "transform": T,
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
    }


def test_sample(h5: h5py.File, sid: str) -> list[dict]:
    if sid not in h5:
        return [{"sample": sid, "error": "not_found"}]
    g = h5[sid]
    if not g.attrs.get("detected", False):
        return [{"sample": sid, "error": "not_detected"}]

    depth_path = ROOT / g.attrs["depth_path"]
    depth_m = load_depth_meter(depth_path)

    class_names = list(h5.attrs["class_names"])
    classes = np.asarray(g["classes"])
    confs = np.asarray(g["confidences"])
    poly_ds = g["mask_poly"]

    results: list[dict] = []
    for k in range(len(classes)):
        cls_id = int(classes[k])
        cls_name = class_names[cls_id]
        conf = float(confs[k])
        if conf < 0.5:
            continue

        poly = np.asarray(poly_ds[k]).reshape(-1, 2)
        if poly.size < 6:
            results.append({"sample": sid, "object": k, "class": cls_name,
                            "skipped": "empty_polygon"})
            continue

        mask = poly_to_mask(poly)
        scene_pts = depth_mask_to_pc(depth_m, mask)
        if len(scene_pts) < 100:
            results.append({"sample": sid, "object": k, "class": cls_name,
                            "skipped": "too_few_points",
                            "n_points": int(len(scene_pts))})
            continue

        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(scene_pts)

        ply_name = CLASS_TO_PLY.get(cls_name)
        if ply_name is None:
            results.append({"sample": sid, "object": k, "class": cls_name,
                            "skipped": "no_ply_mapping"})
            continue
        model_pcd, model_extent = load_ply_centered(PLY_DIR / ply_name)

        # Init: translate PLY to scene centroid
        scene_centroid = scene_pts.mean(axis=0)
        init = np.eye(4)
        init[:3, 3] = scene_centroid

        icp = multiscale_icp(model_pcd, scene_pcd, init=init)

        results.append({
            "sample": sid,
            "object": k,
            "class": cls_name,
            "conf": conf,
            "n_scene_points": int(len(scene_pts)),
            "model_extent_m": model_extent.tolist(),
            "scene_centroid": scene_centroid.tolist(),
            "fitness": icp["fitness"],
            "inlier_rmse": icp["inlier_rmse"],
            "translation": icp["transform"][:3, 3].tolist(),
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

    report_path = OUT_DIR / "test_report.json"
    report_path.write_text(json.dumps(all_results, indent=2, default=float))
    print(f"\nwrote {report_path}")

    # Summary
    gated = [r for r in all_results if "fitness" in r]
    if gated:
        fits = [r["fitness"] for r in gated]
        print(f"\nfitness stats over {len(gated)} objects: "
              f"mean={np.mean(fits):.3f} min={min(fits):.3f} max={max(fits):.3f}")
        pass_gate = sum(1 for r in gated if r["fitness"] >= 0.02
                        and r["inlier_rmse"] <= 0.005)
        print(f"gate (fit>=0.02 & rmse<=0.005): {pass_gate}/{len(gated)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
