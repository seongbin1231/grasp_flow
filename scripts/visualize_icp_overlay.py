"""
Visualize ICP alignment quality.

For each tested sample, render a 3D scatter of the scene point cloud
plus the ICP-transformed PLY model. Save as PNG for headless review.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
DET_H5 = ROOT / "img_dataset/yolo_cache_v3/detections.h5"
PLY_DIR = Path("/home/robotics/Competition/RoboCup_ARM/models/ply")
OUT_DIR = ROOT / "scripts/_icp_test_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_FX = K_FY = 1109.0
K_CX, K_CY = 640.0, 360.0
IMG_W, IMG_H = 1280, 720

CLASS_TO_PLY = {
    "bottle": "blueBottle.ply", "can": "greenCan.ply",
    "cube_blue": "cube.ply", "cube_green": "cube.ply",
    "cube_p": "cube.ply", "cube_red": "cube.ply",
    "marker": "marker.ply", "spam": "Simsort_SPAM.ply",
}

TEST_SAMPLES = ["sample_random1_1", "sample_random2_5", "sample_random3_10"]


def load_depth_meter(p: Path) -> np.ndarray:
    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return raw.astype(np.float32) / 1000.0


def poly_to_mask(poly: np.ndarray) -> np.ndarray:
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


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


def multiscale_icp(source, target, init, voxels=(0.012, 0.006, 0.003),
                   max_iters=(200, 200, 500)):
    T = init.copy()
    result = None
    for v, it in zip(voxels, max_iters):
        s = source.voxel_down_sample(v)
        t = target.voxel_down_sample(v)
        result = o3d.pipelines.registration.registration_icp(
            s, t, v * 2.0, T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it))
        T = result.transformation
    return T, float(result.fitness), float(result.inlier_rmse)


def render_sample(ax, scene_pts: np.ndarray, objects: list[dict]):
    """scene (gray) + each aligned PLY (colored)."""
    if len(scene_pts) > 8000:
        idx = np.random.choice(len(scene_pts), 8000, replace=False)
        scene_pts = scene_pts[idx]
    ax.scatter(scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2],
               c="lightgray", s=1, alpha=0.3, label="scene")
    colors = ["red", "green", "blue", "orange", "purple", "cyan", "magenta"]
    for i, obj in enumerate(objects):
        pts = obj["aligned_pts"]
        if len(pts) > 2000:
            pts = pts[np.random.choice(len(pts), 2000, replace=False)]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=colors[i % len(colors)], s=2, alpha=0.7,
                   label=f"{obj['class']} fit={obj['fitness']:.2f}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(fontsize=7, loc="upper right")


def main():
    with h5py.File(DET_H5, "r") as h5:
        class_names = list(h5.attrs["class_names"])
        for sid in TEST_SAMPLES:
            g = h5[sid]
            depth_m = load_depth_meter(ROOT / g.attrs["depth_path"])
            classes = np.asarray(g["classes"])
            confs = np.asarray(g["confidences"])
            poly_ds = g["mask_poly"]

            fig = plt.figure(figsize=(14, 6))
            ax1 = fig.add_subplot(121, projection="3d")
            ax1.set_title(f"{sid} — masked scene only")
            ax2 = fig.add_subplot(122, projection="3d")
            ax2.set_title(f"{sid} — scene + ICP-aligned PLY")

            # union scene from all masks for ax1
            all_scene: list[np.ndarray] = []
            objects: list[dict] = []
            for k in range(len(classes)):
                if float(confs[k]) < 0.5:
                    continue
                poly = np.asarray(poly_ds[k]).reshape(-1, 2)
                if poly.size < 6:
                    continue
                mask = poly_to_mask(poly)
                scene_pts = depth_mask_to_pc(depth_m, mask)
                if len(scene_pts) < 100:
                    continue
                all_scene.append(scene_pts)

                cls_name = class_names[int(classes[k])]
                ply_name = CLASS_TO_PLY.get(cls_name)
                if ply_name is None:
                    continue
                model = load_ply_centered(PLY_DIR / ply_name)
                scene_pcd = o3d.geometry.PointCloud()
                scene_pcd.points = o3d.utility.Vector3dVector(scene_pts)
                init = np.eye(4)
                init[:3, 3] = scene_pts.mean(axis=0)
                T, fit, rmse = multiscale_icp(model, scene_pcd, init)
                aligned = model.transform(T)
                aligned_pts = np.asarray(aligned.points)
                objects.append({"class": cls_name, "fitness": fit,
                                "rmse": rmse, "aligned_pts": aligned_pts})

            if not all_scene:
                print(f"skip {sid}")
                continue
            scene_all = np.vstack(all_scene)
            render_sample(ax1, scene_all, [])
            render_sample(ax2, scene_all, objects)

            # match camera-ish viewpoint (top-down camera frame)
            for ax in (ax1, ax2):
                ax.view_init(elev=-60, azim=-90)

            out = OUT_DIR / f"{sid}_overlay.png"
            plt.tight_layout()
            plt.savefig(out, dpi=120)
            plt.close(fig)
            print(f"wrote {out}  objects={len(objects)}")


if __name__ == "__main__":
    main()
