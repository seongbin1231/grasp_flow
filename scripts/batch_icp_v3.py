"""
Batch ICP v3 — 588장 전체에 v3 로직 적용, poses.h5 + quality_report.json 생성.

v3 = PCA 4-flip + identity (총 5 init) + MATLAB 고정 distance (0.048/0.020/0.005)
     + statistical outlier removal + SVD cleanup + bbox margin 필터
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import traceback

import cv2
import h5py
import numpy as np
import open3d as o3d

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
DET_H5 = ROOT / "img_dataset/yolo_cache_v3/detections.h5"
PLY_DIR = Path("/home/robotics/Competition/RoboCup_ARM/models/ply")
OUT_DIR = ROOT / "img_dataset/icp_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
POSES_H5 = OUT_DIR / "poses.h5"
REPORT_JSON = OUT_DIR / "quality_report.json"
LOG_JSON = OUT_DIR / "errors.jsonl"

K_FX = K_FY = 1109.0
K_CX, K_CY = 640.0, 360.0
IMG_W, IMG_H = 1280, 720

CONF_THRESHOLD = 0.5
MIN_SCENE_POINTS = 100
FITNESS_GATE = 0.30
RMSE_GATE_M = 0.003

# class별 YOLO mask 픽셀 하위 10% threshold — 기존 YOLO cache v3 에서 계산됨.
# bbox_margin=20px (31% drop) 대체. 화면 가장자리여도 mask 가 충분히 크면 ICP 진행.
MASK_PX_P10 = {
    "bottle": 7878,
    "can": 11114,
    "cube_blue": 3623,
    "cube_green": 3631,
    "cube_p": 5903,
    "cube_red": 5038,
    "marker": 1599,
    "spam": 3517,
}

VOXELS = (0.012, 0.006, 0.003)
DISTS = (0.048, 0.020, 0.005)
ITERS = (200, 400, 1000)

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

# -----------------------------------------------------------


def load_depth_meter(p: Path) -> np.ndarray:
    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return raw.astype(np.float32) / 1000.0


def poly_to_mask(poly: np.ndarray) -> np.ndarray:
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def depth_mask_to_pc(depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask & (depth_m > 0.1) & (depth_m < 2.0))
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth_m[ys, xs]
    x = (xs - K_CX) * z / K_FX
    y = (ys - K_CY) * z / K_FY
    return np.stack([x, y, z], axis=1).astype(np.float32)


_PLY_CACHE: dict[str, o3d.geometry.PointCloud] = {}
_PLY_META: dict[str, dict] = {}


def load_ply(name: str) -> tuple[o3d.geometry.PointCloud, dict]:
    if name in _PLY_CACHE:
        return _PLY_CACHE[name], _PLY_META[name]
    pcd = o3d.io.read_point_cloud(str(PLY_DIR / name))
    pts = np.asarray(pcd.points)
    if pts.ptp(axis=0).max() > 1.0:
        pcd.scale(0.001, center=(0, 0, 0))
        pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    pcd.translate(-centroid)
    pts = np.asarray(pcd.points)
    extent = pts.ptp(axis=0).astype(np.float32)
    long_axis = int(np.argmax(extent))
    _PLY_CACHE[name] = pcd
    _PLY_META[name] = {"extent": extent, "long_axis": long_axis,
                       "n_points": len(pts)}
    return pcd, _PLY_META[name]


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


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """R (3x3) -> [qw, qx, qy, qz]. Shepperd's method, stable."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float32)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def build_inits(model_pts: np.ndarray,
                scene_pts: np.ndarray) -> list[tuple[str, np.ndarray]]:
    sc = scene_pts.mean(axis=0)
    mc = model_pts.mean(axis=0)
    R_base = svd_cleanup(pca_axes(scene_pts) @ pca_axes(model_pts).T)
    flips = [np.diag([1.0, 1.0, 1.0]),
             np.diag([-1.0, -1.0, 1.0]),
             np.diag([-1.0, 1.0, -1.0]),
             np.diag([1.0, -1.0, -1.0])]
    out = []
    for i, F in enumerate(flips):
        R = svd_cleanup(R_base @ F)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = sc - R @ mc
        out.append((f"pca_flip{i}", T))
    T_id = np.eye(4)
    T_id[:3, 3] = sc - mc
    out.append(("identity", T_id))
    return out


def multiscale_icp(source: o3d.geometry.PointCloud,
                   target: o3d.geometry.PointCloud,
                   init: np.ndarray) -> tuple[np.ndarray, float, float]:
    T = init.copy()
    fit, rmse = 0.0, 0.0
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


def align_object(scene_pts_raw: np.ndarray,
                 model_pcd: o3d.geometry.PointCloud) -> dict:
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_pts_raw)
    if len(scene_pts_raw) >= 5:
        scene_pcd, _ = scene_pcd.remove_statistical_outlier(4, 2.0)
    scene_clean = np.asarray(scene_pcd.points)
    if len(scene_clean) < 20:
        return {"skipped": "too_few_after_denoise",
                "scene_n_raw": len(scene_pts_raw),
                "scene_n_clean": len(scene_clean)}
    model_pts = np.asarray(model_pcd.points)

    inits = build_inits(model_pts, scene_clean)
    best: dict | None = None
    candidates: list[dict] = []
    for name, init in inits:
        T, fit, rmse = multiscale_icp(model_pcd, scene_pcd, init)
        candidates.append({"init": name, "fitness": fit, "rmse": rmse})
        cand = {"init": name, "T": T, "fitness": fit, "rmse": rmse}
        if best is None:
            best = cand
        elif (cand["fitness"] > best["fitness"] + 1e-6
              or (abs(cand["fitness"] - best["fitness"]) < 1e-6
                  and cand["rmse"] < best["rmse"])):
            best = cand

    return {
        "best": best,
        "candidates": candidates,
        "scene_n_raw": int(len(scene_pts_raw)),
        "scene_n_clean": int(len(scene_clean)),
        "scene_centroid": scene_clean.mean(axis=0).astype(np.float32),
    }


# -----------------------------------------------------------


def run():
    start = time.time()
    stats = {
        "samples_total": 0, "samples_with_objs": 0,
        "samples_zero_det": 0,
        "objects_attempted": 0,
        "objects_skipped_low_conf": 0,
        "objects_skipped_mask_px": 0,
        "objects_skipped_poly": 0,
        "objects_skipped_scene_pts": 0,
        "objects_skipped_no_ply": 0,
        "objects_skipped_denoise": 0,
        "objects_aligned": 0,
        "objects_pass_gate": 0,
        "errors": 0,
        "per_class": {},
    }
    err_log = open(LOG_JSON, "w")

    with h5py.File(DET_H5, "r") as src, h5py.File(POSES_H5, "w") as dst:
        class_names = list(src.attrs["class_names"])

        # root attrs
        dst.attrs["camera_fx"] = K_FX
        dst.attrs["camera_fy"] = K_FY
        dst.attrs["camera_cx"] = K_CX
        dst.attrs["camera_cy"] = K_CY
        dst.attrs["image_h"] = IMG_H
        dst.attrs["image_w"] = IMG_W
        dst.attrs["coord_frame"] = "camera"
        dst.attrs["schema_version"] = 1
        dst.attrs["class_names"] = class_names
        dst.attrs["detections_h5"] = str(DET_H5)
        dst.attrs["icp_variant"] = "v3"
        dst.attrs["icp_voxels"] = np.array(VOXELS, dtype=np.float32)
        dst.attrs["icp_dists"] = np.array(DISTS, dtype=np.float32)
        dst.attrs["icp_iters"] = np.array(ITERS, dtype=np.int32)
        dst.attrs["conf_threshold"] = CONF_THRESHOLD
        dst.attrs["fitness_gate"] = FITNESS_GATE
        dst.attrs["rmse_gate_m"] = RMSE_GATE_M
        dst.attrs["mask_px_p10_classes"] = list(MASK_PX_P10.keys())
        dst.attrs["mask_px_p10_values"] = np.array(
            list(MASK_PX_P10.values()), dtype=np.int32)

        sids = list(src.keys())
        n_total = len(sids)
        stats["samples_total"] = n_total
        print(f"[info] total samples: {n_total}")

        for i, sid in enumerate(sids):
            g_src = src[sid]
            if not g_src.attrs.get("detected", False):
                stats["samples_zero_det"] += 1
                continue

            g_dst = dst.create_group(sid)
            g_dst.attrs["rgb_path"] = g_src.attrs["rgb_path"]
            g_dst.attrs["depth_path"] = g_src.attrs["depth_path"]
            g_dst.attrs["scene"] = g_src.attrs["scene"]

            try:
                depth_m = load_depth_meter(ROOT / g_src.attrs["depth_path"])
            except Exception as e:
                err_log.write(json.dumps(
                    {"sample": sid, "error": "depth_load", "msg": str(e)}) + "\n")
                stats["errors"] += 1
                continue

            classes = np.asarray(g_src["classes"])
            confs = np.asarray(g_src["confidences"])
            bboxes = np.asarray(g_src["bboxes"])
            poly_ds = g_src["mask_poly"]
            uv_ctrs = np.asarray(g_src["uv_centroid"])

            n_aligned = 0
            for k in range(len(classes)):
                stats["objects_attempted"] += 1
                cls_id = int(classes[k])
                cls_name = class_names[cls_id]
                conf = float(confs[k])
                bbox = bboxes[k]
                per_cls = stats["per_class"].setdefault(
                    cls_name, {"attempted": 0, "aligned": 0, "passed": 0})
                per_cls["attempted"] += 1

                g_obj = g_dst.create_group(f"object_{k}")
                g_obj.attrs["class_id"] = cls_id
                g_obj.attrs["class_name"] = cls_name
                g_obj.attrs["yolo_conf"] = conf
                g_obj.attrs["bbox_xyxy"] = bbox.astype(np.float32)
                g_obj.create_dataset("uv_centroid",
                                     data=uv_ctrs[k].astype(np.float32))

                if conf < CONF_THRESHOLD:
                    g_obj.attrs["skipped_reason"] = "low_conf"
                    stats["objects_skipped_low_conf"] += 1
                    continue
                poly = np.asarray(poly_ds[k]).reshape(-1, 2)
                if poly.size < 6:
                    g_obj.attrs["skipped_reason"] = "empty_poly"
                    stats["objects_skipped_poly"] += 1
                    continue

                ply_name = CLASS_TO_PLY.get(cls_name)
                if ply_name is None:
                    g_obj.attrs["skipped_reason"] = "no_ply_mapping"
                    stats["objects_skipped_no_ply"] += 1
                    continue
                g_obj.attrs["ply_file"] = ply_name

                try:
                    mask = poly_to_mask(poly)
                    mask_px = int(mask.sum())
                    g_obj.create_dataset("mask_px",
                                         data=np.int32(mask_px))
                    p10_thresh = MASK_PX_P10.get(cls_name, 0)
                    if mask_px < p10_thresh:
                        g_obj.attrs["skipped_reason"] = "mask_px_below_p10"
                        g_obj.attrs["mask_px_p10_threshold"] = p10_thresh
                        stats["objects_skipped_mask_px"] += 1
                        continue
                    scene_pts = depth_mask_to_pc(depth_m, mask)
                    if len(scene_pts) < MIN_SCENE_POINTS:
                        g_obj.attrs["skipped_reason"] = "too_few_scene_pts"
                        g_obj.create_dataset("scene_n_raw",
                                             data=np.int32(len(scene_pts)))
                        stats["objects_skipped_scene_pts"] += 1
                        continue

                    model_pcd, meta = load_ply(ply_name)
                    out = align_object(scene_pts, model_pcd)
                    if "skipped" in out:
                        g_obj.attrs["skipped_reason"] = out["skipped"]
                        g_obj.create_dataset("scene_n_raw",
                                             data=np.int32(out["scene_n_raw"]))
                        g_obj.create_dataset("scene_n_clean",
                                             data=np.int32(out["scene_n_clean"]))
                        stats["objects_skipped_denoise"] += 1
                        continue

                    best = out["best"]
                    T = best["T"]
                    R = T[:3, :3].astype(np.float32)
                    t = T[:3, 3].astype(np.float32)
                    q = rotmat_to_quat_wxyz(R)
                    pose7 = np.concatenate([t, q]).astype(np.float32)

                    stable = (best["fitness"] >= FITNESS_GATE
                              and best["rmse"] <= RMSE_GATE_M)

                    g_obj.attrs["best_init"] = best["init"]
                    g_obj.attrs["stable_flag"] = bool(stable)
                    g_obj.create_dataset("pose_cam", data=pose7)
                    g_obj.create_dataset("R_cam", data=R)
                    g_obj.create_dataset("t_cam", data=t)
                    g_obj.create_dataset("fitness",
                                         data=np.float32(best["fitness"]))
                    g_obj.create_dataset("inlier_rmse",
                                         data=np.float32(best["rmse"]))
                    g_obj.create_dataset("scene_n_raw",
                                         data=np.int32(out["scene_n_raw"]))
                    g_obj.create_dataset("scene_n_clean",
                                         data=np.int32(out["scene_n_clean"]))
                    g_obj.create_dataset("scene_centroid",
                                         data=out["scene_centroid"])
                    g_obj.create_dataset("model_extent",
                                         data=meta["extent"])
                    g_obj.create_dataset("model_long_axis_idx",
                                         data=np.int32(meta["long_axis"]))
                    # Candidates (for debugging, small)
                    cand_fits = np.array(
                        [(c["fitness"], c["rmse"]) for c in out["candidates"]],
                        dtype=np.float32)
                    cand_names = [c["init"] for c in out["candidates"]]
                    g_obj.create_dataset("cand_fit_rmse", data=cand_fits)
                    g_obj.attrs["cand_inits"] = cand_names

                    stats["objects_aligned"] += 1
                    per_cls["aligned"] += 1
                    if stable:
                        stats["objects_pass_gate"] += 1
                        per_cls["passed"] += 1
                    n_aligned += 1

                except Exception as e:
                    g_obj.attrs["skipped_reason"] = "error"
                    err_log.write(json.dumps({
                        "sample": sid, "object": k, "class": cls_name,
                        "error": str(e),
                        "trace": traceback.format_exc()}) + "\n")
                    stats["errors"] += 1

            g_dst.attrs["n_aligned"] = n_aligned
            if n_aligned > 0:
                stats["samples_with_objs"] += 1

            if (i + 1) % 25 == 0 or (i + 1) == n_total:
                elapsed = time.time() - start
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (n_total - (i + 1)) / max(rate, 1e-6)
                print(f"[{i + 1}/{n_total}] elapsed={elapsed:.1f}s "
                      f"eta={eta:.1f}s aligned={stats['objects_aligned']} "
                      f"passed={stats['objects_pass_gate']}", flush=True)

    err_log.close()

    REPORT_JSON.write_text(json.dumps(stats, indent=2, default=str))
    total_elapsed = time.time() - start
    print(f"\n=== DONE ({total_elapsed:.1f}s) ===")
    print(f"samples: {stats['samples_with_objs']}/{stats['samples_total']} with objects")
    print(f"objects: attempted={stats['objects_attempted']}  "
          f"aligned={stats['objects_aligned']}  "
          f"passed_gate={stats['objects_pass_gate']}")
    print(f"skipped: mask_px={stats['objects_skipped_mask_px']}  "
          f"low_conf={stats['objects_skipped_low_conf']}  "
          f"poly={stats['objects_skipped_poly']}  "
          f"scene_pts={stats['objects_skipped_scene_pts']}  "
          f"denoise={stats['objects_skipped_denoise']}  "
          f"no_ply={stats['objects_skipped_no_ply']}")
    print(f"errors: {stats['errors']}")
    print("\nper-class:")
    for cls, d in sorted(stats["per_class"].items()):
        if d["attempted"]:
            print(f"  {cls:12s} attempted={d['attempted']:4d}  "
                  f"aligned={d['aligned']:4d}  passed={d['passed']:4d}  "
                  f"pass_rate={100 * d['passed'] / d['attempted']:.1f}%")


if __name__ == "__main__":
    run()
