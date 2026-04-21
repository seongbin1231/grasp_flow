"""v2: mimic training preprocessing before YOLO inference.

Training data pipeline (Roboflow):
  1280×720 PNG (source)  →  Fit within 640×640  →  640×360 JPG q75

This script reproduces that preprocessing for each captured image before
feeding it to YOLO. Bbox/mask/(u,v) coordinates returned by YOLO are in the
preprocessed 640×360 frame; we upscale by 2× back to 1280×720 so downstream
stages (ICP, grasp synth) stay in native Simulink resolution.

Outputs:
  img_dataset/yolo_cache_v2/
    ├── detections.h5     (same schema as v1, coords in 1280×720)
    ├── summary.json
    ├── errors.jsonl
    └── problems/         (no_detect / all_below_conf / many_detections)
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt
from ultralytics import YOLO

PROJECT_ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
MODEL_PATH = PROJECT_ROOT / "runs/yolov8m_seg_v2/weights/best.pt"
IMG_ROOT = PROJECT_ROOT / "img_dataset/captured_images"
DEPTH_ROOT = PROJECT_ROOT / "img_dataset/captured_images_depth"
OUT_DIR = PROJECT_ROOT / "img_dataset/yolo_cache_v2"
PROBLEMS_DIR = OUT_DIR / "problems"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROBLEMS_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.7
IMGSZ = 640
JPG_QUALITY = 75  # Roboflow default
NATIVE_H, NATIVE_W = 720, 1280
PROC_H, PROC_W = 360, 640
SCALE = NATIVE_W / PROC_W  # = 2.0


def preprocess_roboflow_style(bgr_native: np.ndarray) -> np.ndarray:
    """1280×720 BGR → 640×360 JPG-encoded/decoded BGR (Roboflow 'Fit within 640×640')."""
    resized = cv2.resize(bgr_native, (PROC_W, PROC_H), interpolation=cv2.INTER_LINEAR)
    ok, enc = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
    if not ok:
        return resized
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def three_uv(mask_bool: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        nan = np.array([np.nan, np.nan], dtype=np.float32)
        return nan.copy(), nan.copy(), nan.copy()
    centroid = np.array([xs.mean(), ys.mean()], dtype=np.float32)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    bbox_ctr = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    dt = distance_transform_edt(mask_bool)
    py, px = np.unravel_index(dt.argmax(), dt.shape)
    dt_peak = np.array([px, py], dtype=np.float32)
    return centroid, bbox_ctr, dt_peak


def pca_lite(mask_bool: np.ndarray) -> tuple[float, np.ndarray]:
    ys, xs = np.where(mask_bool)
    if xs.size < 10:
        return float("nan"), np.array([np.nan, np.nan], dtype=np.float32)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_axis = eigvecs[:, -1].astype(np.float32)
    proj_long = centered @ eigvecs[:, -1]
    proj_short = centered @ eigvecs[:, 0]
    long_len = float(proj_long.max() - proj_long.min())
    short_len = float(proj_short.max() - proj_short.min())
    aspect = long_len / max(short_len, 1e-6)
    return aspect, long_axis


def resolve_depth_path(rgb_path: Path) -> Path:
    scene = rgb_path.parent.name
    idx = rgb_path.stem.split("_")[-1]
    return DEPTH_ROOT / f"{scene}_dep" / f"{scene}_depth_{idx}.png"


def collect_images() -> list[Path]:
    imgs = []
    for scene_dir in sorted(IMG_ROOT.iterdir()):
        if scene_dir.is_dir():
            imgs.extend(sorted(scene_dir.glob("*.png")))
    return imgs


def draw_problem_preview(rgb_native: np.ndarray, reason: str, detections: list) -> np.ndarray:
    canvas = rgb_native.copy()
    overlay = rgb_native.copy()
    for d in detections:
        mask = d["mask_bool_native"]
        color = [80, 200, 255]
        overlay[mask] = (0.4 * np.array(color) + 0.6 * overlay[mask]).astype(np.uint8)
        x1, y1, x2, y2 = d["bbox_native"].astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{d['class_name']} {d['conf']:.2f}",
                    (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
    cv2.rectangle(canvas, (10, 10), (720, 46), (0, 0, 0), -1)
    cv2.putText(canvas, f"[PROBLEM v2] {reason}", (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    return canvas


def main():
    imgs = collect_images()
    print(f"[found] {len(imgs)} images")
    model = YOLO(str(MODEL_PATH))
    names = model.names
    out_h5 = OUT_DIR / "detections.h5"
    errors_path = OUT_DIR / "errors.jsonl"
    errors_path.unlink(missing_ok=True)
    problems = []

    class_counts_passed = {v: 0 for v in names.values()}
    class_counts_filtered = {v: 0 for v in names.values()}
    aspect_bucket = {"lying(>3)": 0, "mid(1.5-3)": 0, "low(<1.5)": 0}
    per_img_detect_count = []
    no_detect_samples = []
    filtered_only_samples = []

    t0 = time.time()
    with h5py.File(out_h5, "w") as f, errors_path.open("w") as elog:
        f.attrs["camera_fx"] = 1109.0
        f.attrs["camera_fy"] = 1109.0
        f.attrs["camera_cx"] = 640.0
        f.attrs["camera_cy"] = 360.0
        f.attrs["image_h"] = NATIVE_H
        f.attrs["image_w"] = NATIVE_W
        f.attrs["coord_frame"] = "native_1280x720"
        f.attrs["preprocessing"] = "roboflow_640x360_jpg_q75"
        f.attrs["conf_threshold"] = CONF_THRESHOLD
        f.attrs["imgsz"] = IMGSZ
        f.attrs["model_path"] = str(MODEL_PATH.relative_to(PROJECT_ROOT))
        f.attrs["class_names"] = np.array(
            [names[i] for i in range(len(names))], dtype=h5py.string_dtype())

        vlen_f32 = h5py.vlen_dtype(np.dtype("float32"))

        for idx, img_path in enumerate(imgs):
            sid = img_path.stem
            depth_path = resolve_depth_path(img_path)
            try:
                bgr_native = cv2.imread(str(img_path))
                if bgr_native is None or bgr_native.shape[:2] != (NATIVE_H, NATIVE_W):
                    raise RuntimeError(f"bad rgb shape {None if bgr_native is None else bgr_native.shape}")

                bgr_proc = preprocess_roboflow_style(bgr_native)
                rgb_proc = cv2.cvtColor(bgr_proc, cv2.COLOR_BGR2RGB)

                results = model.predict(source=rgb_proc, imgsz=IMGSZ, conf=0.05, iou=IOU_THRESHOLD,
                                        verbose=False, device=0)
                res = results[0]

                dets_pass, dets_below = [], []
                if res.masks is not None and len(res.boxes) > 0:
                    # masks are in model-internal size; resize to 640×360 (proc frame)
                    masks_np = res.masks.data.cpu().numpy()
                    if masks_np.shape[1:] != (PROC_H, PROC_W):
                        masks_np = np.stack([
                            cv2.resize(m.astype(np.float32), (PROC_W, PROC_H),
                                       interpolation=cv2.INTER_NEAREST) > 0.5
                            for m in masks_np])
                    else:
                        masks_np = masks_np > 0.5

                    boxes_proc = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    classes = res.boxes.cls.cpu().numpy().astype(int)

                    for k in range(len(classes)):
                        cls_name = names[int(classes[k])]
                        # Upsample mask to native 1280×720
                        mask_native = cv2.resize(
                            masks_np[k].astype(np.uint8), (NATIVE_W, NATIVE_H),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
                        bbox_native = boxes_proc[k] * SCALE
                        entry = {
                            "mask_bool_proc": masks_np[k],
                            "mask_bool_native": mask_native,
                            "bbox_proc": boxes_proc[k],
                            "bbox_native": bbox_native,
                            "conf": float(confs[k]),
                            "class_id": int(classes[k]),
                            "class_name": cls_name,
                        }
                        if confs[k] >= CONF_THRESHOLD:
                            dets_pass.append(entry)
                            class_counts_passed[cls_name] += 1
                        else:
                            dets_below.append(entry)
                            class_counts_filtered[cls_name] += 1

                g = f.create_group(f"sample_{sid}")
                g.attrs["scene"] = img_path.parent.name
                g.attrs["rgb_path"] = str(img_path.relative_to(PROJECT_ROOT))
                g.attrs["depth_path"] = str(depth_path.relative_to(PROJECT_ROOT)) if depth_path.exists() else ""
                g.attrs["depth_available"] = depth_path.exists()
                g.attrs["n_above_threshold"] = len(dets_pass)
                g.attrs["n_below_threshold"] = len(dets_below)
                g.attrs["detected"] = len(dets_pass) > 0

                if len(dets_pass) == 0 and len(dets_below) == 0:
                    no_detect_samples.append(sid)
                    problems.append((sid, bgr_native, "no_detection", []))
                elif len(dets_pass) == 0 and len(dets_below) > 0:
                    filtered_only_samples.append(sid)
                    problems.append((sid, bgr_native, "all_below_conf0.5", dets_below))
                per_img_detect_count.append(len(dets_pass))
                if len(dets_pass) >= 5:
                    problems.append((sid, bgr_native, f"many_detections({len(dets_pass)})", dets_pass))

                if not dets_pass:
                    continue

                N = len(dets_pass)
                classes_arr = np.array([d["class_id"] for d in dets_pass], dtype=np.int32)
                confs_arr = np.array([d["conf"] for d in dets_pass], dtype=np.float32)
                bboxes_native = np.stack([d["bbox_native"] for d in dets_pass]).astype(np.float32)
                uv_c = np.zeros((N, 2), dtype=np.float32)
                uv_b = np.zeros((N, 2), dtype=np.float32)
                uv_d = np.zeros((N, 2), dtype=np.float32)
                aspects = np.zeros((N,), dtype=np.float32)
                long_axes = np.zeros((N, 2), dtype=np.float32)
                polys_native = []

                for k, d in enumerate(dets_pass):
                    # compute uv + PCA on native-resolution mask for final coords
                    c, b, dp = three_uv(d["mask_bool_native"])
                    uv_c[k] = c; uv_b[k] = b; uv_d[k] = dp
                    asp, la = pca_lite(d["mask_bool_native"])
                    aspects[k] = asp
                    long_axes[k] = la
                    if asp > 3: aspect_bucket["lying(>3)"] += 1
                    elif asp >= 1.5: aspect_bucket["mid(1.5-3)"] += 1
                    else: aspect_bucket["low(<1.5)"] += 1

                    contours, _ = cv2.findContours(d["mask_bool_native"].astype(np.uint8),
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        c_big = max(contours, key=cv2.contourArea).reshape(-1, 2)
                        polys_native.append(np.ascontiguousarray(c_big.ravel(), dtype=np.float32))
                    else:
                        polys_native.append(np.zeros(2, dtype=np.float32))

                g.create_dataset("classes", data=classes_arr)
                g.create_dataset("confidences", data=confs_arr)
                g.create_dataset("bboxes", data=bboxes_native)
                g.create_dataset("uv_centroid", data=uv_c)
                g.create_dataset("uv_bbox_ctr", data=uv_b)
                g.create_dataset("uv_dt_peak", data=uv_d)
                g.create_dataset("pca_aspect", data=aspects)
                g.create_dataset("pca_long_axis", data=long_axes)
                mp_ds = g.create_dataset("mask_poly", shape=(N,), dtype=vlen_f32)
                for k in range(N):
                    mp_ds[k] = polys_native[k]

            except Exception as exc:
                elog.write(json.dumps({
                    "sample_id": sid,
                    "image": str(img_path),
                    "error": str(exc),
                    "trace": traceback.format_exc(limit=3),
                }) + "\n")
                problems.append((sid, None, f"error: {exc}", []))

            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{len(imgs)}] {(time.time()-t0):.1f}s")

    for sid, rgb_bgr, reason, dets in problems[:30]:
        if rgb_bgr is None: continue
        canvas = draw_problem_preview(rgb_bgr, reason, dets)
        safe = reason.replace(" ", "_").replace(":", "-").replace("(", "").replace(")", "")
        cv2.imwrite(str(PROBLEMS_DIR / f"{sid}__{safe}.png"), canvas)

    summary = {
        "total_images": len(imgs),
        "elapsed_sec": round(time.time() - t0, 1),
        "preprocessing": "roboflow_640x360_jpg_q75",
        "coord_frame": "native_1280x720",
        "conf_threshold": CONF_THRESHOLD,
        "class_counts_passed": class_counts_passed,
        "class_counts_filtered_below_threshold": class_counts_filtered,
        "aspect_buckets_passed": aspect_bucket,
        "per_image_detect_count_stats": {
            "mean": float(np.mean(per_img_detect_count)) if per_img_detect_count else 0,
            "median": float(np.median(per_img_detect_count)) if per_img_detect_count else 0,
            "p90": float(np.percentile(per_img_detect_count, 90)) if per_img_detect_count else 0,
            "max": int(np.max(per_img_detect_count)) if per_img_detect_count else 0,
            "zero_detect_count": int(sum(1 for x in per_img_detect_count if x == 0)),
        },
        "samples_no_detection": no_detect_samples[:50],
        "samples_all_below_threshold": filtered_only_samples[:50],
    }
    with (OUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done v2] {len(imgs)} images, {summary['elapsed_sec']}s")
    print(f"[class passed]    {class_counts_passed}")
    print(f"[class filtered]  {class_counts_filtered}")
    print(f"[aspect buckets]  {aspect_bucket}")
    print(f"[per-img]         {summary['per_image_detect_count_stats']}")
    print(f"[no detection]    {len(no_detect_samples)} / [all below]  {len(filtered_only_samples)}")


if __name__ == "__main__":
    main()
