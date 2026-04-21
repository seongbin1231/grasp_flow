"""v3: YOLOv8m-seg v3 (finetuned on captured 1280×720) re-inference.

Same logic/schema as batch_yolo_cache.py (v1), but:
  - Model: runs/yolov8m_seg_v3_1280/weights/best.pt
  - Output: img_dataset/yolo_cache_v3/
  - Image size: native 1280×720 (no pre-resize)

Used for sanity-check after finetune — expect higher recall on
marker/cube_blue and fewer all_below_conf failures.
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
MODEL_PATH = PROJECT_ROOT / "runs/yolov8m_seg_v3_1280/weights/best.pt"
IMG_ROOT = PROJECT_ROOT / "img_dataset/captured_images"
DEPTH_ROOT = PROJECT_ROOT / "img_dataset/captured_images_depth"
OUT_DIR = PROJECT_ROOT / "img_dataset/yolo_cache_v3"
PROBLEMS_DIR = OUT_DIR / "problems"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROBLEMS_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.7
IMGSZ = 1280  # v3 trained at 1280
IMG_H, IMG_W = 720, 1280


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


def draw_problem_preview(rgb: np.ndarray, reason: str, detections: list) -> np.ndarray:
    canvas = rgb.copy()
    overlay = rgb.copy()
    for d in detections:
        mask = d["mask_bool"]
        color = [80, 200, 255]
        overlay[mask] = (0.4 * np.array(color) + 0.6 * overlay[mask]).astype(np.uint8)
        x1, y1, x2, y2 = d["bbox"].astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{d['class_name']} {d['conf']:.2f}",
                    (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
    cv2.rectangle(canvas, (10, 10), (720, 46), (0, 0, 0), -1)
    cv2.putText(canvas, f"[v3 PROBLEM] {reason}", (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    return canvas


def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] v3 best.pt not found: {MODEL_PATH}")
        return

    imgs = collect_images()
    print(f"[found] {len(imgs)} images")
    model = YOLO(str(MODEL_PATH))
    names = model.names
    print(f"[model] {MODEL_PATH.name}, classes: {names}")

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
        f.attrs["image_h"] = IMG_H
        f.attrs["image_w"] = IMG_W
        f.attrs["coord_frame"] = "native_1280x720"
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
                bgr = cv2.imread(str(img_path))
                if bgr is None or bgr.shape[:2] != (IMG_H, IMG_W):
                    raise RuntimeError(f"bad rgb shape")
                # ultralytics: numpy 입력은 BGR로 가정 (소스: predictor.py preprocess).
                # cv2.imread가 이미 BGR이므로 그대로 전달. RGB 변환 금지.
                results = model.predict(source=bgr, imgsz=IMGSZ, conf=0.05, iou=IOU_THRESHOLD,
                                        verbose=False, device=0)
                res = results[0]

                dets_pass, dets_below = [], []
                if res.masks is not None and len(res.boxes) > 0:
                    masks_np = res.masks.data.cpu().numpy()
                    if masks_np.shape[1:] != (IMG_H, IMG_W):
                        masks_np = np.stack([
                            cv2.resize(m.astype(np.float32), (IMG_W, IMG_H),
                                       interpolation=cv2.INTER_NEAREST) > 0.5
                            for m in masks_np])
                    else:
                        masks_np = masks_np > 0.5
                    boxes = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    classes = res.boxes.cls.cpu().numpy().astype(int)
                    for k in range(len(classes)):
                        cls_name = names[int(classes[k])]
                        entry = {
                            "mask_bool": masks_np[k],
                            "bbox": boxes[k],
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
                    problems.append((sid, bgr, "no_detection", []))
                elif len(dets_pass) == 0 and len(dets_below) > 0:
                    filtered_only_samples.append(sid)
                    problems.append((sid, bgr, "all_below_conf0.5", dets_below))
                per_img_detect_count.append(len(dets_pass))
                if len(dets_pass) >= 6:
                    problems.append((sid, bgr, f"many_detections({len(dets_pass)})", dets_pass))

                if not dets_pass:
                    continue

                N = len(dets_pass)
                classes_arr = np.array([d["class_id"] for d in dets_pass], dtype=np.int32)
                confs_arr = np.array([d["conf"] for d in dets_pass], dtype=np.float32)
                bboxes_arr = np.stack([d["bbox"] for d in dets_pass]).astype(np.float32)
                uv_c = np.zeros((N, 2), dtype=np.float32)
                uv_b = np.zeros((N, 2), dtype=np.float32)
                uv_d = np.zeros((N, 2), dtype=np.float32)
                aspects = np.zeros((N,), dtype=np.float32)
                long_axes = np.zeros((N, 2), dtype=np.float32)
                polys = []
                for k, d in enumerate(dets_pass):
                    c, b, dp = three_uv(d["mask_bool"])
                    uv_c[k] = c; uv_b[k] = b; uv_d[k] = dp
                    asp, la = pca_lite(d["mask_bool"])
                    aspects[k] = asp
                    long_axes[k] = la
                    if asp > 3: aspect_bucket["lying(>3)"] += 1
                    elif asp >= 1.5: aspect_bucket["mid(1.5-3)"] += 1
                    else: aspect_bucket["low(<1.5)"] += 1
                    contours, _ = cv2.findContours(d["mask_bool"].astype(np.uint8),
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        c_big = max(contours, key=cv2.contourArea).reshape(-1, 2)
                        polys.append(np.ascontiguousarray(c_big.ravel(), dtype=np.float32))
                    else:
                        polys.append(np.zeros(2, dtype=np.float32))

                g.create_dataset("classes", data=classes_arr)
                g.create_dataset("confidences", data=confs_arr)
                g.create_dataset("bboxes", data=bboxes_arr)
                g.create_dataset("uv_centroid", data=uv_c)
                g.create_dataset("uv_bbox_ctr", data=uv_b)
                g.create_dataset("uv_dt_peak", data=uv_d)
                g.create_dataset("pca_aspect", data=aspects)
                g.create_dataset("pca_long_axis", data=long_axes)
                mp_ds = g.create_dataset("mask_poly", shape=(N,), dtype=vlen_f32)
                for k in range(N):
                    mp_ds[k] = polys[k]

            except Exception as exc:
                elog.write(json.dumps({
                    "sample_id": sid, "image": str(img_path),
                    "error": str(exc),
                    "trace": traceback.format_exc(limit=3),
                }) + "\n")
                problems.append((sid, None, f"error: {exc}", []))

            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{len(imgs)}] {(time.time()-t0):.1f}s")

    for sid, bgr, reason, dets in problems[:30]:
        if bgr is None: continue
        canvas = draw_problem_preview(bgr, reason, dets)
        safe = reason.replace(" ", "_").replace(":", "-").replace("(", "").replace(")", "")
        cv2.imwrite(str(PROBLEMS_DIR / f"{sid}__{safe}.png"), canvas)

    summary = {
        "total_images": len(imgs),
        "elapsed_sec": round(time.time() - t0, 1),
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

    print(f"\n[done v3] {len(imgs)} images, {summary['elapsed_sec']}s")
    print(f"[class passed]   {class_counts_passed}")
    print(f"[class filtered] {class_counts_filtered}")
    print(f"[aspect]         {aspect_bucket}")
    print(f"[per-img]        {summary['per_image_detect_count_stats']}")
    print(f"[no detection]   {len(no_detect_samples)} / [all below] {len(filtered_only_samples)}")


if __name__ == "__main__":
    main()
