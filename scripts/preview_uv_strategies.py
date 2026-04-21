"""Preview (u,v) extraction strategies on a few captured images.

Visual verification before batch inference. Produces:
  runs/uv_preview/{image_name}.png  — RGB with masks + 3 (u,v) points + PCA axes
  runs/uv_preview/summary.json      — per-detection numeric info
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from ultralytics import YOLO

PROJECT_ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
MODEL_PATH = PROJECT_ROOT / "runs/yolov8m_seg_v2/weights/best.pt"
IMG_ROOT = PROJECT_ROOT / "img_dataset/captured_images"
OUT_DIR = PROJECT_ROOT / "runs/uv_preview"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_K = np.array(
    [[1109.0, 0.0, 640.0], [0.0, 1109.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float32
)
IMG_H, IMG_W = 720, 1280
N_LONG_AXIS_SAMPLES = 4  # lying object: how many points along principal axis


def three_uv(mask_bool: np.ndarray) -> dict:
    """Compute centroid / bbox_center / distance-transform peak."""
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        nan = np.array([np.nan, np.nan], dtype=np.float32)
        return {"centroid": nan, "bbox_ctr": nan, "dt_peak": nan}
    centroid = np.array([xs.mean(), ys.mean()], dtype=np.float32)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    bbox_ctr = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    dt = distance_transform_edt(mask_bool)
    py, px = np.unravel_index(dt.argmax(), dt.shape)
    dt_peak = np.array([px, py], dtype=np.float32)
    return {"centroid": centroid, "bbox_ctr": bbox_ctr, "dt_peak": dt_peak}


def pca_axes_and_samples(mask_bool: np.ndarray, n_samples: int = N_LONG_AXIS_SAMPLES) -> dict:
    """PCA on mask pixels → principal axis + N interpolated samples along it.

    Returns {long_axis, short_axis, long_len, short_len, samples (n,2), aspect}.
    """
    ys, xs = np.where(mask_bool)
    if xs.size < 10:
        return {}
    pts = np.stack([xs, ys], axis=1).astype(np.float32)  # (M,2)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    # Largest eigenvalue -> principal (long) axis
    long_axis = eigvecs[:, -1]
    short_axis = eigvecs[:, 0]
    # Project points onto axes to get extents
    proj_long = centered @ long_axis
    proj_short = centered @ short_axis
    long_len = float(proj_long.max() - proj_long.min())
    short_len = float(proj_short.max() - proj_short.min())
    aspect = long_len / max(short_len, 1e-6)
    # Samples along long axis, exclude 10% at each end
    t_min, t_max = proj_long.min(), proj_long.max()
    t_margin = 0.1 * (t_max - t_min)
    ts = np.linspace(t_min + t_margin, t_max - t_margin, n_samples)
    samples = mean[None, :] + ts[:, None] * long_axis[None, :]  # (n,2)
    return {
        "long_axis": long_axis.astype(np.float32),
        "short_axis": short_axis.astype(np.float32),
        "long_len": long_len,
        "short_len": short_len,
        "aspect": aspect,
        "mean": mean.astype(np.float32),
        "samples": samples.astype(np.float32),
    }


def draw_overlay(rgb: np.ndarray, detections: list) -> np.ndarray:
    """Draw masks (semi-transparent), bboxes, 3 uv points, and PCA long axis."""
    canvas = rgb.copy()
    overlay = rgb.copy()
    rng = np.random.default_rng(0)

    for det in detections:
        color = rng.integers(80, 255, size=3).tolist()
        mask = det["mask_bool"]
        # semi-transparent mask
        overlay[mask] = (0.5 * np.array(color) + 0.5 * overlay[mask]).astype(np.uint8)

        # bbox
        x1, y1, x2, y2 = det["bbox"].astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # 3 uv points
        uvs = det["uvs"]
        for label, point, pt_color in [
            ("C", uvs["centroid"], (0, 0, 255)),  # red
            ("B", uvs["bbox_ctr"], (255, 0, 0)),  # blue
            ("D", uvs["dt_peak"], (0, 255, 255)),  # yellow
        ]:
            if not np.any(np.isnan(point)):
                px, py = int(point[0]), int(point[1])
                cv2.circle(canvas, (px, py), 6, pt_color, -1)
                cv2.circle(canvas, (px, py), 8, (0, 0, 0), 1)
                cv2.putText(canvas, label, (px + 8, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, pt_color, 2)

        # PCA long-axis samples (green)
        pca = det.get("pca", {})
        if pca and "samples" in pca:
            mean = pca["mean"].astype(int)
            half = 0.5 * pca["long_len"] * pca["long_axis"]
            p1 = (mean - half).astype(int)
            p2 = (mean + half).astype(int)
            cv2.line(canvas, tuple(p1), tuple(p2), (0, 255, 0), 2)
            for s in pca["samples"].astype(int):
                cv2.circle(canvas, tuple(s), 4, (0, 255, 0), -1)

        # class label
        cls_name = det["class_name"]
        conf = det["conf"]
        cv2.putText(canvas, f"{cls_name} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # blend overlay with mask alpha
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)

    # legend
    legend_y = 30
    cv2.rectangle(canvas, (10, 10), (330, 130), (0, 0, 0), -1)
    cv2.putText(canvas, "Red C = Centroid", (20, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(canvas, "Blue B = Bbox center", (20, legend_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
    cv2.putText(canvas, "Yellow D = DT peak", (20, legend_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(canvas, "Green = PCA long axis + samples", (20, legend_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return canvas


def process_image(model: YOLO, img_path: Path) -> dict:
    rgb = cv2.imread(str(img_path))  # BGR
    rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    assert rgb.shape[:2] == (IMG_H, IMG_W), f"Unexpected shape: {rgb.shape}"

    # Predict with training imgsz=640 (model trained on 640×360 letterboxed to 640×640).
    # Passing the raw 1280×720 and letting ultralytics letterbox matches the training
    # distribution; masks/boxes come back in original 1280×720 coordinates.
    results = model.predict(source=rgb_rgb, imgsz=640, conf=0.25, iou=0.7,
                            verbose=False, device=0)
    res = results[0]

    detections = []
    if res.masks is None or len(res.boxes) == 0:
        return {"image": img_path.name, "n_detected": 0, "detections": []}

    masks_np = res.masks.data.cpu().numpy()  # (N, h, w) bool-like
    # Ultralytics masks may be at model resolution — resize to image
    if masks_np.shape[1:] != (IMG_H, IMG_W):
        resized = []
        for m in masks_np:
            m_rs = cv2.resize(m.astype(np.float32), (IMG_W, IMG_H),
                              interpolation=cv2.INTER_NEAREST)
            resized.append(m_rs > 0.5)
        masks_np = np.stack(resized)
    else:
        masks_np = masks_np > 0.5

    boxes = res.boxes.xyxy.cpu().numpy()  # (N,4)
    confs = res.boxes.conf.cpu().numpy()
    classes = res.boxes.cls.cpu().numpy().astype(int)
    names = res.names

    for k in range(len(classes)):
        mask = masks_np[k]
        uvs = three_uv(mask)
        pca = pca_axes_and_samples(mask)
        detections.append({
            "mask_bool": mask,
            "bbox": boxes[k],
            "conf": float(confs[k]),
            "class_id": int(classes[k]),
            "class_name": names[int(classes[k])],
            "uvs": uvs,
            "pca": pca,
        })

    overlay = draw_overlay(rgb, detections)
    out_path = OUT_DIR / f"{img_path.stem}_preview.png"
    cv2.imwrite(str(out_path), overlay)

    summary = {
        "image": img_path.name,
        "n_detected": len(detections),
        "output_preview": str(out_path.relative_to(PROJECT_ROOT)),
        "detections": [
            {
                "class_name": d["class_name"],
                "conf": d["conf"],
                "bbox": d["bbox"].tolist(),
                "uv_centroid": d["uvs"]["centroid"].tolist(),
                "uv_bbox_ctr": d["uvs"]["bbox_ctr"].tolist(),
                "uv_dt_peak": d["uvs"]["dt_peak"].tolist(),
                "pca_aspect": float(d["pca"].get("aspect", -1)) if d["pca"] else None,
                "pca_long_len_px": float(d["pca"].get("long_len", -1)) if d["pca"] else None,
                "pca_short_len_px": float(d["pca"].get("short_len", -1)) if d["pca"] else None,
                "pca_samples": d["pca"]["samples"].tolist() if d["pca"] else None,
            }
            for d in detections
        ],
    }
    return summary


def main():
    random.seed(42)
    model = YOLO(str(MODEL_PATH))
    print(f"[load] {MODEL_PATH}")
    print(f"[classes] {model.names}")

    # Pick 2 images from different scenes for diversity
    scene_dirs = sorted([p for p in IMG_ROOT.iterdir() if p.is_dir()])
    picked = []
    for sd in random.sample(scene_dirs, min(2, len(scene_dirs))):
        imgs = sorted(sd.glob("*.png"))
        if imgs:
            picked.append(random.choice(imgs))
    print(f"[preview] running on {len(picked)} images")

    summaries = []
    for p in picked:
        s = process_image(model, p)
        summaries.append(s)
        print(f"  ✓ {p.name}: {s['n_detected']} objects → {s.get('output_preview', 'N/A')}")

    out_json = OUT_DIR / "summary.json"
    with out_json.open("w") as f:
        json.dump({"camera_K": CAMERA_K.tolist(), "image_size": [IMG_H, IMG_W],
                   "model": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
                   "summaries": summaries}, f, indent=2)
    print(f"[save] {out_json}")


if __name__ == "__main__":
    main()
