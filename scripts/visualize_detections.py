"""Visualize YOLO segmentation results + 3 (u,v) strategies on a diverse set of samples.

Reads from img_dataset/yolo_cache/detections.h5 and the original RGB images.
Produces visualizations at runs/detection_viz/.

Picks one image per scene to cover diversity (random1 ~ random6).
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import h5py
import numpy as np

PROJECT_ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
H5_PATH = PROJECT_ROOT / "img_dataset/yolo_cache/detections.h5"
OUT_DIR = PROJECT_ROOT / "runs/detection_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# cube_* 는 학습 시엔 별도 클래스였으나 하류에서는 depth 형상상 동일 cube 한 종류로 취급.
CUBE_CLASSES = {"cube_blue", "cube_green", "cube_p", "cube_red"}

CLASS_COLORS = {
    "bottle":     (96, 160, 255),
    "can":        (255, 180, 60),
    "marker":     (220, 60, 180),
    "spam":       (60, 220, 120),
    "cube":       (80, 240, 240),  # 통합
}


def unified_class_name(raw: str) -> str:
    return "cube" if raw in CUBE_CLASSES else raw


def poly_to_array(poly_flat: np.ndarray) -> np.ndarray:
    """(2K,) float32 → (K,2) int32"""
    if poly_flat.size < 4:
        return np.empty((0, 2), dtype=np.int32)
    return poly_flat.reshape(-1, 2).astype(np.int32)


def draw_sample(rgb: np.ndarray, sample: h5py.Group, class_names: list[str]) -> np.ndarray:
    canvas = rgb.copy()
    overlay = rgb.copy()

    classes = sample["classes"][:]
    confs = sample["confidences"][:]
    bboxes = sample["bboxes"][:]
    uv_c = sample["uv_centroid"][:]
    uv_b = sample["uv_bbox_ctr"][:]
    uv_d = sample["uv_dt_peak"][:]
    aspects = sample["pca_aspect"][:]
    long_axes = sample["pca_long_axis"][:]
    polys = sample["mask_poly"][:]

    for k in range(len(classes)):
        raw_name = class_names[int(classes[k])]
        disp = unified_class_name(raw_name)
        color = CLASS_COLORS.get(disp, (200, 200, 200))

        # Fill polygon (semi-transparent mask)
        pts = poly_to_array(polys[k])
        if pts.shape[0] >= 3:
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

        # bbox
        x1, y1, x2, y2 = bboxes[k].astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)

        # 3 uv points — distinct, clearly labeled
        for point, pt_color, label in [
            (uv_c[k], (0, 0, 255),  "C"),   # red
            (uv_b[k], (255, 100, 0), "B"),  # blue (BGR)
            (uv_d[k], (0, 255, 255), "D"),  # yellow
        ]:
            if not np.any(np.isnan(point)):
                px, py = int(point[0]), int(point[1])
                cv2.circle(canvas, (px, py), 7, pt_color, -1)
                cv2.circle(canvas, (px, py), 9, (0, 0, 0), 1)
                cv2.putText(canvas, label, (px + 10, py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, pt_color, 2)

        # PCA long axis line (thin)
        if not np.any(np.isnan(long_axes[k])) and pts.shape[0] >= 3:
            center = uv_c[k]
            # length heuristic from bbox diagonal
            diag = np.hypot(x2 - x1, y2 - y1) * 0.45
            p1 = (center - diag * long_axes[k]).astype(int)
            p2 = (center + diag * long_axes[k]).astype(int)
            cv2.line(canvas, tuple(p1), tuple(p2), (0, 255, 0), 1)

        # label: class (unified) + conf + aspect
        asp_str = f"{aspects[k]:.2f}" if np.isfinite(aspects[k]) else "NaN"
        text = f"{disp} {confs[k]:.2f} a={asp_str}"
        cv2.putText(canvas, text, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # blend overlay
    canvas = cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0)

    # Legend
    cv2.rectangle(canvas, (8, 8), (415, 118), (0, 0, 0), -1)
    cv2.putText(canvas, "Red C  = Centroid",          (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255),  2)
    cv2.putText(canvas, "Blue B = Bbox center",       (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2)
    cv2.putText(canvas, "Yellow D = DT peak",         (14, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(canvas, "Green line = PCA long axis", (14, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 255, 0),   2)

    # Footer: image dims
    h, w = canvas.shape[:2]
    cv2.putText(canvas, f"{w}x{h} (native 1280x720)", (w - 260, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas


def pick_diverse_samples(f: h5py.File, per_scene: int = 2) -> list[str]:
    """Pick `per_scene` samples per scene (random1..random6) among those with detections."""
    random.seed(7)
    buckets: dict[str, list[str]] = {}
    for name in f.keys():
        if not name.startswith("sample_"):
            continue
        grp = f[name]
        if not grp.attrs.get("detected", False):
            continue
        scene = grp.attrs.get("scene")
        if isinstance(scene, bytes):
            scene = scene.decode()
        buckets.setdefault(scene, []).append(name)
    out = []
    for scene in sorted(buckets.keys()):
        items = buckets[scene]
        out.extend(random.sample(items, k=min(per_scene, len(items))))
    return out


def main():
    with h5py.File(H5_PATH, "r") as f:
        class_names = [s.decode() if isinstance(s, bytes) else s
                       for s in f.attrs["class_names"]]
        picks = pick_diverse_samples(f, per_scene=2)
        print(f"[viz] {len(picks)} samples across {len(set(f[p].attrs['scene'] for p in picks))} scenes")

        for name in picks:
            grp = f[name]
            rgb_path = PROJECT_ROOT / grp.attrs["rgb_path"]
            rgb = cv2.imread(str(rgb_path))
            if rgb is None:
                print(f"  skip {name}: cannot read {rgb_path}")
                continue
            canvas = draw_sample(rgb, grp, class_names)
            sid = name.replace("sample_", "")
            out_path = OUT_DIR / f"{sid}.png"
            cv2.imwrite(str(out_path), canvas)
            print(f"  ✓ {sid} → {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
