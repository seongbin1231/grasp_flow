"""Why does YOLO work well on valid/ but poorly on captured/?

Hypothesis: the model was trained on 640×360 JPG (Roboflow-preprocessed).
Captured images are 1280×720 PNG. Even though Ultralytics letterboxes to 640,
the combination of (a) different resize algorithm and (b) PNG vs JPG compression
may shift the input distribution enough to degrade detection.

Test: take ONE image name that exists in both valid/ and captured/.
Run inference under 4 conditions:
  A) original captured (1280×720 PNG) + imgsz=640  [= detection_viz path]
  B) captured → resize 640×360 PNG + imgsz=640
  C) captured → resize 640×360 JPG(q=75) + imgsz=640  [mimic Roboflow]
  D) valid JPG directly + imgsz=640                   [= test_center path]

Compare: #detections, per-class confidence, bbox positions.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT = Path("/home/robotics/Competition/YOLO_Grasp")
MODEL = PROJECT / "runs/yolov8m_seg_v2/weights/best.pt"
CAP_ROOT = PROJECT / "img_dataset/captured_images"
VAL_DIR = PROJECT / "dataset/valid/images"
OUT_DIR = PROJECT / "runs/diagnose_preproc"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_pair() -> tuple[Path, Path] | None:
    """Find an image that exists both in valid/ (640×360 JPG) and captured/ (1280×720 PNG)."""
    # collect captured stems
    cap_map = {}
    for scene in CAP_ROOT.iterdir():
        if scene.is_dir():
            for f in scene.glob("*.png"):
                cap_map[f.stem] = f

    # valid files look like: random3_33_png.rf.<hash>.jpg  → stem random3_33
    for f in sorted(VAL_DIR.glob("*.jpg")):
        m = re.match(r"(random\d+_\d+)_png\.rf\.", f.name)
        if not m:
            continue
        stem = m.group(1)
        if stem in cap_map:
            return cap_map[stem], f
    return None


def run_yolo(model: YOLO, img: np.ndarray, tag: str) -> dict:
    res = model.predict(source=img, imgsz=640, conf=0.05, iou=0.7, verbose=False, device=0)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return {"tag": tag, "n": 0, "detections": [], "annotated": img}
    classes = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    boxes = res.boxes.xyxy.cpu().numpy()
    names = res.names
    det = [{"class": names[int(c)], "conf": float(p),
            "bbox": tuple(round(float(x), 1) for x in b)}
           for c, p, b in zip(classes, confs, boxes)]
    annotated = res.plot()
    return {"tag": tag, "n": len(det), "detections": det, "annotated": annotated}


def main():
    pair = find_pair()
    if pair is None:
        print("no overlap")
        return
    cap_path, val_path = pair
    print(f"[pair] cap: {cap_path.name} ({cap_path.parent.name})")
    print(f"[pair] val: {val_path.name}")

    # Read
    cap_bgr = cv2.imread(str(cap_path))                 # 1280x720 PNG
    val_bgr = cv2.imread(str(val_path))                 # 640x360 JPG

    # B) resize captured to 640x360 PNG-equivalent bytes (just pass array)
    cap_resized_png = cv2.resize(cap_bgr, (640, 360), interpolation=cv2.INTER_LINEAR)

    # C) mimic Roboflow: resize 640x360 → encode JPG q75 → decode (injects JPG artifacts)
    _, enc = cv2.imencode(".jpg", cap_resized_png, [cv2.IMWRITE_JPEG_QUALITY, 75])
    cap_resized_jpg = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    model = YOLO(str(MODEL))

    cases = {
        "A_captured_1280x720_PNG":    cap_bgr,
        "B_captured_resized_640x360_PNG": cap_resized_png,
        "C_captured_resized_640x360_JPG_q75": cap_resized_jpg,
        "D_valid_640x360_JPG":        val_bgr,
    }

    results = {}
    for tag, arr in cases.items():
        r = run_yolo(model, arr, tag)
        results[tag] = r
        cv2.imwrite(str(OUT_DIR / f"{tag}_annot.png"), r["annotated"])

    # Report
    print("\n" + "=" * 80)
    print(f"{'Condition':40s} | {'N':3s} | Classes (conf)")
    print("-" * 80)
    for tag, r in results.items():
        cls_str = ", ".join(f"{d['class']}:{d['conf']:.2f}" for d in r["detections"])
        print(f"{tag:40s} | {r['n']:3d} | {cls_str}")
    print("=" * 80)

    # Save raw comparison table
    with (OUT_DIR / "report.txt").open("w") as f:
        f.write(f"Pair: cap={cap_path.name}, val={val_path.name}\n\n")
        for tag, r in results.items():
            f.write(f"--- {tag} ---\n")
            f.write(f"  detections={r['n']}\n")
            for d in r["detections"]:
                f.write(f"    {d['class']:12s} conf={d['conf']:.3f}  bbox={d['bbox']}\n")
            f.write("\n")
    print(f"\nsaved: {OUT_DIR}/")


if __name__ == "__main__":
    main()
