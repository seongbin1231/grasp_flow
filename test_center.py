"""세그멘테이션 중심점(centroid) 계산 + 시각화"""
from ultralytics import YOLO
import cv2
import numpy as np
import os
import random

random.seed(42)

base = os.path.dirname(__file__)
ckpt = os.path.join(base, "runs", "yolov8m_seg_v2", "weights", "best.pt")
val_dir = os.path.join(base, "dataset", "valid", "images")
out_dir = os.path.join(base, "runs", "test_center")
os.makedirs(out_dir, exist_ok=True)

samples = random.sample(sorted(os.listdir(val_dir)), 10)
sample_paths = [os.path.join(val_dir, f) for f in samples]

model = YOLO(ckpt)
results = model.predict(source=sample_paths, conf=0.25, iou=0.7, imgsz=640, verbose=False)

for r, path in zip(results, sample_paths):
    img = cv2.imread(path)
    H, W = img.shape[:2]

    if r.masks is None:
        continue

    masks = r.masks.data.cpu().numpy()   # (N, h, w) — 모델 출력 해상도
    classes = r.boxes.cls.cpu().numpy().astype(int)
    boxes = r.boxes.xyxy.cpu().numpy()

    print(f"\n=== {os.path.basename(path)} ===")
    for i, mask in enumerate(masks):
        # 원본 해상도로 리사이즈
        mask_full = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        # (1) Mask centroid — 픽셀 좌표 평균
        ys, xs = np.where(mask_full)
        if len(xs) == 0:
            continue
        cx_mask = int(xs.mean())
        cy_mask = int(ys.mean())

        # (2) Bbox center (비교용)
        x1, y1, x2, y2 = boxes[i]
        cx_box, cy_box = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # (3) Distance transform peak — 내부 가장 안쪽 지점
        dist = cv2.distanceTransform(mask_full.astype(np.uint8), cv2.DIST_L2, 5)
        cy_dt, cx_dt = np.unravel_index(dist.argmax(), dist.shape)

        label = model.names[classes[i]]
        print(f"  [{label}] centroid=({cx_mask},{cy_mask})  bbox=({cx_box},{cy_box})  dt_peak=({cx_dt},{cy_dt})")

        # 시각화: mask overlay + 3가지 중심점
        color_mask = np.zeros_like(img)
        color_mask[mask_full] = (0, 255, 0)
        img = cv2.addWeighted(img, 1.0, color_mask, 0.4, 0)

        cv2.circle(img, (cx_mask, cy_mask), 6, (0, 0, 255), -1)   # red: centroid
        cv2.circle(img, (cx_box, cy_box), 6, (255, 0, 0), -1)     # blue: bbox
        cv2.circle(img, (cx_dt, cy_dt), 6, (0, 255, 255), -1)     # yellow: DT peak
        cv2.putText(img, label, (cx_mask + 8, cy_mask),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(os.path.join(out_dir, os.path.basename(path)), img)

print(f"\n저장: {out_dir}/")
print("Red=centroid, Blue=bbox center, Yellow=distance-transform peak")
