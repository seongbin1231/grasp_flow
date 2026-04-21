"""Val 이미지 10장 추론 + 시각화"""
from ultralytics import YOLO
import os
import random

random.seed(42)

base = os.path.dirname(__file__)
ckpt = os.path.join(base, "runs", "yolov8m_seg_v2", "weights", "best.pt")
val_dir = os.path.join(base, "dataset", "valid", "images")
out_dir = os.path.join(base, "runs", "test_results")

# 무작위 10장 선택
images = sorted(os.listdir(val_dir))
samples = random.sample(images, 10)
sample_paths = [os.path.join(val_dir, f) for f in samples]

model = YOLO(ckpt)
results = model.predict(
    source=sample_paths,
    conf=0.25,         # confidence threshold
    iou=0.7,
    imgsz=640,
    save=True,         # 시각화 이미지 자동 저장
    save_txt=True,     # 예측 라벨 .txt 저장
    save_conf=True,
    project=out_dir,
    name="val10",
    exist_ok=True,
)

# 결과 요약
for r, path in zip(results, sample_paths):
    n = len(r.boxes) if r.boxes is not None else 0
    classes = [model.names[int(c)] for c in r.boxes.cls] if n else []
    print(f"{os.path.basename(path)}: {n} objects → {classes}")

print(f"\n결과 저장: {out_dir}/val10/")
