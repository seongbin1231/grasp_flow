from ultralytics import YOLO
import os

# ── 1. Dataset ──
dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
data_yaml = os.path.join(dataset_dir, "data.yaml")

if not os.path.exists(data_yaml):
    from roboflow import Roboflow
    rf = Roboflow(api_key="6i1faz0bEDlQhAAU1K8g")
    project = rf.workspace("s-workspace-jr1zt").project("yolov8-ntfdh")
    version = project.version(1)
    version.download("yolov8", location=dataset_dir)

# ── 2. Train ──
model = YOLO("yolov8m-seg.pt")

model.train(
    data=data_yaml,
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    patience=30,
    save=True,
    save_period=10,
    project="runs",
    name="yolov8m_seg_v2",
    exist_ok=True,
)
