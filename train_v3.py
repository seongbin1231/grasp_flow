"""Finetune yolov8m-seg on dataset_v2 (1280×720 native) from v2 best weights.

Config rationale (RTX 4070 12GB, ~10GB usable):
  - imgsz=1280        → preserves detail (marker, cube), matches deployment
  - batch=4           → fits ~10GB with amp + cache='ram' off
  - epochs=50         → finetune sufficient
  - patience=15       → early stop if no improvement
  - start from best.pt of v2 → warm start
"""
from pathlib import Path
from ultralytics import YOLO

PROJECT = Path("/home/robotics/Competition/YOLO_Grasp")
DATA_YAML = PROJECT / "dataset_v2/data.yaml"
START_WEIGHTS = PROJECT / "runs/yolov8m_seg_v2/weights/best.pt"

def main():
    model = YOLO(str(START_WEIGHTS))
    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=1280,
        batch=4,
        device=0,
        workers=4,
        patience=15,
        save=True,
        save_period=10,
        amp=True,
        cache=False,
        project="runs",
        name="yolov8m_seg_v3_1280",
        exist_ok=True,
        verbose=True,
    )

if __name__ == "__main__":
    main()
