"""Split Roboflow v2 (all-train) into train/valid 80/20.

Uses a seed-stable shuffle. Keeps the same directory layout as Roboflow expects.
Moves files (doesn't copy) to save disk space.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

ROOT = Path("/home/robotics/Competition/YOLO_Grasp/dataset_v2")
TRAIN_IMG = ROOT / "train/images"
TRAIN_LBL = ROOT / "train/labels"
VALID_IMG = ROOT / "valid/images"
VALID_LBL = ROOT / "valid/labels"

VAL_RATIO = 0.20
SEED = 42


def main():
    VALID_IMG.mkdir(parents=True, exist_ok=True)
    VALID_LBL.mkdir(parents=True, exist_ok=True)

    images = sorted(TRAIN_IMG.glob("*"))
    random.seed(SEED)
    random.shuffle(images)

    n_val = int(round(len(images) * VAL_RATIO))
    val_imgs = images[:n_val]
    print(f"[split] total={len(images)}, val={n_val}, train={len(images) - n_val}")

    moved = 0
    missing_lbl = 0
    for img_path in val_imgs:
        lbl_path = TRAIN_LBL / (img_path.stem + ".txt")
        if not lbl_path.exists():
            missing_lbl += 1
            # still move image to keep pairing consistent
        shutil.move(str(img_path), VALID_IMG / img_path.name)
        if lbl_path.exists():
            shutil.move(str(lbl_path), VALID_LBL / lbl_path.name)
        moved += 1

    remaining_train = len(list(TRAIN_IMG.glob("*")))
    remaining_val = len(list(VALID_IMG.glob("*")))
    print(f"[done] moved={moved}, missing_labels={missing_lbl}")
    print(f"[train/images] {remaining_train}, [train/labels] {len(list(TRAIN_LBL.glob('*')))}")
    print(f"[valid/images] {remaining_val}, [valid/labels] {len(list(VALID_LBL.glob('*')))}")


if __name__ == "__main__":
    main()
