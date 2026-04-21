"""Train/Val 데이터셋 분리 (80/20)"""
import os
import shutil
import random

random.seed(42)

base = os.path.join(os.path.dirname(__file__), "dataset")
train_img = os.path.join(base, "train", "images")
train_lbl = os.path.join(base, "train", "labels")
val_img = os.path.join(base, "valid", "images")
val_lbl = os.path.join(base, "valid", "labels")

os.makedirs(val_img, exist_ok=True)
os.makedirs(val_lbl, exist_ok=True)

images = sorted(os.listdir(train_img))
random.shuffle(images)

val_count = int(len(images) * 0.2)
val_images = images[:val_count]

print(f"Total: {len(images)}, Val: {val_count}, Train: {len(images) - val_count}")

for img_name in val_images:
    # move image
    shutil.move(os.path.join(train_img, img_name), os.path.join(val_img, img_name))
    # move label
    lbl_name = os.path.splitext(img_name)[0] + ".txt"
    lbl_src = os.path.join(train_lbl, lbl_name)
    if os.path.exists(lbl_src):
        shutil.move(lbl_src, os.path.join(val_lbl, lbl_name))

print("Done!")
