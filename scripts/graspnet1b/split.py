"""SKELETON — GraspNet-1B 공식 split.

검증된 split (graspnet.py:91-101):
  scene 0~99   = train
  scene 100~129 = test_seen
  scene 130~159 = test_similar
  scene 160~189 = test_novel

obj 18 제외 (graspnet_dataset.py:251-256 `if i==18: continue`)

⚠️ 주의: 우리는 train_1.zip (scene 0~24) + test_seen/similar/novel zip 만 사용.
  - train_2/3/4 (scene 25~99) 는 디스크 절약 위해 다운 X (사용자 disk 44GB 제약)
  - 1M row subsample 정책에서 train scene 25 정도면 충분.
"""
from __future__ import annotations
import argparse


SPLIT_RANGES = {
    "train": range(0, 100),
    "test_seen": range(100, 130),
    "test_similar": range(130, 160),
    "test_novel": range(160, 190),
}
EXCLUDED_OBJS = {18}


def get_split_for_scene(scene_id: int) -> str:
    for split, rng in SPLIT_RANGES.items():
        if scene_id in rng:
            return split
    raise ValueError(f"scene_id {scene_id} out of range")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_id", type=int, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    print(get_split_for_scene(args.scene_id))


if __name__ == "__main__":
    main()
