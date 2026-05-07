"""SKELETON — GraspNet-1B label/*.png → per-object (u,v) centroid 추출.

graspnet-baseline label 포맷 (검증됨, fact-check 2026-05-07):
  - label/*.png: per-pixel instance id (0=bg, 1~88 obj)
  - meta['cls_indexes']: shape (n_obj,), 0-index → object_id mapping

출력: H5 또는 JSON, scene_id × ann_id × obj_id → (u, v, mask_px)

TODO:
  1. graspnetAPI 설치 + import
  2. for sceneId in scene_list:
       for annId in ann_list:
         seg = imread(label_dir/f"{annId:04d}.png")
         meta = scio.loadmat(meta_dir/f"{annId:04d}.mat")
         for inst_id, oid in enumerate(meta['cls_indexes'].flatten(), 1):
           mask = (seg == inst_id)
           if mask.sum() < 200: continue  # 점유율 낮은 객체 skip
           ys, xs = np.where(mask)
           u, v = xs.mean(), ys.mean()
           # save
"""
from __future__ import annotations
import argparse
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graspnet_root", type=Path, required=True)
    ap.add_argument("--out_h5", type=Path, default=Path("datasets/graspnet1b_uv.h5"))
    ap.add_argument("--camera", choices=["realsense", "kinect"], default="realsense")
    ap.add_argument("--split", choices=["train", "test_seen", "test_similar", "test_novel"],
                    default="train")
    ap.add_argument("--mask_px_min", type=int, default=200)
    return ap.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError("TODO: implement after fact-check section §2 of plan")


if __name__ == "__main__":
    main()
