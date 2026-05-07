"""SKELETON — extract_uv + flatten_grasps + obj_to_cam + k_warp 통합 → grasp_v2 schema.

출력 H5 schema (grasp_v2.h5 와 동일):
  /train, /val (또는 split 명에 따라)
    grasps_cam:      (N, 7)   [x,y,z, qw,qx,qy,qz]   camera frame
    approach_vec:    (N, 3)   unit vector
    yaw_around_app:  (N,)     radians
    object_class:    (N,)     int (0~87 for GraspNet, mode_id=-1=unknown)
    object_mode:     (N,)     int (-1 = unknown for GraspNet)
    object_ref:      (N,)     int (split-wide unique ref)
    grasp_group:     (N,)     int (-1 = no group label for GraspNet)
    uvs:             (N, 2)   per-object centroid (extract_uv 결과)
    depth_ref:       (N,)     int → /scenes/{ref}/depth (1280×720 canonicalized)

Camera K: 1109 (canonicalized from Kinect/RealSense via k_warp.py)

⚠️ mode_id=-1 (unknown bucket): 우리 grasp_v2 의 0=lying/1=standing/2=cube 와 충돌 회피.
   Stage 2 fine-tune 시 mode_id 입력 안 받음 (현 모델 spec) → 영향 없음.

⚠️ Disk constraint (2026-05-07 사용자 경고):
   503GB 중 44GB 만 여유. GraspNet 변환 후 H5 = 약 50GB 예상.
   subsample (1M rows) → ~5GB H5 권장.
"""
from __future__ import annotations
import argparse
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graspnet_root", type=Path, required=True)
    ap.add_argument("--out_h5", type=Path, default=Path("datasets/graspnet1b_v1.h5"))
    ap.add_argument("--max_rows", type=int, default=1_000_000,
                    help="subsample to limit disk (full = ~12M rows ≈ 50GB)")
    ap.add_argument("--camera", choices=["realsense", "kinect"], default="realsense")
    return ap.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError("TODO: depend on extract_uv + flatten_grasps + obj_to_cam + k_warp")


if __name__ == "__main__":
    main()
