"""SKELETON — RegionNormalizedGrasp (THU-VCLab) baseline 평가.

⚠️ Critical fact-check (2026-05-07):
  - 정확한 repo 이름 = `RegionNormalizedGrasp` (RNGNet 아님)
  - URL: https://github.com/THU-VCLab/RegionNormalizedGrasp
  - 마지막 push: 2026-03-28 (NOT 2026-04-22)
  - **NO LICENSE** (= all rights reserved). 코드 modify/redistribute 법적 위험
    → inference-only 사용. 우리 repo 에 baseline 코드 bundle 금지
  - **640×360 internal rescale** (1280×720 native 아님). YOLO (u,v) 좌표 ×0.5 변환 필수
  - `infer_from_rgbd_centers(rgb, depth, centers_2d)` API **없음**
    → demo.py 의 GHM auto-center bypass + 우리 YOLO (u,v) inject 필요
    → monkey-patch 150~250 LoC 예상

전제:
  1. CUDA Toolkit 설치 (현재 nvcc 미설치)
  2. PointNet2 ops 빌드 성공
  3. RegionNormalizedGrasp clone 완료 (외부 디렉토리, e.g. /tmp/RegionNormalizedGrasp)
  4. 사전학습 ckpt: checkpoints/realsense (40MB), checkpoints/kinect (40MB)

평가 절차:
  1. 우리 grasp_v2.h5 val split 로드 (98 scene)
  2. for each scene × object:
       rgb = imread(rgb_path)              # 1280×720
       depth = imread(depth_path) / 1000   # mm → m
       u, v = uv_centroid                   # 우리 YOLO 결과
       u_640 = u * 0.5                      # 1280→640 scale
       v_360 = v * 0.5                      # 720→360 scale
       grasps_6dof = rngnet_inference(rgb, depth, [(u_640, v_360)])
  3. 메트릭 통합: scripts/baselines/eval_baseline_unified.py::compute_metrics_7d()
     → Pos MAE / Ang Err / COV / APD per-mode
  4. Table 1 RegionNormalizedGrasp 행 채움

TODO:
  1. RegionNormalizedGrasp clone + ckpt 다운 (수동, README 참조)
  2. monkey-patch demo.py: GHM/AnchorNet 우회 + custom centers inject
  3. uv_adapter.py 와 충돌 X (PC 입력 아님 — depth + (u,v) 직접)
"""
from __future__ import annotations
import argparse
from pathlib import Path

# 우리 캠라 K → RegionNormalizedGrasp 입력 K 매핑
SCALE_W = 640.0 / 1280.0  # 0.5
SCALE_H = 360.0 / 720.0   # 0.5


def uv_to_rngnet_input(u_1280, v_720):
    """1280×720 → 640×360 좌표 변환."""
    return u_1280 * SCALE_W, v_720 * SCALE_H


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_h5", type=Path, default=Path("datasets/grasp_v2.h5"))
    ap.add_argument("--rngnet_root", type=Path, required=True,
                    help="path to cloned RegionNormalizedGrasp repo")
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="checkpoints/realsense or checkpoints/kinect")
    ap.add_argument("--out_json", type=Path, default=Path("runs/rngnet_eval.json"))
    return ap.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError("TODO: implement after RegionNormalizedGrasp install")


if __name__ == "__main__":
    main()
