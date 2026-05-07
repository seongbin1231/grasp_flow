"""SKELETON — Kinect/RealSense 카메라 K 를 우리 K (1109) 로 canonicalize.

검증된 K (graspnetAPI utils.py:34-37):
  Kinect:    fx=631.55, fy=631.21, cx=638.43, cy=366.50
  RealSense: fx=927.17, fy=927.37, cx=651.32, cy=349.62
우리 K:        fx=fy=1109,  cx=640,    cy=360

⚠️ Critical: 카메라마다 K 다르므로 분기 처리 필수.

K-warp 절차 (depth + (u,v) 동시):
  K_dst = [[1109, 0, 640], [0, 1109, 360], [0, 0, 1]]
  sx = 1109 / K_src[0,0]
  sy = 1109 / K_src[1,1]
  tx = 640 - sx * K_src[0,2]
  ty = 360 - sy * K_src[1,2]
  M = np.float32([[sx, 0, tx], [0, sy, ty]])
  depth_canon = cv2.warpAffine(depth_mm, M, (1280, 720), flags=cv2.INTER_NEAREST)
  u_c = sx * u + tx
  v_c = sy * v + ty
  # 3D grasp (R, t) 는 K-independent → 변경 없음
"""
from __future__ import annotations
import argparse
from pathlib import Path


K_KINECT = (631.55, 631.21, 638.43, 366.50)
K_REALSENSE = (927.17, 927.37, 651.32, 349.62)
K_OURS = (1109.0, 1109.0, 640.0, 360.0)


def detect_camera(scene_path: Path) -> str:
    """scene_path 에서 'kinect' 또는 'realsense' 추출."""
    p = str(scene_path).lower()
    if "kinect" in p:
        return "kinect"
    if "realsense" in p:
        return "realsense"
    raise ValueError(f"camera not detected in path: {scene_path}")


def warp_depth_uv(depth_mm, uv, camera: str):
    """TODO: cv2.warpAffine 으로 depth canonicalize + uv 변환."""
    raise NotImplementedError


def main():
    raise NotImplementedError("TODO: integrate with build_h5.py")


if __name__ == "__main__":
    main()
