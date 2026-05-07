"""SKELETON — 9D Zhou (pos + R[:,0:2]) → graspnetAPI GraspGroup 17-col.

검증된 GraspGroup 17 컬럼 (graspnetAPI grasp.py:10-34):
  arr[:, 0]    = score      # AP ranking 핵심 (constant 시 collapse)
  arr[:, 1]    = width       # gripper opening width
  arr[:, 2]    = height      # finger height (≈0.02)
  arr[:, 3]    = depth       # approach depth (≈0.02)
  arr[:, 4:13] = R(9 flat, row-major)
  arr[:, 13:16]= t (3) translation
  arr[:, 16]   = obj_id

⚠️ Score 컬럼 collapse 위험 (fact-check 2026-05-07):
  - Constant score = AP 곤두박질
  - 권장 score 함수:
      score = YOLO_conf × |cos(angle to estimated normal)|
      또는 graspnet-baseline 의 objectness score head 사후 학습
  - 우리는 flow log-prob 도 사용 가능 (단 추가 적분 비용)

⚠️ Gripper 좌표 convention 확인 필수 (graspnetAPI Grasp.to_open3d_geometry()):
  - graspnet-baseline: R[:,0]=approach (X), R[:,1]=binormal (Y), R[:,2]=minor (Z)
  - 우리 Zhou6D: R[:,2]=approach (Tool Z), R[:,1]=open (Tool Y), R[:,0]=Tool X
  → 변환 시 column 순서 재배열 필요!
"""
from __future__ import annotations
import numpy as np


def zhou6d_to_R(r6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt: r6 = [a1(3), a2(3)] → R 3x3."""
    a1, a2 = r6[:3], r6[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    b2 = a2 - (b1 @ a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])


def remap_axes_to_graspnet(R_ours: np.ndarray) -> np.ndarray:
    """우리 Zhou6D (X=Tool X, Y=Open, Z=Approach) → GraspNet (X=Approach, Y=Binormal, Z=Minor).

    TODO: 정확한 axis remap 검증 필요 (graspnet-baseline source 정독)
    임시 가정: GraspNet X = 우리 Z, GraspNet Y = 우리 Y, GraspNet Z = -우리 X
    """
    raise NotImplementedError("TODO: verify axis convention in graspnet-baseline")


def to_graspgroup17(
    pos: np.ndarray,         # (N, 3)
    r6: np.ndarray,          # (N, 6) Zhou6D
    scores: np.ndarray,      # (N,) — YOLO conf × cos(angle) 권장
    obj_ids: np.ndarray,     # (N,)
    width: float = 0.05,
    height: float = 0.02,
    depth: float = 0.02,
) -> np.ndarray:
    """N grasps → (N, 17) GraspGroup array."""
    N = pos.shape[0]
    arr = np.zeros((N, 17), dtype=np.float32)
    arr[:, 0] = scores
    arr[:, 1] = width
    arr[:, 2] = height
    arr[:, 3] = depth
    for i in range(N):
        R_ours = zhou6d_to_R(r6[i])
        R_gn = remap_axes_to_graspnet(R_ours)   # TODO
        arr[i, 4:13] = R_gn.flatten()           # row-major
    arr[:, 13:16] = pos
    arr[:, 16] = obj_ids
    return arr


def main():
    raise NotImplementedError("TODO: integrate with W3 step 15 GraspNet AP eval pipeline")


if __name__ == "__main__":
    main()
