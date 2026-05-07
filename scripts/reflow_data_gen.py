"""SKELETON — Reflow 1라운드 데이터 생성 (P6, RA-L W3, no distillation).

Liu et al. ICLR 2023 "Flow Straight and Fast: Rectified Flow" (arXiv:2209.03003)
Algorithm 1: Z^(k+1) = RectFlow((Z_0^k, Z_1^k))
  - 1차 모델로 (g_0, g_1) coupled pair 생성 (RK4 적분 ~50 steps)
  - 새 pair 로 2차 모델 학습 → 1-step Euler 정확도 향상

핵심 절차:
  1. zhou_9d_full_250ep ckpt 로드 (1차 모델)
  2. train H5 의 각 (cond, x_0) sample 마다 RK4 50-step 으로 x_1' 적분
  3. 새 H5 (datasets/grasp_v2_reflow1.h5) 저장: g_1 컬럼만 x_1' 로 교체
  4. train_flow.py 로 2차 모델 학습 (~10h GPU)

⚠️ Fact-check (2026-05-07):
  - Liu 2023 의 FID 4.85 (CIFAR-10, 1-step) 는 2-rectified flow + Distill 결과
  - 단순 1-round reflow (no distill) 는 FID 12.21 (Table 1)
  - paper 본문 framing: "We apply one round of Reflow (Liu 2023, no distillation) ..."
    4.85 같은 image-domain 수치 인용 금지
  - Liu 권장: "we find that it is sufficient to only reflow once" (1라운드 OK)

TODO (실행 시):
  1. flow_dataset.GraspDataset 의 train split 로드
  2. 모델 ckpt 로드 (zhou_9d_full_250ep)
  3. RK4 적분기 (scipy.integrate 또는 hand-rolled)
     - x_t' = (1-t) g_0 + t g_1 가 아니라 1차 모델 v_θ 따라 ODE 적분
  4. 새 H5 schema = grasp_v2 와 동일, g_1 컬럼만 변경 (mode_id, group, class 등 보존)
  5. train_flow.py --h5 datasets/grasp_v2_reflow1.h5 --pretrained <1차ckpt> 로 2차 학습
"""
from __future__ import annotations
from pathlib import Path
import argparse
import torch


def integrate_rk4(model, cond, g_0: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
    """1차 모델 v_θ 를 따라 g_0 → g_1' 적분 (RK4)."""
    raise NotImplementedError("TODO: RK4 with model.velocity()")


def build_reflow_h5(
    src_h5: Path,
    out_h5: Path,
    ckpt: Path,
    rk4_steps: int = 50,
):
    """TODO: src H5 의 (cond, g_0=randn) 마다 g_1' 적분 → 새 H5 저장."""
    raise NotImplementedError("TODO: implement after Liu 2023 §2.1 verbatim")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_h5", type=Path, default=Path("datasets/grasp_v2.h5"))
    ap.add_argument("--out_h5", type=Path, default=Path("datasets/grasp_v2_reflow1.h5"))
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="1차 모델 ckpt (예: runs/.../zhou_9d_full_250ep/.../best.pt)")
    ap.add_argument("--rk4_steps", type=int, default=50)
    return ap.parse_args()


def main():
    args = parse_args()
    build_reflow_h5(args.src_h5, args.out_h5, args.ckpt, args.rk4_steps)


if __name__ == "__main__":
    main()
