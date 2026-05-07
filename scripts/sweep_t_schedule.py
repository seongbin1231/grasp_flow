"""SKELETON — t-schedule 4-way ablation (P4, RA-L W1).

목적: Conditional Flow Matching 학습 시 time t 의 sampling 분포를 4종 비교.
  - uniform U[0,1] (현재 baseline)
  - logit-normal(m=0, s=1)        — Esser et al. 2024 (SD3) 표준
  - logit-normal(m=-0.5, s=1)     — early-t 강조
  - mode-cosine                    — 양 끝 약화 (cosine schedule)

50ep × 4 sweep → val_flow ranking → winner 만 250ep 재학습 (별도 단계 P3).

⚠️ 주의:
  - L162 evaluate() 의 t = torch.rand() 는 변경 X (검증 metric 안정화 위해 uniform 유지)
  - L357 학습 본체의 t = torch.rand() 만 _sample_t() 로 분기

⚠️ Fact-check (2026-05-07):
  - SD3 logit-normal 효과 (FID 49.70 → 45.78) 는 image domain 만 검증.
    우리 7-9D grasp 에 transfer 보장 X. 50ep ablation 으로 정량 측정 후 결정.

TODO (실행 시):
  1. _sample_t() 함수를 train_flow.py 에 import 가능하도록 src/ 또는 utils/ 로 이동
  2. train_flow.py L357 한 줄을 _sample_t(args.t_schedule, B, device) 로 변경
  3. wandb sweep YAML 작성 (variant=uniform/lognorm/lognorm_neg/cosine)
  4. 50ep × 4 run 후 val_flow 비교 표 생성
"""
from __future__ import annotations
import math
import torch


def _sample_t(schedule: str, B: int, device) -> torch.Tensor:
    """Time t ∈ [0,1] sampler. 4 schedules supported.

    Returns: (B,) torch.Tensor on `device`.
    """
    if schedule == "uniform":
        return torch.rand(B, device=device)
    if schedule == "lognorm":
        # logit-normal(m=0, s=1) — SD3 default (Esser 2024 Table 2)
        u = torch.randn(B, device=device)
        return torch.sigmoid(u)
    if schedule == "lognorm_neg":
        # m=-0.5, early-t 강조
        u = torch.randn(B, device=device) - 0.5
        return torch.sigmoid(u)
    if schedule == "cosine":
        # mode-cosine, 양 끝 강조 (boundary 학습 강화)
        u = torch.rand(B, device=device)
        return 1.0 - torch.cos(math.pi / 2 * u)
    raise ValueError(f"unknown t-schedule: {schedule}")


def main():
    """TODO: 4-way sweep 실행 wrapper. wandb sweep YAML 또는 단일 multi-run shell."""
    raise NotImplementedError("TODO: implement after train_flow.py integration")


if __name__ == "__main__":
    main()
