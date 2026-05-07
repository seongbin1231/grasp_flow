"""SKELETON — Post-hoc EMA reconstruction (P2, RA-L W2).

Karras et al. CVPR 2024 "Analyzing and Improving the Training Dynamics of Diffusion Models"
(arXiv:2312.02696, NVlabs/edm2). 학습 종료 후 저장된 N 개 EMA snapshot 을
가중합으로 임의의 power-EMA decay 로 재구성.

핵심 절차:
  1. 학습 시 K=10~20 개의 EMA snapshot 저장 (각 snapshot 은 (decay, step, state_dict))
  2. 평가 시: 목표 power-EMA profile p* 를 최소제곱으로 재구성
       w_i = argmin_w || sum_i w_i · profile_i - profile_target ||²
  3. 가중합 model = sum_i w_i · snapshot_i
  4. val_flow 측정, decay sweep 으로 best 선택

⚠️ Fact-check (2026-05-07):
  - Karras 2024 의 FID 2.41 → 1.81 는 post-hoc EMA + magnitude-preserving + DiT 조합 결과.
    post-hoc EMA 단독 Δ 는 paper 에서 isolated 측정 안 됨.
    우리 paper 본문: "We adopt post-hoc EMA; Table 3 reports Δval_flow for our domain."
  - snapshot 5 개는 lower-accuracy regime, 권장 10~20 개

TODO (실행 시):
  1. train_flow.py 의 EMA class 확장: snapshot 저장 옵션 추가
     - 저장 step interval: max(1, total_steps // 15)
     - 저장 위치: runs/<run>/ema_snapshots/snapshot_{step:08d}.pt
     - 각 snapshot: {'state_dict': ..., 'decay': power_ema_decay, 'step': step}
  2. 본 스크립트: snapshot dir → 가중합 → 새 ckpt 저장 + val_flow 측정
  3. Karras 2024 power-EMA reconstruction 수식: profile(t; r) = (1+r) * t^r / (1 - (1-α)^(t+1))
     (정확한 form 은 paper §3 + Appendix C 참조)
"""
from __future__ import annotations
from pathlib import Path
import argparse
import torch


def reconstruct_posthoc_ema(snapshot_dir: Path, target_decay: float):
    """TODO: 저장된 snapshot 들을 가중합으로 target decay 로 재구성.

    Returns: state_dict
    """
    raise NotImplementedError("TODO: implement Karras 2024 §3 weighted reconstruction")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot_dir", type=Path, required=True)
    ap.add_argument("--decay_sweep", type=str, default="0.999,0.9995,0.9999")
    ap.add_argument("--out_dir", type=Path, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    decays = [float(d) for d in args.decay_sweep.split(",")]
    for decay in decays:
        sd = reconstruct_posthoc_ema(args.snapshot_dir, decay)
        # TODO: load val data, compute val_flow, save as ckpt
        print(f"[TODO] decay={decay} val_flow=...")


if __name__ == "__main__":
    main()
