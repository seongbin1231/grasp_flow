"""
Direct MLP baseline for (depth + uv) → 6-DoF single-grasp regression.

비교 대상: Flow Matching (FlowGraspNet).
공정 비교를 위해:
  - Encoder 동일 (GlobalDepthEncoder + LocalCropEncoder + cond_dropout)
  - Aux mode head 동일
  - hidden / n_blocks 동일
  - 차이점: velocity MLP (g_t, cond, t, uv 입력) 대신 cond 직접 → grasp 회귀

이 baseline 은 동일 (depth, uv) 입력에 대해 항상 deterministic 한 단일 grasp 만 출력 →
같은 픽셀에 multi-modal GT (예: standing top-down + side-45 + side-cap) 가 있을 때
mode collapse (평균치로 수렴) 가 시각적·정량적으로 드러남.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.flow_model import (
    GlobalDepthEncoder, LocalCropEncoder, crop_around_uv,
    clip_norm_depth, IMG_H, IMG_W,
)


class ResMLPBlock(nn.Module):
    """Residual MLP block matching AdaLN-Zero capacity (no conditioning gate).
    Capacity: 2 × Linear(dim, dim).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.ln(x)
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class DirectGraspNet(nn.Module):
    """Encoder + Direct MLP head (cond → grasp 8D)."""
    def __init__(
        self,
        g_dim: int = 8,
        global_dim: int = 128,
        local_dim: int = 128,
        hidden: int = 1024,
        n_blocks: int = 12,
        cond_dropout: float = 0.0,   # default OFF (no CFG for deterministic baseline)
    ):
        super().__init__()
        self.global_enc = GlobalDepthEncoder(global_dim)
        self.local_enc = LocalCropEncoder(local_dim)
        cond_dim = global_dim + local_dim   # 256
        self.aux_mode_head = nn.Linear(cond_dim, 3)
        self.cond_dropout = cond_dropout
        # uv_norm(2) 도 cond 에 포함 (Flow 모델과 동일하게 위치 정보 보존)
        self.in_proj = nn.Linear(cond_dim + 2, hidden)
        self.blocks = nn.ModuleList([ResMLPBlock(hidden) for _ in range(n_blocks)])
        self.out = nn.Linear(hidden, g_dim)

    def encode(self, depth: torch.Tensor, uv: torch.Tensor):
        d = clip_norm_depth(depth)
        g_feat = self.global_enc(d)
        crop = crop_around_uv(d, uv)
        l_feat = self.local_enc(crop)
        return torch.cat([g_feat, l_feat], dim=-1)

    def forward_with_aux(self, depth, uv):
        cond = self.encode(depth, uv)
        mode_logits = self.aux_mode_head(cond)
        cond_for_head = cond
        if self.training and self.cond_dropout > 0:
            mask = (torch.rand(cond.shape[0], 1, device=cond.device)
                    > self.cond_dropout).float()
            cond_for_head = cond * mask
        uv_norm = torch.stack([uv[:, 0] / IMG_W, uv[:, 1] / IMG_H], dim=-1)
        h = self.in_proj(torch.cat([cond_for_head, uv_norm], dim=-1))
        for blk in self.blocks:
            h = blk(h)
        grasp = self.out(h)
        return grasp, mode_logits

    def forward(self, depth, uv):
        grasp, _ = self.forward_with_aux(depth, uv)
        return grasp
