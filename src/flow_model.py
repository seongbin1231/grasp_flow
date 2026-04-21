"""
Flow Matching model for (depth + uv) → 6-DoF multi-grasp sampling.

Architecture:
  Depth (1, 720, 1280)
    ├─ Global branch: lightweight CNN → avg pool → 128-d
    └─ Local branch: uv-centric 192×192 crop → CNN → 128-d
  Concat 256 + uv_norm(2) + t_emb(64) → cond
  VelocityMLP: (g_t(8), cond) → v(8)

ONNX-safe: no custom ops, no control flow. Input: depth, uv, g_t, t → velocity.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_H, IMG_W = 720, 1280
CROP = 192
DEPTH_CLIP_MIN = 0.3
DEPTH_CLIP_MAX = 1.5
DEPTH_SCALE_DIV = 1.5


def clip_norm_depth(depth: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W) meter → clip+scale to roughly [0.2, 1.0]."""
    d = torch.clamp(depth, DEPTH_CLIP_MIN, DEPTH_CLIP_MAX)
    return d / DEPTH_SCALE_DIV


def conv_block(ci, co, k=3, s=2, p=1):
    return nn.Sequential(
        nn.Conv2d(ci, co, k, stride=s, padding=p),
        nn.GroupNorm(8, co),
        nn.SiLU(inplace=True),
    )


class GlobalDepthEncoder(nn.Module):
    """Depth (1, 720, 1280) → 128-d. Stride ×32 to 23×40 → avg pool."""
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(1, 32, s=4),      # 180×320
            conv_block(32, 64, s=2),     # 90×160
            conv_block(64, 128, s=2),    # 45×80
            conv_block(128, 128, s=2),   # 23×40
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, out_dim), nn.SiLU(inplace=True),
        )

    def forward(self, d):
        return self.head(self.net(d))


class LocalCropEncoder(nn.Module):
    """Depth crop (1, 192, 192) around (u, v) → 128-d."""
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(1, 32, s=2),      # 96
            conv_block(32, 64, s=2),     # 48
            conv_block(64, 128, s=2),    # 24
            conv_block(128, 128, s=2),   # 12
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, out_dim), nn.SiLU(inplace=True),
        )

    def forward(self, crop):
        return self.head(self.net(crop))


def crop_around_uv(depth: torch.Tensor, uv: torch.Tensor, crop: int = CROP) -> torch.Tensor:
    """Crop (B, 1, H, W) around (u, v) using grid_sample for ONNX compatibility.

    Returns (B, 1, crop, crop).
    """
    B, C, H, W = depth.shape
    half = crop / 2.0
    # make normalized grid (-1..+1 across each axis)
    ys = torch.linspace(-half, half - 1, crop, device=depth.device)
    xs = torch.linspace(-half, half - 1, crop, device=depth.device)
    # grid coords (B, crop, crop, 2) in pixel space
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gx = gx.unsqueeze(0) + uv[:, 0:1, None]       # (B, crop, crop)
    gy = gy.unsqueeze(0) + uv[:, 1:2, None]
    # normalize to [-1, 1]
    gx_n = (gx / (W - 1)) * 2.0 - 1.0
    gy_n = (gy / (H - 1)) * 2.0 - 1.0
    grid = torch.stack([gx_n, gy_n], dim=-1)      # (B, crop, crop, 2)
    return F.grid_sample(depth, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)


def sinusoidal_time_embed(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """t ∈ (B,) → (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / half
    )
    ang = t[:, None] * freqs[None]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class FiLMBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.film = nn.Linear(cond_dim, dim * 2)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, cond):
        h = self.fc1(x)
        h = F.silu(h)
        h = self.fc2(h)
        scale_shift = self.film(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = self.ln(h) * (1 + scale) + shift
        return F.silu(x + h)


class AdaLNZeroBlock(nn.Module):
    """DiT-style AdaLN-Zero. gate α zero-init → block starts as identity.
    Capacity matched to FiLMBlock: two dim→dim Linears."""
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.adaLN = nn.Linear(cond_dim, dim * 3)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x, cond):
        gamma, beta, alpha = self.adaLN(cond).chunk(3, dim=-1)
        h = self.ln(x) * (1 + gamma) + beta
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return x + alpha * h


BLOCK_TYPES = {"film": FiLMBlock, "adaln_zero": AdaLNZeroBlock}


class VelocityMLP(nn.Module):
    def __init__(self, g_dim: int = 8, cond_dim: int = 256, t_dim: int = 64,
                 hidden: int = 512, n_blocks: int = 4, block_type: str = "adaln_zero"):
        super().__init__()
        self.in_proj = nn.Linear(g_dim, hidden)
        self.cond_proj = nn.Linear(cond_dim + t_dim + 2, cond_dim)  # +uv(2)
        Block = BLOCK_TYPES[block_type]
        self.blocks = nn.ModuleList([
            Block(hidden, cond_dim) for _ in range(n_blocks)
        ])
        self.out = nn.Linear(hidden, g_dim)
        self.cond_dim = cond_dim
        self.block_type = block_type

    def forward(self, g_t, cond_feat, t_emb, uv_norm):
        cond = torch.cat([cond_feat, t_emb, uv_norm], dim=-1)
        cond = self.cond_proj(cond)
        h = self.in_proj(g_t)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out(h)


class FlowGraspNet(nn.Module):
    def __init__(
        self,
        g_dim: int = 8,
        global_dim: int = 128,
        local_dim: int = 128,
        t_dim: int = 64,
        hidden: int = 512,
        n_blocks: int = 4,
        cond_dropout: float = 0.2,
        block_type: str = "adaln_zero",
    ):
        super().__init__()
        self.global_enc = GlobalDepthEncoder(global_dim)
        self.local_enc = LocalCropEncoder(local_dim)
        self.cond_dropout = cond_dropout
        self.aux_mode_head = nn.Linear(global_dim + local_dim, 3)  # lying/standing/cube
        self.velocity = VelocityMLP(
            g_dim=g_dim, cond_dim=global_dim + local_dim,
            t_dim=t_dim, hidden=hidden, n_blocks=n_blocks,
            block_type=block_type,
        )
        self.block_type = block_type

    def encode(self, depth: torch.Tensor, uv: torch.Tensor):
        d = clip_norm_depth(depth)
        g_feat = self.global_enc(d)                          # (B, 128)
        crop = crop_around_uv(d, uv)                          # (B, 1, 192, 192)
        l_feat = self.local_enc(crop)                         # (B, 128)
        cond = torch.cat([g_feat, l_feat], dim=-1)            # (B, 256)
        return cond

    def forward(self, depth, uv, g_t, t):
        """ONNX forward. depth (B,1,720,1280), uv (B,2), g_t (B,8), t (B,)."""
        cond = self.encode(depth, uv)
        if self.training and self.cond_dropout > 0:
            mask = (torch.rand(cond.shape[0], 1, device=cond.device)
                    > self.cond_dropout).float()
            cond = cond * mask
        t_emb = sinusoidal_time_embed(t, dim=64)
        uv_norm = torch.stack([uv[:, 0] / IMG_W, uv[:, 1] / IMG_H], dim=-1)
        v = self.velocity(g_t, cond, t_emb, uv_norm)
        return v

    def forward_with_aux(self, depth, uv, g_t, t):
        cond = self.encode(depth, uv)
        # aux_mode_head uses PRE-dropout cond — otherwise unconditional (cond=0) samples
        # would train aux on noise, contaminating mode classification
        mode_logits = self.aux_mode_head(cond)
        cond_for_velocity = cond
        if self.training and self.cond_dropout > 0:
            mask = (torch.rand(cond.shape[0], 1, device=cond.device)
                    > self.cond_dropout).float()
            cond_for_velocity = cond * mask
        t_emb = sinusoidal_time_embed(t, dim=64)
        uv_norm = torch.stack([uv[:, 0] / IMG_W, uv[:, 1] / IMG_H], dim=-1)
        v = self.velocity(g_t, cond_for_velocity, t_emb, uv_norm)
        return v, mode_logits


class EMA:
    """Exponential moving average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return self.shadow

    def load_into(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)
