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


class MultiScaleLocalCropEncoder(nn.Module):
    """[A2] Depth crops at multiple scales (e.g. 64/192/384 px) around (u, v).

    Each scale has its own encoder (specialized for object size).
    Returns concat feature (B, out_dim_per_scale * len(scales))
    + per-scale list (used as separate cross-attn tokens by FlowGraspNet).
    """
    def __init__(self, out_dim_per_scale: int = 128,
                 scales: tuple[int, ...] = (64, 192, 384)):
        super().__init__()
        self.scales = scales
        self.out_dim_per_scale = out_dim_per_scale
        self.encoders = nn.ModuleList([
            self._make_encoder(out_dim_per_scale) for _ in scales
        ])

    @staticmethod
    def _make_encoder(out_dim: int) -> nn.Module:
        return nn.Sequential(
            conv_block(1, 32, s=2),
            conv_block(32, 64, s=2),
            conv_block(64, 128, s=2),
            conv_block(128, 128, s=2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, out_dim), nn.SiLU(inplace=True),
        )

    def forward_per_scale(self, depth: torch.Tensor, uv: torch.Tensor,
                           scale_dropout: float = 0.0) -> list[torch.Tensor]:
        """학습 시 scale_dropout 확률로 random 1 scale 만 활성, 나머지는 zero vector.
        정보 중복 방어 (Stochastic Depth 의 multi-scale 적용)."""
        n_scales = len(self.scales)
        if self.training and scale_dropout > 0 and torch.rand(1).item() < scale_dropout:
            keep_idx = int(torch.randint(0, n_scales, (1,)).item())
            feats = []
            for i, (crop_size, enc) in enumerate(zip(self.scales, self.encoders)):
                if i == keep_idx:
                    crop = crop_around_uv(depth, uv, crop=crop_size)
                    feats.append(enc(crop))
                else:
                    feats.append(torch.zeros(depth.size(0), self.out_dim_per_scale,
                                              device=depth.device, dtype=depth.dtype))
            return feats
        feats = []
        for crop_size, enc in zip(self.scales, self.encoders):
            crop = crop_around_uv(depth, uv, crop=crop_size)
            feats.append(enc(crop))
        return feats

    def forward(self, depth: torch.Tensor, uv: torch.Tensor,
                 scale_dropout: float = 0.0) -> torch.Tensor:
        feats = self.forward_per_scale(depth, uv, scale_dropout=scale_dropout)
        return torch.cat(feats, dim=-1)


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


# Camera intrinsics (fixed for our Simulink scene)
K_FX_CAM, K_FY_CAM = 1109.0, 1109.0
K_CX_CAM, K_CY_CAM = 640.0, 360.0


def depth_to_pc(depth: torch.Tensor, uv: torch.Tensor,
                crop: int = 192, n_points: int = 512) -> torch.Tensor:
    """Back-project a (u,v)-centred depth crop to camera-frame point cloud.

    Returns (B, n_points, 3) in metres. ONNX-safe: only standard ops
    (linspace/grid_sample/stack/index_select). No FPS, no KNN, no random.

    Invalid depth (out of clip range) is replaced with z=-1e3 so MiniPointNet
    max-pool ignores it.
    """
    B = depth.size(0)
    crop_d = crop_around_uv(depth, uv, crop=crop)        # (B, 1, crop, crop)
    z = crop_d.squeeze(1)                                 # (B, crop, crop) — meters

    # pixel grid in image coordinates: (u_img, v_img) for each crop pixel
    ys = torch.linspace(-crop / 2.0, crop / 2.0 - 1, crop, device=depth.device)
    xs = torch.linspace(-crop / 2.0, crop / 2.0 - 1, crop, device=depth.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")        # (crop, crop)
    u_img = gx.unsqueeze(0) + uv[:, 0:1, None]            # (B, crop, crop)
    v_img = gy.unsqueeze(0) + uv[:, 1:2, None]

    x = (u_img - K_CX_CAM) * z / K_FX_CAM
    y = (v_img - K_CY_CAM) * z / K_FY_CAM
    pc = torch.stack([x, y, z], dim=-1)                   # (B, crop, crop, 3)
    pc = pc.reshape(B, crop * crop, 3)

    # deterministic stride sub-sample (ONNX-safe)
    total = crop * crop
    stride = max(1, total // n_points)
    idx = torch.arange(0, n_points, device=depth.device) * stride
    idx = torch.clamp(idx, max=total - 1)
    pc = pc.index_select(1, idx)                           # (B, n_points, 3)

    # mask invalid depth (z outside clip range) so max-pool ignores them
    invalid = (pc[..., 2:3] < DEPTH_CLIP_MIN) | (pc[..., 2:3] > DEPTH_CLIP_MAX)
    pc = torch.where(invalid, torch.full_like(pc, -1e3), pc)
    return pc


class MiniPointNet(nn.Module):
    """PointNet-lite (Qi 2017) without T-Net: per-point Conv1d + max-pool.
    ONNX-safe (only Conv1d / GroupNorm / SiLU / max). ~90K params.
    """

    def __init__(self, in_dim: int = 3, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1), nn.GroupNorm(8, 64), nn.SiLU(inplace=True),
            nn.Conv1d(64, 128, 1),    nn.GroupNorm(8, 128), nn.SiLU(inplace=True),
            nn.Conv1d(128, 256, 1),   nn.GroupNorm(8, 256), nn.SiLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(256, out_dim), nn.SiLU(inplace=True),
        )

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        # pc: (B, N, 3) → (B, 3, N)
        h = self.mlp(pc.transpose(1, 2))             # (B, 256, N)
        h = torch.max(h, dim=-1).values              # (B, 256)
        return self.head(h)                           # (B, out_dim)


# =============================================================================
# v9 PC-only modules (기존 코드 변경 없음, 새로 추가만)
# =============================================================================

def depth_to_anchor3d(depth: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """[v9 fix #1] (u,v) + depth(u,v) back-project → anchor 3D 점.
    학습/추론 동일 방식. depth(u,v) 결측 시 3×3 valid-mean fallback.
    Returns (B, 3) in camera frame metres.
    """
    B = depth.size(0)
    u = uv[:, 0].long().clamp(1, IMG_W - 2)
    v = uv[:, 1].long().clamp(1, IMG_H - 2)

    # 3×3 patch around (u,v) — index_select for ONNX safety
    z_patch = []
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            uu = (u + du).clamp(0, IMG_W - 1)
            vv = (v + dv).clamp(0, IMG_H - 1)
            # depth: (B, 1, H, W) → (B,) at (vv, uu)
            idx = vv * IMG_W + uu
            z_flat = depth.view(B, -1)
            z_patch.append(z_flat.gather(1, idx[:, None]).squeeze(-1))
    z_patch = torch.stack(z_patch, dim=-1)              # (B, 9)
    valid = (z_patch > DEPTH_CLIP_MIN) & (z_patch < DEPTH_CLIP_MAX)
    z_safe = torch.where(valid, z_patch, torch.zeros_like(z_patch))
    n_valid = valid.sum(dim=-1, keepdim=True).clamp(min=1).float()
    z = z_safe.sum(dim=-1, keepdim=True) / n_valid       # (B, 1) — valid-mean

    u_f = uv[:, 0:1].float()
    v_f = uv[:, 1:2].float()
    x = (u_f - K_CX_CAM) * z / K_FX_CAM
    y = (v_f - K_CY_CAM) * z / K_FY_CAM
    return torch.cat([x, y, z], dim=-1)                  # (B, 3)


def depth_to_scene_pc_hybrid(depth: torch.Tensor, uv: torch.Tensor,
                               n_sparse: int = 2048, n_dense: int = 2048,
                               dense_crop: int = 192) -> torch.Tensor:
    """[v9 fix #4] Hybrid PC: sparse (전체 stride) + dense ((u,v) 주변 crop).

    Returns (B, n_sparse + n_dense, 3) in camera frame.
    """
    B, _, H, W = depth.shape

    # 1. Sparse: 전체 image stride sub-sample
    total = H * W
    stride_s = max(1, total // n_sparse)
    idx_s = torch.arange(0, n_sparse, device=depth.device) * stride_s
    idx_s = torch.clamp(idx_s, max=total - 1)
    z_flat = depth.view(B, -1)
    z_s = z_flat.index_select(1, idx_s)                   # (B, n_sparse)
    u_s = (idx_s % W).float()                             # (n_sparse,)
    v_s = (idx_s // W).float()                            # (n_sparse,)
    x_s = (u_s[None] - K_CX_CAM) * z_s / K_FX_CAM         # (B, n_sparse)
    y_s = (v_s[None] - K_CY_CAM) * z_s / K_FY_CAM
    pc_sparse = torch.stack([x_s, y_s, z_s], dim=-1)      # (B, n_sparse, 3)

    # 2. Dense: (u,v) 주변 crop sub-sample
    crop_d = crop_around_uv(depth, uv, crop=dense_crop)    # (B, 1, crop, crop)
    z_d = crop_d.squeeze(1)                                # (B, crop, crop)
    ys = torch.linspace(-dense_crop / 2.0, dense_crop / 2.0 - 1, dense_crop,
                         device=depth.device)
    xs = torch.linspace(-dense_crop / 2.0, dense_crop / 2.0 - 1, dense_crop,
                         device=depth.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    u_img = gx.unsqueeze(0) + uv[:, 0:1, None]
    v_img = gy.unsqueeze(0) + uv[:, 1:2, None]
    x_d = (u_img - K_CX_CAM) * z_d / K_FX_CAM
    y_d = (v_img - K_CY_CAM) * z_d / K_FY_CAM
    pc_d = torch.stack([x_d, y_d, z_d], dim=-1).reshape(B, dense_crop * dense_crop, 3)
    stride_d = max(1, (dense_crop * dense_crop) // n_dense)
    idx_d = torch.arange(0, n_dense, device=depth.device) * stride_d
    idx_d = torch.clamp(idx_d, max=dense_crop * dense_crop - 1)
    pc_dense = pc_d.index_select(1, idx_d)                # (B, n_dense, 3)

    pc = torch.cat([pc_sparse, pc_dense], dim=1)          # (B, n_sparse + n_dense, 3)

    # invalid depth (clip 범위 밖) → z=-1e3 마스킹
    invalid = (pc[..., 2:3] < DEPTH_CLIP_MIN) | (pc[..., 2:3] > DEPTH_CLIP_MAX)
    pc = torch.where(invalid, torch.full_like(pc, -1e3), pc)
    return pc


def ball_query_topk(scene_pc_local: torch.Tensor, n_pts: int) -> torch.Tensor:
    """[v9 fix #2] Anchor-centered scene PC 에서 anchor (=원점) 으로부터
    가장 가까운 n_pts 점 선택. cdist 회피 — squared-norm broadcast 만 사용.

    Args:
        scene_pc_local: (B, N, 3), already anchor-centred (anchor at origin).
    Returns:
        (B, n_pts, 3)
    """
    dist_sq = (scene_pc_local ** 2).sum(dim=-1)            # (B, N)
    # top-k closest (largest=False on negative dist)
    _, idx = (-dist_sq).topk(n_pts, dim=1)                 # (B, n_pts)
    pc_local = scene_pc_local.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
    return pc_local


class MultiScaleBallEncoder(nn.Module):
    """[v9] Multi-scale ball-query around anchor → MiniPointNet × N.
    Anchor-centered input. Returns list of (B, out_dim_per_scale)."""

    def __init__(self, n_pts_per_scale: tuple = (256, 512, 1024),
                  out_dim_per_scale: int = 128):
        super().__init__()
        self.n_pts_per_scale = n_pts_per_scale
        self.encoders = nn.ModuleList([
            MiniPointNet(in_dim=3, out_dim=out_dim_per_scale)
            for _ in n_pts_per_scale
        ])
        self.out_dim_per_scale = out_dim_per_scale
        self.n_scales = len(n_pts_per_scale)

    def forward(self, scene_pc_local: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for n_pts, enc in zip(self.n_pts_per_scale, self.encoders):
            pc = ball_query_topk(scene_pc_local, n_pts)
            feats.append(enc(pc))
        return feats


class FlowGraspNetPC(nn.Module):
    """[v9 PC-only] FlowGraspNet 의 PC 입력 버전. 인터페이스 동일 (depth, uv, g_t, t).
    내부에서 depth → PC 변환 + anchor-centered + multi-scale ball-query.
    기존 FlowGraspNet 은 변경 없음.
    """

    def __init__(
        self,
        g_dim: int = 9,                              # 9D Zhou (default)
        global_dim: int = 128,
        local_dim: int = 128,
        t_dim: int = 64,
        hidden: int = 768,
        n_blocks: int = 8,
        cond_dropout: float = 0.2,
        block_type: str = "adaln_zero",
        use_xattn: bool = True,                      # default ON for v9
        n_pts_per_scale: tuple = (256, 512, 1024),    # multi-scale closest-k
        n_sparse: int = 2048,
        n_dense: int = 2048,
        dense_crop: int = 192,
    ):
        super().__init__()
        self.global_enc = MiniPointNet(in_dim=3, out_dim=global_dim)
        self.local_enc = MultiScaleBallEncoder(
            n_pts_per_scale=n_pts_per_scale, out_dim_per_scale=local_dim)
        self.n_local_scales = len(n_pts_per_scale)
        self.n_sparse = n_sparse
        self.n_dense = n_dense
        self.dense_crop = dense_crop

        cond_dim = global_dim + local_dim * self.n_local_scales   # 128 + 384 = 512
        self.cond_dropout = cond_dropout
        self.aux_mode_head = nn.Linear(cond_dim, 3)

        self.use_xattn = use_xattn
        if use_xattn:
            self.global_token_proj = nn.Linear(global_dim, cond_dim)
            self.local_token_projs = nn.ModuleList([
                nn.Linear(local_dim, cond_dim) for _ in range(self.n_local_scales)
            ])
            self.time_token_proj = nn.Linear(t_dim, cond_dim)
            self.uv_token_proj = nn.Linear(2, cond_dim)
        self.velocity = VelocityMLP(
            g_dim=g_dim, cond_dim=cond_dim,
            t_dim=t_dim, hidden=hidden, n_blocks=n_blocks,
            block_type=block_type, use_xattn=use_xattn,
        )
        self.block_type = block_type

    def _encode_features(self, depth: torch.Tensor, uv: torch.Tensor):
        """Returns (g_feat, l_feats, l_concat, anchor3d).
        anchor3d 는 추론 시 위치 복원용으로 외부 노출."""
        anchor3d = depth_to_anchor3d(depth, uv)              # (B, 3)
        scene_pc = depth_to_scene_pc_hybrid(
            depth, uv, n_sparse=self.n_sparse, n_dense=self.n_dense,
            dense_crop=self.dense_crop)                       # (B, N, 3)
        # [v9 fix #3] anchor-centered: PC 좌표를 anchor 원점 기준으로
        scene_pc_local = scene_pc - anchor3d[:, None, :]      # (B, N, 3)
        g_feat = self.global_enc(scene_pc_local)              # (B, 128)
        l_feats = self.local_enc(scene_pc_local)              # list of (B, 128)
        l_concat = torch.cat(l_feats, dim=-1)                 # (B, 384)
        return g_feat, l_feats, l_concat, anchor3d

    def encode(self, depth: torch.Tensor, uv: torch.Tensor):
        """추론용: cond 만 반환 (B, cond_dim)."""
        g_feat, _, l_concat, _ = self._encode_features(depth, uv)
        return torch.cat([g_feat, l_concat], dim=-1)

    def _build_cond_tokens(self, g_feat, l_feats, t_emb, uv_norm):
        tokens = [self.global_token_proj(g_feat)]
        for proj, l in zip(self.local_token_projs, l_feats):
            tokens.append(proj(l))
        tokens.append(self.time_token_proj(t_emb))
        tokens.append(self.uv_token_proj(uv_norm))
        return torch.stack(tokens, dim=1)

    def forward(self, depth, uv, g_t, t):
        g_feat, l_feats, l_concat, _ = self._encode_features(depth, uv)
        cond = torch.cat([g_feat, l_concat], dim=-1)
        if self.training and self.cond_dropout > 0:
            mask = (torch.rand(cond.shape[0], 1, device=cond.device)
                    > self.cond_dropout).float()
            cond = cond * mask
        t_emb = sinusoidal_time_embed(t, dim=64)
        uv_norm = torch.stack([uv[:, 0] / IMG_W, uv[:, 1] / IMG_H], dim=-1)
        if self.use_xattn:
            cond_tokens = self._build_cond_tokens(g_feat, l_feats, t_emb, uv_norm)
            if self.training and self.cond_dropout > 0:
                cond_tokens = cond_tokens * mask.unsqueeze(-1)
            return self.velocity(g_t, cond, t_emb, uv_norm, cond_tokens=cond_tokens)
        return self.velocity(g_t, cond, t_emb, uv_norm)

    def forward_with_aux(self, depth, uv, g_t, t):
        g_feat, l_feats, l_concat, _ = self._encode_features(depth, uv)
        cond = torch.cat([g_feat, l_concat], dim=-1)
        mode_logits = self.aux_mode_head(cond)
        cond_for_velocity = cond
        mask = None
        if self.training and self.cond_dropout > 0:
            mask = (torch.rand(cond.shape[0], 1, device=cond.device)
                    > self.cond_dropout).float()
            cond_for_velocity = cond * mask
        t_emb = sinusoidal_time_embed(t, dim=64)
        uv_norm = torch.stack([uv[:, 0] / IMG_W, uv[:, 1] / IMG_H], dim=-1)
        if self.use_xattn:
            cond_tokens = self._build_cond_tokens(g_feat, l_feats, t_emb, uv_norm)
            if mask is not None:
                cond_tokens = cond_tokens * mask.unsqueeze(-1)
            v = self.velocity(g_t, cond_for_velocity, t_emb, uv_norm,
                               cond_tokens=cond_tokens)
        else:
            v = self.velocity(g_t, cond_for_velocity, t_emb, uv_norm)
        return v, mode_logits


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


class CrossAttnBlock(nn.Module):
    """[A1] g_t (query) ↔ cond_tokens (key/value) cross-attention.

    Zero-init out_proj → block starts as identity (DiT 관례: 학습 초반 v7 동작 보존).
    ONNX-safe: 명시적 einsum + reshape, scaled_dot_product_attention 미사용 (opset 17 호환).
    """
    def __init__(self, dim: int, token_dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0, f"hidden dim {dim} not divisible by n_heads {n_heads}"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(token_dim, dim, bias=False)
        self.v_proj = nn.Linear(token_dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(token_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        """x: (B, dim) query feature.
        cond_tokens: (B, T, token_dim) — multiple condition tokens.
        """
        B = x.size(0)
        T = cond_tokens.size(1)
        q = self.q_proj(self.ln_q(x))                                 # (B, dim)
        kv = self.ln_kv(cond_tokens)
        k = self.k_proj(kv)                                            # (B, T, dim)
        v = self.v_proj(kv)
        # multi-head reshape
        q = q.view(B, self.n_heads, 1, self.head_dim)                  # (B, H, 1, d)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, d)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.einsum("bhqd,bhkd->bhqk", q, k) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        ctx = torch.einsum("bhqk,bhkd->bhqd", attn, v)                 # (B, H, 1, d)
        ctx = ctx.squeeze(2).reshape(B, self.dim)
        return x + self.out_proj(ctx)


BLOCK_TYPES = {"film": FiLMBlock, "adaln_zero": AdaLNZeroBlock}


class VelocityMLP(nn.Module):
    def __init__(self, g_dim: int = 8, cond_dim: int = 256, t_dim: int = 64,
                 hidden: int = 512, n_blocks: int = 4, block_type: str = "adaln_zero",
                 use_xattn: bool = False):
        super().__init__()
        self.in_proj = nn.Linear(g_dim, hidden)
        self.cond_proj = nn.Linear(cond_dim + t_dim + 2, cond_dim)  # +uv(2)
        Block = BLOCK_TYPES[block_type]
        self.blocks = nn.ModuleList([
            Block(hidden, cond_dim) for _ in range(n_blocks)
        ])
        self.use_xattn = use_xattn
        if use_xattn:
            self.cross_blocks = nn.ModuleList([
                CrossAttnBlock(hidden, cond_dim) for _ in range(n_blocks)
            ])
        self.out = nn.Linear(hidden, g_dim)
        self.cond_dim = cond_dim
        self.block_type = block_type

    def forward(self, g_t, cond_feat, t_emb, uv_norm, cond_tokens=None):
        cond = torch.cat([cond_feat, t_emb, uv_norm], dim=-1)
        cond = self.cond_proj(cond)
        h = self.in_proj(g_t)
        if self.use_xattn:
            assert cond_tokens is not None, "cond_tokens required for use_xattn"
            for blk, cattn in zip(self.blocks, self.cross_blocks):
                h = cattn(h, cond_tokens)        # 1) Cross-Attn (token routing)
                h = blk(h, cond)                  # 2) AdaLN-Zero (variable scale/shift)
        else:
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
        use_xattn: bool = False,                     # v8 [A1] Cross-Attention
        multiscale_local_scales: tuple | None = None,  # v8 [A2] Multi-Scale Crop, e.g. (96,192,384)
        scale_dropout: float = 0.0,                   # v8 [A2 안전장치] random scale drop
        use_pc: bool = False,                         # v9 [B1] Point Cloud token
        pc_n_points: int = 512,                       # v9 [B1] PC subsample size
        pc_dim: int = 128,                            # v9 [B1] PC feature dim
        pc_crop: int = 192,                           # v9 [B1] PC source crop size
    ):
        super().__init__()
        self.global_enc = GlobalDepthEncoder(global_dim)
        self.use_multiscale = multiscale_local_scales is not None
        if self.use_multiscale:
            self.local_enc = MultiScaleLocalCropEncoder(
                out_dim_per_scale=local_dim,
                scales=tuple(multiscale_local_scales))
            self.n_local_scales = len(multiscale_local_scales)
        else:
            self.local_enc = LocalCropEncoder(local_dim)
            self.n_local_scales = 1
        self.scale_dropout = scale_dropout
        cond_dim = global_dim + local_dim * self.n_local_scales   # v7: 256, v8 multiscale 3: 512
        self.cond_dropout = cond_dropout
        self.aux_mode_head = nn.Linear(cond_dim, 3)  # lying/standing/cube
        self.use_xattn = use_xattn
        # v9 PC branch (Option B: cross-attn token only, cond_dim 유지)
        self.use_pc = use_pc
        self.pc_n_points = pc_n_points
        self.pc_crop = pc_crop
        if use_pc:
            self.pc_enc = MiniPointNet(in_dim=3, out_dim=pc_dim)
        if use_xattn:
            # cond tokens: global(1) + local×N + time(1) + uv(1) [+ pc(1) if v9]
            self.global_token_proj = nn.Linear(global_dim, cond_dim)
            self.local_token_projs = nn.ModuleList([
                nn.Linear(local_dim, cond_dim) for _ in range(self.n_local_scales)
            ])
            self.time_token_proj = nn.Linear(t_dim, cond_dim)
            self.uv_token_proj = nn.Linear(2, cond_dim)
            if use_pc:
                self.pc_token_proj = nn.Linear(pc_dim, cond_dim)
        self.velocity = VelocityMLP(
            g_dim=g_dim, cond_dim=cond_dim,
            t_dim=t_dim, hidden=hidden, n_blocks=n_blocks,
            block_type=block_type, use_xattn=use_xattn,
        )
        self.block_type = block_type

    def _encode_features(self, depth: torch.Tensor, uv: torch.Tensor):
        """Returns (g_feat, l_feats_list, l_feat_concat). l_feats_list 길이 = n_local_scales."""
        d = clip_norm_depth(depth)
        g_feat = self.global_enc(d)
        if self.use_multiscale:
            l_feats = self.local_enc.forward_per_scale(d, uv,
                                                        scale_dropout=self.scale_dropout)
            l_feat_concat = torch.cat(l_feats, dim=-1)
        else:
            crop = crop_around_uv(d, uv)
            l_feat = self.local_enc(crop)
            l_feat_concat = l_feat
            l_feats = [l_feat]
        return g_feat, l_feats, l_feat_concat

    def _encode_pc_feat(self, depth: torch.Tensor, uv: torch.Tensor):
        """[v9 PC 추가 분기] use_pc 시에만 호출. 별도 함수로 분리해 기존 _encode_features 변경 없음."""
        pc = depth_to_pc(depth, uv, crop=self.pc_crop, n_points=self.pc_n_points)
        return self.pc_enc(pc)

    def encode(self, depth: torch.Tensor, uv: torch.Tensor):
        """추론용 backward-compatible: cond 만 반환 (B, cond_dim)."""
        g_feat, _, l_feat_concat = self._encode_features(depth, uv)
        return torch.cat([g_feat, l_feat_concat], dim=-1)

    def _build_cond_tokens(self, g_feat, l_feats, t_emb, uv_norm, pc_feat=None):
        """Cross-attn 용 토큰 생성. (B, T, cond_dim) where
        T = 1 (global) + n_local_scales + 2 (time, uv) [+ 1 if pc]."""
        tokens = [self.global_token_proj(g_feat)]
        for proj, l in zip(self.local_token_projs, l_feats):
            tokens.append(proj(l))
        tokens.append(self.time_token_proj(t_emb))
        tokens.append(self.uv_token_proj(uv_norm))
        if self.use_pc and pc_feat is not None:
            tokens.append(self.pc_token_proj(pc_feat))
        return torch.stack(tokens, dim=1)

    def forward(self, depth, uv, g_t, t):
        g_feat, l_feats, l_concat = self._encode_features(depth, uv)
        cond = torch.cat([g_feat, l_concat], dim=-1)
        if self.training and self.cond_dropout > 0:
            mask = (torch.rand(cond.shape[0], 1, device=cond.device)
                    > self.cond_dropout).float()
            cond = cond * mask
        t_emb = sinusoidal_time_embed(t, dim=64)
        uv_norm = torch.stack([uv[:, 0] / IMG_W, uv[:, 1] / IMG_H], dim=-1)
        if self.use_xattn:
            pc_feat = self._encode_pc_feat(depth, uv) if self.use_pc else None
            cond_tokens = self._build_cond_tokens(g_feat, l_feats, t_emb, uv_norm,
                                                    pc_feat=pc_feat)
            if self.training and self.cond_dropout > 0:
                cond_tokens = cond_tokens * mask.unsqueeze(-1)
            return self.velocity(g_t, cond, t_emb, uv_norm, cond_tokens=cond_tokens)
        return self.velocity(g_t, cond, t_emb, uv_norm)

    def forward_with_aux(self, depth, uv, g_t, t):
        g_feat, l_feats, l_concat = self._encode_features(depth, uv)
        cond = torch.cat([g_feat, l_concat], dim=-1)
        # aux_mode_head uses PRE-dropout cond
        mode_logits = self.aux_mode_head(cond)
        cond_for_velocity = cond
        mask = None
        if self.training and self.cond_dropout > 0:
            mask = (torch.rand(cond.shape[0], 1, device=cond.device)
                    > self.cond_dropout).float()
            cond_for_velocity = cond * mask
        t_emb = sinusoidal_time_embed(t, dim=64)
        uv_norm = torch.stack([uv[:, 0] / IMG_W, uv[:, 1] / IMG_H], dim=-1)
        if self.use_xattn:
            pc_feat = self._encode_pc_feat(depth, uv) if self.use_pc else None
            cond_tokens = self._build_cond_tokens(g_feat, l_feats, t_emb, uv_norm,
                                                    pc_feat=pc_feat)
            if mask is not None:
                cond_tokens = cond_tokens * mask.unsqueeze(-1)
            v = self.velocity(g_t, cond_for_velocity, t_emb, uv_norm,
                               cond_tokens=cond_tokens)
        else:
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
