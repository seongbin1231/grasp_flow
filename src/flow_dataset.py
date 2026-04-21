"""
Dataset / Sampler for Flow Matching grasp model.

각 행 = grasp. 같은 object의 모든 grasp는 같은 (depth, uv) 공유 → flow가 다봉 학습.

Augmentation (online):
  1. Yaw 회전 (±180°) around cam Z (이미지 중앙 기준) — depth + uv + grasp 동시 회전
  2. uv jitter (±3 px)
  3. Depth noise (σ=1.5 mm on valid pixels)
  4. Depth random erasing (1~3개 patch, 24~48 px)
  5. Grasp GT jitter: ±2° yaw + ±2 mm xyz

Sampling:
  WeightedSampler: weight ∝ 1 / freq^0.5  (standing underfit 방지)
  + Mode-balanced epochs (옵션)
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

IMG_H, IMG_W = 720, 1280
CAM_CX, CAM_CY = 640.0, 360.0
K_FX = K_FY = 1109.0


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def rotate_grasp_by_cam_z(pos, approach, yaw, theta):
    """Rotate 3D grasp to match image rotation by cv2 angle `theta`.

    cv2.getRotationMatrix2D with positive theta rotates image CCW (visual, y-down).
    This corresponds to camera rotating by +theta around +Z (right-hand).
    3D points in cam frame then transform by R_z(-theta).
    In matrix form: [[ cos,  sin, 0],
                     [-sin,  cos, 0],
                     [   0,    0, 1]]
    """
    c, s = math.cos(theta), math.sin(theta)
    Rz = np.array([[ c,  s, 0],
                   [-s,  c, 0],
                   [ 0,  0, 1]], dtype=np.float64)
    new_pos = Rz @ pos
    R_tool = _build_R_tool(approach, yaw)
    R_tool_new = Rz @ R_tool
    app_new = R_tool_new[:, 2]
    yaw_new = _yaw_from_Rtool(R_tool_new)
    return new_pos, app_new, yaw_new


def _build_R_tool(approach, yaw):
    a = approach / (np.linalg.norm(approach) + 1e-9)
    ref = np.array([1.0, 0, 0.0]) if abs(a[0]) < 0.95 else np.array([0, 1.0, 0])
    b0 = ref - (ref @ a) * a; b0 /= np.linalg.norm(b0) + 1e-9
    n0 = np.cross(a, b0)
    b = b0 * math.cos(yaw) + n0 * math.sin(yaw); b /= np.linalg.norm(b) + 1e-9
    x = np.cross(b, a)
    return np.column_stack([x, b, a])


def _yaw_from_Rtool(R_tool):
    a = R_tool[:, 2]
    b_actual = R_tool[:, 1]
    ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.95 else np.array([0, 1.0, 0])
    b0 = ref - (ref @ a) * a; b0 /= np.linalg.norm(b0) + 1e-9
    n0 = np.cross(a, b0)
    cy = b_actual @ b0
    sy = b_actual @ n0
    return math.atan2(sy, cy)


def rotate_depth_around_center(depth: np.ndarray, theta_rad: float) -> np.ndarray:
    """Rotate depth image around (cx, cy) by theta (cw in pixel coords)."""
    M = cv2.getRotationMatrix2D((CAM_CX, CAM_CY), math.degrees(theta_rad), 1.0)
    return cv2.warpAffine(depth, M, (IMG_W, IMG_H),
                          flags=cv2.INTER_LINEAR, borderValue=0.0)


def rotate_uv_around_center(uv: np.ndarray, theta_rad: float) -> np.ndarray:
    """Rotate uv pixel with the same cv2 affine as the depth rotation, guaranteed consistent.

    cv2.getRotationMatrix2D returns M = [[c, s, ...], [-s, c, ...]] (2×3).
    Apply it directly to (u, v, 1) so uv follows wherever depth content moves.
    """
    M = cv2.getRotationMatrix2D((CAM_CX, CAM_CY), math.degrees(theta_rad), 1.0)
    pt = np.array([uv[0], uv[1], 1.0], dtype=np.float64)
    return (M @ pt).astype(np.float32)


def xy_img_rot_matches_xy_cam_rot(theta_rad: float) -> float:
    """Camera X in image = +x (right), Y = +y (down). Camera frame X = right, Y = down. So image rotation by theta == camera-frame rotation around +Z by theta (cam Z = into scene)."""
    return theta_rad


class GraspDataset(Dataset):
    """Reads grasp_v2.h5. Applies augmentation online.

    Returns dict:
        depth: (1, 720, 1280) float32
        uv:    (2,) float32 pixel
        g1:    (8,) float32  = [x_norm,y_norm,z_norm, ax,ay,az, sinψ, cosψ]
               (position z-scored using train-set stats; approach/sincos raw)
        mode:  () int64 (0 lying, 1 standing, 2 cube)
        group: () int64 (0 top-down, 1 side-cap, 2 lying, 3 cube)
        class: () int64 (0..4)

    Stats accessible via self.pos_mean (3,), self.pos_std (3,) — shared
    between train/val via always computing from train split.
    """

    def __init__(self, h5_path: str, split: str = "train", augment: bool = True,
                 yaw_aug: bool = True, uv_jitter_px: float = 3.0,
                 depth_noise_mm: float = 1.5, erase_prob: float = 0.5,
                 grasp_jitter_yaw_deg: float = 2.0, grasp_jitter_xyz_mm: float = 2.0,
                 preload_depth: bool = True, normalize_pos: bool = True):
        self.h5_path = h5_path
        self.split = split
        self.augment = augment
        self.yaw_aug = yaw_aug
        self.uv_jit = uv_jitter_px
        self.depth_noise_m = depth_noise_mm / 1000.0
        self.erase_prob = erase_prob
        self.jit_yaw = math.radians(grasp_jitter_yaw_deg)
        self.jit_xyz = grasp_jitter_xyz_mm / 1000.0
        self.normalize_pos = normalize_pos

        with h5py.File(h5_path, "r") as f:
            g = f[split]
            self.depth_ref = g["depth_ref"][:]
            self.uvs = g["uvs"][:]
            self.grasps_cam = g["grasps_cam"][:]
            self.approach_vec = g["approach_vec"][:]
            self.yaw_arr = g["yaw_around_app"][:]
            self.mode = g["object_mode"][:]
            self.group = g["grasp_group"][:]
            self.cls = g["object_class"][:]
            self.object_ref = g["object_ref"][:]
            if preload_depth:
                self.depths = g["depths"][:]
            else:
                self.depths = None

            # compute pos stats from TRAIN split (always; shared across train/val)
            train_pos = f["train/grasps_cam"][:, :3].astype(np.float32)
            self.pos_mean = train_pos.mean(axis=0)
            self.pos_std = train_pos.std(axis=0) + 1e-6
        self._h5_depth = None  # lazy for workers

        self.N = len(self.depth_ref)

    def _get_depth(self, dref: int) -> np.ndarray:
        if self.depths is not None:
            return self.depths[dref].copy()
        if self._h5_depth is None:
            self._h5_depth = h5py.File(self.h5_path, "r")[self.split]["depths"]
        return self._h5_depth[dref][:].copy()

    def __len__(self):
        return self.N

    def _apply_erase(self, depth):
        h, w = depth.shape
        n_patches = np.random.randint(1, 4)
        for _ in range(n_patches):
            ph = np.random.randint(24, 48)
            pw = np.random.randint(24, 48)
            y0 = np.random.randint(0, h - ph)
            x0 = np.random.randint(0, w - pw)
            depth[y0:y0+ph, x0:x0+pw] = 0.0
        return depth

    def __getitem__(self, i):
        dref = int(self.depth_ref[i])
        depth = self._get_depth(dref)          # (720, 1280) float32
        uv = self.uvs[i].astype(np.float32)
        pos = self.grasps_cam[i, :3].astype(np.float64)
        q = self.grasps_cam[i, 3:7].astype(np.float64)
        app = self.approach_vec[i].astype(np.float64)
        yaw = float(self.yaw_arr[i])

        if self.augment and self.yaw_aug:
            theta = np.random.uniform(-math.pi, math.pi)
            depth = rotate_depth_around_center(depth, theta)
            uv = rotate_uv_around_center(uv, theta)
            pos, app, yaw = rotate_grasp_by_cam_z(pos, app, yaw, theta)

        if self.augment and self.uv_jit > 0:
            uv = uv + np.random.uniform(-self.uv_jit, self.uv_jit, 2).astype(np.float32)
            uv[0] = np.clip(uv[0], 0, IMG_W - 1)
            uv[1] = np.clip(uv[1], 0, IMG_H - 1)

        if self.augment and self.depth_noise_m > 0:
            noise = np.random.normal(0, self.depth_noise_m, depth.shape).astype(np.float32)
            mask = depth > 0.1
            depth = depth + noise * mask.astype(np.float32)

        if self.augment and np.random.rand() < self.erase_prob:
            depth = self._apply_erase(depth)

        if self.augment and self.jit_yaw > 0:
            yaw = yaw + np.random.uniform(-self.jit_yaw, self.jit_yaw)
        if self.augment and self.jit_xyz > 0:
            pos = pos + np.random.uniform(-self.jit_xyz, self.jit_xyz, 3)

        # build 8D parameterization
        app_u = app / (np.linalg.norm(app) + 1e-9)
        pos_out = pos.astype(np.float32)
        if self.normalize_pos:
            pos_out = (pos_out - self.pos_mean) / self.pos_std
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)
        g1 = np.concatenate([pos_out,
                             app_u.astype(np.float32),
                             np.array([sin_yaw, cos_yaw], dtype=np.float32)])
        # symmetric alternative for lying: yaw+π (same pos, same approach, flipped sincos)
        # lying mode (id=0) has 180° yaw pairs at same position in GT
        mode_i = int(self.mode[i])
        if mode_i == 0:
            g1_alt = np.concatenate([pos_out,
                                     app_u.astype(np.float32),
                                     np.array([-sin_yaw, -cos_yaw], dtype=np.float32)])
        else:
            g1_alt = g1.copy()

        return {
            "depth": torch.from_numpy(depth).float().unsqueeze(0),  # (1, H, W)
            "uv": torch.from_numpy(uv).float(),
            "g1": torch.from_numpy(g1).float(),
            "g1_alt": torch.from_numpy(g1_alt).float(),
            "mode": torch.tensor(mode_i, dtype=torch.long),
            "group": torch.tensor(int(self.group[i]), dtype=torch.long),
            "class": torch.tensor(int(self.cls[i]), dtype=torch.long),
            "object_ref": torch.tensor(int(self.object_ref[i]), dtype=torch.long),
        }


def make_weighted_sampler(dataset: GraspDataset, mode_balance: bool = True,
                           power: float = 0.5,
                           class_boost: dict[int, float] | None = None
                           ) -> WeightedRandomSampler:
    """weight ∝ 1 / freq^power. Options:
      - mode_balance=True: inv-freq of mode (0 lying / 1 standing / 2 cube)
      - class_boost: {class_id: multiplier} — extra upweight for specific classes.
        e.g. {3: 2.0, 4: 1.5} means marker rows ×2, spam rows ×1.5.
    """
    if mode_balance:
        freq = np.bincount(dataset.mode, minlength=3).astype(np.float64)
        inv = 1.0 / np.maximum(freq, 1) ** power
        inv = inv / inv.sum()
        w = inv[dataset.mode]
    else:
        freq = np.bincount(dataset.group, minlength=4).astype(np.float64)
        inv = 1.0 / np.maximum(freq, 1) ** power
        w = inv[dataset.group]

    # class boost (multiplicative)
    if class_boost:
        boost = np.ones(5, dtype=np.float64)
        for k, v in class_boost.items():
            if 0 <= k < 5:
                boost[k] = float(v)
        w = w * boost[dataset.cls]

    # normalize
    w = w / w.sum() * len(dataset)
    return WeightedRandomSampler(torch.from_numpy(w).double(),
                                 num_samples=len(dataset), replacement=True)
