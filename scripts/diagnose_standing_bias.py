"""Standing bottle 의 mode 편향 진단 (학습 없음, v7 추론만).

가설: v7 가 같은 (depth, uv) 입력에 대해 N=64 노이즈로 샘플링했을 때
top-down / side-cap / side-45 가 GT 의 8/8/8 비율과 비슷하게 나오는가?
사용자 보고: side 만 압도적으로 나옴.

분석:
  1. 5 standing bottle 후보에 대해 N=64 grasp 샘플링 (CFG=2.5, 1.0, 0.0 비교)
  2. approach z 성분으로 자동 분류:
       top-down  : |a_z| > 0.7
       side      : |a_z| < 0.3
       mid       : 0.3 ≤ |a_z| ≤ 0.7
  3. 분포 비교 (예측 vs GT)
  4. 가능한 원인 가설 검증:
     a. CFG 가 한쪽 모드 증폭하는가? (CFG sweep)
     b. NOISE_TEMP 가 부족한가? (1.0/1.2/1.5 비교)
     c. cond dropout 학습 효과로 1 모드 우세?
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import torch
import h5py
from collections import Counter

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
sys.path.insert(0, str(ROOT))
from src.flow_model import FlowGraspNet, sinusoidal_time_embed, IMG_W, IMG_H

GRASP_H5 = ROOT / "img_dataset/grasp_cache/grasps.h5"
POSES_H5 = ROOT / "img_dataset/icp_cache/poses.h5"
FLOW_CKPT = ROOT / "runs/yolograsp_v2/v7_v4policy_big/adaln_zero_lr0.001_nb12_h1024/checkpoints/best.pt"
DATASET = ROOT / "datasets/grasp_v2.h5"

CASES = [
    ("sample_random6_31", 2),
    ("sample_random6_40", 0),
    ("sample_random6_32", 1),
    ("sample_random6_33", 2),
    ("sample_random6_43", 3),
]
N_SAMPLES = 64
T_EULER = 32

CFG_SWEEP = [0.0, 1.0, 2.5, 3.5]
TEMP_SWEEP = [0.8, 1.0, 1.2]


def load_flow():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(FLOW_CKPT, weights_only=False, map_location=device)
    cfg = ck["cfg"]["args"]
    model = FlowGraspNet(block_type=cfg["block"], n_blocks=cfg["n_blocks"],
                         hidden=cfg["hidden"], cond_dropout=cfg["cond_dropout"]).to(device)
    model.load_state_dict(ck["ema"], strict=False)
    model.eval()
    print(f"[load] Flow ep={ck['epoch']} val={ck.get('val_loss'):.4f}")
    return model, ck["norm_stats"], device


@torch.no_grad()
def sample_flow(model, depth, uv, n=64, t_steps=32, temp=0.8, w_cfg=2.5, seed=0):
    device = next(model.parameters()).device
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    d = torch.from_numpy(depth)[None, None].float().to(device)
    u = torch.from_numpy(uv)[None].float().to(device)
    d_b = d.expand(n, -1, -1, -1).contiguous()
    u_b = u.expand(n, -1).contiguous()
    cond_on = model.encode(d_b, u_b)
    cond_off = torch.zeros_like(cond_on)
    uv_norm = torch.stack([u_b[:, 0] / IMG_W, u_b[:, 1] / IMG_H], dim=-1)
    g_t = torch.randn(n, 8, device=device) * temp
    for k in range(t_steps):
        t = torch.full((n,), k / t_steps, device=device)
        t_emb = sinusoidal_time_embed(t, dim=64)
        v_on = model.velocity(g_t, cond_on, t_emb, uv_norm)
        if w_cfg > 0:
            v_off = model.velocity(g_t, cond_off, t_emb, uv_norm)
            v = v_off + w_cfg * (v_on - v_off)
        else:
            v = v_on
        g_t = g_t + v / t_steps
    return g_t.cpu().numpy()


def classify_modes(grasps_8d):
    """approach vec (cols 3,4,5) z 성분으로 분류. 학습 공간에서 a_z 는 [-1,1]."""
    app = grasps_8d[:, 3:6]
    app = app / (np.linalg.norm(app, axis=1, keepdims=True) + 1e-9)
    az = np.abs(app[:, 2])
    bins = np.where(az > 0.7, 'top',
                    np.where(az < 0.3, 'side', 'mid'))
    return Counter(bins.tolist()), app


def get_uv(g_h5_path, sid, idx):
    with h5py.File(g_h5_path, "r") as g:
        return np.asarray(g[sid][f"object_{idx}"]["uv_centroid"], dtype=np.float32)


def get_depth(p_h5_path, sid):
    import cv2
    with h5py.File(p_h5_path, "r") as p:
        depth_path = ROOT / p[sid].attrs["depth_path"]
    return cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0


def main():
    model, norm, device = load_flow()
    pos_mean = np.array(norm["pos_mean"]); pos_std = np.array(norm["pos_std"])

    print("\n" + "=" * 78)
    print("DIAGNOSIS 1: Default settings (CFG=2.5, temp=0.8, seed=0) per case")
    print("=" * 78)
    print("GT 분포: top=33% (8/24), side-cap=33% (8/24), side-45=33% (8/24)")
    print("자동 분류 임계값: top |a_z|>0.7,  mid 0.3-0.7,  side |a_z|<0.3")
    print()

    for sid, idx in CASES:
        depth = get_depth(POSES_H5, sid)
        uv = get_uv(GRASP_H5, sid, idx)
        grasps = sample_flow(model, depth, uv, n=N_SAMPLES, w_cfg=2.5, temp=0.8, seed=0)
        cnt, app = classify_modes(grasps)
        total = sum(cnt.values())
        az_dist = np.abs(app[:, 2])
        print(f"  [{sid}/obj{idx}]  N={N_SAMPLES}  "
              f"top={cnt.get('top',0):2d}({100*cnt.get('top',0)/total:.0f}%)  "
              f"mid={cnt.get('mid',0):2d}({100*cnt.get('mid',0)/total:.0f}%)  "
              f"side={cnt.get('side',0):2d}({100*cnt.get('side',0)/total:.0f}%)  "
              f"|a_z|: mean={az_dist.mean():.3f} std={az_dist.std():.3f}")

    print("\n" + "=" * 78)
    print("DIAGNOSIS 2: CFG sweep (sample_random6_31 obj 2 한 case)")
    print("=" * 78)
    sid, idx = CASES[0]
    depth = get_depth(POSES_H5, sid); uv = get_uv(GRASP_H5, sid, idx)
    for w_cfg in CFG_SWEEP:
        grasps = sample_flow(model, depth, uv, n=N_SAMPLES, w_cfg=w_cfg, temp=0.8, seed=0)
        cnt, _ = classify_modes(grasps)
        total = sum(cnt.values())
        print(f"  CFG={w_cfg:>3.1f}  "
              f"top={cnt.get('top',0):2d}({100*cnt.get('top',0)/total:.0f}%)  "
              f"mid={cnt.get('mid',0):2d}({100*cnt.get('mid',0)/total:.0f}%)  "
              f"side={cnt.get('side',0):2d}({100*cnt.get('side',0)/total:.0f}%)")

    print("\n" + "=" * 78)
    print("DIAGNOSIS 3: NOISE_TEMP sweep (CFG=2.5 fixed)")
    print("=" * 78)
    for temp in TEMP_SWEEP:
        grasps = sample_flow(model, depth, uv, n=N_SAMPLES, w_cfg=2.5, temp=temp, seed=0)
        cnt, _ = classify_modes(grasps)
        total = sum(cnt.values())
        print(f"  temp={temp:>3.1f}  "
              f"top={cnt.get('top',0):2d}({100*cnt.get('top',0)/total:.0f}%)  "
              f"mid={cnt.get('mid',0):2d}({100*cnt.get('mid',0)/total:.0f}%)  "
              f"side={cnt.get('side',0):2d}({100*cnt.get('side',0)/total:.0f}%)")

    print("\n" + "=" * 78)
    print("DIAGNOSIS 4: 다른 seed 5 회 평균 (분포 안정성)")
    print("=" * 78)
    sid, idx = CASES[0]
    depth = get_depth(POSES_H5, sid); uv = get_uv(GRASP_H5, sid, idx)
    accumulated = Counter()
    for seed in range(5):
        grasps = sample_flow(model, depth, uv, n=N_SAMPLES, w_cfg=2.5, temp=0.8, seed=seed)
        cnt, _ = classify_modes(grasps)
        accumulated.update(cnt)
    total = sum(accumulated.values())
    print(f"  5 seeds × 64 samples = 320 grasps")
    print(f"  top  = {accumulated.get('top',0):3d} ({100*accumulated.get('top',0)/total:.1f}%)")
    print(f"  mid  = {accumulated.get('mid',0):3d} ({100*accumulated.get('mid',0)/total:.1f}%)")
    print(f"  side = {accumulated.get('side',0):3d} ({100*accumulated.get('side',0)/total:.1f}%)")
    print()

    print("=" * 78)
    print("DIAGNOSIS 5: GT (학습 데이터) 의 standing bottle 분포 — sanity check")
    print("=" * 78)
    with h5py.File(DATASET, "r") as f:
        train = f["train"]
        cls = train["object_class"][:]
        mode = train["object_mode"][:]
        groups = train["grasp_group"][:]
        # bottle (cls=0) + standing (mode=1)
        mask = (cls == 0) & (mode == 1)
        gt_groups = groups[mask]
        gt_cnt = Counter(int(g) for g in gt_groups)
        total = len(gt_groups)
        print(f"  train standing bottle grasps: {total}")
        for g_id, name in [(0, "top-down"), (1, "side-cap"), (4, "side-45")]:
            c = gt_cnt.get(g_id, 0)
            print(f"    group {g_id} ({name:<10}): {c} ({100*c/total:.1f}%)")


if __name__ == "__main__":
    main()
