"""Table 1 — 정량 비교 (Pos MAE / Ang Err / Mode Coverage).

Direct MLP 학습 완료 후 실행.
val split (random6) 전체 object 에 대해:
  - Direct: 1 grasp 출력 → GT 집합 중 nearest 1개와 비교
  - Flow:   N=32 grasps → GT 집합과 매칭 (Hungarian or set-coverage)
  - Mode Coverage: GT 의 grasp_group 중 예측이 표현한 비율

출력: paper_figs/table1.json + table1.md (LaTeX/markdown 둘 다)
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import sys
import cv2
import h5py
import numpy as np
import torch

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
sys.path.insert(0, str(ROOT))
from src.flow_model import FlowGraspNet, IMG_H, IMG_W
from src.direct_model import DirectGraspNet

DATASET = ROOT / "datasets/grasp_v2.h5"
OUT = ROOT / "paper_figs"
OUT.mkdir(parents=True, exist_ok=True)

FLOW_CKPT = ROOT / "runs/yolograsp_v2/v7_v4policy_big/adaln_zero_lr0.001_nb12_h1024/checkpoints/best.pt"
DIRECT_CKPT = ROOT / "runs/yolograsp_v2/v7_direct_mlp_big/direct_mlp_lr0.001_nb12_h1024/checkpoints/best.pt"

N_SAMPLES = 32
T_EULER = 32
NOISE_TEMP = 0.8
CFG_W = 2.5                           # Fig 3 와 통일
DIRECT_UV_JITTER_PX = 2.0
DIST_FILTER_M = 0.15                  # 객체 중심 outlier 제거 (lying 6cm, cube 5cm 보다 여유)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_ckpt", default=str(FLOW_CKPT))
    ap.add_argument("--direct_ckpt", default=str(DIRECT_CKPT))
    ap.add_argument("--max_objs", type=int, default=300,
                    help="평가할 val object 개수 (속도 절충)")
    return ap.parse_args()


@torch.no_grad()
def infer_flow(model, depth, uv, n=32, t_steps=32, temp=0.8, w_cfg=3.5):
    """CFG: cond_dropout 학습 일관성 위해 cond 벡터를 zero out (depth zero 가 아님)."""
    from src.flow_model import sinusoidal_time_embed, IMG_W as W, IMG_H as H
    device = next(model.parameters()).device
    d = torch.from_numpy(depth)[None, None].float().to(device)
    u = torch.from_numpy(uv)[None].float().to(device)
    d_b = d.expand(n, -1, -1, -1).contiguous()
    u_b = u.expand(n, -1).contiguous()
    cond_on = model.encode(d_b, u_b)
    cond_off = torch.zeros_like(cond_on)
    uv_norm = torch.stack([u_b[:, 0] / W, u_b[:, 1] / H], dim=-1)
    g_t = torch.randn(n, 8, device=device) * temp
    for k in range(t_steps):
        t = torch.full((n,), k / t_steps, device=device)
        t_emb = sinusoidal_time_embed(t, dim=64)
        v_on = model.velocity(g_t, cond_on, t_emb, uv_norm)
        v_off = model.velocity(g_t, cond_off, t_emb, uv_norm)
        v = v_off + w_cfg * (v_on - v_off)
        g_t = g_t + v / t_steps
    return g_t.cpu().numpy()


@torch.no_grad()
def infer_direct(model, depth, uv, n=1, uv_jitter_px=0.0):
    """Direct 도 N 개: uv ±jitter (target 검출 불확실성 모사)."""
    device = next(model.parameters()).device
    d = torch.from_numpy(depth)[None, None].float().to(device)
    u = torch.from_numpy(uv)[None].float().to(device)
    d_b = d.expand(n, -1, -1, -1).contiguous()
    u_b = u.expand(n, -1).contiguous()
    if uv_jitter_px > 0 and n > 1:
        noise = torch.randn(n, 2, device=device) * uv_jitter_px
        noise[0] = 0.0
        u_b = u_b + noise
    g, _ = model.forward_with_aux(d_b, u_b)
    return g.cpu().numpy()


POS_TH_M = 0.05      # 5 cm  (Contact-GraspNet ICRA 2021 / ACRONYM 관례)
ANG_TH_DEG = 30.0    # 30°


def compute_metrics(pred_8d, gt_grasps_7d, gt_groups, pos_mean, pos_std):
    """pred_8d (M, 8) — pos 정규화 해제.
    gt_grasps_7d (G, 7) cam frame [x,y,z, qw..qz]
    gt_groups (G,) int

    Returns:
      pos_mae(cm)   — pred-side: 각 pred 의 nearest GT 까지 평균 위치 오차
      ang_err(deg)  — pred-side: 각 pred 의 nearest GT 와의 approach 각도 오차
      coverage(%)   — GT-side: GT group 중 (pos<5cm & ang<30°) 인 pred 가 하나 이상인 비율 [Achlioptas ICML 2018]
      apd(cm)       — Average Pairwise Distance: 예측들 간 평균 쌍별 위치 거리 (다양성)
    """
    pred = pred_8d.copy()
    pred[:, :3] = pred[:, :3] * pos_std + pos_mean
    pos_pred = pred[:, :3]                              # (M, 3)
    app_pred = pred[:, 3:6] / (np.linalg.norm(pred[:, 3:6], axis=1, keepdims=True) + 1e-9)

    pos_gt = gt_grasps_7d[:, :3]                        # (G, 3)
    app_gt = np.zeros((len(gt_grasps_7d), 3), dtype=np.float32)
    for i, q in enumerate(gt_grasps_7d[:, 3:7]):
        w, x, y, z = q
        app_gt[i] = [2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)]

    # ---- pred-side: per-pred nearest GT ----
    pos_errs, ang_errs = [], []
    for m in range(len(pred)):
        d_pos = np.linalg.norm(pos_gt - pos_pred[m], axis=1)
        nearest = int(np.argmin(d_pos))
        pos_errs.append(float(d_pos[nearest]) * 100)
        cos = float(np.clip(app_pred[m] @ app_gt[nearest], -1, 1))
        ang_errs.append(float(np.degrees(np.arccos(abs(cos)))))

    # ---- GT-side group coverage: GT group 별로 어떤 pred 라도 (5cm, 30°) 안에 있나? ----
    unique_groups = sorted(set(int(g) for g in gt_groups))
    covered = set()
    for grp in unique_groups:
        gt_mask = (gt_groups == grp)
        for gi in np.where(gt_mask)[0]:
            d_pos = np.linalg.norm(pos_pred - pos_gt[gi], axis=1)
            cos = np.clip(app_pred @ app_gt[gi], -1, 1)
            ang = np.degrees(np.arccos(np.abs(cos)))
            if np.any((d_pos < POS_TH_M) & (ang < ANG_TH_DEG)):
                covered.add(grp); break

    coverage = 100.0 * len(covered) / max(len(unique_groups), 1)

    # ---- APD: 예측들 간 평균 쌍별 위치 거리 (cm) ----
    if len(pos_pred) > 1:
        diffs = pos_pred[:, None, :] - pos_pred[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        iu = np.triu_indices(len(pos_pred), k=1)
        apd = float(dists[iu].mean()) * 100
    else:
        apd = 0.0

    return float(np.mean(pos_errs)), float(np.mean(ang_errs)), coverage, len(unique_groups), apd


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    fck = torch.load(args.flow_ckpt, weights_only=False, map_location=device)
    cfg_f = fck["cfg"]["args"]
    flow = FlowGraspNet(block_type=cfg_f["block"], n_blocks=cfg_f["n_blocks"],
                        hidden=cfg_f["hidden"], cond_dropout=cfg_f["cond_dropout"]).to(device)
    flow.load_state_dict(fck["ema"], strict=False); flow.eval()
    flow_norm = fck["norm_stats"]
    print(f"[load] Flow ep={fck['epoch']} val={fck.get('val_loss'):.4f}")

    dck = torch.load(args.direct_ckpt, weights_only=False, map_location=device)
    cfg_d = dck["cfg"]["args"]
    direct = DirectGraspNet(n_blocks=cfg_d["n_blocks"], hidden=cfg_d["hidden"],
                            cond_dropout=cfg_d.get("cond_dropout", 0.0)).to(device)
    direct.load_state_dict(dck["ema"], strict=False); direct.eval()
    direct_norm = dck["norm_stats"]
    print(f"[load] Direct ep={dck['epoch']} val={dck.get('val_loss'):.4f}")

    # Load val data
    f = h5py.File(DATASET, "r")
    val = f["val"]
    depths = val["depths"][:]              # (n_unique_depths, H, W)
    depth_ref = val["depth_ref"][:]         # (N_rows,)
    uvs = val["uvs"][:]                     # (N_rows, 2)
    grasps_cam = val["grasps_cam"][:]       # (N_rows, 7)
    grasp_group = val["grasp_group"][:]     # (N_rows,)
    object_ref = val["object_ref"][:]       # (N_rows,) per-split global obj id

    # group rows by object_ref
    obj_ids = np.unique(object_ref)[: args.max_objs]
    print(f"[eval] {len(obj_ids)} val objects")

    object_mode = val["object_mode"][:]
    MODE_NAMES = {0: "lying", 1: "standing", 2: "cube"}

    # results: per (model, mode_name) → list of metric tuples
    results = {("flow", m): [] for m in ("standing","lying","cube","all")}
    results.update({("direct", m): [] for m in ("standing","lying","cube","all")})

    for oi, oid in enumerate(obj_ids):
        idx = np.where(object_ref == oid)[0]
        depth = depths[depth_ref[idx[0]]]
        uv = uvs[idx[0]]
        gt_grasps = grasps_cam[idx]
        gt_groups = grasp_group[idx]
        mode_name = MODE_NAMES.get(int(object_mode[idx[0]]), "?")

        g_flow_raw = infer_flow(flow, depth, uv,
                                n=N_SAMPLES, t_steps=T_EULER,
                                temp=NOISE_TEMP, w_cfg=CFG_W)
        # 객체 중심 distance 필터 (Fig 3 와 동일한 outlier 제거)
        pos_dn = (g_flow_raw[:, :3] * np.array(flow_norm["pos_std"])
                  + np.array(flow_norm["pos_mean"]))
        gt_center = gt_grasps[:, :3].mean(axis=0)
        keep = np.linalg.norm(pos_dn - gt_center, axis=1) < DIST_FILTER_M
        g_flow = g_flow_raw[keep] if keep.any() else g_flow_raw
        g_direct = infer_direct(direct, depth, uv,
                                n=N_SAMPLES, uv_jitter_px=DIRECT_UV_JITTER_PX)

        m_flow = compute_metrics(g_flow, gt_grasps, gt_groups,
                                  np.array(flow_norm["pos_mean"]),
                                  np.array(flow_norm["pos_std"]))
        m_direct = compute_metrics(g_direct, gt_grasps, gt_groups,
                                    np.array(direct_norm["pos_mean"]),
                                    np.array(direct_norm["pos_std"]))
        results[("flow", mode_name)].append(m_flow)
        results[("flow", "all")].append(m_flow)
        results[("direct", mode_name)].append(m_direct)
        results[("direct", "all")].append(m_direct)
        if (oi+1) % 50 == 0:
            print(f"  [{oi+1}/{len(obj_ids)}]  ({mode_name:<8}) "
                  f"flow=({m_flow[0]:.1f}cm,{m_flow[1]:.1f}°,COV={m_flow[2]:.0f}%/{m_flow[3]}grp,APD={m_flow[4]:.2f}cm)  "
                  f"direct=({m_direct[0]:.1f}cm,{m_direct[1]:.1f}°,COV={m_direct[2]:.0f}%/{m_direct[3]}grp,APD={m_direct[4]:.2f}cm)")

    f.close()

    summary = {}
    for name in ["flow", "direct"]:
        summary[name] = {}
        for mode in ("standing","lying","cube","all"):
            arr = np.array(results[(name, mode)]) if results[(name, mode)] else np.zeros((0,5))
            if len(arr) == 0:
                summary[name][mode] = {"n": 0}; continue
            summary[name][mode] = {
                "pos_mae_cm": float(arr[:, 0].mean()),
                "ang_err_deg": float(arr[:, 1].mean()),
                "mode_coverage_pct": float(arr[:, 2].mean()),
                "n": int(len(arr)),
                "mean_groups_in_gt": float(arr[:, 3].mean()),
                "apd_cm": float(arr[:, 4].mean()),
            }

    out_json = OUT / "table1.json"
    out_json.write_text(json.dumps(summary, indent=2))

    md = "# Table 1. Quantitative comparison (val split, random6)\n\n"
    md += f"_Pos threshold {POS_TH_M*100:.0f} cm, Ang threshold {ANG_TH_DEG:.0f}° "
    md += "(Contact-GraspNet ICRA 2021 / ACRONYM)_\n"
    md += f"_N_samples={N_SAMPLES}, CFG={CFG_W}, T_euler={T_EULER}._\n\n"
    md += "| Mode | Model | n_obj | GT groups | Pos. MAE [cm] ↓ | Ang. Err. [°] ↓ | **COV [%] ↑** | **APD [cm] ↑** |\n"
    md += "|---|---|---|---|---|---|---|---|\n"
    for mode in ("standing","lying","cube","all"):
        for name, label in [("direct","Direct MLP"), ("flow","**Ours (Flow)**")]:
            s = summary[name][mode]
            if s.get("n", 0) == 0: continue
            md += (f"| {mode} | {label} | {s['n']} | {s['mean_groups_in_gt']:.2f} | "
                   f"{s['pos_mae_cm']:.2f} | {s['ang_err_deg']:.2f} | "
                   f"{s['mode_coverage_pct']:.1f} | {s['apd_cm']:.2f} |\n")
    md += "\n_COV = Coverage [Achlioptas, ICML 2018]. APD = Average Pairwise Distance among predictions (diversity)._\n"
    (OUT / "table1.md").write_text(md)

    print("\n=== Table 1 ===")
    print(md)


if __name__ == "__main__":
    main()
