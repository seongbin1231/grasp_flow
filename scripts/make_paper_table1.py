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
CFG_W = 2.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_ckpt", default=str(FLOW_CKPT))
    ap.add_argument("--direct_ckpt", default=str(DIRECT_CKPT))
    ap.add_argument("--max_objs", type=int, default=300,
                    help="평가할 val object 개수 (속도 절충)")
    return ap.parse_args()


@torch.no_grad()
def infer_flow(model, depth, uv, n=32, t_steps=32, temp=0.8, w_cfg=2.0):
    device = next(model.parameters()).device
    d = torch.from_numpy(depth)[None, None].float().to(device)
    u = torch.from_numpy(uv)[None].float().to(device)
    d_b = d.expand(n, -1, -1, -1).contiguous()
    u_b = u.expand(n, -1).contiguous()
    g_t = torch.randn(n, 8, device=device) * temp
    for k in range(t_steps):
        t = torch.full((n,), k / t_steps, device=device)
        v_on = model(d_b, u_b, g_t, t)
        v_off = model(torch.zeros_like(d_b), u_b, g_t, t)
        v = v_off + w_cfg * (v_on - v_off)
        g_t = g_t + v / t_steps
    return g_t.cpu().numpy()


@torch.no_grad()
def infer_direct(model, depth, uv):
    device = next(model.parameters()).device
    d = torch.from_numpy(depth)[None, None].float().to(device)
    u = torch.from_numpy(uv)[None].float().to(device)
    g, _ = model.forward_with_aux(d, u)
    return g.cpu().numpy()  # (1, 8)


def compute_metrics(pred_8d, gt_grasps_7d, gt_groups, pos_mean, pos_std):
    """pred_8d (M, 8) — denormalize pos.
    gt_grasps_7d (G, 7) cam frame [x,y,z, qw..qz]
    gt_groups (G,) int

    Returns: pos_mae(cm), ang_err(deg), mode_coverage(%)
    """
    pred = pred_8d.copy()
    pred[:, :3] = pred[:, :3] * pos_std + pos_mean
    pos_pred = pred[:, :3]                              # (M, 3)
    app_pred = pred[:, 3:6] / (np.linalg.norm(pred[:, 3:6], axis=1, keepdims=True) + 1e-9)

    pos_gt = gt_grasps_7d[:, :3]                        # (G, 3)
    # GT approach = R_gt[:,2] (Tool Z column from quat)
    app_gt = np.zeros((len(gt_grasps_7d), 3), dtype=np.float32)
    for i, q in enumerate(gt_grasps_7d[:, 3:7]):
        w, x, y, z = q
        app_gt[i] = [2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)]

    # for each pred, find nearest GT (by pos)
    metrics = []
    matched_groups = set()
    for m in range(len(pred)):
        d_pos = np.linalg.norm(pos_gt - pos_pred[m], axis=1)
        nearest = int(np.argmin(d_pos))
        pos_err_cm = float(d_pos[nearest]) * 100
        cos = float(np.clip(app_pred[m] @ app_gt[nearest], -1, 1))
        ang_err_deg = float(np.degrees(np.arccos(abs(cos))))  # symmetric (180° flip OK)
        metrics.append((pos_err_cm, ang_err_deg))
        matched_groups.add(int(gt_groups[nearest]))

    mae_pos = float(np.mean([m[0] for m in metrics]))
    mae_ang = float(np.mean([m[1] for m in metrics]))
    n_groups_gt = len(set(int(g) for g in gt_groups))
    coverage = 100.0 * len(matched_groups) / max(n_groups_gt, 1)
    return mae_pos, mae_ang, coverage


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

    results = {"flow": [], "direct": []}
    for oi, oid in enumerate(obj_ids):
        idx = np.where(object_ref == oid)[0]
        depth = depths[depth_ref[idx[0]]]
        uv = uvs[idx[0]]
        gt_grasps = grasps_cam[idx]
        gt_groups = grasp_group[idx]

        g_flow = infer_flow(flow, depth, uv,
                            n=N_SAMPLES, t_steps=T_EULER,
                            temp=NOISE_TEMP, w_cfg=CFG_W)
        g_direct = infer_direct(direct, depth, uv)

        m_flow = compute_metrics(g_flow, gt_grasps, gt_groups,
                                  np.array(flow_norm["pos_mean"]),
                                  np.array(flow_norm["pos_std"]))
        m_direct = compute_metrics(g_direct, gt_grasps, gt_groups,
                                    np.array(direct_norm["pos_mean"]),
                                    np.array(direct_norm["pos_std"]))
        results["flow"].append(m_flow)
        results["direct"].append(m_direct)
        if (oi+1) % 50 == 0:
            print(f"  [{oi+1}/{len(obj_ids)}]  "
                  f"flow=({m_flow[0]:.1f}cm, {m_flow[1]:.1f}°, cov={m_flow[2]:.0f}%)  "
                  f"direct=({m_direct[0]:.1f}cm, {m_direct[1]:.1f}°, cov={m_direct[2]:.0f}%)")

    f.close()

    summary = {}
    for name in ["flow", "direct"]:
        arr = np.array(results[name])
        summary[name] = {
            "pos_mae_cm": float(arr[:, 0].mean()),
            "ang_err_deg": float(arr[:, 1].mean()),
            "mode_coverage_pct": float(arr[:, 2].mean()),
            "n_objs": int(len(arr)),
        }

    out_json = OUT / "table1.json"
    out_json.write_text(json.dumps(summary, indent=2))

    md = "# Table 1. Quantitative comparison (val split, random6)\n\n"
    md += "| Model | Pos. MAE [cm] ↓ | Ang. Err. [°] ↓ | Mode Cov. [%] ↑ |\n"
    md += "|---|---|---|---|\n"
    md += (f"| Regression (Direct MLP) | "
           f"{summary['direct']['pos_mae_cm']:.2f} | "
           f"{summary['direct']['ang_err_deg']:.2f} | "
           f"{summary['direct']['mode_coverage_pct']:.1f} |\n")
    md += (f"| **Ours (Flow Matching)** | "
           f"{summary['flow']['pos_mae_cm']:.2f} | "
           f"{summary['flow']['ang_err_deg']:.2f} | "
           f"**{summary['flow']['mode_coverage_pct']:.1f}** |\n")
    md += f"\n_Evaluated on {summary['flow']['n_objs']} val objects._\n"
    (OUT / "table1.md").write_text(md)

    print("\n=== Table 1 ===")
    print(md)


if __name__ == "__main__":
    main()
