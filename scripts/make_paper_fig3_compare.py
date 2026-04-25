"""Fig 3 — 기준(Direct MLP) vs 제안(Flow Matching) 파지 자세 비교 (3-panel × 2-row).

Direct MLP 학습 완료 후 실행. 두 모델로 같은 (depth, uv) → grasp 추론 후 비교.
top row: Standing cylinder, Lying cylinder, Cube  (Direct, 1 grasp 한 점 수렴)
bot row: 동일 case  (Flow, N=32 multi-modal 분포)

Usage:
    python scripts/make_paper_fig3_compare.py
"""
from __future__ import annotations
from pathlib import Path
import argparse
import sys
import matplotlib
matplotlib.use("Agg")
import cv2
import h5py
import numpy as np
import open3d as o3d
import torch
import matplotlib.pyplot as plt

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
sys.path.insert(0, str(ROOT))
from src.flow_model import FlowGraspNet, IMG_H, IMG_W
from src.direct_model import DirectGraspNet

POSES_H5 = ROOT / "img_dataset/icp_cache/poses.h5"
DET_H5 = ROOT / "img_dataset/yolo_cache_v3/detections.h5"
PLY_DIR = Path("/home/robotics/Competition/RoboCup_ARM/models/ply")
OUT = ROOT / "paper_figs"
OUT.mkdir(parents=True, exist_ok=True)

K_FX = K_FY = 1109.0
K_CX, K_CY = 640.0, 360.0
GRIPPER_HALF = 0.0425
FINGER_LEN = 0.040
PALM_BACK = 0.025

# v7 Flow ckpt
FLOW_CKPT = ROOT / "runs/yolograsp_v2/v7_v4policy_big/adaln_zero_lr0.001_nb12_h1024/checkpoints/best.pt"
DIRECT_CKPT = ROOT / "runs/yolograsp_v2/v7_direct_mlp_big/direct_mlp_lr0.001_nb12_h1024/checkpoints/best.pt"

# 3 case: standing bottle, lying can, cube_red
CASES = [
    ("Standing cylinder",  "sample_random6_31", 2, "blueBottle.ply"),
    ("Lying cylinder",     "sample_random6_11", 0, "greenCan.ply"),
    ("Cube",               "sample_random6_30", 7, "cube.ply"),
]

N_SAMPLES = 32
T_EULER = 32
NOISE_TEMP = 0.8
CFG_W = 2.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_ckpt", default=str(FLOW_CKPT))
    ap.add_argument("--direct_ckpt", default=str(DIRECT_CKPT))
    return ap.parse_args()


def poly_to_mask(poly):
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def depth_mask_to_pc(depth_m, mask):
    ys, xs = np.where(mask & (depth_m > 0.1) & (depth_m < 2.0))
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth_m[ys, xs]
    x = (xs - K_CX) * z / K_FX
    y = (ys - K_CY) * z / K_FY
    return np.stack([x, y, z], axis=1).astype(np.float32)


def load_ply(name):
    pcd = o3d.io.read_point_cloud(str(PLY_DIR / name))
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.ptp(axis=0).max() > 1.0:
        pts *= 0.001
    pts -= pts.mean(axis=0)
    if len(pts) > 1500:
        pts = pts[np.random.default_rng(0).choice(len(pts), 1500, replace=False)]
    return pts


def grasp_8d_to_pose7(g8):
    """g(8) = [x,y,z, ax,ay,az, sin_yaw, cos_yaw] → pose(7) = [x,y,z, qw,qx,qy,qz]."""
    pos = g8[:3]
    app = g8[3:6] / (np.linalg.norm(g8[3:6]) + 1e-9)
    yaw = np.arctan2(g8[6], g8[7])
    # Tool frame: Z = approach
    ref = np.array([1.0, 0, 0]) if abs(app[0]) < 0.95 else np.array([0, 1, 0])
    b0 = ref - (ref @ app) * app
    b0 /= np.linalg.norm(b0) + 1e-9
    n0 = np.cross(app, b0)
    b = b0 * np.cos(yaw) + n0 * np.sin(yaw)
    b /= np.linalg.norm(b) + 1e-9
    x = np.cross(b, app)
    R = np.column_stack([x, b, app])
    # Shepperd quat
    tr = R.trace()
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s; qx = (R[2,1]-R[1,2])/s; qy = (R[0,2]-R[2,0])/s; qz = (R[1,0]-R[0,1])/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2
        qw=(R[2,1]-R[1,2])/s; qx=0.25*s; qy=(R[0,1]+R[1,0])/s; qz=(R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2
        qw=(R[0,2]-R[2,0])/s; qx=(R[0,1]+R[1,0])/s; qy=0.25*s; qz=(R[1,2]+R[2,1])/s
    else:
        s = np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2
        qw=(R[1,0]-R[0,1])/s; qx=(R[0,2]+R[2,0])/s; qy=(R[1,2]+R[2,1])/s; qz=0.25*s
    q = np.array([qw, qx, qy, qz])
    if q[0] < 0: q = -q
    q /= np.linalg.norm(q)
    return np.concatenate([pos, q])


def quat_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),     2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z),   2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),     1-2*(x*x+y*y)],
    ])


def gripper_segs(pos, q):
    R = quat_R(q)
    a = R[:, 2]; b = R[:, 1]
    tip1 = pos + b*GRIPPER_HALF; tip2 = pos - b*GRIPPER_HALF
    base1 = tip1 - a*FINGER_LEN; base2 = tip2 - a*FINGER_LEN
    palm = pos - a*FINGER_LEN
    wrist = palm - a*PALM_BACK
    return [(wrist, palm), (base1, base2), (base1, tip1), (base2, tip2)]


def render(ax, scene_pts, ply_pts, R_obj, t_obj, grasps_8d, color, title,
           pos_mean, pos_std):
    obj_pts = (ply_pts @ R_obj.T) + t_obj

    if len(scene_pts) > 600:
        scene_pts = scene_pts[np.random.default_rng(0).choice(len(scene_pts), 600, replace=False)]

    ax.scatter(scene_pts[:,0], scene_pts[:,1], scene_pts[:,2],
               s=2, c="#4a90e2", alpha=0.30, edgecolors='none')
    ax.scatter(obj_pts[:,0], obj_pts[:,1], obj_pts[:,2],
               s=1, c="#888", alpha=0.30, edgecolors='none')

    for g8 in grasps_8d:
        # denorm pos
        g8 = g8.copy()
        g8[:3] = g8[:3] * pos_std + pos_mean
        pose7 = grasp_8d_to_pose7(g8)
        for a, b in gripper_segs(pose7[:3], pose7[3:7]):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                    color=color, lw=0.9, alpha=0.85)

    c = scene_pts.mean(axis=0)
    r = max(float(scene_pts.ptp(axis=0).max()), 0.10) * 0.65
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=-60, azim=-90)
    ax.set_title(title, fontsize=10, pad=2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1,1,1,0))
        axis.pane.set_edgecolor((0.85,)*3+(0.5,))


@torch.no_grad()
def infer_flow(model, depth, uv, n=32, t_steps=32, temp=0.8, w_cfg=2.0):
    device = next(model.parameters()).device
    d = torch.from_numpy(depth)[None, None].float().to(device)
    u = torch.from_numpy(uv)[None].float().to(device)
    # batch N for parallel sampling
    d_b = d.expand(n, -1, -1, -1).contiguous()
    u_b = u.expand(n, -1).contiguous()
    g_t = torch.randn(n, 8, device=device) * temp
    for k in range(t_steps):
        t = torch.full((n,), k / t_steps, device=device)
        v_on = model(d_b, u_b, g_t, t)
        v_off = model(torch.zeros_like(d_b), u_b, g_t, t)  # null cond proxy
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


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Flow
    fck = torch.load(args.flow_ckpt, weights_only=False, map_location=device)
    cfg_f = fck["cfg"]["args"]
    flow = FlowGraspNet(block_type=cfg_f["block"], n_blocks=cfg_f["n_blocks"],
                        hidden=cfg_f["hidden"], cond_dropout=cfg_f["cond_dropout"]).to(device)
    flow.load_state_dict(fck["ema"], strict=False); flow.eval()
    flow_norm = fck["norm_stats"]
    print(f"[load] Flow ep={fck['epoch']} val={fck.get('val_loss'):.4f}")

    # Load Direct
    dck = torch.load(args.direct_ckpt, weights_only=False, map_location=device)
    cfg_d = dck["cfg"]["args"]
    direct = DirectGraspNet(n_blocks=cfg_d["n_blocks"], hidden=cfg_d["hidden"],
                            cond_dropout=cfg_d.get("cond_dropout", 0.0)).to(device)
    direct.load_state_dict(dck["ema"], strict=False); direct.eval()
    direct_norm = dck["norm_stats"]
    print(f"[load] Direct ep={dck['epoch']} val={dck.get('val_loss'):.4f}")

    fig = plt.figure(figsize=(12, 7), dpi=150)

    with h5py.File(POSES_H5, "r") as p, h5py.File(DET_H5, "r") as d:
        for col, (label, sid, idx, ply_name) in enumerate(CASES, start=1):
            p_o = p[sid][f"object_{idx}"]
            depth_path = ROOT / p[sid].attrs["depth_path"]
            depth_m = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            poly = np.asarray(d[sid]["mask_poly"][idx]).reshape(-1, 2)
            mask = poly_to_mask(poly)
            scene_pts = depth_mask_to_pc(depth_m, mask)
            uv = np.asarray(p_o["uv_centroid"], dtype=np.float32)
            ply_pts = load_ply(ply_name)
            pose = np.asarray(p_o["pose_cam"])
            R_obj = quat_R(pose[3:7]); t_obj = pose[:3]

            # Direct: 32 noise → 32 same outputs (deterministic)
            g_direct = infer_direct(direct, depth_m, uv)
            g_direct_n = np.tile(g_direct, (N_SAMPLES, 1))  # show N copies (collapsed)
            # Flow: N samples
            g_flow = infer_flow(flow, depth_m, uv, n=N_SAMPLES,
                                t_steps=T_EULER, temp=NOISE_TEMP, w_cfg=CFG_W)

            # Top row: Direct
            ax_top = fig.add_subplot(2, 3, col, projection="3d")
            render(ax_top, scene_pts, ply_pts, R_obj, t_obj, g_direct_n,
                   color="#c0392b",
                   title=f"Baseline (Direct MLP)\n{label}",
                   pos_mean=np.array(direct_norm["pos_mean"]),
                   pos_std=np.array(direct_norm["pos_std"]))

            # Bottom row: Flow
            ax_bot = fig.add_subplot(2, 3, 3 + col, projection="3d")
            render(ax_bot, scene_pts, ply_pts, R_obj, t_obj, g_flow,
                   color="#27ae60",
                   title=f"Ours (Flow Matching, N={N_SAMPLES})\n{label}",
                   pos_mean=np.array(flow_norm["pos_mean"]),
                   pos_std=np.array(flow_norm["pos_std"]))

    fig.suptitle("Fig 3. Baseline (single regression) vs Ours (multi-modal sampling)",
                 fontsize=11, weight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = OUT / "fig3_compare.png"
    out_pdf = OUT / "fig3_compare.pdf"
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"[fig3] {out_png}")


if __name__ == "__main__":
    main()
