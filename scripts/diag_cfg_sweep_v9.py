"""CFG sweep diagnostic — 9D Zhou + depth model 의 CFG 값 변화에 따른
standing can grasp distribution 측정/시각화.

각 CFG 값 별로 N=32 grasp 을 추출, mode 별 색상 분류:
  top-down (|a_z|>0.85)  — 파랑
  side-45  (0.5<|a_z|≤0.85) — 초록
  side-cap (|a_z|≤0.5)   — 빨강

출력: paper_figs/diag_cfg_sweep.png — 2×3 grid (CFG=0.0/1.0/1.5/2.0/2.5/3.5)
"""
from __future__ import annotations
from pathlib import Path
import math
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

POSES_H5 = ROOT / "img_dataset/icp_cache/poses.h5"
DET_H5 = ROOT / "img_dataset/yolo_cache_v3/detections.h5"
GRASP_H5 = ROOT / "img_dataset/grasp_cache/grasps.h5"
PLY_DIR = Path("/home/robotics/Competition/RoboCup_ARM/models/ply")
OUT = ROOT / "paper_figs"

K_FX = K_FY = 1109.0
K_CX, K_CY = 640.0, 360.0
GRIPPER_HALF = 0.0425 * 0.65
FINGER_LEN = 0.040 * 0.65
PALM_BACK = 0.025 * 0.65

FLOW_CKPT = ROOT / "runs/yolograsp_v2/zhou_9d_full_250ep/adaln_zero_lr0.001_nb8_h768/checkpoints/best.pt"

# Standing can (Fig 3 에서 사용한 동일 case)
SID = "sample_random4_56"
OBJ_IDX = 0
PLY_NAME = "greenCan.ply"
ELEV, AZIM = -15, 205

# CFG sweep
CFG_VALUES = [0.0, 1.0, 1.5, 2.0, 2.5, 3.5]
N_SAMPLES = 32
T_EULER = 32
NOISE_TEMP = 0.8
SEED = 7

# Mode classification thresholds (a_z 기준)
# top-down ≈ a_z=1.0 (정확)
# side-45  ≈ a_z=0.7071
# side-cap ≈ a_z=0.0
TH_TOP = 0.85   # top-down: |a_z|>0.85 (cos 32° 보다 위 — top-down 만)
TH_SIDE = 0.5   # side-45 vs side-cap 경계 (cos 60°)

MODE_COLORS = {
    "top":  "#1976d2",   # 파랑
    "s45":  "#2e7d32",   # 초록
    "cap":  "#c0392b",   # 빨강
}


def poly_to_mask(poly):
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def depth_mask_to_pc_rgb(depth_m, mask, rgb_bgr):
    ys, xs = np.where(mask & (depth_m > 0.1) & (depth_m < 2.0))
    if len(xs) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    z = depth_m[ys, xs]
    x = (xs - K_CX) * z / K_FX
    y = (ys - K_CY) * z / K_FY
    pc = np.stack([x, y, z], axis=1).astype(np.float32)
    bgr = rgb_bgr[ys, xs]
    rgb = bgr[:, ::-1].astype(np.float32) / 255.0
    return pc, rgb


def load_ply(name):
    pcd = o3d.io.read_point_cloud(str(PLY_DIR / name))
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.ptp(axis=0).max() > 1.0:
        pts *= 0.001
    pts -= pts.mean(axis=0)
    if len(pts) > 1500:
        pts = pts[np.random.default_rng(0).choice(len(pts), 1500, replace=False)]
    return pts


def quat_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def gram_schmidt_to_R(r6):
    a1, a2 = r6[:3], r6[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    b2 = a2 - (b1 @ a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])


def gripper_segs(pos, R):
    a = R[:, 2]; b = R[:, 1]
    tip1 = pos + b*GRIPPER_HALF; tip2 = pos - b*GRIPPER_HALF
    base1 = tip1 - a*FINGER_LEN; base2 = tip2 - a*FINGER_LEN
    palm = pos - a*FINGER_LEN
    wrist = palm - a*PALM_BACK
    return [(wrist, palm), (base1, base2), (base1, tip1), (base2, tip2)]


def rgb_path_for_depth(depth_path: Path) -> Path:
    name = depth_path.name.replace("_depth_", "_")
    folder = depth_path.parent.name.replace("_dep", "")
    base = depth_path.parent.parent.parent / "captured_images"
    return base / folder / name


@torch.no_grad()
def infer_flow_cfg(model, depth, uv, n, t_steps, temp, w_cfg, seed, g_dim=9):
    """N samples for given CFG. Returns g_t (N, g_dim) - 정규화 상태."""
    from src.flow_model import sinusoidal_time_embed
    device = next(model.parameters()).device
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    d = torch.from_numpy(depth)[None, None].float().to(device)
    u = torch.from_numpy(uv)[None].float().to(device)
    d_b = d.expand(n, -1, -1, -1).contiguous()
    u_b = u.expand(n, -1).contiguous()
    cond_on = model.encode(d_b, u_b)
    cond_off = torch.zeros_like(cond_on)
    uv_norm = torch.stack([u_b[:, 0] / IMG_W, u_b[:, 1] / IMG_H], dim=-1)
    g_t = torch.randn(n, g_dim, device=device) * temp
    for k in range(t_steps):
        t = torch.full((n,), k / t_steps, device=device)
        t_emb = sinusoidal_time_embed(t, dim=64)
        v_on = model.velocity(g_t, cond_on, t_emb, uv_norm)
        v_off = model.velocity(g_t, cond_off, t_emb, uv_norm)
        v = v_off + w_cfg * (v_on - v_off)
        g_t = g_t + v / t_steps
    return g_t.cpu().numpy()


def classify_mode(a_z):
    az = abs(a_z)
    if az > TH_TOP:
        return "top"
    if az > TH_SIDE:
        return "s45"
    return "cap"


def render_panel(ax, scene_pts, scene_rgb, ply_pts, R_obj, t_obj,
                 grasps_9d, pos_mean, pos_std, elev, azim):
    obj_pts = (ply_pts @ R_obj.T) + t_obj

    if len(scene_pts) > 1500:
        sel = np.random.default_rng(0).choice(len(scene_pts), 1500, replace=False)
        scene_pts = scene_pts[sel]; scene_rgb = scene_rgb[sel]

    ax.computed_zorder = False
    if len(scene_pts) > 0:
        ax.scatter(scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2],
                   s=1.5, c=scene_rgb, alpha=0.30, edgecolors='none',
                   depthshade=False, zorder=1)
    if len(obj_pts) > 0:
        ax.scatter(obj_pts[:, 0], obj_pts[:, 1], obj_pts[:, 2],
                   s=0.5, c='#bdbdbd', alpha=0.10, edgecolors='none',
                   depthshade=False, zorder=2)

    counts = {"top": 0, "s45": 0, "cap": 0}
    grasp_pts_all = []
    for g9 in grasps_9d:
        g = g9.copy()
        g[:3] = g[:3] * pos_std + pos_mean
        R = gram_schmidt_to_R(g[3:9])
        a_z = R[2, 2]
        m = classify_mode(a_z)
        counts[m] += 1
        c = MODE_COLORS[m]
        for sa, sb in gripper_segs(g[:3], R):
            ax.plot([sa[0], sb[0]], [sa[1], sb[1]], [sa[2], sb[2]],
                    color=c, lw=1.5, alpha=0.95, zorder=10)
            grasp_pts_all.append(sa); grasp_pts_all.append(sb)

    grasp_pts_all = np.array(grasp_pts_all) if grasp_pts_all else np.zeros((0, 3))
    pts_for_bound = np.vstack([scene_pts, grasp_pts_all]) if len(grasp_pts_all) else scene_pts
    c_bound = pts_for_bound.mean(axis=0)
    r = max(float(pts_for_bound.ptp(axis=0).max()), 0.12) * 0.6
    ax.set_xlim(c_bound[0]-r, c_bound[0]+r)
    ax.set_ylim(c_bound[1]-r, c_bound[1]+r)
    ax.set_zlim(c_bound[2]-r, c_bound[2]+r)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim, vertical_axis='y')
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((0.85,)*3 + (0.5,))
    return counts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load 9D Zhou model
    fck = torch.load(str(FLOW_CKPT), weights_only=False, map_location=device)
    cfg = fck["cfg"]["args"]
    rot_repr = cfg.get("rot_repr", "approach_yaw")
    g_dim = 9 if rot_repr == "zhou6d" else 8
    assert g_dim == 9, f"Expected 9D Zhou, got g_dim={g_dim}"
    flow = FlowGraspNet(block_type=cfg["block"], n_blocks=cfg["n_blocks"],
                        hidden=cfg["hidden"], cond_dropout=cfg["cond_dropout"],
                        g_dim=g_dim).to(device)
    flow.load_state_dict(fck["ema"], strict=False); flow.eval()
    norm = fck["norm_stats"]
    pos_mean = np.array(norm["pos_mean"]); pos_std = np.array(norm["pos_std"])
    print(f"[load] 9D Zhou ep={fck['epoch']} val={fck.get('val_loss'):.4f}")

    # Scene load
    with h5py.File(POSES_H5, "r") as p, h5py.File(DET_H5, "r") as d, \
         h5py.File(GRASP_H5, "r") as g:
        p_o = p[SID][f"object_{OBJ_IDX}"]
        depth_path = ROOT / p[SID].attrs["depth_path"]
        rgb_path = rgb_path_for_depth(depth_path)
        depth_m = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        poly = np.asarray(d[SID]["mask_poly"][OBJ_IDX]).reshape(-1, 2)
        mask = poly_to_mask(poly)
        scene_pts, scene_rgb = depth_mask_to_pc_rgb(depth_m, mask, rgb_bgr)
        uv = np.asarray(g[SID][f"object_{OBJ_IDX}"]["uv_centroid"], dtype=np.float32)
        ply_pts = load_ply(PLY_NAME)
        pose = np.asarray(p_o["pose_cam"])
        R_obj = quat_R(pose[3:7]); t_obj = pose[:3]

    # Figure: 2x3 grid
    fig = plt.figure(figsize=(15, 9.5), dpi=160)
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 18,
    })

    print(f"\n=== CFG sweep on Standing can ({SID}/obj_{OBJ_IDX}), N={N_SAMPLES} per CFG ===")
    print(f"  Mode bins: top |a_z|>{TH_TOP}, side-45 ({TH_SIDE},{TH_TOP}], cap ≤{TH_SIDE}\n")

    for i, w_cfg in enumerate(CFG_VALUES):
        g_samples = infer_flow_cfg(flow, depth_m, uv, n=N_SAMPLES,
                                    t_steps=T_EULER, temp=NOISE_TEMP,
                                    w_cfg=w_cfg, seed=SEED, g_dim=g_dim)
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        counts = render_panel(ax, scene_pts, scene_rgb, ply_pts, R_obj, t_obj,
                               g_samples, pos_mean, pos_std, ELEV, AZIM)
        n_tot = N_SAMPLES
        title = (f"CFG = {w_cfg:.1f}\n"
                 f"top {100*counts['top']/n_tot:.0f}%  "
                 f"s45 {100*counts['s45']/n_tot:.0f}%  "
                 f"cap {100*counts['cap']/n_tot:.0f}%")
        ax.set_title(title, fontsize=16, pad=4, fontweight='bold')
        print(f"  CFG={w_cfg:.1f}  top={counts['top']:2d} s45={counts['s45']:2d} cap={counts['cap']:2d}")

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], color=MODE_COLORS["top"], lw=2.5,
               label=f"top-down (|a_z|>{TH_TOP})"),
        Line2D([0],[0], color=MODE_COLORS["s45"], lw=2.5,
               label=f"side-45  ({TH_SIDE}<|a_z|≤{TH_TOP})"),
        Line2D([0],[0], color=MODE_COLORS["cap"], lw=2.5,
               label=f"side-cap (|a_z|≤{TH_SIDE})"),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=3,
               fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("CFG sweep — Standing can / N=32 / 9D Zhou+depth (val 0.2419) — GT: 33:33:33",
                 fontsize=15, y=0.99, fontweight='bold')
    plt.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.05,
                        wspace=0.10, hspace=0.20)
    out_png = OUT / "diag_cfg_sweep.png"
    out_pdf = OUT / "diag_cfg_sweep.pdf"
    plt.savefig(out_png, dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(out_pdf, bbox_inches='tight', pad_inches=0.05)
    print(f"\n[saved] {out_png}")
    print(f"[saved] {out_pdf}")


if __name__ == "__main__":
    main()
