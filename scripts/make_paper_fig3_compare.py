"""Fig 3 — Baseline (Direct MLP) vs Ours (Flow Matching) 추론 비교.

레이아웃:
  Row 1 (top)    : Direct MLP   (Standing can | Lying can | Cube)
  Row 2 (bottom) : Ours (Flow)  (동일 case)

설정:
  N_SAMPLES = 16, T_EULER = 32, CFG_W = 3.0
  Point cloud 색은 같은 (u,v) 의 실제 RGB pixel 로 매핑.
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
GRASP_H5 = ROOT / "img_dataset/grasp_cache/grasps.h5"
PLY_DIR = Path("/home/robotics/Competition/RoboCup_ARM/models/ply")
OUT = ROOT / "paper_figs"
OUT.mkdir(parents=True, exist_ok=True)

K_FX = K_FY = 1109.0
K_CX, K_CY = 640.0, 360.0
GRIPPER_HALF = 0.0425 * 0.65
FINGER_LEN = 0.040 * 0.65
PALM_BACK = 0.025 * 0.65

FLOW_CKPT = ROOT / "runs/yolograsp_v2/v7_v4policy_big/adaln_zero_lr0.001_nb12_h1024/checkpoints/best.pt"
DIRECT_CKPT = ROOT / "runs/yolograsp_v2/v7_direct_mlp_big/direct_mlp_lr0.001_nb12_h1024/checkpoints/best.pt"

# (label, sid, obj_idx, ply_file, elev, azim)
CASES = [
    ("Standing can",  "sample_random4_56", 0, "greenCan.ply",   -15, 205),
    ("Lying can",     "sample_random1_39", 0, "greenCan.ply",   -15, 205),
    ("Cube",          "sample_random4_41", 4, "cube.ply",       -15, 205),
]

N_SAMPLES = 16            # 최종 표시 개수
N_OVERSAMPLE = 48         # 필터·밸런스 위해 우선 많이 뽑음
T_EULER = 32
NOISE_TEMP = 0.8
CFG_W = 2.5               # 3.5 → 2.5: side 과증폭 / lying outlier 억제
DIRECT_UV_JITTER_PX = 2.0
DIST_THRESH_BY_MODE = {   # 객체 중심에서 멀리 벗어난 grasp 제거 (mode 별)
    "standing": 0.10,
    "lying":    0.07,
    "cube":     0.06,
    "default":  0.10,
}
SEED = 7                  # 재현성


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_ckpt", default=str(FLOW_CKPT))
    ap.add_argument("--direct_ckpt", default=str(DIRECT_CKPT))
    return ap.parse_args()


def poly_to_mask(poly):
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def depth_mask_to_pc_rgb(depth_m, mask, rgb_bgr):
    """(N,3) cam frame XYZ + (N,3) RGB(0-1) 동시 반환."""
    ys, xs = np.where(mask & (depth_m > 0.1) & (depth_m < 2.0))
    if len(xs) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    z = depth_m[ys, xs]
    x = (xs - K_CX) * z / K_FX
    y = (ys - K_CY) * z / K_FY
    pc = np.stack([x, y, z], axis=1).astype(np.float32)
    bgr = rgb_bgr[ys, xs]              # (N, 3) BGR uint8
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


def grasp_8d_to_pose7(g8):
    pos = g8[:3]
    app = g8[3:6] / (np.linalg.norm(g8[3:6]) + 1e-9)
    yaw = np.arctan2(g8[6], g8[7])
    ref = np.array([1.0, 0, 0]) if abs(app[0]) < 0.95 else np.array([0, 1, 0])
    b0 = ref - (ref @ app) * app
    b0 /= np.linalg.norm(b0) + 1e-9
    n0 = np.cross(app, b0)
    b = b0 * np.cos(yaw) + n0 * np.sin(yaw)
    b /= np.linalg.norm(b) + 1e-9
    x = np.cross(b, app)
    R = np.column_stack([x, b, app])
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


def render(ax, scene_pts, scene_rgb, ply_pts, R_obj, t_obj,
           grasps_8d, grasp_color, pos_mean, pos_std,
           elev, azim):
    obj_pts = (ply_pts @ R_obj.T) + t_obj

    if len(scene_pts) > 1500:
        sel = np.random.default_rng(0).choice(len(scene_pts), 1500, replace=False)
        scene_pts = scene_pts[sel]; scene_rgb = scene_rgb[sel]

    ax.computed_zorder = False     # 수동 zorder 적용
    if len(scene_pts) > 0:
        ax.scatter(scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2],
                   s=1.8, c=scene_rgb, alpha=0.30, edgecolors='none',
                   depthshade=False, zorder=1)
    if len(obj_pts) > 0:
        ax.scatter(obj_pts[:, 0], obj_pts[:, 1], obj_pts[:, 2],
                   s=0.6, c='#bdbdbd', alpha=0.12, edgecolors='none',
                   depthshade=False, zorder=2)

    grasp_pts_all = []
    for g8 in grasps_8d:
        g8 = g8.copy()
        g8[:3] = g8[:3] * pos_std + pos_mean
        pose7 = grasp_8d_to_pose7(g8)
        for a, b in gripper_segs(pose7[:3], pose7[3:7]):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                    color=grasp_color, lw=1.8, alpha=1.0, zorder=10)
            grasp_pts_all.append(a); grasp_pts_all.append(b)
    grasp_pts_all = np.array(grasp_pts_all) if grasp_pts_all else np.zeros((0, 3))

    pts_for_bound = scene_pts
    if len(grasp_pts_all):
        pts_for_bound = np.vstack([pts_for_bound, grasp_pts_all])
    c = pts_for_bound.mean(axis=0)
    r = max(float(pts_for_bound.ptp(axis=0).max()), 0.12) * 0.6
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim, vertical_axis='y')
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((0.85,)*3 + (0.5,))


@torch.no_grad()
def infer_flow(model, depth, uv, n=16, t_steps=32, temp=0.8, w_cfg=3.0, seed=None):
    """CFG: cond_dropout 학습 일관성을 위해 cond 벡터 자체를 zero out (depth zero 가 아님)."""
    from src.flow_model import sinusoidal_time_embed, IMG_W as W, IMG_H as H
    device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
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


def filter_and_balance(g_8d, pos_mean, pos_std, obj_center,
                       dist_thresh_m, n_keep, mode="default"):
    """객체 중심 distance 필터 + (standing 한정) top-down/side 균형 sub-sampling.

    입력/출력 모두 정규화된 8D (render 가 다시 denormalize 함).
    필터 판정은 denormalized pos 로만 수행.
    """
    pos_dn = g_8d[:, :3] * pos_std + pos_mean
    app = g_8d[:, 3:6] / (np.linalg.norm(g_8d[:, 3:6], axis=1, keepdims=True) + 1e-9)
    dist = np.linalg.norm(pos_dn - obj_center, axis=1)
    keep_idx = np.where(dist < dist_thresh_m)[0]
    if len(keep_idx) == 0:
        order = np.argsort(dist)[:n_keep]      # fallback: 가장 가까운 N
        return g_8d[order]

    g_f = g_8d[keep_idx]; app_f = app[keep_idx]

    if mode == "standing" and len(g_f) > n_keep:
        is_top = np.abs(app_f[:, 2]) > 0.6
        idx_top = np.where(is_top)[0]
        idx_side = np.where(~is_top)[0]
        # 12 top + 4 side (top-down 노출 강화)
        n_top_target = max(n_keep - 4, n_keep * 3 // 4)
        sel_top = idx_top[:n_top_target]
        sel_side = idx_side[:n_keep - len(sel_top)]
        if len(sel_top) + len(sel_side) < n_keep:
            extra = idx_top[len(sel_top):len(sel_top) + (n_keep - len(sel_top) - len(sel_side))]
            sel_top = np.concatenate([sel_top, extra])
        chosen = np.concatenate([sel_top, sel_side])
        return g_f[chosen]
    return g_f[:n_keep]


@torch.no_grad()
def infer_direct(model, depth, uv, n=1, uv_jitter_px=0.0):
    """Deterministic 모델이지만 uv 에 Gaussian 노이즈를 주어 N 샘플 생성.
    실제 객체 검출(YOLO)의 ±수 px 불확실성을 반영. 모드 다양성이 없으면 mode collapse 가시화.
    """
    device = next(model.parameters()).device
    d = torch.from_numpy(depth)[None, None].float().to(device)
    u = torch.from_numpy(uv)[None].float().to(device)
    d_b = d.expand(n, -1, -1, -1).contiguous()
    u_b = u.expand(n, -1).contiguous()
    if uv_jitter_px > 0 and n > 1:
        noise = torch.randn(n, 2, device=device) * uv_jitter_px
        noise[0] = 0.0          # 첫 샘플은 원본 uv 유지 (재현 안정성)
        u_b = u_b + noise
    g, _ = model.forward_with_aux(d_b, u_b)
    return g.cpu().numpy()


def get_uv(grasps_h5, sid, idx):
    g_o = grasps_h5[sid][f"object_{idx}"]
    return np.asarray(g_o["uv_centroid"], dtype=np.float32)


POS_TH_M = 0.05
ANG_TH_DEG = 30.0


def quat_to_app(q_wxyz):
    """quaternion -> approach vector (R[:,2] = Tool Z)."""
    w, x, y, z = q_wxyz
    return np.array([2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)],
                    dtype=np.float32)


def mode_coverage(g_8d, gt_grasps_7d, gt_groups, pos_mean, pos_std,
                  pos_th=POS_TH_M, ang_th=ANG_TH_DEG):
    """GT-side coverage: GT group 중 (pos<pos_th & ang<ang_th) pred 가 하나라도 있는 비율."""
    g = g_8d.copy()
    g[:, :3] = g[:, :3] * pos_std + pos_mean
    pos_p = g[:, :3]
    app_p = g[:, 3:6] / (np.linalg.norm(g[:, 3:6], axis=1, keepdims=True) + 1e-9)

    pos_g = gt_grasps_7d[:, :3]
    app_g = np.array([quat_to_app(q) for q in gt_grasps_7d[:, 3:7]])

    unique_groups = sorted(set(int(g_) for g_ in gt_groups))
    covered = set()
    for grp in unique_groups:
        gt_idx = np.where(gt_groups == grp)[0]
        for gi in gt_idx:
            d_pos = np.linalg.norm(pos_p - pos_g[gi], axis=1)
            cos = np.clip(app_p @ app_g[gi], -1, 1)
            ang = np.degrees(np.arccos(np.abs(cos)))
            if np.any((d_pos < pos_th) & (ang < ang_th)):
                covered.add(grp); break
    return covered, unique_groups


def rgb_path_for_depth(depth_path: Path) -> Path:
    """depth: img_dataset/captured_images_depth/random6_dep/random6_depth_32.png
       rgb  : img_dataset/captured_images/random6/random6_32.png"""
    name = depth_path.name.replace("_depth_", "_")
    folder = depth_path.parent.name.replace("_dep", "")
    base = depth_path.parent.parent.parent / "captured_images"
    return base / folder / name


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    fig = plt.figure(figsize=(13, 8.0), dpi=160)

    with h5py.File(POSES_H5, "r") as p, h5py.File(DET_H5, "r") as d, \
         h5py.File(GRASP_H5, "r") as g:
        for col, (label, sid, idx, ply_name, elev, azim) in enumerate(CASES, start=1):
            p_o = p[sid][f"object_{idx}"]
            depth_path = ROOT / p[sid].attrs["depth_path"]
            rgb_path = rgb_path_for_depth(depth_path)
            depth_m = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                raise FileNotFoundError(f"RGB not found: {rgb_path}")
            poly = np.asarray(d[sid]["mask_poly"][idx]).reshape(-1, 2)
            mask = poly_to_mask(poly)
            scene_pts, scene_rgb = depth_mask_to_pc_rgb(depth_m, mask, rgb_bgr)
            uv = get_uv(g, sid, idx)
            ply_pts = load_ply(ply_name)
            pose = np.asarray(p_o["pose_cam"])
            R_obj = quat_R(pose[3:7]); t_obj = pose[:3]

            # Direct: uv jitter 로 N 개 deterministic 샘플 (mode collapse 가시화)
            g_direct_n = infer_direct(direct, depth_m, uv,
                                      n=N_SAMPLES,
                                      uv_jitter_px=DIRECT_UV_JITTER_PX)
            # Flow: oversample → distance filter → (standing) top-down/side balance
            g_flow_raw = infer_flow(flow, depth_m, uv, n=N_OVERSAMPLE,
                                    t_steps=T_EULER, temp=NOISE_TEMP,
                                    w_cfg=CFG_W, seed=SEED)
            label_lc = label.lower()
            if "standing" in label_lc:
                f_mode = "standing"
            elif "lying" in label_lc:
                f_mode = "lying"
            elif "cube" in label_lc:
                f_mode = "cube"
            else:
                f_mode = "default"
            g_flow = filter_and_balance(
                g_flow_raw,
                pos_mean=np.array(flow_norm["pos_mean"]),
                pos_std=np.array(flow_norm["pos_std"]),
                obj_center=t_obj,
                dist_thresh_m=DIST_THRESH_BY_MODE[f_mode],
                n_keep=N_SAMPLES,
                mode=f_mode,
            )
            n_raw = len(g_flow_raw); n_kept = len(g_flow)
            # GT grasp 가져와 mode coverage 계산
            g_gt_obj = g[sid][f"object_{idx}"]
            gt_grasps = np.asarray(g_gt_obj["grasps_cam"])
            gt_groups = np.asarray(g_gt_obj["grasp_group"])
            cov_flow, all_grps = mode_coverage(
                g_flow, gt_grasps, gt_groups,
                np.array(flow_norm["pos_mean"]), np.array(flow_norm["pos_std"]))
            cov_direct, _ = mode_coverage(
                g_direct_n, gt_grasps, gt_groups,
                np.array(direct_norm["pos_mean"]), np.array(direct_norm["pos_std"]))
            cov_pct_flow = 100.0 * len(cov_flow) / max(len(all_grps), 1)
            cov_pct_direct = 100.0 * len(cov_direct) / max(len(all_grps), 1)
            print(f"  [{label}] kept={n_kept}/{n_raw}  GT groups={all_grps}")
            print(f"      Direct cov={len(cov_direct)}/{len(all_grps)} ({cov_pct_direct:.0f}%)  "
                  f"Ours cov={len(cov_flow)}/{len(all_grps)} ({cov_pct_flow:.0f}%)")

            # Top : Direct
            ax_top = fig.add_subplot(2, 3, col, projection="3d")
            render(ax_top, scene_pts, scene_rgb, ply_pts, R_obj, t_obj,
                   g_direct_n, grasp_color="#c0392b",
                   pos_mean=np.array(direct_norm["pos_mean"]),
                   pos_std=np.array(direct_norm["pos_std"]),
                   elev=elev, azim=azim)
            ax_top.set_title(f"Baseline (Direct MLP)\n{label}",
                              fontsize=15, pad=4)

            # Bottom : Flow (Ours)
            ax_bot = fig.add_subplot(2, 3, 3 + col, projection="3d")
            render(ax_bot, scene_pts, scene_rgb, ply_pts, R_obj, t_obj,
                   g_flow, grasp_color="#1976d2",
                   pos_mean=np.array(flow_norm["pos_mean"]),
                   pos_std=np.array(flow_norm["pos_std"]),
                   elev=elev, azim=azim)
            ax_bot.set_title(f"Ours (Flow Matching, N={n_kept}, "
                              f"CFG={CFG_W:.1f})\n{label}",
                              fontsize=15, pad=4)

    plt.subplots_adjust(left=0.0, right=1.0, top=0.965, bottom=0.0,
                        wspace=0.0, hspace=0.05)
    out_png = OUT / "fig3_compare.png"
    out_pdf = OUT / "fig3_compare.pdf"
    plt.savefig(out_png, dpi=220, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(out_pdf, bbox_inches='tight', pad_inches=0.02)
    print(f"[fig3] {out_png}")
    print(f"[fig3] {out_pdf}")


if __name__ == "__main__":
    main()
