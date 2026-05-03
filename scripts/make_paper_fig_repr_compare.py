"""Paper figure — 8D approach_yaw vs 9D Zhou 6D rotation 추론 비교.

Layout (fig3 동일):
  Row 1 (top)    : 8D approach_yaw  (Standing can | Lying can | Cube)
  Row 2 (bottom) : 9D Zhou          (동일 case)

두 모델 모두 30 epoch, hidden=768, n_blocks=8, 동일 hyperparam.
표현 차이만 다름.
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

CKPT_8D = ROOT / "runs/yolograsp_v2/repr_compare/approach_yaw_8d/checkpoints/best.pt"
CKPT_9D = ROOT / "runs/yolograsp_v2/repr_compare/zhou_9d/checkpoints/best.pt"

CASES = [
    ("Standing can",  "sample_random4_56", 0, "greenCan.ply",   -15, 205),
    ("Lying can",     "sample_random1_39", 0, "greenCan.ply",   -15, 205),
    ("Cube",          "sample_random4_41", 4, "cube.ply",       -15, 205),
]

N_SAMPLES = 16
N_OVERSAMPLE = 48
T_EULER = 32
NOISE_TEMP = 0.8
CFG_W = 2.5
DIST_THRESH_BY_MODE = {"standing": 0.10, "lying": 0.07, "cube": 0.06, "default": 0.10}
SEED = 7


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_8d", default=str(CKPT_8D))
    ap.add_argument("--ckpt_9d", default=str(CKPT_9D))
    return ap.parse_args()


def poly_to_mask(poly):
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def depth_mask_to_pc_rgb(depth_m, mask, rgb_bgr):
    H, W = depth_m.shape
    ys, xs = np.mgrid[0:H, 0:W]
    valid = (depth_m > 0.1) & (depth_m < 2.0)
    m_full = mask & valid
    z = depth_m[m_full]; xs = xs[m_full]; ys = ys[m_full]
    x = (xs - K_CX) * z / K_FX
    y = (ys - K_CY) * z / K_FY
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    rgb = rgb_bgr[ys, xs][:, ::-1] / 255.0
    if len(pts) > 4500:
        idx = np.random.default_rng(0).choice(len(pts), 4500, replace=False)
        pts = pts[idx]; rgb = rgb[idx]
    return pts, rgb


def load_ply(name):
    pcd = o3d.io.read_point_cloud(str(PLY_DIR / name))
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.size == 0:
        m = o3d.io.read_triangle_mesh(str(PLY_DIR / name))
        pts = np.asarray(m.vertices, dtype=np.float32)
    if len(pts) > 1500:
        pts = pts[np.random.default_rng(0).choice(len(pts), 1500, replace=False)]
    return pts


def _gram_schmidt_R(r6):
    a1, a2 = r6[:3], r6[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    b2 = a2 - (b1 @ a2) * b1
    b2 /= np.linalg.norm(b2) + 1e-9
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])


def grasp_to_pose7(g, rot_repr):
    """g (8 or 9,) -> [pos(3), quat_wxyz(4)]"""
    pos = g[:3]
    if rot_repr == "zhou6d":
        R = _gram_schmidt_R(g[3:9])
    else:
        app = g[3:6] / (np.linalg.norm(g[3:6]) + 1e-9)
        yaw = np.arctan2(g[6], g[7])
        ref = np.array([1.0, 0, 0]) if abs(app[0]) < 0.95 else np.array([0, 1.0, 0])
        b0 = ref - (ref @ app) * app
        b0 /= np.linalg.norm(b0) + 1e-9
        n0 = np.cross(app, b0)
        b = b0 * np.cos(yaw) + n0 * np.sin(yaw); b /= np.linalg.norm(b) + 1e-9
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
           grasps_g, grasp_color, pos_mean, pos_std, rot_repr, elev, azim):
    ax.set_proj_type("ortho")
    ax.scatter(scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2],
               c=scene_rgb, s=5, marker=".", linewidths=0,
               depthshade=False, alpha=0.85, zorder=1)
    obj_world = (R_obj @ ply_pts.T).T + t_obj
    ax.scatter(obj_world[:, 0], obj_world[:, 1], obj_world[:, 2],
               s=2, c="#3a3a3a", alpha=0.30, depthshade=False, zorder=0)
    pts_g = []
    for g in grasps_g:
        gd = g.copy()
        gd[:3] = gd[:3] * pos_std + pos_mean
        pose7 = grasp_to_pose7(gd, rot_repr)
        for s, e in gripper_segs(pose7[:3], pose7[3:7]):
            ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]],
                    color=grasp_color, linewidth=2.0, zorder=10)
            pts_g.extend([s, e])
    if pts_g:
        pts_g = np.array(pts_g)
        all_pts = np.vstack([scene_pts, pts_g])
    else:
        all_pts = scene_pts
    ax.set_box_aspect((1, 1, 1))
    cx, cy, cz = all_pts.mean(axis=0)
    rng = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() * 0.55
    ax.set_xlim(cx - rng, cx + rng); ax.set_ylim(cy - rng, cy + rng); ax.set_zlim(cz - rng, cz + rng)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


@torch.no_grad()
def infer_flow(model, depth, uv, g_dim, n=16, t_steps=32, temp=0.8, w_cfg=2.5,
                rot_repr="approach_yaw", seed=None):
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
    g_t = torch.randn(n, g_dim, device=device) * temp
    for k in range(t_steps):
        t = torch.full((n,), k / t_steps, device=device)
        t_emb = sinusoidal_time_embed(t, dim=64)
        v_on = model.velocity(g_t, cond_on, t_emb, uv_norm)
        v_off = model.velocity(g_t, cond_off, t_emb, uv_norm)
        v = v_off + w_cfg * (v_on - v_off)
        g_t = g_t + v / t_steps
    return g_t.cpu().numpy()


def approach_from_g(g, rot_repr):
    if rot_repr == "zhou6d":
        return _gram_schmidt_R(g[3:9])[:, 2]
    a = g[3:6]
    return a / (np.linalg.norm(a) + 1e-9)


def filter_and_balance(g_arr, pos_mean, pos_std, obj_center, dist_thresh_m,
                        n_keep, mode, rot_repr):
    pos_dn = g_arr[:, :3] * pos_std + pos_mean
    apps = np.array([approach_from_g(g, rot_repr) for g in g_arr])
    dist = np.linalg.norm(pos_dn - obj_center, axis=1)
    keep_idx = np.where(dist < dist_thresh_m)[0]
    if len(keep_idx) == 0:
        return g_arr[np.argsort(dist)[:n_keep]]
    g_f = g_arr[keep_idx]; app_f = apps[keep_idx]
    if mode == "standing" and len(g_f) > n_keep:
        is_top = np.abs(app_f[:, 2]) > 0.6
        idx_top = np.where(is_top)[0]
        idx_side = np.where(~is_top)[0]
        n_top_target = max(n_keep - 4, n_keep * 3 // 4)
        sel_top = idx_top[:n_top_target]
        sel_side = idx_side[:n_keep - len(sel_top)]
        if len(sel_top) + len(sel_side) < n_keep:
            extra = idx_top[len(sel_top):len(sel_top) + (n_keep - len(sel_top) - len(sel_side))]
            sel_top = np.concatenate([sel_top, extra])
        return g_f[np.concatenate([sel_top, sel_side])]
    return g_f[:n_keep]


def get_uv(grasps_h5, sid, idx):
    g_o = grasps_h5[sid][f"object_{idx}"]
    return np.asarray(g_o["uv_centroid"], dtype=np.float32)


def rgb_path_for_depth(depth_path: Path) -> Path:
    name = depth_path.name.replace("_depth_", "_")
    folder = depth_path.parent.name.replace("_dep", "")
    base = depth_path.parent.parent.parent / "captured_images"
    return base / folder / name


def load_flow_model(ckpt_path, device):
    ck = torch.load(ckpt_path, weights_only=False, map_location=device)
    cfg = ck["cfg"]["args"]
    rot_repr = cfg.get("rot_repr", "approach_yaw")
    g_dim = 9 if rot_repr == "zhou6d" else 8
    model = FlowGraspNet(
        g_dim=g_dim,
        block_type=cfg["block"], n_blocks=cfg["n_blocks"],
        hidden=cfg["hidden"], cond_dropout=cfg["cond_dropout"],
    ).to(device)
    model.load_state_dict(ck["ema"], strict=False); model.eval()
    norm = ck["norm_stats"]
    print(f"[load] {ckpt_path.name} ep={ck['epoch']} val={ck.get('val_loss'):.4f} "
          f"rot_repr={rot_repr} g_dim={g_dim}")
    return model, rot_repr, g_dim, norm


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m8, rep8, gd8, norm8 = load_flow_model(Path(args.ckpt_8d), device)
    m9, rep9, gd9, norm9 = load_flow_model(Path(args.ckpt_9d), device)

    fig = plt.figure(figsize=(13, 8.0), dpi=160)

    with h5py.File(POSES_H5, "r") as p, h5py.File(DET_H5, "r") as d, \
         h5py.File(GRASP_H5, "r") as g:
        for col, (label, sid, idx, ply_name, elev, azim) in enumerate(CASES, start=1):
            p_o = p[sid][f"object_{idx}"]
            depth_path = ROOT / p[sid].attrs["depth_path"]
            rgb_path = rgb_path_for_depth(depth_path)
            depth_m = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            poly = np.asarray(d[sid]["mask_poly"][idx]).reshape(-1, 2)
            mask = poly_to_mask(poly)
            scene_pts, scene_rgb = depth_mask_to_pc_rgb(depth_m, mask, rgb_bgr)
            uv = get_uv(g, sid, idx)
            ply_pts = load_ply(ply_name)
            pose = np.asarray(p_o["pose_cam"])
            R_obj = quat_R(pose[3:7]); t_obj = pose[:3]

            label_lc = label.lower()
            f_mode = ("standing" if "standing" in label_lc else
                      "lying" if "lying" in label_lc else "cube")

            # 8D
            g8_raw = infer_flow(m8, depth_m, uv, gd8, n=N_OVERSAMPLE, t_steps=T_EULER,
                                 temp=NOISE_TEMP, w_cfg=CFG_W, rot_repr=rep8, seed=SEED)
            g8 = filter_and_balance(g8_raw, np.array(norm8["pos_mean"]),
                                     np.array(norm8["pos_std"]), t_obj,
                                     DIST_THRESH_BY_MODE[f_mode], N_SAMPLES,
                                     f_mode, rep8)

            # 9D
            g9_raw = infer_flow(m9, depth_m, uv, gd9, n=N_OVERSAMPLE, t_steps=T_EULER,
                                 temp=NOISE_TEMP, w_cfg=CFG_W, rot_repr=rep9, seed=SEED)
            g9 = filter_and_balance(g9_raw, np.array(norm9["pos_mean"]),
                                     np.array(norm9["pos_std"]), t_obj,
                                     DIST_THRESH_BY_MODE[f_mode], N_SAMPLES,
                                     f_mode, rep9)
            print(f"  [{label}] 8D kept={len(g8)}/{N_OVERSAMPLE}  "
                  f"9D kept={len(g9)}/{N_OVERSAMPLE}")

            # Top: 8D
            ax_top = fig.add_subplot(2, 3, col, projection="3d")
            render(ax_top, scene_pts, scene_rgb, ply_pts, R_obj, t_obj,
                   g8, "#c0392b",
                   np.array(norm8["pos_mean"]), np.array(norm8["pos_std"]),
                   rep8, elev, azim)
            ax_top.set_title(f"8D approach+yaw\n{label}", fontsize=15, pad=4)

            # Bottom: 9D
            ax_bot = fig.add_subplot(2, 3, 3 + col, projection="3d")
            render(ax_bot, scene_pts, scene_rgb, ply_pts, R_obj, t_obj,
                   g9, "#1976d2",
                   np.array(norm9["pos_mean"]), np.array(norm9["pos_std"]),
                   rep9, elev, azim)
            ax_bot.set_title(f"9D Zhou 6D\n{label}", fontsize=15, pad=4)

    plt.subplots_adjust(left=0.0, right=1.0, top=0.965, bottom=0.0,
                        wspace=0.0, hspace=0.05)
    out_png = OUT / "fig_repr_compare.png"
    out_pdf = OUT / "fig_repr_compare.pdf"
    plt.savefig(out_png, dpi=220, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(out_pdf, bbox_inches='tight', pad_inches=0.02)
    print(f"[fig] {out_png}")
    print(f"[fig] {out_pdf}")


if __name__ == "__main__":
    main()
