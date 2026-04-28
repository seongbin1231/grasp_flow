"""
End-to-end inference demo:
  RGB → YOLO → mask centroid (u,v) → Flow Matching (N=32) → 3D HTML

Produces per-category HTMLs in scripts/_infer_viz/ with 2 cases each.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
sys.path.insert(0, str(ROOT))

from src.flow_dataset import (
    CAM_CX, CAM_CY, IMG_H, IMG_W, K_FX, K_FY, _build_R_tool,
)
from src.flow_model import FlowGraspNet

import os
_default_ckpt = "runs/yolograsp_v2/v2_posnorm/checkpoints/best.pt"
_legacy_ckpt = "runs/yolograsp_v2/sweep_v1/adaln_zero_lr0.001_nb8_h512/checkpoints/best.pt"
_ckpt_candidate = ROOT / os.environ.get("YGRASP_CKPT", _default_ckpt)
CKPT = _ckpt_candidate if _ckpt_candidate.exists() else (ROOT / _legacy_ckpt)
YOLO_WEIGHTS = ROOT / "runs/yolov8m_seg_v3_1280/weights/best.pt"
YOLO_CACHE = ROOT / "img_dataset/yolo_cache_v3/detections.h5"
POSES = ROOT / "img_dataset/icp_cache/poses.h5"
OUT_DIR = ROOT / os.environ.get("YGRASP_OUT", "deploy/viz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = int(os.environ.get("YGRASP_N", 32))
GRIPPER_HALF = 0.0425
FINGER_LEN = 0.040
PALM_BACK = 0.025
APPROACH_STEM = 0.050

# --- collision / contact filter ---
# Body = wrist + approach-stem samples (palm excluded — for a side grasp on a
# narrow bottle the palm sits ~5 mm outside the cylinder surface and would
# false-trigger collision).
# Contact is checked by SWEEPING each fingertip toward the TCP (= closure path)
# and asking whether any sample point comes within TIP_RADIUS of the scene PC.
# Raw tip-to-PC distance over-rejects narrow objects: a Robotiq finger (4.25 cm
# half-opening) around a 1.5 cm-radius bottle starts 2.75 cm clear of the
# visible surface; the closure path, not the open tip, is what makes contact.
FILTER_ENABLED = os.environ.get("YGRASP_FILTER", "on").lower() != "off"
BODY_MARGIN = float(os.environ.get("YGRASP_BODY_MARGIN", 0.005))   # 5mm
TIP_RADIUS = float(os.environ.get("YGRASP_TIP_RADIUS", 0.015))    # 15mm, along sweep
STEM_SAMPLES = 4
SWEEP_SAMPLES = 6

# ---------- model load ----------

N_EULER_STEPS = int(os.environ.get("YGRASP_STEPS", 32))      # Tier 1: 16 → 32
NOISE_TEMP = float(os.environ.get("YGRASP_TEMP", 0.8))       # Tier 1: 1.0 → 0.8 (concentrate around modes)
GUIDANCE_SCALE = float(os.environ.get("YGRASP_GUIDANCE", 2.0))  # CFG: 1.0 = off, 2.0 = 2× condition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CKPT, weights_only=False, map_location=device)
cfg_args = ckpt["cfg"]["args"]

ms_local_arg = cfg_args.get("multiscale_local", "") or ""
ms_scales = tuple(int(x) for x in ms_local_arg.split(",")) if ms_local_arg else None

model = FlowGraspNet(
    block_type=cfg_args["block"],
    n_blocks=cfg_args["n_blocks"],
    hidden=cfg_args["hidden"],
    cond_dropout=cfg_args["cond_dropout"],
    use_xattn=bool(cfg_args.get("xattn", False)),
    multiscale_local_scales=ms_scales,
    scale_dropout=float(cfg_args.get("scale_dropout", 0.0)),
).to(device)
model.load_state_dict(ckpt["ema"], strict=False)
model.eval()
print(f"[load] {CKPT.relative_to(ROOT)} ep={ckpt['epoch']} val_loss={ckpt.get('val_loss'):.4f}"
      f"  xattn={cfg_args.get('xattn', False)} ms={ms_scales}")

# norm stats (new checkpoints store these; fallback to training-set defaults)
if "norm_stats" in ckpt:
    POS_MEAN = np.asarray(ckpt["norm_stats"]["pos_mean"], dtype=np.float32)
    POS_STD = np.asarray(ckpt["norm_stats"]["pos_std"], dtype=np.float32)
    NORMALIZE_POS = bool(ckpt["norm_stats"]["normalize_pos"])
    print(f"[load] pos normalization: mean={POS_MEAN.tolist()} std={POS_STD.tolist()}")
else:
    # legacy checkpoint (no normalization)
    POS_MEAN = np.zeros(3, dtype=np.float32)
    POS_STD = np.ones(3, dtype=np.float32)
    NORMALIZE_POS = False
    print(f"[load] legacy ckpt — no pos normalization")

# ---------- case selection ----------

CASES = {
    "standing_bottle": [("sample_random6_30", "object_3"),
                        ("sample_random6_31", "object_2")],
    "standing_can":    [("sample_random6_30", "object_1"),
                        ("sample_random6_31", "object_0")],
    "lying_bottle":    [("sample_random6_1",  "object_2"),
                        ("sample_random6_10", "object_1")],
    "lying_can":       [("sample_random6_11", "object_0"),
                        ("sample_random6_16", "object_1")],
    "lying_marker":    [("sample_random6_1",  "object_3"),
                        ("sample_random6_10", "object_5")],
    "lying_spam":      [("sample_random6_16", "object_0"),
                        ("sample_random6_20", "object_0")],
    "cube":            [("sample_random6_30", "object_7"),  # cube_red
                        ("sample_random6_30", "object_4")], # cube_blue
}

# ---------- helpers ----------

def back_project(depth, stride=3, mask=None):
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[ys, xs]
    valid = (z > 0.1) & (z < 2.0)
    if mask is not None:
        valid &= mask[ys, xs]
    z = z[valid]; xs = xs[valid]; ys = ys[valid]
    x = (xs - CAM_CX) * z / K_FX
    y = (ys - CAM_CY) * z / K_FY
    return np.stack([x, y, z], axis=1).astype(np.float32)


def project_3d(P):
    return np.array([K_FX * P[0] / P[2] + CAM_CX,
                     K_FY * P[1] / P[2] + CAM_CY])


def load_depth(sid_group):
    path = ROOT / sid_group.attrs["depth_path"]
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return d.astype(np.float32) / 1000.0


def poly_to_mask(poly):
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


@torch.no_grad()
def flow_sample(depth_np, uv_np, n=N_SAMPLES, steps=N_EULER_STEPS,
                guidance=GUIDANCE_SCALE):
    """Multi-step Euler with Classifier-Free Guidance.

    CFG at each step:
        v_uncond = velocity(g_t, t, cond=0)
        v_cond   = velocity(g_t, t, cond=encoder)
        v_guided = v_uncond + guidance · (v_cond - v_uncond)
                 = (1-guidance) · v_uncond + guidance · v_cond

    - guidance=1.0 → pure conditional (no CFG, 1× MLP call per step)
    - guidance>1.0 → amplify conditional signal (2× MLP calls per step)
    """
    d = torch.from_numpy(depth_np).float().unsqueeze(0).unsqueeze(0).to(device)
    uv = torch.tensor([[uv_np[0], uv_np[1]]], dtype=torch.float32, device=device)

    use_xattn = bool(getattr(model, "use_xattn", False))
    # encode once, reuse across steps and N samples
    if use_xattn:
        g_feat, l_feats, l_concat = model._encode_features(d, uv)
        cond_on = torch.cat([g_feat, l_concat], dim=-1)
        g_feat_b = g_feat.expand(n, -1)
        l_feats_b = [lf.expand(n, -1) for lf in l_feats]
    else:
        cond_on = model.encode(d, uv)                        # (1, 256)
    cond_on_b = cond_on.expand(n, -1)
    cond_off_b = torch.zeros_like(cond_on_b)                 # "null" condition
    uv_b = uv.expand(n, -1)
    uv_norm = torch.stack([uv_b[:, 0] / IMG_W, uv_b[:, 1] / IMG_H], dim=-1)

    from src.flow_model import sinusoidal_time_embed
    g_t = torch.randn(n, 8, device=device) * NOISE_TEMP
    dt = 1.0 / steps
    for k in range(steps):
        t_val = k * dt
        t_b = torch.full((n,), t_val, device=device)
        t_emb = sinusoidal_time_embed(t_b, dim=64)
        if use_xattn:
            tokens_on = model._build_cond_tokens(g_feat_b, l_feats_b, t_emb, uv_norm)
            tokens_off = torch.zeros_like(tokens_on)
            v_cond = model.velocity(g_t, cond_on_b, t_emb, uv_norm, cond_tokens=tokens_on)
            if abs(guidance - 1.0) < 1e-6:
                v = v_cond
            else:
                v_uncond = model.velocity(g_t, cond_off_b, t_emb, uv_norm,
                                           cond_tokens=tokens_off)
                v = v_uncond + guidance * (v_cond - v_uncond)
        else:
            v_cond = model.velocity(g_t, cond_on_b, t_emb, uv_norm)
            if abs(guidance - 1.0) < 1e-6:
                v = v_cond
            else:
                v_uncond = model.velocity(g_t, cond_off_b, t_emb, uv_norm)
                v = v_uncond + guidance * (v_cond - v_uncond)
        g_t = g_t + v * dt

    g_1 = g_t.cpu().numpy()
    if NORMALIZE_POS:
        g_1[:, :3] = g_1[:, :3] * POS_STD + POS_MEAN
    return g_1


def g8_to_components(g):
    """g: (8,) → (pos(3), approach_unit(3), yaw)"""
    pos = g[:3]
    app = g[3:6]
    app = app / (np.linalg.norm(app) + 1e-9)
    yaw = math.atan2(g[6], g[7])
    return pos, app, yaw


def gripper_u(pos, approach, yaw):
    R = _build_R_tool(approach, yaw)
    a = R[:, 2]; b = R[:, 1]
    tip1 = pos + b * GRIPPER_HALF
    tip2 = pos - b * GRIPPER_HALF
    base1 = tip1 - a * FINGER_LEN
    base2 = tip2 - a * FINGER_LEN
    palm = pos - a * FINGER_LEN
    wrist = palm - a * PALM_BACK
    app_start = wrist - a * APPROACH_STEM
    return dict(
        u_poly=np.stack([tip1, base1, base2, tip2]),
        palm=np.stack([palm, wrist]),
        app_stem=np.stack([app_start, wrist]),
        cone_tail=app_start, cone_dir=a, tcp=pos,
        wrist_pt=wrist, app_start_pt=app_start,
        tip1=tip1, tip2=tip2,
    )


def build_scene_kdtree(depth, stride=3):
    """Full scene PC (incl. target) → KDTree."""
    pc = back_project(depth, stride=stride)
    return cKDTree(pc), pc


def grasp_filter_single(w, tree):
    """Return (flag, d_body_min, d_sweep_min).

    flag ∈ {'kept', 'body_collision', 'no_contact'}
    Sweep = samples along tip→TCP closure path, for both fingers.
    """
    wrist = w["wrist_pt"]
    app_start = w["app_start_pt"]
    stem = np.linspace(wrist, app_start, STEM_SAMPLES)
    body = np.vstack([wrist[None, :], stem])
    d_body, _ = tree.query(body, k=1)
    d_body_min = float(d_body.min())
    if d_body_min < BODY_MARGIN:
        return "body_collision", d_body_min, float("inf")

    tcp = w["tcp"]
    d_sweep_min = float("inf")
    for tip in (w["tip1"], w["tip2"]):
        samples = np.linspace(tip, tcp, SWEEP_SAMPLES)
        d, _ = tree.query(samples, k=1)
        d_sweep_min = min(d_sweep_min, float(d.min()))

    if d_sweep_min > TIP_RADIUS:
        return "no_contact", d_body_min, d_sweep_min
    return "kept", d_body_min, d_sweep_min


def filter_grasps(grasps_8d, depth):
    if not FILTER_ENABLED:
        flags = ["kept"] * len(grasps_8d)
        return np.arange(len(grasps_8d)), flags, None
    tree, _ = build_scene_kdtree(depth, stride=3)
    flags = []
    for g in grasps_8d:
        pos, app, yaw = g8_to_components(g)
        w = gripper_u(pos, app, yaw)
        flag, _, _ = grasp_filter_single(w, tree)
        flags.append(flag)
    kept_idx = np.array([i for i, f in enumerate(flags) if f == "kept"], dtype=np.int64)
    return kept_idx, flags, tree

# ---------- scene traces ----------

def _add_grasp_group(fig, row, col, grasps_8d, color, width_main, dash_main,
                     legendgroup, name, showlegend, show_cone=True):
    if len(grasps_8d) == 0:
        return
    ux, uy, uz = [], [], []
    sx, sy, sz = [], [], []
    ax, ay, az = [], [], []
    cx, cy, cz, cu, cv, cw = [], [], [], [], [], []
    tx, ty, tz = [], [], []
    for g in grasps_8d:
        pos, app, yaw = g8_to_components(g)
        w = gripper_u(pos, app, yaw)
        ux.extend([*w["u_poly"][:, 0], None])
        uy.extend([*w["u_poly"][:, 1], None])
        uz.extend([*w["u_poly"][:, 2], None])
        sx.extend([*w["palm"][:, 0], None])
        sy.extend([*w["palm"][:, 1], None])
        sz.extend([*w["palm"][:, 2], None])
        ax.extend([*w["app_stem"][:, 0], None])
        ay.extend([*w["app_stem"][:, 1], None])
        az.extend([*w["app_stem"][:, 2], None])
        cx.append(w["cone_tail"][0]); cy.append(w["cone_tail"][1]); cz.append(w["cone_tail"][2])
        cu.append(w["cone_dir"][0]);  cv.append(w["cone_dir"][1]);  cw.append(w["cone_dir"][2])
        tx.append(w["tcp"][0]); ty.append(w["tcp"][1]); tz.append(w["tcp"][2])

    fig.add_trace(go.Scatter3d(
        x=ux, y=uy, z=uz, mode="lines",
        line=dict(color=color, width=width_main, dash=dash_main),
        name=name, legendgroup=legendgroup, showlegend=showlegend,
    ), row=row, col=col)
    fig.add_trace(go.Scatter3d(
        x=sx, y=sy, z=sz, mode="lines",
        line=dict(color=color, width=max(1.5, width_main - 1.5), dash=dash_main),
        legendgroup=legendgroup, showlegend=False,
    ), row=row, col=col)
    fig.add_trace(go.Scatter3d(
        x=ax, y=ay, z=az, mode="lines",
        line=dict(color=color, width=max(1.0, width_main - 2.5), dash="dot"),
        legendgroup=legendgroup, showlegend=False,
    ), row=row, col=col)
    if show_cone:
        fig.add_trace(go.Cone(
            x=cx, y=cy, z=cz, u=cu, v=cv, w=cw,
            sizemode="absolute", sizeref=0.012, anchor="tail",
            colorscale=[[0, color], [1, color]], showscale=False,
            legendgroup=legendgroup, showlegend=False, hoverinfo="skip",
        ), row=row, col=col)
    fig.add_trace(go.Scatter3d(
        x=tx, y=ty, z=tz, mode="markers",
        marker=dict(size=3, color=color, line=dict(color="white", width=0.5)),
        legendgroup=legendgroup, showlegend=False,
    ), row=row, col=col)


def add_scene(fig, row, col, depth, uv_px, obj_mask, grasps_8d, flags,
              show_legend=False, best_idx=-1):
    # scene PC (non-object)
    pc_bg = back_project(depth, stride=4, mask=~obj_mask)
    if len(pc_bg) > 6000:
        idx = np.random.choice(len(pc_bg), 6000, replace=False); pc_bg = pc_bg[idx]
    fig.add_trace(go.Scatter3d(
        x=pc_bg[:, 0], y=pc_bg[:, 1], z=pc_bg[:, 2], mode="markers",
        marker=dict(size=1.5, color="lightgray", opacity=0.45),
        name="scene", showlegend=show_legend,
    ), row=row, col=col)

    # object PC (from YOLO mask)
    pc_obj = back_project(depth, stride=3, mask=obj_mask)
    if len(pc_obj) > 3000:
        idx = np.random.choice(len(pc_obj), 3000, replace=False); pc_obj = pc_obj[idx]
    fig.add_trace(go.Scatter3d(
        x=pc_obj[:, 0], y=pc_obj[:, 1], z=pc_obj[:, 2], mode="markers",
        marker=dict(size=2.4, color="#2ca02c", opacity=0.85),
        name="target obj (YOLO mask)", showlegend=show_legend,
    ), row=row, col=col)

    # uv 3D marker
    u, v = int(round(uv_px[0])), int(round(uv_px[1]))
    u = max(6, min(IMG_W - 7, u)); v = max(6, min(IMG_H - 7, v))
    patch = depth[v-5:v+6, u-5:u+6]
    vals = patch[(patch > 0.1) & (patch < 2.0)]
    if vals.size > 0:
        z = float(np.median(vals))
        uv3 = np.array([(uv_px[0]-CAM_CX)*z/K_FX,
                        (uv_px[1]-CAM_CY)*z/K_FY, z])
        fig.add_trace(go.Scatter3d(
            x=[uv3[0]], y=[uv3[1]], z=[uv3[2]], mode="markers",
            marker=dict(size=8, color="magenta", symbol="x", line=dict(color="white", width=1)),
            name="uv (YOLO centroid)", showlegend=show_legend,
        ), row=row, col=col)

    kept_mask = np.array([f == "kept"           for f in flags])
    coll_mask = np.array([f == "body_collision" for f in flags])
    air_mask  = np.array([f == "no_contact"     for f in flags])
    # exclude best from "kept" set so best renders on top with distinct color
    best_mask = np.zeros(len(grasps_8d), dtype=bool)
    if 0 <= best_idx < len(grasps_8d):
        best_mask[best_idx] = True
    kept_mask = kept_mask & ~best_mask
    n_kept = int(kept_mask.sum())
    n_coll = int(coll_mask.sum())
    n_air  = int(air_mask.sum())

    # order: air → collision → kept → best (top). Distinct colors + cone.
    if air_mask.any():
        _add_grasp_group(
            fig, row, col, grasps_8d[air_mask],
            color="rgba(90,140,230,0.85)", width_main=2.5, dash_main="dot",
            legendgroup="no_contact",
            name=f"no_contact ({n_air})",
            showlegend=show_legend, show_cone=True,
        )
    if coll_mask.any():
        _add_grasp_group(
            fig, row, col, grasps_8d[coll_mask],
            color="rgba(255,150,40,0.90)", width_main=2.5, dash_main="dash",
            legendgroup="collision",
            name=f"body_collision ({n_coll})",
            showlegend=show_legend, show_cone=True,
        )
    if kept_mask.any():
        _add_grasp_group(
            fig, row, col, grasps_8d[kept_mask],
            color="#e41a1c", width_main=4.5, dash_main="solid",
            legendgroup="kept",
            name=f"kept ({n_kept}/{len(grasps_8d)})",
            showlegend=show_legend, show_cone=True,
        )
    if best_mask.any():
        _add_grasp_group(
            fig, row, col, grasps_8d[best_mask],
            color="#ffd700", width_main=7.0, dash_main="solid",
            legendgroup="best",
            name="best (top-down + nearest uv)",
            showlegend=show_legend, show_cone=True,
        )
        # extra emphasis: large star marker at best TCP
        bp = grasps_8d[best_idx][:3]
        fig.add_trace(go.Scatter3d(
            x=[bp[0]], y=[bp[1]], z=[bp[2]], mode="markers",
            marker=dict(size=11, color="#ffd700", symbol="diamond",
                        line=dict(color="black", width=1)),
            name="best TCP", legendgroup="best", showlegend=False,
        ), row=row, col=col)


def pick_yolo_centroid(det_group, obj_idx):
    """Use YOLO cache uv_centroid and mask_poly."""
    uv = det_group["uv_centroid"][obj_idx]
    poly = np.asarray(det_group["mask_poly"][obj_idx]).reshape(-1, 2)
    return uv.astype(np.float32), poly


def _uv_to_3d(depth, uv_px):
    """Back-project (u,v) pixel to camera-frame 3D using median depth around uv."""
    u, v = int(round(uv_px[0])), int(round(uv_px[1]))
    u = max(6, min(IMG_W - 7, u)); v = max(6, min(IMG_H - 7, v))
    patch = depth[v-5:v+6, u-5:u+6]
    vals = patch[(patch > 0.1) & (patch < 2.0)]
    if vals.size == 0:
        return None
    z = float(np.median(vals))
    return np.array([(uv_px[0]-CAM_CX)*z/K_FX, (uv_px[1]-CAM_CY)*z/K_FY, z])


def pick_best(grasps_8d, flags, uv_3d, az_top=0.7):
    """Best = top-down priority (|a_z|>az_top), then closest TCP to uv_3d.
    Returns best index in grasps_8d, or -1 if no kept."""
    kept = [i for i, f in enumerate(flags) if f == "kept"]
    if not kept or uv_3d is None:
        return -1
    top_pool = []
    for i in kept:
        _, a, _ = g8_to_components(grasps_8d[i])
        if abs(a[2]) > az_top:
            top_pool.append(i)
    pool = top_pool if top_pool else kept
    dists = [float(np.linalg.norm(grasps_8d[i][:3] - uv_3d)) for i in pool]
    return pool[int(np.argmin(dists))]


def _side_topdown_split(grasps_8d, flags):
    """For standing cases: count kept grasps split by approach direction.
    top-down ~ approach ∥ +camZ (|a_z| > 0.7), side ~ horizontal (|a_z| < 0.3)."""
    n_top, n_side, n_mid = 0, 0, 0
    for g, f in zip(grasps_8d, flags):
        if f != "kept":
            continue
        _, app, _ = g8_to_components(g)
        az = abs(app[2])
        if az > 0.7:
            n_top += 1
        elif az < 0.3:
            n_side += 1
        else:
            n_mid += 1
    return n_top, n_side, n_mid


def run_category(cat_name, pairs):
    titles = [f"{cat_name}  case {i+1}" for i in range(len(pairs))]
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}] * 2],
                        subplot_titles=titles, horizontal_spacing=0.02)
    with h5py.File(POSES, "r") as poses, h5py.File(YOLO_CACHE, "r") as dets:
        for ci, (sid, oname) in enumerate(pairs):
            p_sid = poses[sid]; d_sid = dets[sid]
            obj = p_sid[oname]
            cls = obj.attrs["class_name"]
            obj_idx = int(oname.split("_")[1])
            uv, poly = pick_yolo_centroid(d_sid, obj_idx)
            mask = poly_to_mask(poly)
            depth = load_depth(p_sid)

            grasps = flow_sample(depth, uv, n=N_SAMPLES)
            kept_idx, flags, _ = filter_grasps(grasps, depth)
            n_kept = len(kept_idx)
            n_coll = sum(1 for f in flags if f == "body_collision")
            n_air = sum(1 for f in flags if f == "no_contact")
            uv_3d = _uv_to_3d(depth, uv)
            best_idx = pick_best(grasps, flags, uv_3d)
            extra = ""
            if cat_name.startswith("standing"):
                nt, ns, nm = _side_topdown_split(grasps, flags)
                extra = f"  [kept split: topdown={nt} side={ns} mid={nm}]"
            best_info = ""
            if best_idx >= 0:
                bp = grasps[best_idx][:3]; _, ba, _ = g8_to_components(grasps[best_idx])
                d_uv = float(np.linalg.norm(bp - uv_3d)) if uv_3d is not None else -1
                best_info = (f"  best=idx{best_idx} pos=({bp[0]:+.3f},{bp[1]:+.3f},{bp[2]:+.3f}) "
                             f"|a_z|={abs(ba[2]):.2f}  Δuv3d={d_uv*100:.1f}cm")
            print(f"[{cat_name}] {sid}/{oname} cls={cls} "
                  f"uv=({uv[0]:.0f},{uv[1]:.0f}) "
                  f"→ N={N_SAMPLES}  kept={n_kept}  "
                  f"collision={n_coll}  no_contact={n_air}{extra}{best_info}")
            add_scene(fig, 1, ci + 1, depth, uv, mask, grasps, flags,
                      show_legend=(ci == 0), best_idx=best_idx)

    scene_prop = dict(
        aspectmode="data",
        xaxis=dict(title="X (m)", backgroundcolor="#fafafa",
                   gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(title="Y (m)", backgroundcolor="#fafafa",
                   gridcolor="rgba(0,0,0,0.08)"),
        zaxis=dict(title="Z (m, depth→)", backgroundcolor="#fafafa",
                   gridcolor="rgba(0,0,0,0.08)"),
        camera=dict(up=dict(x=0, y=-1, z=0),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.2, y=-1.2, z=-1.2)),
        dragmode="turntable",
    )
    for name in ["scene", "scene2"]:
        if name in fig.layout:
            fig.layout[name].update(**scene_prop)

    filt_str = (f"  filter: body<{BODY_MARGIN*1000:.0f}mm + sweep<{TIP_RADIUS*1000:.0f}mm"
                if FILTER_ENABLED else "  filter: OFF")
    fig.update_layout(
        title=f"<b>Flow Matching inference — {cat_name}</b>   "
              f"N={N_SAMPLES}  steps={N_EULER_STEPS}  temp={NOISE_TEMP}  "
              f"CFG w={GUIDANCE_SCALE}{filt_str}",
        width=1600, height=800, template="plotly_white",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="gray",
                    borderwidth=1, itemsizing="constant"),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    out = OUT_DIR / f"infer_{cat_name}.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  wrote {out}  ({out.stat().st_size/1e6:.2f} MB)\n")


def main():
    np.random.seed(0); torch.manual_seed(0)
    for cat_name, pairs in CASES.items():
        run_category(cat_name, pairs)


if __name__ == "__main__":
    main()
