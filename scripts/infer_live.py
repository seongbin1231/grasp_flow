"""Live inference: saved RGB+Depth  →  YOLO  →  Flow Matching  →  filter  →  HTML.

Expects:
  img_dataset/live_capture/rgb.png     (BGR, 1280x720, uint8)
  img_dataset/live_capture/depth.png   (uint16, millimeters)
Outputs:
  deploy/viz/infer_live.html

Run (conda base):
  /home/robotics/anaconda3/bin/python scripts/infer_live.py
"""
from __future__ import annotations

import os, sys, math
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
sys.path.insert(0, str(ROOT))
from src.flow_dataset import CAM_CX, CAM_CY, IMG_H, IMG_W, K_FX, K_FY, _build_R_tool
from src.flow_model import FlowGraspNet, sinusoidal_time_embed

CKPT = ROOT / "runs/yolograsp_v2/v6_150ep/adaln_zero_lr0.001_nb8_h768/checkpoints/best.pt"
YOLO_W = ROOT / "runs/yolov8m_seg_v3_1280/weights/best.pt"
RGB_PATH = ROOT / "img_dataset/live_capture/rgb.png"
DEPTH_PATH = ROOT / "img_dataset/live_capture/depth.png"
OUT_DIR = ROOT / "deploy/viz/live"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Self-contained upload? Set YGRASP_EMBED=1 to inline plotly.js (+3MB/file)
EMBED_PLOTLYJS = bool(int(os.environ.get("YGRASP_EMBED", "0")))

N_SAMPLES = int(os.environ.get("YGRASP_N", 32))
N_EULER_STEPS = int(os.environ.get("YGRASP_STEPS", 32))
NOISE_TEMP = float(os.environ.get("YGRASP_TEMP", 0.8))
GUIDANCE_SCALE = float(os.environ.get("YGRASP_GUIDANCE", 2.0))
GRIPPER_HALF, FINGER_LEN, PALM_BACK, APPROACH_STEM = 0.0425, 0.040, 0.025, 0.050
BODY_MARGIN, TIP_RADIUS = 0.005, 0.015
STEM_SAMPLES, SWEEP_SAMPLES = 4, 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- model ----------
ck = torch.load(CKPT, weights_only=False, map_location=device)
cfg = ck["cfg"]["args"]
model = FlowGraspNet(block_type=cfg["block"], n_blocks=cfg["n_blocks"],
                     hidden=cfg["hidden"], cond_dropout=cfg["cond_dropout"]).to(device)
model.load_state_dict(ck["ema"], strict=False); model.eval()
print(f"[flow] {CKPT.name} ep={ck['epoch']} val={ck.get('val_loss'):.4f}")
ns = ck.get("norm_stats", {})
POS_MEAN = np.asarray(ns.get("pos_mean", [0,0,0]), np.float32)
POS_STD  = np.asarray(ns.get("pos_std",  [1,1,1]), np.float32)
NORMALIZE_POS = bool(ns.get("normalize_pos", False))

# ---------- helpers ----------
def back_project(depth, stride=3, mask=None):
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[ys, xs]
    valid = (z > 0.1) & (z < 2.0)
    if mask is not None: valid &= mask[ys, xs]
    z = z[valid]; xs = xs[valid]; ys = ys[valid]
    x = (xs - CAM_CX) * z / K_FX
    y = (ys - CAM_CY) * z / K_FY
    return np.stack([x, y, z], axis=1).astype(np.float32)


@torch.no_grad()
def flow_sample(depth, uv, n=N_SAMPLES, steps=N_EULER_STEPS, guidance=GUIDANCE_SCALE):
    d = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0).to(device)
    uv_t = torch.tensor([[uv[0], uv[1]]], dtype=torch.float32, device=device)
    cond_on = model.encode(d, uv_t).expand(n, -1)
    cond_off = torch.zeros_like(cond_on)
    uv_b = uv_t.expand(n, -1)
    uv_norm = torch.stack([uv_b[:,0]/IMG_W, uv_b[:,1]/IMG_H], dim=-1)
    g_t = torch.randn(n, 8, device=device) * NOISE_TEMP
    dt = 1.0 / steps
    for k in range(steps):
        t_b = torch.full((n,), k*dt, device=device)
        t_emb = sinusoidal_time_embed(t_b, dim=64)
        v_on = model.velocity(g_t, cond_on, t_emb, uv_norm)
        v_off = model.velocity(g_t, cond_off, t_emb, uv_norm)
        v = v_off + guidance * (v_on - v_off)
        g_t = g_t + v * dt
    g_1 = g_t.cpu().numpy()
    if NORMALIZE_POS: g_1[:, :3] = g_1[:, :3] * POS_STD + POS_MEAN
    return g_1


def g8_to_components(g):
    pos = g[:3]; a = g[3:6]; a = a / (np.linalg.norm(a) + 1e-9)
    return pos, a, math.atan2(g[6], g[7])


def gripper_geom(pos, app, yaw):
    R = _build_R_tool(app, yaw); a = R[:,2]; b = R[:,1]
    tip1 = pos + b*GRIPPER_HALF; tip2 = pos - b*GRIPPER_HALF
    base1 = tip1 - a*FINGER_LEN; base2 = tip2 - a*FINGER_LEN
    palm = pos - a*FINGER_LEN
    wrist = palm - a*PALM_BACK
    app_start = wrist - a*APPROACH_STEM
    return dict(u_poly=np.stack([tip1,base1,base2,tip2]),
                palm=np.stack([palm,wrist]), app_stem=np.stack([app_start,wrist]),
                cone_tail=app_start, cone_dir=a, tcp=pos,
                tip1=tip1, tip2=tip2, wrist_pt=wrist, app_start_pt=app_start)


def uv_to_3d(depth, uv):
    u, v = int(round(uv[0])), int(round(uv[1]))
    u = max(6, min(IMG_W-7, u)); v = max(6, min(IMG_H-7, v))
    patch = depth[v-5:v+6, u-5:u+6]
    vals = patch[(patch>0.1)&(patch<2.0)]
    if vals.size == 0: return None
    z = float(np.median(vals))
    return np.array([(uv[0]-CAM_CX)*z/K_FX, (uv[1]-CAM_CY)*z/K_FY, z])


def pick_best(grasps, flags, uv_3d, az_top=0.7):
    kept = [i for i, f in enumerate(flags) if f == "kept"]
    if not kept or uv_3d is None: return -1
    top_pool = [i for i in kept if abs(g8_to_components(grasps[i])[1][2]) > az_top]
    pool = top_pool if top_pool else kept
    dists = [float(np.linalg.norm(grasps[i][:3] - uv_3d)) for i in pool]
    return pool[int(np.argmin(dists))]


def filter_one(w, tree):
    stem = np.linspace(w["wrist_pt"], w["app_start_pt"], STEM_SAMPLES)
    body = np.vstack([w["wrist_pt"][None], stem])
    db, _ = tree.query(body, k=1)
    if float(db.min()) < BODY_MARGIN: return "body_collision"
    dsweep = np.inf
    for tip in (w["tip1"], w["tip2"]):
        s = np.linspace(tip, w["tcp"], SWEEP_SAMPLES)
        d, _ = tree.query(s, k=1)
        dsweep = min(dsweep, float(d.min()))
    return "kept" if dsweep <= TIP_RADIUS else "no_contact"


def _add_grasp_group(fig, row, col, grasps, color, w, dash, name, lg, show,
                     show_cone=True, tcp_size=4):
    if len(grasps) == 0: return
    ux,uy,uz=[],[],[]; sx,sy,sz=[],[],[]; ax,ay,az=[],[],[]
    cx_,cy_,cz_,cu,cv,cw=[],[],[],[],[],[]; tx,ty,tz=[],[],[]
    for g in grasps:
        pos,a,yaw = g8_to_components(g); gg = gripper_geom(pos,a,yaw)
        ux.extend([*gg["u_poly"][:,0],None]); uy.extend([*gg["u_poly"][:,1],None]); uz.extend([*gg["u_poly"][:,2],None])
        sx.extend([*gg["palm"][:,0],None]); sy.extend([*gg["palm"][:,1],None]); sz.extend([*gg["palm"][:,2],None])
        ax.extend([*gg["app_stem"][:,0],None]); ay.extend([*gg["app_stem"][:,1],None]); az.extend([*gg["app_stem"][:,2],None])
        cx_.append(gg["cone_tail"][0]); cy_.append(gg["cone_tail"][1]); cz_.append(gg["cone_tail"][2])
        cu.append(gg["cone_dir"][0]); cv.append(gg["cone_dir"][1]); cw.append(gg["cone_dir"][2])
        tx.append(gg["tcp"][0]); ty.append(gg["tcp"][1]); tz.append(gg["tcp"][2])
    fig.add_trace(go.Scatter3d(x=ux,y=uy,z=uz,mode="lines",
        line=dict(color=color,width=w,dash=dash),name=name,legendgroup=lg,showlegend=show),
        row=row,col=col)
    fig.add_trace(go.Scatter3d(x=sx,y=sy,z=sz,mode="lines",
        line=dict(color=color,width=max(1.5,w-1.5),dash=dash),legendgroup=lg,showlegend=False),row=row,col=col)
    fig.add_trace(go.Scatter3d(x=ax,y=ay,z=az,mode="lines",
        line=dict(color=color,width=max(1.0,w-2.5),dash="dot"),legendgroup=lg,showlegend=False),row=row,col=col)
    if show_cone:
        fig.add_trace(go.Cone(x=cx_,y=cy_,z=cz_,u=cu,v=cv,w=cw,
            sizemode="absolute", sizeref=0.010, anchor="tail",
            colorscale=[[0,color],[1,color]], showscale=False,
            legendgroup=lg, showlegend=False, hoverinfo="skip"),row=row,col=col)
    fig.add_trace(go.Scatter3d(x=tx,y=ty,z=tz,mode="markers",
        marker=dict(size=tcp_size,color=color,line=dict(color="white",width=0.5)),
        legendgroup=lg,showlegend=False),row=row,col=col)


# ---------- YOLO ----------
print("[yolo] loading...")
from ultralytics import YOLO
yolo = YOLO(str(YOLO_W))

bgr = cv2.imread(str(RGB_PATH), cv2.IMREAD_COLOR)
depth = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
print(f"[img] rgb={bgr.shape} dtype={bgr.dtype}  depth={depth.shape} range=[{depth.min():.3f},{depth.max():.3f}]m")

res = yolo.predict(bgr, conf=0.4, iou=0.5, verbose=False)[0]
if res.masks is None or len(res.boxes) == 0:
    print("[yolo] no detections"); sys.exit(0)
classes = res.boxes.cls.cpu().numpy().astype(int)
names = [res.names[c] for c in classes]
confs = res.boxes.conf.cpu().numpy()
masks = res.masks.data.cpu().numpy()           # (N, h, w) — 640-ish
# resize masks to 1280x720
masks_full = np.stack([cv2.resize(m.astype(np.uint8), (IMG_W, IMG_H),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
                       for m in masks])
print(f"[yolo] detections: {len(classes)}")
for i, (n, c) in enumerate(zip(names, confs)):
    ys, xs = np.where(masks_full[i])
    u = float(xs.mean()); v = float(ys.mean())
    print(f"  obj_{i:2d} {n:12s} conf={c:.3f}  uv=({u:.0f},{v:.0f})  mask_px={masks_full[i].sum()}")

# ---------- one HTML per detection + index.html ----------
sprop = dict(aspectmode="data",
    xaxis=dict(title="X (m)", backgroundcolor="#fafafa"),
    yaxis=dict(title="Y (m)", backgroundcolor="#fafafa"),
    zaxis=dict(title="Z (m)", backgroundcolor="#fafafa"),
    camera=dict(up=dict(x=0,y=-1,z=0), eye=dict(x=1.2,y=-1.2,z=-1.2)),
    dragmode="turntable")

kept_summary = []
per_obj_files = []

for i in range(len(classes)):
    mask = masks_full[i]
    ys, xs = np.where(mask)
    uv = np.array([xs.mean(), ys.mean()], dtype=np.float32)

    grasps = flow_sample(depth, uv)
    tree = cKDTree(back_project(depth, stride=3))
    flags = []
    for g in grasps:
        pos, a, yaw = g8_to_components(g)
        flags.append(filter_one(gripper_geom(pos, a, yaw), tree))
    n_kept = sum(1 for f in flags if f == "kept")
    n_coll = sum(1 for f in flags if f == "body_collision")
    n_air = sum(1 for f in flags if f == "no_contact")
    azs = [abs(g8_to_components(g)[1][2]) for g, f in zip(grasps, flags) if f == "kept"]
    top_n = sum(1 for z in azs if z > 0.7); side_n = sum(1 for z in azs if z < 0.3)
    kept_summary.append((names[i], confs[i], uv.tolist(), n_kept, n_coll, n_air, top_n, side_n))

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
    pc_bg = back_project(depth, stride=4, mask=~mask)
    if len(pc_bg) > 6000:
        pc_bg = pc_bg[np.random.choice(len(pc_bg), 6000, replace=False)]
    pc_ob = back_project(depth, stride=3, mask=mask)
    if len(pc_ob) > 3000:
        pc_ob = pc_ob[np.random.choice(len(pc_ob), 3000, replace=False)]
    fig.add_trace(go.Scatter3d(x=pc_bg[:,0], y=pc_bg[:,1], z=pc_bg[:,2],
        mode="markers", marker=dict(size=1.5, color="lightgray", opacity=0.45),
        name="scene"))
    fig.add_trace(go.Scatter3d(x=pc_ob[:,0], y=pc_ob[:,1], z=pc_ob[:,2],
        mode="markers", marker=dict(size=2.4, color="#2ca02c", opacity=0.85),
        name=f"target {names[i]}"))
    u, v = int(uv[0]), int(uv[1])
    u = max(6,min(IMG_W-7,u)); v = max(6,min(IMG_H-7,v))
    patch = depth[v-5:v+6, u-5:u+6]
    vals = patch[(patch>0.1)&(patch<2.0)]
    if vals.size > 0:
        z = float(np.median(vals))
        uv3 = np.array([(uv[0]-CAM_CX)*z/K_FX, (uv[1]-CAM_CY)*z/K_FY, z])
        fig.add_trace(go.Scatter3d(x=[uv3[0]], y=[uv3[1]], z=[uv3[2]], mode="markers",
            marker=dict(size=8, color="magenta", symbol="x"), name="YOLO uv"))
    grasps_arr = np.asarray(grasps)
    uv_3d = uv_to_3d(depth, uv)
    best_idx = pick_best(grasps_arr, flags, uv_3d)
    kept_mask = np.array([f == "kept"           for f in flags])
    coll_mask = np.array([f == "body_collision" for f in flags])
    air_mask  = np.array([f == "no_contact"     for f in flags])
    best_mask = np.zeros(len(grasps_arr), dtype=bool)
    if best_idx >= 0:
        best_mask[best_idx] = True
        kept_mask = kept_mask & ~best_mask
    if air_mask.any():
        _add_grasp_group(fig, 1, 1, grasps_arr[air_mask],
                         "rgba(90,140,230,0.85)", 2.5, "dot",
                         f"no_contact ({n_air})", "no_contact", True,
                         show_cone=True, tcp_size=3)
    if coll_mask.any():
        _add_grasp_group(fig, 1, 1, grasps_arr[coll_mask],
                         "rgba(255,150,40,0.90)", 2.5, "dash",
                         f"body_collision ({n_coll})", "collision", True,
                         show_cone=True, tcp_size=3)
    if kept_mask.any():
        _add_grasp_group(fig, 1, 1, grasps_arr[kept_mask],
                         "#e41a1c", 4.5, "solid",
                         f"kept ({kept_mask.sum()}/{N_SAMPLES})", "kept", True,
                         show_cone=True, tcp_size=5)
    if best_mask.any():
        _add_grasp_group(fig, 1, 1, grasps_arr[best_mask],
                         "#ffd700", 7.0, "solid",
                         "best (top-down + nearest uv)", "best", True,
                         show_cone=True, tcp_size=8)
        bp = grasps_arr[best_idx][:3]
        fig.add_trace(go.Scatter3d(x=[bp[0]], y=[bp[1]], z=[bp[2]],
            mode="markers",
            marker=dict(size=11, color="#ffd700", symbol="diamond",
                        line=dict(color="black", width=1)),
            name="best TCP", legendgroup="best", showlegend=False), row=1, col=1)
    if "scene" in fig.layout: fig.layout["scene"].update(**sprop)
    fig.update_layout(
        title=(f"<b>obj_{i} {names[i]} (conf {confs[i]:.2f})</b> &nbsp; "
               f"uv=({uv[0]:.0f},{uv[1]:.0f})  kept={n_kept}/{N_SAMPLES}  "
               f"(coll {n_coll}, air {n_air})  top/side={top_n}/{side_n}"),
        width=1400, height=780, template="plotly_white",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="gray", borderwidth=1),
        margin=dict(l=10, r=10, t=60, b=10))
    fname = f"obj_{i:02d}_{names[i]}.html"
    fpath = OUT_DIR / fname
    fig.write_html(str(fpath),
                   include_plotlyjs=(True if EMBED_PLOTLYJS else "cdn"))
    per_obj_files.append((fname, names[i], confs[i], n_kept, n_coll, n_air))

# ---------- index.html with tab nav ----------
tabs_json = "[\n" + ",\n".join(
    f'    {{"id":"{fname.replace(".html","")}","label":"obj_{i} {nm} ({cf:.2f}) — kept {k}/{N_SAMPLES}",'
    f'"file":"{fname}"}}'
    for i, (fname, nm, cf, k, _, _) in enumerate(per_obj_files)
) + "\n  ]"

HEADER = (f"ckpt: <b>v6_150ep/best.pt</b> ep=118 val={ck.get('val_loss'):.4f}  ·  "
          f"inference: N={N_SAMPLES}, {N_EULER_STEPS}-step Euler, temp={NOISE_TEMP}, CFG={GUIDANCE_SCALE}  ·  "
          f"filter: body&lt;{BODY_MARGIN*1000:.0f}mm + sweep&lt;{TIP_RADIUS*1000:.0f}mm  ·  "
          f"source: ROS /camera/[rgb,depth] @ DOMAIN_ID=13  ·  "
          f"embed: {'ON (offline-ok)' if EMBED_PLOTLYJS else 'CDN (needs internet)'}")

INDEX_HTML = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<title>YOLO-Grasp live inference</title>
<style>
  html, body {{ margin:0; padding:0; height:100%; font-family: ui-sans-serif, system-ui, sans-serif; background:#0f1419; color:#e6e6e6; }}
  header {{ background:#1a2028; padding: 10px 16px; border-bottom: 1px solid #2a3038; }}
  h1 {{ margin: 0 0 6px 0; font-size: 15px; font-weight: 600; }}
  .meta {{ font-size: 12px; color:#9ab; }}
  nav {{ background:#151a1f; padding: 8px; border-bottom: 1px solid #2a3038; display:flex; gap:4px; flex-wrap:wrap; }}
  nav button {{
    background:#222a33; color:#ccc; border: 1px solid #2a3038; padding: 7px 12px;
    border-radius: 4px; cursor: pointer; font-size: 13px; font-weight: 500;
  }}
  nav button:hover {{ background:#2c3540; color:#fff; }}
  nav button.active {{ background: #e41a1c; color: white; border-color: #e41a1c; }}
  main {{ height: calc(100vh - 108px); }}
  iframe {{ width:100%; height:100%; border: none; background:white; }}
  .kbd {{ background:#2a3038; padding: 1px 5px; border-radius: 3px; font-family: monospace; font-size: 11px; }}
</style></head>
<body>
<header>
  <h1>YOLO-Grasp Flow Matching — live inference (per object)</h1>
  <div class="meta">{HEADER}  ·  nav: <span class="kbd">← →</span> or click</div>
</header>
<nav id="tabs"></nav>
<main><iframe id="viewer" src=""></iframe></main>
<script>
  const CASES = {tabs_json};
  const nav = document.getElementById("tabs");
  const viewer = document.getElementById("viewer");
  let active = 0;
  function setActive(i) {{
    active = i;
    viewer.src = CASES[i].file;
    [...nav.children].forEach((b, j) => b.classList.toggle("active", j === i));
  }}
  CASES.forEach((c, i) => {{
    const btn = document.createElement("button");
    btn.textContent = c.label;
    btn.onclick = () => setActive(i);
    nav.appendChild(btn);
  }});
  document.addEventListener("keydown", (e) => {{
    if (e.key === "ArrowRight") setActive((active + 1) % CASES.length);
    else if (e.key === "ArrowLeft") setActive((active - 1 + CASES.length) % CASES.length);
  }});
  setActive(0);
</script>
</body></html>
"""
(OUT_DIR / "index.html").write_text(INDEX_HTML)

# ---------- print summary ----------
print()
print(f"=== LIVE INFERENCE ({len(classes)} detections) ===")
print(f"{'obj':<14} {'conf':>5} {'uv':>12} {'kept':>6} {'coll':>5} {'air':>5} {'top/side':>10}")
for nm, cf, uv, k, co, a_, t, s in kept_summary:
    print(f"{nm:<14} {cf:5.2f} ({uv[0]:4.0f},{uv[1]:4.0f}) {k:4d}/{N_SAMPLES:<3d} {co:5d} {a_:5d}  {t:3d}/{s:<3d}")

print(f"\n[save] {OUT_DIR}/")
for fname, _, _, _, _, _ in per_obj_files:
    p = OUT_DIR / fname; print(f"   {fname}   ({p.stat().st_size/1e6:.2f} MB)")
idx = OUT_DIR / "index.html"
print(f"   index.html   ({idx.stat().st_size/1e3:.1f} KB)")
print(f"\nopen: xdg-open {idx}")
