"""GT grasp visualization — same 7 demo cases as demo_inference.py,
but drawing the synthesized ground-truth grasps from datasets/grasp_v2.h5.

Pred (red)  vs  GT (green solid, thick).  Same scene PC + uv marker.
"""
from __future__ import annotations

import math, os, sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
sys.path.insert(0, str(ROOT))
from src.flow_dataset import (CAM_CX, CAM_CY, IMG_H, IMG_W, K_FX, K_FY,
                              quat_wxyz_to_R, _build_R_tool, _yaw_from_Rtool)

DATASET = ROOT / "datasets/grasp_v2.h5"
POSES = ROOT / "img_dataset/icp_cache/poses.h5"
YOLO_CACHE = ROOT / "img_dataset/yolo_cache_v3/detections.h5"
OUT_DIR = ROOT / "deploy/viz/gt"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMBED = bool(int(os.environ.get("YGRASP_EMBED", "0")))

GRIPPER_HALF, FINGER_LEN, PALM_BACK, APPROACH_STEM = 0.0425, 0.040, 0.025, 0.050

# Same cases as demo_inference.py
CASES = {
    "standing_bottle": [("sample_random6_30", 3), ("sample_random6_31", 2)],
    "standing_can":    [("sample_random6_30", 1), ("sample_random6_31", 0)],
    "lying_bottle":    [("sample_random6_1",  2), ("sample_random6_10", 1)],
    "lying_can":       [("sample_random6_11", 0), ("sample_random6_16", 1)],
    "lying_marker":    [("sample_random6_1",  3), ("sample_random6_10", 5)],
    "lying_spam":      [("sample_random6_16", 0), ("sample_random6_20", 0)],
    "cube":            [("sample_random6_30", 7), ("sample_random6_30", 4)],
}


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


def load_depth(poses_group):
    path = ROOT / poses_group.attrs["depth_path"]
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0


def poly_to_mask(poly):
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def pick_yolo_uv_mask(det_group, obj_idx):
    uv = det_group["uv_centroid"][obj_idx]
    poly = np.asarray(det_group["mask_poly"][obj_idx]).reshape(-1, 2)
    return uv.astype(np.float32), poly_to_mask(poly)


def gripper_u_from_quat(pos, q_wxyz):
    R = quat_wxyz_to_R(q_wxyz)
    a = R[:, 2]; b = R[:, 1]
    tip1 = pos + b*GRIPPER_HALF; tip2 = pos - b*GRIPPER_HALF
    base1 = tip1 - a*FINGER_LEN; base2 = tip2 - a*FINGER_LEN
    palm = pos - a*FINGER_LEN
    wrist = palm - a*PALM_BACK
    app_start = wrist - a*APPROACH_STEM
    return dict(
        u_poly=np.stack([tip1, base1, base2, tip2]),
        palm=np.stack([palm, wrist]),
        app_stem=np.stack([app_start, wrist]),
        cone_tail=app_start, cone_dir=a, tcp=pos,
    )


def add_gt_grasps(fig, row, col, grasps_7d, color, name, show_legend):
    """grasps_7d: (N, 7) = [x,y,z,qw,qx,qy,qz]"""
    if len(grasps_7d) == 0: return
    ux,uy,uz=[],[],[]; sx,sy,sz=[],[],[]; ax,ay,az=[],[],[]
    cx_,cy_,cz_,cu,cv,cw=[],[],[],[],[],[]; tx,ty,tz=[],[],[]
    for g in grasps_7d:
        pos = g[:3]; q = g[3:7]
        w = gripper_u_from_quat(pos, q)
        ux.extend([*w["u_poly"][:,0],None]); uy.extend([*w["u_poly"][:,1],None]); uz.extend([*w["u_poly"][:,2],None])
        sx.extend([*w["palm"][:,0],None]); sy.extend([*w["palm"][:,1],None]); sz.extend([*w["palm"][:,2],None])
        ax.extend([*w["app_stem"][:,0],None]); ay.extend([*w["app_stem"][:,1],None]); az.extend([*w["app_stem"][:,2],None])
        cx_.append(w["cone_tail"][0]); cy_.append(w["cone_tail"][1]); cz_.append(w["cone_tail"][2])
        cu.append(w["cone_dir"][0]); cv.append(w["cone_dir"][1]); cw.append(w["cone_dir"][2])
        tx.append(w["tcp"][0]); ty.append(w["tcp"][1]); tz.append(w["tcp"][2])
    fig.add_trace(go.Scatter3d(x=ux,y=uy,z=uz,mode="lines",
        line=dict(color=color,width=5), name=name,
        legendgroup="gt", showlegend=show_legend), row=row, col=col)
    fig.add_trace(go.Scatter3d(x=sx,y=sy,z=sz,mode="lines",
        line=dict(color=color,width=3), legendgroup="gt", showlegend=False),row=row,col=col)
    fig.add_trace(go.Scatter3d(x=ax,y=ay,z=az,mode="lines",
        line=dict(color=color,width=2,dash="dot"), legendgroup="gt", showlegend=False),row=row,col=col)
    fig.add_trace(go.Cone(x=cx_,y=cy_,z=cz_,u=cu,v=cv,w=cw,
        sizemode="absolute", sizeref=0.012, anchor="tail",
        colorscale=[[0,color],[1,color]], showscale=False,
        legendgroup="gt", showlegend=False, hoverinfo="skip"), row=row, col=col)
    fig.add_trace(go.Scatter3d(x=tx,y=ty,z=tz,mode="markers",
        marker=dict(size=4,color=color,line=dict(color="white",width=0.5)),
        legendgroup="gt", showlegend=False), row=row, col=col)


def add_scene_pcs(fig, row, col, depth, uv, obj_mask, show_legend):
    pc_bg = back_project(depth, stride=4, mask=~obj_mask)
    if len(pc_bg) > 6000:
        pc_bg = pc_bg[np.random.choice(len(pc_bg), 6000, replace=False)]
    pc_ob = back_project(depth, stride=3, mask=obj_mask)
    if len(pc_ob) > 3000:
        pc_ob = pc_ob[np.random.choice(len(pc_ob), 3000, replace=False)]
    fig.add_trace(go.Scatter3d(x=pc_bg[:,0],y=pc_bg[:,1],z=pc_bg[:,2],mode="markers",
        marker=dict(size=1.5,color="lightgray",opacity=0.45),
        name="scene", showlegend=show_legend), row=row, col=col)
    fig.add_trace(go.Scatter3d(x=pc_ob[:,0],y=pc_ob[:,1],z=pc_ob[:,2],mode="markers",
        marker=dict(size=2.4,color="#2ca02c",opacity=0.85),
        name="target obj (YOLO mask)", showlegend=show_legend), row=row, col=col)
    u,v = int(round(uv[0])), int(round(uv[1]))
    u = max(6,min(IMG_W-7,u)); v = max(6,min(IMG_H-7,v))
    patch = depth[v-5:v+6,u-5:u+6]
    vals = patch[(patch>0.1)&(patch<2.0)]
    if vals.size > 0:
        z = float(np.median(vals))
        uv3 = np.array([(uv[0]-CAM_CX)*z/K_FX,(uv[1]-CAM_CY)*z/K_FY,z])
        fig.add_trace(go.Scatter3d(x=[uv3[0]],y=[uv3[1]],z=[uv3[2]],mode="markers",
            marker=dict(size=8,color="magenta",symbol="x",line=dict(color="white",width=1)),
            name="uv (YOLO centroid)", showlegend=show_legend), row=row, col=col)


# ---------- main ----------
def main():
    # Load GT from val split
    f_ds = h5py.File(DATASET, "r")
    sref_all = f_ds["val/sample_ref"][:]
    if sref_all.dtype.kind == "O":
        sref_all = np.array([s.decode() if isinstance(s, bytes) else s for s in sref_all])
    uvs_all    = f_ds["val/uvs"][:]            # (N, 2) per row, same for same object
    grasps_all = f_ds["val/grasps_cam"][:]
    cls_all    = f_ds["val/object_class"][:]
    mode_all   = f_ds["val/object_mode"][:]
    CLASS_NAMES = ["bottle","can","cube","marker","spam"]
    MODE_NAMES  = ["lying","standing","cube"]

    poses_f = h5py.File(POSES, "r")
    dets_f = h5py.File(YOLO_CACHE, "r")

    for cat, pairs in CASES.items():
        titles = [f"{cat}  case {i+1}" for i in range(len(pairs))]
        fig = make_subplots(rows=1, cols=2, specs=[[{"type":"scene"}]*2],
                            subplot_titles=titles, horizontal_spacing=0.02)
        for ci, (sid, obj_idx) in enumerate(pairs):
            # scene + YOLO uv (per-scene obj_idx)
            p_sid = poses_f[sid]; d_sid = dets_f[sid]
            uv, obj_mask = pick_yolo_uv_mask(d_sid, obj_idx)
            depth = load_depth(p_sid)

            # Match grasp_v2 rows by (sample_ref == sid) AND uv close to YOLO uv
            scene_mask = sref_all == sid
            d_uv = np.linalg.norm(uvs_all - uv[None, :], axis=1)   # (N,) pixel distance
            sel = scene_mask & (d_uv < 2.0)                        # <2px tolerance
            if not sel.any():
                # relax tolerance for fallback
                sel = scene_mask & (d_uv < 10.0)
            if not sel.any():
                print(f"  [{cat}] {sid}/obj_{obj_idx} uv=({uv[0]:.0f},{uv[1]:.0f}): "
                      f"NO GT rows (closest uv dist in scene = "
                      f"{d_uv[scene_mask].min() if scene_mask.any() else 'n/a':.1f}px)")
                # still draw scene so subplot isn't empty
                add_scene_pcs(fig, 1, ci+1, depth, uv, obj_mask, show_legend=(ci==0))
                continue
            gt = grasps_all[sel]
            cls_name = CLASS_NAMES[int(cls_all[np.argmax(sel)])]
            mode_name = MODE_NAMES[int(mode_all[np.argmax(sel)])]
            print(f"[{cat}] {sid}/obj_{obj_idx} uv=({uv[0]:.0f},{uv[1]:.0f}) "
                  f"cls={cls_name} mode={mode_name} → {len(gt)} GT grasps")

            add_scene_pcs(fig, 1, ci+1, depth, uv, obj_mask, show_legend=(ci==0))
            add_gt_grasps(fig, 1, ci+1, gt, color="#2ca02c",
                          name=f"GT grasps ({len(gt)})", show_legend=(ci==0))

        sprop = dict(aspectmode="data",
            xaxis=dict(title="X (m)", backgroundcolor="#fafafa"),
            yaxis=dict(title="Y (m)", backgroundcolor="#fafafa"),
            zaxis=dict(title="Z (m)", backgroundcolor="#fafafa"),
            camera=dict(up=dict(x=0,y=-1,z=0), eye=dict(x=1.2,y=-1.2,z=-1.2)),
            dragmode="turntable")
        for sn in ("scene","scene2"):
            if sn in fig.layout: fig.layout[sn].update(**sprop)
        fig.update_layout(
            title=f"<b>GT grasps — {cat}</b>   "
                  f"(val scene random6, from datasets/grasp_v2.h5)",
            width=1600, height=800, template="plotly_white",
            legend=dict(yanchor="top",y=0.98,xanchor="left",x=0.01,
                        bgcolor="rgba(255,255,255,0.85)",bordercolor="gray",borderwidth=1),
            margin=dict(l=10,r=10,t=60,b=10))
        out = OUT_DIR / f"gt_{cat}.html"
        fig.write_html(str(out), include_plotlyjs=(True if EMBED else "cdn"))
        print(f"  wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")

    f_ds.close(); poses_f.close(); dets_f.close()

    # Index
    INDEX = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>GT grasps</title>
<style>
  html,body{margin:0;padding:0;height:100%;font-family:ui-sans-serif,system-ui,sans-serif;background:#0f1419;color:#e6e6e6;}
  header{background:#1a2028;padding:10px 16px;border-bottom:1px solid #2a3038;}
  h1{margin:0 0 6px 0;font-size:15px;font-weight:600;}
  .meta{font-size:12px;color:#9ab;}
  nav{background:#151a1f;padding:8px;border-bottom:1px solid #2a3038;display:flex;gap:4px;flex-wrap:wrap;}
  nav button{background:#222a33;color:#ccc;border:1px solid #2a3038;padding:8px 14px;border-radius:4px;cursor:pointer;font-size:13px;font-weight:500;}
  nav button:hover{background:#2c3540;color:#fff;}
  nav button.active{background:#2ca02c;color:white;border-color:#2ca02c;}
  main{height:calc(100vh - 108px);}
  iframe{width:100%;height:100%;border:none;background:white;}
  .kbd{background:#2a3038;padding:1px 5px;border-radius:3px;font-family:monospace;font-size:11px;}
</style></head>
<body>
<header>
  <h1>YOLO-Grasp — Ground Truth grasps (val scene random6)</h1>
  <div class="meta">
    source: <b>datasets/grasp_v2.h5</b> (schema v2, 6-DoF SE(3))  &nbsp;·&nbsp;
    policy: standing 16 (top8+side8), lying bottle/can 8, lying marker/spam 6, cube 2  &nbsp;·&nbsp;
    pred viz: <a style="color:#9cf" href="../index.html">../index.html</a>
    &nbsp;·&nbsp; nav: <span class="kbd">← →</span> 또는 클릭
  </div>
</header>
<nav id="tabs"></nav>
<main><iframe id="viewer" src=""></iframe></main>
<script>
  const CASES = [
    {id:"standing_bottle", label:"Standing Bottle"},
    {id:"standing_can",    label:"Standing Can"},
    {id:"lying_bottle",    label:"Lying Bottle"},
    {id:"lying_can",       label:"Lying Can"},
    {id:"lying_marker",    label:"Lying Marker"},
    {id:"lying_spam",      label:"Lying Spam"},
    {id:"cube",            label:"Cube"},
  ];
  const nav = document.getElementById("tabs");
  const viewer = document.getElementById("viewer");
  let active = 0;
  function setActive(i){active=i;viewer.src=`gt_${CASES[i].id}.html`;
    [...nav.children].forEach((b,j)=>b.classList.toggle("active",j===i));}
  CASES.forEach((c,i)=>{const b=document.createElement("button");
    b.textContent=c.label;b.onclick=()=>setActive(i);nav.appendChild(b);});
  document.addEventListener("keydown",e=>{
    if(e.key==="ArrowRight")setActive((active+1)%CASES.length);
    else if(e.key==="ArrowLeft")setActive((active-1+CASES.length)%CASES.length);});
  setActive(0);
</script></body></html>"""
    (OUT_DIR / "index.html").write_text(INDEX)
    print(f"\n[save] {OUT_DIR}/index.html")


if __name__ == "__main__":
    main()
