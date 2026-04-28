"""Standing/Lying can 후보를 한 그림에 모아 시각 비교 (대체 GT 선정용)."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import cv2, h5py, numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
GRASP_H5 = ROOT / "img_dataset/grasp_cache/grasps.h5"
POSES_H5 = ROOT / "img_dataset/icp_cache/poses.h5"
DET_H5 = ROOT / "img_dataset/yolo_cache_v3/detections.h5"
PLY_DIR = Path("/home/robotics/Competition/RoboCup_ARM/models/ply")
OUT = ROOT / "paper_figs"

K_FX = K_FY = 1109.0
K_CX, K_CY = 640.0, 360.0
IMG_H, IMG_W = 720, 1280
GRIPPER_HALF = 0.0425; FINGER_LEN = 0.040; PALM_BACK = 0.025

STANDING = [
    ("sample_random6_32", 0), ("sample_random6_33", 0), ("sample_random6_36", 0),
    ("sample_random6_41", 0), ("sample_random6_40", 1), ("sample_random6_31", 0),
]
LYING = [
    ("sample_random6_9",  0), ("sample_random6_11", 0), ("sample_random6_8",  0),
    ("sample_random6_69", 0), ("sample_random6_58", 1), ("sample_random6_5",  0),
]


def poly_to_mask(poly):
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1,1,2)], 1)
    return m.astype(bool)


def depth_mask_to_pc(depth_m, mask):
    ys, xs = np.where(mask & (depth_m > 0.1) & (depth_m < 2.0))
    if not len(xs): return np.zeros((0,3), np.float32)
    z = depth_m[ys, xs]
    x = (xs - K_CX) * z / K_FX; y = (ys - K_CY) * z / K_FY
    return np.stack([x, y, z], axis=1).astype(np.float32)


def quat_R(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ], np.float32)


def gripper_segs(pos, q):
    R = quat_R(q); a = R[:,2]; b = R[:,1]
    tip1 = pos + b*GRIPPER_HALF; tip2 = pos - b*GRIPPER_HALF
    base1 = tip1 - a*FINGER_LEN; base2 = tip2 - a*FINGER_LEN
    palm = pos - a*FINGER_LEN; wrist = palm - a*PALM_BACK
    return [(wrist,palm),(base1,base2),(base1,tip1),(base2,tip2)]


_ply = {}
def load_ply(name):
    if name in _ply: return _ply[name]
    pcd = o3d.io.read_point_cloud(str(PLY_DIR / name))
    pts = np.asarray(pcd.points, np.float32)
    if pts.ptp(axis=0).max() > 1.0: pts *= 0.001
    pts -= pts.mean(axis=0)
    if len(pts) > 1500:
        pts = pts[np.random.default_rng(0).choice(len(pts),1500,replace=False)]
    _ply[name] = pts; return pts


def render_one(ax, sid, obj_idx, det, grasps_h5, poses_h5, title):
    g_o = grasps_h5[sid][f"object_{obj_idx}"]
    p_o = poses_h5[sid][f"object_{obj_idx}"]
    depth_path = ROOT / grasps_h5[sid].attrs["depth_path"]
    depth_m = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    poly = np.asarray(det[sid]["mask_poly"][obj_idx]).reshape(-1,2)
    mask = poly_to_mask(poly)
    scene = depth_mask_to_pc(depth_m, mask)
    if len(scene) > 1200:
        scene = scene[np.random.default_rng(0).choice(len(scene),1200,replace=False)]
    ply_pts = load_ply(p_o.attrs["ply_file"])
    pose = np.asarray(p_o["pose_cam"])
    R = quat_R(pose[3:7]); obj_pts = (ply_pts @ R.T) + pose[:3]

    if len(scene):
        h = -scene[:,1]; hn = (h-h.min())/max(h.ptp(),1e-6)
        ax.scatter(scene[:,0],scene[:,1],scene[:,2], s=4, c=hn, cmap='viridis',
                   alpha=0.75, edgecolors='none')
    if len(obj_pts):
        ax.scatter(obj_pts[:,0],obj_pts[:,1],obj_pts[:,2], s=1.0,
                   c='#bdbdbd', alpha=0.30, edgecolors='none')

    grasps = np.asarray(g_o["grasps_cam"])
    groups = np.asarray(g_o["grasp_group"])
    cmap = {0:"#2ecc71",1:"#f39c12",2:"#2ecc71",3:"#2ecc71",4:"#e74c3c"}
    for k, grp in enumerate(groups):
        col = cmap.get(int(grp),"#666")
        for a,b in gripper_segs(grasps[k,:3], grasps[k,3:7]):
            ax.plot([a[0],b[0]],[a[1],b[1]],[a[2],b[2]], color=col, lw=1.2, alpha=0.92)

    if len(scene) > 5:
        c = scene.mean(axis=0); r = max(float(scene.ptp(axis=0).max()),0.10)*0.65
    else:
        c = pose[:3]; r = 0.10
    ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=-15, azim=205, vertical_axis='y'); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.pane.set_facecolor((1,1,1,0)); axis.pane.set_edgecolor((0.85,)*3+(0.5,))
    ax.set_title(title, fontsize=9, pad=2)


def main():
    fig = plt.figure(figsize=(18, 6.5), dpi=130)
    with h5py.File(GRASP_H5,"r") as g, h5py.File(POSES_H5,"r") as p, h5py.File(DET_H5,"r") as d:
        for i,(sid,oi) in enumerate(STANDING, start=1):
            ax = fig.add_subplot(2, 6, i, projection="3d")
            render_one(ax, sid, oi, d, g, p, f"STAND  {sid}/obj{oi}")
        for i,(sid,oi) in enumerate(LYING, start=1):
            ax = fig.add_subplot(2, 6, 6+i, projection="3d")
            render_one(ax, sid, oi, d, g, p, f"LYING  {sid}/obj{oi}")
    plt.tight_layout()
    out = OUT / "scan_can_candidates.png"
    plt.savefig(out, dpi=160, bbox_inches='tight')
    print(f"[scan] {out}")


if __name__ == "__main__":
    main()
