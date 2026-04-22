"""
Batch grasp synthesis for all stable ICP objects.

입력:  img_dataset/icp_cache/poses.h5  (2238 stable objects)
출력:  img_dataset/grasp_cache/grasps.h5  +  synthesis_report.json

정책 (하네스 확정 2026-04-19):
  - standing (bottle/can/marker/spam, 긴축 cam Z 평행):
      top-down 8 yaw + side-horizontal 8 azimuth = 16 grasp/object
  - lying bottle/can (긴축 cam XY 평면):
      긴축 N=4 × 180° 대칭 = 8 grasp/object
  - lying marker/spam:
      긴축 N=3 × 180° 대칭 = 6 grasp/object
  - cube_* (4종):
      R_icp edge column 정렬 top face 2 yaw (대각선 금지) = 2 grasp/object

저장: grasps_cam(n,7) = [x,y,z, qw,qx,qy,qz], approach_vec(n,3), yaw_around_app(n,),
      grasp_group(n,) int32, collision_ok(n,) bool, uv_centroid(2,) (공통)

충돌 필터 (collision_ok):
  1. 그리퍼 폭: 물체 단면 ≤ (0.085 − 0.005)  (5mm 여유)
  2. 테이블 관통: 핑거 tip이 테이블 관통하면 reject
     - top-down: tcp.z + 0.005 margin 이 z_table (scene PC z 95%) 보다 깊으면 reject
     - side-horizontal: tcp.z 가 z_table 보다 깊으면 reject
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
POSES_H5 = ROOT / "img_dataset/icp_cache/poses.h5"
DET_H5 = ROOT / "img_dataset/yolo_cache_v3/detections.h5"
OUT_DIR = ROOT / "img_dataset/grasp_cache"
OUT_H5 = OUT_DIR / "grasps.h5"
REPORT_PATH = OUT_DIR / "synthesis_report.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_FX = K_FY = 1109.0
K_CX, K_CY = 640.0, 360.0
IMG_W, IMG_H = 1280, 720

GRIPPER_MAX_OPEN = 0.085
GRIPPER_CLEARANCE = 0.005  # 물체 단면 여유
FINGER_LEN = 0.040
TCP_MARGIN = 0.003         # top-down 테이블 관통 허용 margin

SIDE_AZIMUTHS = np.linspace(0, 2 * np.pi, 8, endpoint=False)
SIDE_CAP_OFFSET_FROM_TIP_M = 0.015

# Lying tilt: long 축 주위로 approach 을 회전시켜 곡면 추종
# 클래스별 tilt 범위: bottle/can 큼 (±30°), marker/spam 작음 (±15°)
LYING_TILT_BOTTLE_CAN_DEG = [-30.0, 0.0, 30.0]
LYING_TILT_MARKER_SPAM_DEG = [-15.0, 0.0, 15.0]

# Standing cap-grasp: 꼭대기(0°) / 중간(45°) / 옆면(90°) 3단 tilt
# θ=45° 는 p_cap_top, p_cap_side 의 선형 보간 위치에 TCP 배치
STANDING_TILT_LAYERS = [(0.0, "top-down"), (45.0, "side-45"),
                        (90.0, "side-cap")]

GROUP_ID = {"top-down": 0, "side-cap": 1, "lying": 2, "cube": 3,
            "side-45": 4}


# ---------- geometry helpers ----------

def build_R(approach, yaw):
    """Tool rotation (cam frame) with columns [Tool_X, Tool_Y, Tool_Z].
    Tool Z = approach (unit). Tool Y = open direction (binormal)."""
    a = np.asarray(approach, dtype=np.float64)
    a = a / np.linalg.norm(a)
    ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.95 else np.array([0.0, 1.0, 0.0])
    b0 = ref - (ref @ a) * a
    b0 /= np.linalg.norm(b0)
    n0 = np.cross(a, b0)
    b = b0 * np.cos(yaw) + n0 * np.sin(yaw)
    b /= np.linalg.norm(b)
    x = np.cross(b, a)
    return np.column_stack([x, b, a])


def R_to_quat_wxyz(R):
    """Shepperd method; returns [qw, qx, qy, qz] with qw >= 0, unit-norm."""
    R = np.asarray(R, dtype=np.float64)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    if q[0] < 0:
        q = -q
    q /= np.linalg.norm(q)
    return q


def decide_mode(R, long_axis_idx, cls_name):
    if cls_name.startswith("cube"):
        return "cube"
    e = np.zeros(3); e[long_axis_idx] = 1.0
    return "standing" if abs((R @ e)[2]) > 0.7 else "lying"


# SPAM 뚜껑(라벨면)이 top view 에 보이는지: 짧은 축(두께)이 카메라 +Z 방향
# 과 정렬되어 있어야 함. threshold 0.7 = 약 45° 이내.
SPAM_CAP_UP_THRESHOLD = 0.7


def grasp_policy_valid(cls_name, mode, R, extent, long_axis_idx):
    """해당 (cls, mode, pose) 가 grasp 정책상 유효한지.
    유효하지 않으면 (False, reason) — grasp 생성 자체를 스킵.
    """
    # marker 는 서있는 경우가 실제로 없음 (물리적으로 굴러감)
    if cls_name == "marker" and mode == "standing":
        return False, "marker_cannot_stand"
    # SPAM 도 서있는 경우 실제 없음
    if cls_name == "spam" and mode == "standing":
        return False, "spam_cannot_stand"
    # SPAM lying 중 "옆면으로 누운" (뚜껑이 측면) 케이스 제외.
    # PLY 축 분석: long=X(102mm), medium=Z(83mm), short=Y(60mm).
    # 뚜껑(라벨)면은 long×short 사각형이고, 그 면 normal = medium 축.
    # 따라서 medium 축이 카메라 +Z 와 정렬돼야 뚜껑이 top view 에 노출.
    if cls_name == "spam" and mode == "lying":
        others = [a for a in range(3) if a != long_axis_idx]
        a0, a1 = others
        mid_axis = a0 if float(extent[a0]) > float(extent[a1]) else a1
        e_mid = np.zeros(3); e_mid[mid_axis] = 1.0
        cap_up = abs(float((R @ e_mid)[2]))
        if cap_up < SPAM_CAP_UP_THRESHOLD:
            return False, "spam_cap_not_up"
    return True, None


# ---------- depth helpers (for z_table) ----------

def poly_to_mask(poly):
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def estimate_z_table_from_depth(depth_m, obj_masks):
    """Scene depth의 95-percentile을 테이블 z로 근사.
    obj 픽셀은 제외해야 bottle/can top이 z_table 로 오염되지 않는다."""
    valid = (depth_m > 0.1) & (depth_m < 2.0)
    if obj_masks:
        union = np.zeros_like(valid)
        for m in obj_masks:
            union |= m
        valid &= ~union
    z = depth_m[valid]
    if z.size < 100:
        return float(np.percentile(depth_m[depth_m > 0.1], 95))
    return float(np.percentile(z, 95))


# ---------- grasp generation ----------

def gen_grasps(R_icp, t_icp, extent, long_axis_idx, mode, cls_name):
    """Return list of (pos(3,), approach(3,), yaw, group_str)."""
    grasps: list[tuple] = []
    L = float(extent[long_axis_idx])
    e_long = np.zeros(3); e_long[long_axis_idx] = 1.0
    long_cam = R_icp @ e_long
    sign_toward_cam = -1.0 if long_cam[2] > 0 else 1.0

    if mode == "standing":
        top_off_obj = sign_toward_cam * (L * 0.5) * e_long
        p_cap_top = (t_icp + R_icp @ top_off_obj).copy()
        p_cap_top[2] += 0.003

        side_off_obj = sign_toward_cam * (L * 0.5 - SIDE_CAP_OFFSET_FROM_TIP_M) * e_long
        p_cap_side = (t_icp + R_icp @ side_off_obj).copy()

        # Top-down (θ=0°): 8 yaws around vertical (azimuth irrelevant)
        app_top = np.array([0.0, 0.0, 1.0])
        for yaw in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            grasps.append((p_cap_top, app_top, float(yaw), "top-down"))

        # Side tilts (θ=45° + 90°): 8 azimuths, TCP 선형보간으로 꼭대기↔옆면 사이.
        # yaw 는 "Tool Y ⊥ cap 축" 이 되도록 계산 → 핑거가 cap 을 straddle
        cap_axis = long_cam / (np.linalg.norm(long_cam) + 1e-9)
        for tilt_deg, grp_name in STANDING_TILT_LAYERS:
            if tilt_deg == 0.0:
                continue  # top-down 은 이미 위에서 처리
            theta = np.radians(tilt_deg)
            w = tilt_deg / 90.0  # 0 at top, 1 at side
            p_tcp = (1.0 - w) * p_cap_top + w * p_cap_side
            for az in SIDE_AZIMUTHS:
                app = np.array([np.cos(az) * np.sin(theta),
                                np.sin(az) * np.sin(theta),
                                np.cos(theta)])
                app = app / np.linalg.norm(app)

                # Tool Y target: cap 축과 approach 에 모두 수직 (핑거가 cap 횡단면 straddle)
                y_target = np.cross(cap_axis, app)
                yn = np.linalg.norm(y_target)
                if yn < 1e-6:
                    continue  # approach ∥ cap axis → 퇴화
                y_target = y_target / yn

                # build_R(app, yaw) 가 해당 y_target 을 생성하는 yaw 계산
                ref = (np.array([1.0, 0.0, 0.0]) if abs(app[0]) < 0.95
                       else np.array([0.0, 1.0, 0.0]))
                b0 = ref - (ref @ app) * app
                b0 = b0 / np.linalg.norm(b0)
                n0 = np.cross(app, b0)
                yaw = float(np.arctan2(float(y_target @ n0),
                                        float(y_target @ b0)))
                grasps.append((p_tcp.copy(), app, yaw, grp_name))

    elif mode == "lying":
        N = 3 if cls_name in ("marker", "spam") else 4
        ss = np.linspace(-0.35 * L, 0.35 * L, N)
        short = np.array([float(extent[i]) for i in range(3) if i != long_axis_idx])
        short_r = short.mean() / 2

        long_u = long_cam / (np.linalg.norm(long_cam) + 1e-9)
        a0 = np.array([0.0, 0.0, 1.0])

        # 클래스별 tilt 범위
        if cls_name in ("marker", "spam"):
            tilt_angles = np.radians(LYING_TILT_MARKER_SPAM_DEG)
        else:
            tilt_angles = np.radians(LYING_TILT_BOTTLE_CAN_DEG)

        for s in ss:
            off = np.zeros(3); off[long_axis_idx] = s
            p_center = t_icp + R_icp @ off
            for phi in tilt_angles:
                # Rodrigues: a0 을 long_u 축 주위로 φ 회전 → tilted approach
                c, sn = np.cos(phi), np.sin(phi)
                k_cross_a = np.cross(long_u, a0)
                k_dot_a = float(long_u @ a0)
                app = a0 * c + k_cross_a * sn + long_u * k_dot_a * (1 - c)
                app = app / np.linalg.norm(app)

                # TCP: 물체 중심에서 -approach 방향으로 short_r (원통 곡면 추종)
                p = p_center - app * short_r

                # 오프닝 방향 Y = long 축과 approach 에 모두 수직 (cylinder 종축 횡방향)
                y_dir = np.cross(app, long_u)
                yn = np.linalg.norm(y_dir)
                if yn < 1e-6:
                    # 퇴화: long 축이 approach 와 평행 (lying 에선 드물지만 fallback)
                    continue
                y_dir = y_dir / yn
                # build_R(app, yaw) 가 이 target_y 를 만드는 yaw 계산
                ref = (np.array([1.0, 0.0, 0.0]) if abs(app[0]) < 0.95
                       else np.array([0.0, 1.0, 0.0]))
                b0 = ref - (ref @ app) * app
                b0 = b0 / np.linalg.norm(b0)
                n0 = np.cross(app, b0)
                yaw_base = float(np.arctan2(float(y_dir @ n0), float(y_dir @ b0)))

                for sym in (0.0, np.pi):
                    grasps.append((p, app, yaw_base + sym, "lying"))

    elif mode == "cube":
        cam_z_proj = R_icp[2, :]
        vertical_col = int(np.argmax(np.abs(cam_z_proj)))
        edge_cols = [i for i in range(3) if i != vertical_col]
        cube_half = float(extent[vertical_col]) / 2
        p = np.array([t_icp[0], t_icp[1], t_icp[2] - cube_half + 0.002])
        app = np.array([0.0, 0.0, 1.0])
        for ec in edge_cols:
            edge_cam = R_icp[:, ec]
            yaw = float(np.arctan2(edge_cam[1], edge_cam[0]))
            grasps.append((p, app, yaw, "cube"))
    return grasps


# ---------- collision ----------

def gripper_width_cross_section(extent, long_axis_idx, mode, group):
    """Return the cross-section width (meter) that the two fingers must span."""
    if mode == "standing" and group in ("side-cap", "top-down", "side-45"):
        # cap 직경 = 짧은 extent 평균 (tilt 무관, 모두 cap cross-section 기준)
        short = [float(extent[i]) for i in range(3) if i != long_axis_idx]
        return float(np.mean(short))
    if mode == "lying":
        short = [float(extent[i]) for i in range(3) if i != long_axis_idx]
        return float(np.mean(short))
    if mode == "cube":
        # 나머지 extent (edge-aligned 잡기) 중 짧은 쪽
        # 간단히 extent 전체 평균으로 처리
        return float(np.mean(extent))
    return float(np.max(extent))


def collision_check(pos, approach, group, extent, long_axis_idx, mode, z_table):
    """Return (ok: bool, reason: str|None)."""
    # 1) gripper width
    cs = gripper_width_cross_section(extent, long_axis_idx, mode, group)
    if cs + GRIPPER_CLEARANCE > GRIPPER_MAX_OPEN:
        return False, "grip_width"

    # 2) table penetration (finger tip swept volume)
    # tip = pos (TCP), finger back = pos - approach * FINGER_LEN.
    # tip과 back 중 z 최대값이 z_table + safety 이하면 OK.
    tip_back_z = pos[2] - approach[2] * FINGER_LEN  # 팜쪽 z
    tip_front_z = pos[2]                             # tcp z (tip center)
    max_z = max(float(tip_front_z), float(tip_back_z))
    # safety = 2mm 이내 까지만 허용
    if max_z > z_table + 0.002:
        return False, "table"

    return True, None


# ---------- main ----------

def main():
    with h5py.File(POSES_H5, "r") as poses, h5py.File(DET_H5, "r") as dets, \
         h5py.File(OUT_H5, "w") as out:

        # root attrs passthrough
        for k, v in poses.attrs.items():
            out.attrs[k] = v
        out.attrs["policy_version"] = "6dof-v4"
        out.attrs["lying_tilt_bottle_can_deg"] = np.array(
            LYING_TILT_BOTTLE_CAN_DEG, dtype=np.float32)
        out.attrs["lying_tilt_marker_spam_deg"] = np.array(
            LYING_TILT_MARKER_SPAM_DEG, dtype=np.float32)
        out.attrs["standing_tilt_layers_deg"] = np.array(
            [t for t, _ in STANDING_TILT_LAYERS], dtype=np.float32)
        out.attrs["policy_excludes"] = (
            "marker_standing=never, spam_standing=never, "
            f"spam_lying requires |R@e_mid . z| > {SPAM_CAP_UP_THRESHOLD} "
            "(cap/label face up; e_mid = PLY extent 중 중간 축)")
        out.attrs["created_utc"] = np.string_(__import__("datetime").datetime.utcnow().isoformat())
        out.attrs["gripper_max_open_m"] = GRIPPER_MAX_OPEN
        out.attrs["finger_len_m"] = FINGER_LEN

        sample_ids = list(poses.keys())
        print(f"[batch] {len(sample_ids)} samples, source={POSES_H5.name}")

        report = {
            "total_samples": len(sample_ids),
            "total_objects": 0,
            "total_objects_stable": 0,
            "total_objects_excluded_by_policy": 0,
            "excluded_reasons": Counter(),
            "per_mode": Counter(),
            "per_group": Counter(),
            "total_grasps": 0,
            "total_collision_ok": 0,
            "reject_reasons": Counter(),
        }

        for si, sid in enumerate(sample_ids):
            g_pose = poses[sid]
            g_det = dets[sid] if sid in dets else None
            if g_det is None:
                continue

            # 이 sample의 depth → z_table 계산 (모든 stable mask 집합 제외)
            depth_path_rel = g_pose.attrs["depth_path"]
            depth_path = ROOT / depth_path_rel
            depth_m = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

            obj_masks = []
            for oname in g_pose.keys():
                g_obj = g_pose[oname]
                if not g_obj.attrs.get("stable_flag", False):
                    continue
                k = int(oname.split("_")[1])
                poly = np.asarray(g_det["mask_poly"][k]).reshape(-1, 2)
                obj_masks.append(poly_to_mask(poly))
            z_table = estimate_z_table_from_depth(depth_m, obj_masks)

            # per-sample group
            out_sid = out.create_group(sid)
            out_sid.attrs["depth_path"] = depth_path_rel
            out_sid.attrs["rgb_path"] = g_pose.attrs.get("rgb_path", "")
            out_sid.attrs["scene"] = g_pose.attrs.get("scene", "")
            out_sid.attrs["z_table_m"] = float(z_table)

            for oname in g_pose.keys():
                g_obj = g_pose[oname]
                report["total_objects"] += 1
                if not g_obj.attrs.get("stable_flag", False):
                    continue
                report["total_objects_stable"] += 1

                cls_name = str(g_obj.attrs["class_name"])
                R = np.asarray(g_obj["R_cam"])
                t = np.asarray(g_obj["t_cam"])
                extent = np.asarray(g_obj["model_extent"])
                long_idx = int(np.asarray(g_obj["model_long_axis_idx"]))
                uv_centroid = np.asarray(g_obj["uv_centroid"], dtype=np.float32)
                fitness = float(np.asarray(g_obj["fitness"]))

                mode = decide_mode(R, long_idx, cls_name)

                valid, excl_reason = grasp_policy_valid(
                    cls_name, mode, R, extent, long_idx)
                if not valid:
                    report["total_objects_excluded_by_policy"] += 1
                    report["excluded_reasons"][excl_reason] += 1
                    out_obj = out_sid.create_group(oname)
                    out_obj.attrs["class_id"] = int(g_obj.attrs["class_id"])
                    out_obj.attrs["class_name"] = cls_name
                    out_obj.attrs["mode"] = mode
                    out_obj.attrs["n_grasps"] = 0
                    out_obj.attrs["excluded_reason"] = excl_reason
                    out_obj.create_dataset("uv_centroid", data=uv_centroid)
                    continue

                grasps = gen_grasps(R, t, extent, long_idx, mode, cls_name)
                report["per_mode"][mode] += 1

                n = len(grasps)
                grasps_cam = np.zeros((n, 7), dtype=np.float32)
                approach_vec = np.zeros((n, 3), dtype=np.float32)
                yaw_arr = np.zeros(n, dtype=np.float32)
                group_arr = np.zeros(n, dtype=np.int32)
                ok_arr = np.zeros(n, dtype=bool)

                for i, (pos, app, yaw, grp) in enumerate(grasps):
                    app_u = np.asarray(app, dtype=np.float64)
                    app_u = app_u / np.linalg.norm(app_u)
                    R_tool = build_R(app_u, yaw)
                    q = R_to_quat_wxyz(R_tool)
                    grasps_cam[i] = np.concatenate([pos.astype(np.float32), q.astype(np.float32)])
                    approach_vec[i] = app_u.astype(np.float32)
                    yaw_arr[i] = float(yaw)
                    group_arr[i] = GROUP_ID[grp]
                    ok, reason = collision_check(pos, app_u, grp, extent, long_idx, mode, z_table)
                    ok_arr[i] = ok
                    if not ok:
                        report["reject_reasons"][reason] += 1
                    report["per_group"][grp] += 1

                report["total_grasps"] += n
                report["total_collision_ok"] += int(ok_arr.sum())

                out_obj = out_sid.create_group(oname)
                out_obj.attrs["class_id"] = int(g_obj.attrs["class_id"])
                out_obj.attrs["class_name"] = cls_name
                out_obj.attrs["mode"] = mode
                out_obj.attrs["n_grasps"] = n
                out_obj.attrs["ply_file"] = str(g_obj.attrs.get("ply_file", ""))
                out_obj.attrs["fitness_src"] = fitness
                out_obj.attrs["inlier_rmse_src"] = float(np.asarray(g_obj["inlier_rmse"]))
                out_obj.create_dataset("uv_centroid", data=uv_centroid)
                out_obj.create_dataset("grasps_cam", data=grasps_cam, compression="gzip", compression_opts=4)
                out_obj.create_dataset("approach_vec", data=approach_vec)
                out_obj.create_dataset("yaw_around_app", data=yaw_arr)
                out_obj.create_dataset("grasp_group", data=group_arr)
                out_obj.create_dataset("collision_ok", data=ok_arr)

            if (si + 1) % 50 == 0 or si + 1 == len(sample_ids):
                print(f"[batch] {si + 1}/{len(sample_ids)} samples, "
                      f"objs={report['total_objects_stable']}, grasps={report['total_grasps']}, "
                      f"ok={report['total_collision_ok']}")

    # JSON 저장
    report_json = {
        "total_samples": report["total_samples"],
        "total_objects": report["total_objects"],
        "total_objects_stable": report["total_objects_stable"],
        "total_objects_excluded_by_policy": report["total_objects_excluded_by_policy"],
        "excluded_reasons": dict(report["excluded_reasons"]),
        "per_mode": dict(report["per_mode"]),
        "per_group": dict(report["per_group"]),
        "total_grasps": report["total_grasps"],
        "total_collision_ok": report["total_collision_ok"],
        "collision_reject_rate": (
            1.0 - report["total_collision_ok"] / max(report["total_grasps"], 1)
        ),
        "reject_reasons": dict(report["reject_reasons"]),
        "policy_version": "6dof-v4",
        "source_poses_h5": str(POSES_H5.relative_to(ROOT)),
        "output_h5": str(OUT_H5.relative_to(ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report_json, indent=2))
    print("\n=== synthesis_report.json ===")
    print(json.dumps(report_json, indent=2))
    print(f"\nwrote {OUT_H5} ({OUT_H5.stat().st_size/1e6:.2f} MB)")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
