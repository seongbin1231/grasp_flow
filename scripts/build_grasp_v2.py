"""
Build datasets/grasp_v2.h5 — unrolled rows for Flow Matching training.

입력:
  img_dataset/icp_cache/poses.h5      (stable object metadata, class, ply, R, t)
  img_dataset/grasp_cache/grasps.h5   (per-object n grasps, 7D quat, approach, yaw, group)
  img_dataset/captured_images_depth/  (depth PNG uint16 mm)

출력:
  datasets/grasp_v2.h5
    /train/, /val/   (unrolled rows; cube 4종 통합; scene-level split)
    /metadata/
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
POSES_H5 = ROOT / "img_dataset/icp_cache/poses.h5"
GRASPS_H5 = ROOT / "img_dataset/grasp_cache/grasps.h5"
OUT_DIR = ROOT / "datasets"
OUT_H5 = OUT_DIR / "grasp_v2.h5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# YOLO 8종 → 학습용 5종 통합
UNIFY_MAP = {
    0: (0, "bottle"),
    1: (1, "can"),
    2: (2, "cube"),   # cube_blue
    3: (2, "cube"),   # cube_green
    4: (2, "cube"),   # cube_p
    5: (2, "cube"),   # cube_red
    6: (3, "marker"),
    7: (4, "spam"),
}
UNIFIED_NAMES = ["bottle", "can", "cube", "marker", "spam"]
MODE_ID = {"lying": 0, "standing": 1, "cube": 2}

VAL_SCENES = {"random6"}   # 6 씬 중 1개 → 약 16.7% val
TRAIN_SCENES = {"random1", "random2", "random3", "random4", "random5"}


def load_depth_m(depth_path: Path) -> np.ndarray:
    d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(str(depth_path))
    return d.astype(np.float32) / 1000.0


def main():
    print(f"[build] reading {POSES_H5.name} + {GRASPS_H5.name}")

    # 1) 유니크 depth 수집 (sid → depth_ref)
    with h5py.File(POSES_H5, "r") as poses:
        sample_ids = sorted(poses.keys())
        sid_to_depth_path = {
            sid: poses[sid].attrs["depth_path"] for sid in sample_ids
        }
        sid_to_scene = {
            sid: poses[sid].attrs.get("scene", "") for sid in sample_ids
        }

    train_sids = [s for s in sample_ids if sid_to_scene[s] in TRAIN_SCENES]
    val_sids = [s for s in sample_ids if sid_to_scene[s] in VAL_SCENES]
    assert set(train_sids).isdisjoint(val_sids), "scene leak"
    print(f"[build] train={len(train_sids)} samples, val={len(val_sids)} samples")

    splits = {"train": train_sids, "val": val_sids}

    # 2) depth ref 맵 per split (메모리 절약: split 간 depth 공유 않음)
    per_split_rows = {"train": [], "val": []}
    per_split_depths = {"train": [], "val": []}

    with h5py.File(GRASPS_H5, "r") as grasps, h5py.File(POSES_H5, "r") as poses:
        for split, sids in splits.items():
            depth_idx_map: dict[str, int] = {}
            for sid in sids:
                g_sid = grasps[sid]
                p_sid = poses[sid]
                depth_path = ROOT / p_sid.attrs["depth_path"]
                if sid not in depth_idx_map:
                    depth_idx_map[sid] = len(per_split_depths[split])
                    per_split_depths[split].append(load_depth_m(depth_path))
                dref = depth_idx_map[sid]

                # object_ref 는 split 내 global counter (같은 obj grasp 묶기용)
                for oname in sorted(g_sid.keys(), key=lambda x: int(x.split("_")[1])):
                    g_obj = g_sid[oname]
                    raw_cls = int(g_obj.attrs["class_id"])
                    uni_cls, uni_name = UNIFY_MAP[raw_cls]
                    mode = str(g_obj.attrs["mode"])
                    mode_id = MODE_ID[mode]
                    fitness = float(g_obj.attrs["fitness_src"])
                    rmse = float(g_obj.attrs["inlier_rmse_src"])

                    uv = np.asarray(g_obj["uv_centroid"], dtype=np.float32)
                    grasps_cam = np.asarray(g_obj["grasps_cam"], dtype=np.float32)
                    approach_vec = np.asarray(g_obj["approach_vec"], dtype=np.float32)
                    yaw_arr = np.asarray(g_obj["yaw_around_app"], dtype=np.float32)
                    group_arr = np.asarray(g_obj["grasp_group"], dtype=np.int32)
                    ok_arr = np.asarray(g_obj["collision_ok"], dtype=bool)

                    # object_ref = split 내 object index
                    obj_ref = len(set(r[-1] for r in per_split_rows[split]))  # 느릴 수 있음
                    # 간단히: 연속 증가 id 사용
                    n = grasps_cam.shape[0]
                    obj_ref_val = len(per_split_rows[split])  # temp, rebuild later

                    for i in range(n):
                        per_split_rows[split].append((
                            dref,
                            uv.copy(),
                            grasps_cam[i].copy(),
                            approach_vec[i].copy(),
                            float(yaw_arr[i]),
                            uni_cls,
                            mode_id,
                            int(group_arr[i]),
                            int(parse_scene_id(sid_to_scene[sid])),
                            sid,                                  # sample_ref
                            oname,                                # for later object_ref rebuild
                            fitness,
                            rmse,
                            bool(ok_arr[i]),
                        ))

    # 3) object_ref 재계산 (split 내 unique (sid, oname) → int id)
    for split in ("train", "val"):
        key_to_id: dict[tuple, int] = {}
        new_rows = []
        for r in per_split_rows[split]:
            key = (r[9], r[10])  # (sample_ref, object_ref name)
            if key not in key_to_id:
                key_to_id[key] = len(key_to_id)
            oid = key_to_id[key]
            new_rows.append((*r[:10], oid, r[11], r[12], r[13]))
        per_split_rows[split] = new_rows
        print(f"[build] split={split}: rows={len(new_rows)}, objects={len(key_to_id)}, "
              f"depths={len(per_split_depths[split])}")

    # 4) write HDF5
    print(f"[build] writing {OUT_H5}")
    OUT_H5.unlink(missing_ok=True)
    with h5py.File(OUT_H5, "w") as out:
        # metadata
        md = out.create_group("metadata")
        md.attrs["camera_K"] = np.array([[1109.0, 0, 640.0],
                                          [0, 1109.0, 360.0],
                                          [0, 0, 1.0]], dtype=np.float32)
        md.attrs["image_size"] = np.array([720, 1280], dtype=np.int32)
        md.attrs["coord_frame"] = "camera"
        md.attrs["grasp_dof"] = 6
        md.attrs["schema_version"] = "v2"
        str_dt = h5py.string_dtype()
        md.attrs.create("class_names", UNIFIED_NAMES, dtype=str_dt)
        md.attrs.create("mode_names", ["lying", "standing", "cube"], dtype=str_dt)
        md.attrs.create("group_names", ["top-down", "side-cap", "lying", "cube"],
                         dtype=str_dt)
        md.attrs["depth_clip_min"] = 0.3
        md.attrs["depth_clip_max"] = 1.5
        md.attrs["depth_scale_div"] = 1.5
        md.attrs.create("val_scenes", sorted(VAL_SCENES), dtype=str_dt)
        md.attrs.create("train_scenes", sorted(TRAIN_SCENES), dtype=str_dt)

        stats = {"schema_version": "v2", "splits": {}}

        for split, rows in per_split_rows.items():
            g = out.create_group(split)
            depths = np.stack(per_split_depths[split]).astype(np.float32)
            g.create_dataset("depths", data=depths,
                             compression="gzip", compression_opts=4,
                             chunks=(1, 720, 1280))
            N = len(rows)
            depth_ref = np.array([r[0] for r in rows], dtype=np.int32)
            uvs = np.stack([r[1] for r in rows]).astype(np.float32)
            grasps_cam = np.stack([r[2] for r in rows]).astype(np.float32)
            approach_vec = np.stack([r[3] for r in rows]).astype(np.float32)
            yaw_arr = np.array([r[4] for r in rows], dtype=np.float32)
            obj_cls = np.array([r[5] for r in rows], dtype=np.int32)
            obj_mode = np.array([r[6] for r in rows], dtype=np.int32)
            grasp_group = np.array([r[7] for r in rows], dtype=np.int32)
            scene_id = np.array([r[8] for r in rows], dtype=np.int32)
            sample_ref = np.array([r[9] for r in rows], dtype=h5py.string_dtype())
            object_ref = np.array([r[10] for r in rows], dtype=np.int32)
            fitness = np.array([r[11] for r in rows], dtype=np.float32)
            rmse = np.array([r[12] for r in rows], dtype=np.float32)
            collision_ok = np.array([r[13] for r in rows], dtype=bool)

            g.create_dataset("depth_ref", data=depth_ref)
            g.create_dataset("uvs", data=uvs)
            g.create_dataset("grasps_cam", data=grasps_cam)
            g.create_dataset("approach_vec", data=approach_vec)
            g.create_dataset("yaw_around_app", data=yaw_arr)
            g.create_dataset("object_class", data=obj_cls)
            g.create_dataset("object_mode", data=obj_mode)
            g.create_dataset("grasp_group", data=grasp_group)
            g.create_dataset("scene_id", data=scene_id)
            g.create_dataset("sample_ref", data=sample_ref)
            g.create_dataset("object_ref", data=object_ref)
            g.create_dataset("fitness", data=fitness)
            g.create_dataset("inlier_rmse", data=rmse)
            g.create_dataset("collision_ok", data=collision_ok)

            # split stats
            cls_bins = np.bincount(obj_cls, minlength=len(UNIFIED_NAMES)).tolist()
            mode_bins = np.bincount(obj_mode, minlength=3).tolist()
            group_bins = np.bincount(grasp_group, minlength=4).tolist()
            stats["splits"][split] = {
                "rows": int(N),
                "unique_depths": int(depths.shape[0]),
                "unique_objects": int(object_ref.max() + 1),
                "class_dist": dict(zip(UNIFIED_NAMES, cls_bins)),
                "mode_dist": dict(zip(["lying", "standing", "cube"], mode_bins)),
                "group_dist": dict(zip(["top-down", "side-cap", "lying", "cube"],
                                       group_bins)),
                "fitness_mean": float(fitness.mean()),
                "fitness_p10": float(np.percentile(fitness, 10)),
                "fitness_p90": float(np.percentile(fitness, 90)),
            }
            print(f"[build] wrote /{split}: rows={N}, depths={depths.shape[0]}, "
                  f"objects={int(object_ref.max() + 1)}")

        md.attrs["stats"] = json.dumps(stats)

    size_mb = OUT_H5.stat().st_size / 1e6
    print(f"\n[build] done. {OUT_H5}  ({size_mb:.1f} MB)")
    print(json.dumps(stats, indent=2))


def parse_scene_id(scene_name: str) -> int:
    return int(scene_name.replace("random", ""))


if __name__ == "__main__":
    main()
