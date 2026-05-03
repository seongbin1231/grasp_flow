"""
eval_baseline_unified.py — read a baseline's prediction JSON in our unified format,
evaluate on grasp_v2.h5 val with the same Pos MAE / Ang Err / COV / APD metrics
used in scripts/make_paper_table1.py.

Unified prediction JSON format (one entry per (val row, prediction sample)):
[
  {
    "row":         <int, val row index 0..7239>,
    "uv":          [<u float>, <v float>],
    "scene_id":    <int>,
    "object_ref":  <int>,
    "grasps_cam":  [[x,y,z, qw,qx,qy,qz], ...],   // (N_pred, 7) camera frame
    "scores":      [<float>, ...],                 // (N_pred,) optional
  },
  ...
]

Usage:
  python scripts/baselines/eval_baseline_unified.py \\
    --pred runs/baseline_eval/graspldm.json \\
    --baseline_name "GraspLDM" \\
    --output runs/baseline_eval/graspldm_metrics.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
H5 = ROOT / "datasets/grasp_v2.h5"

POS_TH_M = 0.05
ANG_TH_DEG = 30.0


def quat_to_approach_vec(quat_wxyz: np.ndarray) -> np.ndarray:
    """quat (..., 4) [w,x,y,z] → approach axis (..., 3) (gripper +Z in camera frame)."""
    w, x, y, z = quat_wxyz[..., 0], quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3]
    ax = 2 * (x * z + w * y)
    ay = 2 * (y * z - w * x)
    az = 1 - 2 * (x * x + y * y)
    return np.stack([ax, ay, az], axis=-1)


def compute_metrics_7d(pred_7d: np.ndarray, gt_7d: np.ndarray, gt_groups: np.ndarray):
    """Compute the four headline metrics for one (uv, object) prediction.

    Args:
        pred_7d:   (N_pred, 7) [x,y,z, qw,qx,qy,qz] in camera frame
        gt_7d:     (G,      7) GT grasps for this same object
        gt_groups: (G,)        integer group IDs (mode/sub-mode)

    Returns: (pos_mae_cm, ang_err_deg, cov_pct, n_groups, apd_cm)
    """
    if pred_7d.shape[0] == 0 or gt_7d.shape[0] == 0:
        return float("nan"), float("nan"), 0.0, 0, 0.0

    pos_pred = pred_7d[:, :3]
    pos_gt = gt_7d[:, :3]
    app_pred = quat_to_approach_vec(pred_7d[:, 3:7])
    app_pred /= np.linalg.norm(app_pred, axis=-1, keepdims=True) + 1e-9
    app_gt = quat_to_approach_vec(gt_7d[:, 3:7])
    app_gt /= np.linalg.norm(app_gt, axis=-1, keepdims=True) + 1e-9

    # pred-side per-prediction nearest GT
    pos_errs, ang_errs = [], []
    for m in range(len(pred_7d)):
        d_pos = np.linalg.norm(pos_gt - pos_pred[m], axis=1)
        nearest = int(np.argmin(d_pos))
        pos_errs.append(float(d_pos[nearest]) * 100.0)
        cos = float(np.clip(app_pred[m] @ app_gt[nearest], -1, 1))
        ang_errs.append(float(np.degrees(np.arccos(abs(cos)))))

    # GT-side group coverage
    unique_groups = sorted({int(g) for g in gt_groups})
    covered: set[int] = set()
    for grp in unique_groups:
        gt_mask = gt_groups == grp
        for gi in np.where(gt_mask)[0]:
            d_pos = np.linalg.norm(pos_pred - pos_gt[gi], axis=1)
            cos = np.clip(app_pred @ app_gt[gi], -1, 1)
            ang = np.degrees(np.arccos(np.abs(cos)))
            if np.any((d_pos < POS_TH_M) & (ang < ANG_TH_DEG)):
                covered.add(grp)
                break

    coverage = 100.0 * len(covered) / max(len(unique_groups), 1)

    # APD
    if len(pos_pred) > 1:
        diffs = pos_pred[:, None, :] - pos_pred[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        iu = np.triu_indices(len(pos_pred), k=1)
        apd = float(dists[iu].mean()) * 100.0
    else:
        apd = 0.0

    return (float(np.mean(pos_errs)), float(np.mean(ang_errs)),
            coverage, len(unique_groups), apd)


def load_gt_for_row(val: h5py.Group, row: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Look up GT grasps that share the same (scene_id, object_ref) as `row`.

    Returns (gt_grasps_7d, gt_groups, mode_id).
    """
    scene_id = int(val["scene_id"][row])
    obj_ref = int(val["object_ref"][row])
    mode_id = int(val["object_mode"][row])

    sids = val["scene_id"][:]
    orefs = val["object_ref"][:]
    same = np.where((sids == scene_id) & (orefs == obj_ref))[0]
    gt = val["grasps_cam"][same]
    groups = val["grasp_group"][same]
    return gt, groups, mode_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="baseline prediction JSON")
    ap.add_argument("--baseline_name", default="baseline")
    ap.add_argument("--output", default="", help="output metrics JSON")
    args = ap.parse_args()

    pred_path = Path(args.pred)
    with pred_path.open() as f:
        preds = json.load(f)
    print(f"[eval] {args.baseline_name}: {len(preds)} prediction entries")

    mode_names = {0: "lying", 1: "standing", 2: "cube"}
    per_mode: dict[str, list[dict]] = {n: [] for n in mode_names.values()}

    with h5py.File(H5, "r") as f:
        val = f["val"]
        for entry in preds:
            row = int(entry["row"])
            pred_7d = np.asarray(entry["grasps_cam"], dtype=np.float32)
            gt_7d, gt_groups, mode = load_gt_for_row(val, row)
            if gt_7d.shape[0] == 0:
                continue
            pos_mae, ang_err, cov, n_groups, apd = compute_metrics_7d(
                pred_7d, gt_7d, gt_groups)
            per_mode[mode_names.get(mode, "unknown")].append({
                "row": row, "n_groups": n_groups,
                "pos_mae_cm": pos_mae, "ang_err_deg": ang_err,
                "cov_pct": cov, "apd_cm": apd,
            })

    summary: dict[str, dict] = {}
    for name, rows in per_mode.items():
        if not rows:
            summary[name] = {"n": 0}
            continue
        summary[name] = {
            "n": len(rows),
            "pos_mae_cm":  float(np.nanmean([r["pos_mae_cm"]  for r in rows])),
            "ang_err_deg": float(np.nanmean([r["ang_err_deg"] for r in rows])),
            "cov_pct":     float(np.nanmean([r["cov_pct"]     for r in rows])),
            "apd_cm":      float(np.nanmean([r["apd_cm"]      for r in rows])),
            "mean_groups": float(np.nanmean([r["n_groups"]    for r in rows])),
        }

    print(f"\n=== {args.baseline_name} ===")
    print(f"{'Mode':<10} {'n':>5} {'#GT-grp':>8} "
          f"{'PosMAE(cm)':>11} {'AngErr(°)':>10} {'COV(%)':>8} {'APD(cm)':>8}")
    for name, s in summary.items():
        if s.get("n", 0) == 0:
            continue
        print(f"{name:<10} {s['n']:>5} {s['mean_groups']:>8.2f} "
              f"{s['pos_mae_cm']:>11.2f} {s['ang_err_deg']:>10.2f} "
              f"{s['cov_pct']:>8.1f} {s['apd_cm']:>8.2f}")

    out_path = Path(args.output) if args.output else pred_path.with_suffix(".metrics.json")
    out_path.write_text(json.dumps({
        "baseline": args.baseline_name,
        "pred_path": str(pred_path),
        "summary": summary,
        "n_total": sum(s.get("n", 0) for s in summary.values()),
    }, indent=2))
    print(f"\n[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
