"""
uv_adapter.py — depth + (u,v) → segmented point cloud (camera frame).

Used by all PC-input grasp baselines (GraspLDM, SE(3)-DiffusionFields,
6-DoF GraspNet, Contact-GraspNet, GraspGen) so that we can compare them
against our paradigm on the same val set with consistent segmentation.

Camera intrinsics (default = our 1280×720 setup):
    K = [[1109, 0, 640],
         [   0, 1109, 360],
         [   0, 0,   1]]

Depth format: uint16 PNG, mm units (the user's pipeline convention).
"""
from __future__ import annotations
import numpy as np

DEFAULT_K = np.array([[1109.0, 0.0, 640.0],
                      [0.0, 1109.0, 360.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32)


def uv_to_segmented_pc(
    depth_mm: np.ndarray,
    uv: tuple[float, float] | np.ndarray,
    K: np.ndarray = DEFAULT_K,
    yolo_mask: np.ndarray | None = None,
    radius_px: int = 80,
    z_min_m: float = 0.20,
    z_max_m: float = 1.50,
    max_points: int | None = 20000,
) -> np.ndarray:
    """Back-project a region of `depth_mm` around (u,v) into a 3D point cloud.

    Args:
        depth_mm: (H, W) uint16 or float, depth in millimetres.
        uv: (u, v) pixel coordinate (column, row).
        K: (3, 3) camera intrinsics (DEFAULT_K = our 1109/1280×720 setup).
        yolo_mask: (H, W) bool. If given, used directly as the segmentation;
                   otherwise a circular patch of `radius_px` around (u,v).
        radius_px: radius of the fallback circular patch (default 80).
        z_min_m, z_max_m: keep points whose depth is in [z_min, z_max] metres
                          (default 0.20–1.50 m, table-top range).
        max_points: subsample to this many points if exceeded (default 20000).

    Returns:
        pc: (N, 3) float32 point cloud in camera frame, units = metres.
    """
    H, W = depth_mm.shape
    u, v = float(uv[0]), float(uv[1])

    if yolo_mask is None:
        ys, xs = np.mgrid[0:H, 0:W]
        mask = ((xs - u) ** 2 + (ys - v) ** 2) < (radius_px ** 2)
    else:
        if yolo_mask.shape != (H, W):
            raise ValueError(f"yolo_mask shape {yolo_mask.shape} != depth shape {(H, W)}")
        mask = yolo_mask.astype(bool)
        ys, xs = np.mgrid[0:H, 0:W]

    z = depth_mm[mask].astype(np.float32) / 1000.0  # mm → m

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs[mask] - cx) * z / fx
    y = (ys[mask] - cy) * z / fy
    pc = np.stack([x, y, z], axis=-1)  # (M, 3)

    valid = (pc[:, 2] >= z_min_m) & (pc[:, 2] <= z_max_m)
    pc = pc[valid]

    if max_points is not None and pc.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pc.shape[0], max_points, replace=False)
        pc = pc[idx]

    return pc.astype(np.float32)


def smoke_test() -> None:
    """Numerical sanity check: synthetic depth + known (u,v) → expected geometry."""
    print("[uv_adapter] smoke test")
    H, W = 720, 1280
    depth_mm = np.zeros((H, W), dtype=np.uint16)
    # 0.5 m flat surface in a 200×200 patch at the image centre
    depth_mm[260:460, 540:740] = 500
    pc = uv_to_segmented_pc(depth_mm, (640, 360), radius_px=120)
    print(f"  PC shape: {pc.shape}, points: {pc.shape[0]}")
    print(f"  z range: [{pc[:, 2].min():.3f}, {pc[:, 2].max():.3f}] m  (expected ~0.500)")
    print(f"  x range: [{pc[:, 0].min():.3f}, {pc[:, 0].max():.3f}] m")
    print(f"  y range: [{pc[:, 1].min():.3f}, {pc[:, 1].max():.3f}] m")
    centre = pc.mean(axis=0)
    print(f"  centroid: ({centre[0]:+.4f}, {centre[1]:+.4f}, {centre[2]:.4f}) m  "
          f"(expected ~(0, 0, 0.500))")

    # Assertions
    assert pc.shape[0] > 0, "point cloud is empty"
    assert abs(pc[:, 2].mean() - 0.500) < 0.001, "z mean wrong"
    assert abs(centre[0]) < 0.01 and abs(centre[1]) < 0.01, "centroid not at optical axis"
    print("[uv_adapter] PASS ✓")


def real_data_test() -> None:
    """Run adapter on real val frames from grasp_v2.h5 + verify density.

    Schema reminder:
      val/depths: (98, 720, 1280) float32 in metres
      val/uvs:    (7240, 2)
      val/depth_ref: (7240,) int32 -> index into val/depths
    """
    import h5py
    from pathlib import Path

    H5 = Path("/home/robotics/Competition/YOLO_Grasp/datasets/grasp_v2.h5")
    if not H5.exists():
        print(f"[uv_adapter] real test skipped (missing {H5})")
        return

    print("[uv_adapter] real-data test (5 random val rows)")
    rng = np.random.default_rng(0)
    with h5py.File(H5, "r") as f:
        val = f["val"]
        n_rows = val["uvs"].shape[0]
        n_depths = val["depths"].shape[0]
        for k in range(5):
            row = int(rng.integers(0, n_rows))
            d_idx = int(val["depth_ref"][row])
            depth_m = val["depths"][d_idx]
            depth_mm = (depth_m * 1000.0).astype(np.float32)   # m → mm
            uv = val["uvs"][row]
            pc = uv_to_segmented_pc(depth_mm, uv)
            print(f"  row={row:5d} scene={d_idx:3d}/{n_depths}  "
                  f"uv=({uv[0]:6.1f},{uv[1]:6.1f})  "
                  f"PC=({pc.shape[0]:5d} pts, "
                  f"z=[{pc[:, 2].min():.3f},{pc[:, 2].max():.3f}] m)")
            assert pc.shape[0] >= 100, f"row {row}: too few points ({pc.shape[0]})"
            assert pc[:, 2].min() > 0.10, f"row {row}: z below table"
            assert pc[:, 2].max() < 1.50, f"row {row}: z above ceiling"
    print("[uv_adapter] real-data PASS ✓")


if __name__ == "__main__":
    smoke_test()
    real_data_test()
