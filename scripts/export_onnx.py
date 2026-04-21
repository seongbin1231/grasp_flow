"""
Export Flow Matching grasp model to ONNX for MATLAB integration.

Two files are exported (matches the CLAUDE.md contract):

  encoder.onnx  (depth, uv)  →  cond
      Run ONCE per scene. Heavy conv stack (global + local crop).

  velocity.onnx (g_t, cond, t, uv_norm)  →  v
      Run T_steps × 2 times per scene (×2 for CFG: cond_on / cond_off).
      Lightweight MLP. Sinusoidal time embed is baked INSIDE this graph.

MATLAB runtime is then:
  1. cond = encode(depth, uv);                  (once)
  2. cond_rep = repmat(cond, [N, 1]);            (N = # samples, e.g. 32)
  3. g_t = randn(N, 8) * temp;                   (noise init)
  4. for k = 0 .. T-1
        t = k/T;
        v_cond   = velocity(g_t, cond_rep,           t, uv_norm_rep);
        v_uncond = velocity(g_t, zeros(N,256),       t, uv_norm_rep);
        v = v_uncond + w * (v_cond - v_uncond);     (CFG, w ~ 2.0)
        g_t = g_t + v / T;
  5. g_t[:, :3] = g_t[:, :3] .* pos_std + pos_mean;   (denormalize pos)
  6. grasp = [pos (3), approach_unit (3), sin_yaw, cos_yaw]   → build SE(3) for each
  7. Apply collision + contact filter → pick best.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
sys.path.insert(0, str(ROOT))

from src.flow_model import (
    FlowGraspNet,
    IMG_H, IMG_W,
    sinusoidal_time_embed,
)


class EncoderWrapper(nn.Module):
    """depth, uv → cond."""
    def __init__(self, flow: FlowGraspNet):
        super().__init__()
        self.flow = flow

    def forward(self, depth: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        return self.flow.encode(depth, uv)


class VelocityWrapper(nn.Module):
    """g_t (N,8), cond (N,256), t (N,), uv_norm (N,2) → v (N,8).

    Sinusoidal time embedding is inlined so MATLAB only has to supply scalar t.
    """
    def __init__(self, flow: FlowGraspNet, t_dim: int = 64):
        super().__init__()
        self.velocity = flow.velocity
        half = t_dim // 2
        # Pre-compute the (constant) log-linear frequency ladder once so ONNX sees
        # a buffer instead of arange/exp ops. This trims a couple of nodes and
        # avoids any opset-version edge cases around torch.exp(arange).
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half).float() / half
        )
        self.register_buffer("freqs", freqs)  # (half,)

    def forward(self, g_t: torch.Tensor, cond: torch.Tensor,
                t: torch.Tensor, uv_norm: torch.Tensor) -> torch.Tensor:
        ang = t[:, None] * self.freqs[None]     # (N, half)
        t_emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (N, t_dim)
        return self.velocity(g_t, cond, t_emb, uv_norm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/yolograsp_v2/v6_150ep/"
                    "adaln_zero_lr0.001_nb8_h768/checkpoints/best.pt")
    ap.add_argument("--out_dir", default="deploy/onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--n", type=int, default=32,
                    help="default batch for velocity dummy input")
    args = ap.parse_args()

    ckpt_path = (ROOT / args.ckpt).resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]["args"]
    print(f"[load] {ckpt_path}  ep={ck['epoch']}  val_loss={ck.get('val_loss'):.4f}")
    print(f"[cfg]  block={cfg['block']} hidden={cfg['hidden']} n_blocks={cfg['n_blocks']}")

    model = FlowGraspNet(
        block_type=cfg["block"], n_blocks=cfg["n_blocks"],
        hidden=cfg["hidden"], cond_dropout=cfg["cond_dropout"],
    )
    model.load_state_dict(ck["ema"], strict=False)
    model.eval()

    enc = EncoderWrapper(model).eval()
    vel = VelocityWrapper(model, t_dim=64).eval()

    # ---------- encoder ----------
    depth = torch.zeros(1, 1, IMG_H, IMG_W, dtype=torch.float32)
    uv = torch.tensor([[IMG_W / 2, IMG_H / 2]], dtype=torch.float32)
    enc_path = out_dir / "encoder.onnx"
    torch.onnx.export(
        enc, (depth, uv), str(enc_path),
        input_names=["depth", "uv"],
        output_names=["cond"],
        opset_version=args.opset,
        dynamic_axes={"depth": {0: "B"}, "uv": {0: "B"}, "cond": {0: "B"}},
        do_constant_folding=True,
    )
    print(f"[export] encoder → {enc_path.relative_to(ROOT)} "
          f"({enc_path.stat().st_size/1e6:.2f} MB)")

    # ---------- velocity ----------
    N = args.n
    g_t = torch.zeros(N, 8, dtype=torch.float32)
    cond = torch.zeros(N, 256, dtype=torch.float32)
    t = torch.zeros(N, dtype=torch.float32)
    uv_norm = torch.zeros(N, 2, dtype=torch.float32)
    vel_path = out_dir / "velocity.onnx"
    torch.onnx.export(
        vel, (g_t, cond, t, uv_norm), str(vel_path),
        input_names=["g_t", "cond", "t", "uv_norm"],
        output_names=["v"],
        opset_version=args.opset,
        dynamic_axes={
            "g_t": {0: "N"}, "cond": {0: "N"},
            "t": {0: "N"}, "uv_norm": {0: "N"}, "v": {0: "N"},
        },
        do_constant_folding=True,
    )
    print(f"[export] velocity → {vel_path.relative_to(ROOT)} "
          f"({vel_path.stat().st_size/1e6:.2f} MB)")

    # ---------- metadata (denorm stats, constants) ----------
    ns = ck.get("norm_stats", {})
    meta = {
        "schema_version": "v6_150ep",
        "source_ckpt": str(ckpt_path.relative_to(ROOT)),
        "epoch": int(ck["epoch"]),
        "val_loss": float(ck.get("val_loss", float("nan"))),
        "image": {"H": IMG_H, "W": IMG_W,
                  "camera_K": {"fx": 1109.0, "fy": 1109.0, "cx": 640.0, "cy": 360.0}},
        "depth": {"unit_in": "meter (float32)",
                  "clip_min": 0.3, "clip_max": 1.5,
                  "scale_div": 1.5},
        "normalize_pos": bool(ns.get("normalize_pos", False)),
        "pos_mean": list(map(float, ns.get("pos_mean", [0.0, 0.0, 0.0]))),
        "pos_std":  list(map(float, ns.get("pos_std",  [1.0, 1.0, 1.0]))),
        "noise_temp_train": 1.0,        # train uses N(0,1); inference prefers 0.8
        "recommended_inference": {
            "N_samples": 32, "T_euler_steps": 32,
            "noise_temp": 0.8, "cfg_guidance": 2.0,
        },
        "grasp_param_8d": [
            "x", "y", "z",
            "approach_x", "approach_y", "approach_z",
            "sin_yaw", "cos_yaw",
        ],
        "gripper": {
            "GRIPPER_HALF": 0.0425, "FINGER_LEN": 0.040,
            "PALM_BACK": 0.025, "APPROACH_STEM": 0.050,
            "TCP_to_flange_offset_z_tool": 0.14,
        },
        "filter": {
            "BODY_MARGIN_m": 0.005,
            "TIP_SWEEP_RADIUS_m": 0.015,
            "STEM_SAMPLES": 4, "SWEEP_SAMPLES": 6,
        },
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[export] meta  → {meta_path.relative_to(ROOT)}")

    # ---------- numeric round-trip ----------
    try:
        import onnxruntime as ort
        print("[verify] running onnxruntime round-trip...")
        sess_e = ort.InferenceSession(str(enc_path), providers=["CPUExecutionProvider"])
        sess_v = ort.InferenceSession(str(vel_path), providers=["CPUExecutionProvider"])
        depth_np = np.random.rand(1, 1, IMG_H, IMG_W).astype(np.float32) * 0.5 + 0.4
        uv_np = np.array([[640.0, 360.0]], dtype=np.float32)
        cond_np = sess_e.run(["cond"], {"depth": depth_np, "uv": uv_np})[0]
        assert cond_np.shape == (1, 256)
        g_t_np = np.random.randn(N, 8).astype(np.float32)
        cond_rep = np.broadcast_to(cond_np, (N, 256)).copy()
        t_np = np.full((N,), 0.5, dtype=np.float32)
        uv_norm_np = np.tile(np.array([[0.5, 0.5]], dtype=np.float32), (N, 1))
        v_np = sess_v.run(["v"], {
            "g_t": g_t_np, "cond": cond_rep,
            "t": t_np, "uv_norm": uv_norm_np,
        })[0]
        assert v_np.shape == (N, 8)
        # Compare against torch
        with torch.no_grad():
            cond_t = enc(torch.from_numpy(depth_np), torch.from_numpy(uv_np))
            v_t = vel(torch.from_numpy(g_t_np), cond_t.expand(N, -1),
                      torch.from_numpy(t_np), torch.from_numpy(uv_norm_np))
        err_cond = float((torch.from_numpy(cond_np) - cond_t).abs().max())
        err_v = float((torch.from_numpy(v_np) - v_t).abs().max())
        print(f"[verify] cond max|Δ|={err_cond:.2e}   v max|Δ|={err_v:.2e}")
        assert err_cond < 1e-4 and err_v < 1e-4, "ONNX numerical mismatch!"
        print("[verify] OK")
    except ImportError:
        print("[verify] onnxruntime not installed — skipping round-trip")


if __name__ == "__main__":
    main()
