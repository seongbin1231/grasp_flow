"""
Flow Matching training for (depth + uv) → 6-DoF multi-grasp.

Loss:    ||v_θ(g_t, t, c) − (g_1 − g_0)||²   where g_t = (1-t)g_0 + t g_1, g_0 ~ N(0, I)
Sampler: mode-balanced WeightedRandomSampler
EMA:     decay=0.999
Aux:     mode classification head (λ=0.1)

Usage:
  python scripts/train_flow.py --epochs 30 --batch_size 16 --lr 3e-4 \
                               --run runs/yolograsp_v2/v1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

ROOT = Path("/home/robotics/Competition/YOLO_Grasp")
sys.path.insert(0, str(ROOT))

from src.flow_dataset import GraspDataset, make_weighted_sampler
from src.flow_model import FlowGraspNet, FlowGraspNetPC, EMA


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=str(ROOT / "datasets/grasp_v2.h5"))
    ap.add_argument("--run", default=str(ROOT / "runs/yolograsp_v2/v1"))
    ap.add_argument("--epochs", type=int, default=30,
                    help="default bumped from 15 → 30 for full flow convergence")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--aux_mode_weight", type=float, default=0.1)
    ap.add_argument("--cond_dropout", type=float, default=0.2)
    ap.add_argument("--block", choices=["film", "adaln_zero"], default="adaln_zero")
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--symmetry_loss", action="store_true",
                    help="min-target loss for lying 180° pairs (Tier 2)")
    ap.add_argument("--rot_loss_weight", type=float, default=1.0,
                    help="per-dim loss weight for approach(3) + sincos(2). pos weight fixed at 1.0")
    ap.add_argument("--marker_boost", type=float, default=1.0,
                    help="sampler weight multiplier for marker class (cls=3)")
    ap.add_argument("--spam_boost", type=float, default=1.0,
                    help="sampler weight multiplier for spam class (cls=4)")
    ap.add_argument("--wandb", action="store_true", help="enable wandb logging")
    ap.add_argument("--wandb_project", default="yolograsp-v2")
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--smoke", type=int, default=0,
                    help="limit batches per epoch (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    # v8 architecture options (default OFF → v7 동작 100% 보존)
    ap.add_argument("--xattn", action="store_true",
                    help="[v8 A1] Enable Cross-Attention block (PixArt-α / Hunyuan-DiT 식)")
    ap.add_argument("--multiscale_local", type=str, default="",
                    help="[v8 A2] comma-separated crop scales e.g. '96,192,384'. "
                         "empty = v7 single 192px")
    ap.add_argument("--rot_repr", choices=["approach_yaw", "zhou6d"],
                    default="approach_yaw",
                    help="grasp rotation representation: 8D approach+sincos vs 9D Zhou 6D")
    ap.add_argument("--scale_dropout", type=float, default=0.0,
                    help="[v8 A2 안전장치] prob of using only 1 random scale during training")
    ap.add_argument("--use_pc", action="store_true",
                    help="[v9 B1 hybrid] Add Mini-PointNet token from depth-back-projected PC")
    ap.add_argument("--pc_n_points", type=int, default=512,
                    help="[v9 B1] sub-sample size for PC")
    ap.add_argument("--pc_dim", type=int, default=128,
                    help="[v9 B1] PC feature dim")
    ap.add_argument("--pc_crop", type=int, default=192,
                    help="[v9 B1] crop size for PC back-projection source")
    ap.add_argument("--use_pc_only", action="store_true",
                    help="[v9 PC-only] Replace depth CNN encoder with PC encoder. "
                         "Uses FlowGraspNetPC (anchor-centered, multi-scale ball-query, "
                         "hybrid sparse+dense scene PC). Mutually exclusive with --use_pc.")
    ap.add_argument("--pretrained", type=str, default="",
                    help="warm-start path to .pt (loads model weights only, "
                         "no optimizer/EMA state). Use for stage 2/3 progressive "
                         "training or ablation variants from a v7/v8 checkpoint.")
    return ap.parse_args()


def flow_matching_loss(v_pred, g_1, g_0):
    """v_pred: (B, 8) velocity. GT velocity = g_1 - g_0."""
    v_target = g_1 - g_0
    return F.mse_loss(v_pred, v_target)


def _dim_weights(rot_weight: float, device, g_dim: int = 8) -> torch.Tensor:
    """Per-dim MSE weights.
      8D (approach_yaw): pos(3)×1, approach(3)×rw, sincos(2)×rw
      9D (zhou6d):       pos(3)×1, R[:,0](3)×rw,  R[:,1](3)×rw
    """
    if g_dim == 8:
        w = [1.0] * 3 + [rot_weight] * 5
    elif g_dim == 9:
        w = [1.0] * 3 + [rot_weight] * 6
    else:
        raise ValueError(f"unsupported g_dim={g_dim}")
    return torch.tensor(w, device=device, dtype=torch.float32)


def weighted_flow_loss(v_pred, v_target, dim_w):
    """sum over dims of weighted MSE, mean over batch. Returns (B,)."""
    return ((v_pred - v_target) ** 2 * dim_w).mean(dim=-1)


def symmetric_min_loss(v_pred, g_1, g_1_alt, g_0, dim_w):
    """For lying 180° pairs, pick the closer target per-sample.
    For non-lying, g_1 == g_1_alt so min == either."""
    v_tgt = g_1 - g_0
    v_tgt_alt = g_1_alt - g_0
    err = weighted_flow_loss(v_pred, v_tgt, dim_w)
    err_alt = weighted_flow_loss(v_pred, v_tgt_alt, dim_w)
    return torch.minimum(err, err_alt).mean()


def build_lr_scheduler(optim, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None, dim_w=None, symmetry=False):
    model.eval()
    losses = []
    mode_correct = 0
    mode_total = 0
    per_mode_loss: dict[int, list[float]] = {0: [], 1: [], 2: []}
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        depth = batch["depth"].to(device, non_blocking=True)
        uv = batch["uv"].to(device, non_blocking=True)
        g_1 = batch["g1"].to(device, non_blocking=True)
        g_1_alt = batch["g1_alt"].to(device, non_blocking=True)
        mode = batch["mode"].to(device)
        B = g_1.size(0)
        g_0 = torch.randn_like(g_1)
        t = torch.rand(B, device=device)
        g_t = (1 - t)[:, None] * g_0 + t[:, None] * g_1
        v_pred, mode_logits = model.forward_with_aux(depth, uv, g_t, t)
        if dim_w is None:
            dim_w = torch.ones(g_1.size(-1), device=device)
        if symmetry:
            per_sample_loss = torch.minimum(
                weighted_flow_loss(v_pred, g_1 - g_0, dim_w),
                weighted_flow_loss(v_pred, g_1_alt - g_0, dim_w),
            )
        else:
            per_sample_loss = weighted_flow_loss(v_pred, g_1 - g_0, dim_w)
        losses.append(per_sample_loss.mean().item())
        pred = mode_logits.argmax(-1)
        mode_correct += (pred == mode).sum().item()
        mode_total += B
        for m, l in zip(mode.cpu().numpy(), per_sample_loss.detach().cpu().numpy()):
            per_mode_loss[int(m)].append(float(l))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "mode_acc": mode_correct / max(mode_total, 1),
        "loss_lying": float(np.mean(per_mode_loss[0])) if per_mode_loss[0] else None,
        "loss_standing": float(np.mean(per_mode_loss[1])) if per_mode_loss[1] else None,
        "loss_cube": float(np.mean(per_mode_loss[2])) if per_mode_loss[2] else None,
    }


def main():
    args = parse_args()

    # wandb: use unique run dir per sweep trial
    wandb_run = None
    if args.wandb:
        if not _HAS_WANDB:
            raise ImportError("wandb not installed: pip install wandb")
        wandb_run = wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"{args.block}_lr{args.lr:g}_nb{args.n_blocks}_h{args.hidden}",
        )
        # override run_dir with wandb run id for isolation
        sweep_base = Path(args.run)
        sweep_base.mkdir(parents=True, exist_ok=True)
        args.run = str(sweep_base / wandb.run.name)

    run_dir = Path(args.run)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "metrics.jsonl"
    cfg_path = run_dir / "config.json"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    train_ds = GraspDataset(args.h5, split="train", augment=True, preload_depth=True,
                            rot_repr=args.rot_repr)
    val_ds = GraspDataset(args.h5, split="val", augment=False, preload_depth=True,
                          rot_repr=args.rot_repr)
    print(f"[train] rot_repr={args.rot_repr}  g_dim={train_ds.g_dim}")

    # mode 분포 확인
    train_mode_bins = np.bincount(train_ds.mode, minlength=3)
    val_mode_bins = np.bincount(val_ds.mode, minlength=3)
    print(f"[train] train mode dist (lying/standing/cube): {train_mode_bins}")
    print(f"[train] val   mode dist (lying/standing/cube): {val_mode_bins}")

    class_boost = {}
    if args.marker_boost != 1.0:
        class_boost[3] = args.marker_boost
    if args.spam_boost != 1.0:
        class_boost[4] = args.spam_boost
    sampler = make_weighted_sampler(train_ds, mode_balance=True, power=0.5,
                                     class_boost=(class_boost or None))
    if class_boost:
        print(f"[train] class_boost applied: {class_boost}")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    ms_scales = None
    if args.multiscale_local:
        ms_scales = tuple(int(s) for s in args.multiscale_local.split(","))
        print(f"[train] [v8] multiscale local crop scales: {ms_scales}")
    if args.xattn:
        print(f"[train] [v8] Cross-Attention ENABLED")
    if args.scale_dropout > 0:
        print(f"[train] [v8] scale_dropout = {args.scale_dropout}")
    if args.use_pc:
        print(f"[train] [v9 hybrid] Point Cloud token ENABLED  "
              f"N={args.pc_n_points}  dim={args.pc_dim}  crop={args.pc_crop}")
    if args.use_pc_only:
        print(f"[train] [v9 PC-only] FlowGraspNetPC: depth → PC encoder 전면 교체")

    if args.use_pc_only:
        if args.use_pc:
            raise ValueError("--use_pc_only 와 --use_pc 동시 사용 불가")
        model = FlowGraspNetPC(
            g_dim=train_ds.g_dim,
            cond_dropout=args.cond_dropout, block_type=args.block,
            n_blocks=args.n_blocks, hidden=args.hidden,
            use_xattn=args.xattn,    # cfg.args.xattn 따름 (sweep 에서 ON/OFF)
        ).to(device)
    else:
        model = FlowGraspNet(
            g_dim=train_ds.g_dim,
            cond_dropout=args.cond_dropout, block_type=args.block,
            n_blocks=args.n_blocks, hidden=args.hidden,
            use_xattn=args.xattn,
            multiscale_local_scales=ms_scales,
            scale_dropout=args.scale_dropout,
            use_pc=args.use_pc,
            pc_n_points=args.pc_n_points,
            pc_dim=args.pc_dim,
            pc_crop=args.pc_crop,
        ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params: {n_params/1e6:.2f} M")

    if args.pretrained:
        if not Path(args.pretrained).exists():
            raise FileNotFoundError(f"--pretrained not found: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location='cpu')
        # ckpt could be: {'model':..., 'ema':..., 'opt':...} or just state_dict
        if isinstance(ckpt, dict) and 'model' in ckpt:
            sd = ckpt['model']
        elif isinstance(ckpt, dict) and 'ema' in ckpt:
            sd = ckpt['ema']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[train] [pretrained] loaded {args.pretrained}")
        print(f"[train] [pretrained] missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"[train] [pretrained] missing keys (sample): {missing[:5]}")
        if unexpected:
            print(f"[train] [pretrained] unexpected keys (sample): {unexpected[:5]}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.99))
    batches_per_epoch = args.smoke if args.smoke > 0 else len(train_loader)
    total_steps = batches_per_epoch * args.epochs
    warmup = int(total_steps * args.warmup_frac)
    sched = build_lr_scheduler(optim, total_steps, warmup)
    ema = EMA(model, decay=args.ema_decay)

    cfg = {
        "args": vars(args),
        "model_params_M": n_params / 1e6,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "train_mode_bins": train_mode_bins.tolist(),
        "val_mode_bins": val_mode_bins.tolist(),
        "batches_per_epoch": batches_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup,
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"[train] cfg saved: {cfg_path}")

    best_val = float("inf")
    global_step = 0
    jsonl_f = open(jsonl_path, "w")

    dim_w = _dim_weights(args.rot_loss_weight, device, g_dim=train_ds.g_dim)
    print(f"[train] dim weights = {dim_w.tolist()}  (rot_weight={args.rot_loss_weight})")
    if args.symmetry_loss:
        print(f"[train] symmetry-aware loss ENABLED (lying 180° pairs)")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_start = time.time()
        ep_losses = {"flow": [], "mode_aux": []}
        for bi, batch in enumerate(train_loader):
            if args.smoke and bi >= args.smoke:
                break
            depth = batch["depth"].to(device, non_blocking=True)
            uv = batch["uv"].to(device, non_blocking=True)
            g_1 = batch["g1"].to(device, non_blocking=True)
            g_1_alt = batch["g1_alt"].to(device, non_blocking=True)
            mode = batch["mode"].to(device, non_blocking=True)
            B = g_1.size(0)

            g_0 = torch.randn_like(g_1)
            t = torch.rand(B, device=device)
            g_t = (1 - t)[:, None] * g_0 + t[:, None] * g_1

            v_pred, mode_logits = model.forward_with_aux(depth, uv, g_t, t)
            if args.symmetry_loss:
                loss_flow = symmetric_min_loss(v_pred, g_1, g_1_alt, g_0, dim_w)
            else:
                loss_flow = weighted_flow_loss(v_pred, g_1 - g_0, dim_w).mean()
            loss_mode = F.cross_entropy(mode_logits, mode)
            loss = loss_flow + args.aux_mode_weight * loss_mode

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            ema.update(model)

            ep_losses["flow"].append(loss_flow.item())
            ep_losses["mode_aux"].append(loss_mode.item())
            global_step += 1

            if bi % 50 == 0:
                lr = optim.param_groups[0]["lr"]
                print(f"[train] ep{epoch:02d} b{bi:04d}/{batches_per_epoch} "
                      f"loss_flow={loss_flow.item():.4f} "
                      f"loss_mode={loss_mode.item():.3f} lr={lr:.2e}")

        # eval with EMA weights
        raw_state = {k: v.clone() for k, v in model.state_dict().items()}
        ema.load_into(model)
        val_metrics = evaluate(model, val_loader, device,
                               max_batches=(args.smoke if args.smoke else None),
                               dim_w=dim_w, symmetry=args.symmetry_loss)
        model.load_state_dict(raw_state)

        ep_time = time.time() - t_start
        rec = {
            "epoch": epoch,
            "step": global_step,
            "train_flow_loss_mean": float(np.mean(ep_losses["flow"])),
            "train_mode_aux_loss_mean": float(np.mean(ep_losses["mode_aux"])),
            "val_flow_loss": val_metrics["loss"],
            "val_mode_acc": val_metrics["mode_acc"],
            "val_loss_lying": val_metrics["loss_lying"],
            "val_loss_standing": val_metrics["loss_standing"],
            "val_loss_cube": val_metrics["loss_cube"],
            "epoch_sec": ep_time,
            "lr": optim.param_groups[0]["lr"],
        }
        jsonl_f.write(json.dumps(rec) + "\n")
        jsonl_f.flush()
        if wandb_run is not None:
            wandb.log(rec, step=epoch)
        print(f"[train] ep{epoch:02d} DONE  "
              f"train_flow={rec['train_flow_loss_mean']:.4f}  "
              f"val_flow={rec['val_flow_loss']:.4f} "
              f"val_mode_acc={rec['val_mode_acc']:.3f}  "
              f"({ep_time:.1f}s)")

        # best
        norm_stats = {
            "pos_mean": train_ds.pos_mean.tolist(),
            "pos_std": train_ds.pos_std.tolist(),
            "normalize_pos": bool(train_ds.normalize_pos),
        }
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "epoch": epoch,
                "val_loss": best_val,
                "cfg": cfg,
                "norm_stats": norm_stats,
            }, ckpt_dir / "best.pt")
            print(f"[train] ✓ new best (val_flow={best_val:.4f})")
        torch.save({
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "epoch": epoch,
            "cfg": cfg,
            "norm_stats": norm_stats,
        }, ckpt_dir / "last.pt")

    jsonl_f.close()
    if wandb_run is not None:
        wandb.run.summary["best_val_flow"] = best_val
        wandb.finish()
    print(f"[train] done. best val_flow = {best_val:.4f}")


if __name__ == "__main__":
    main()
