"""
Direct MLP baseline training for (depth + uv) → 6-DoF single-grasp regression.

Flow Matching (train_flow.py) 와 정확히 같은 조건에서 학습:
  - 같은 dataset (datasets/grasp_v2.h5)
  - 같은 dataloader / sampler / mode_balance / class_boost
  - 같은 hyperparams (lr, batch, epochs, warmup, EMA, weight_decay)
  - 같은 dim weights, symmetric loss (lying 180° pair 처리), aux mode head

차이점:
  - 모델 구조: VelocityMLP → ResMLPBlock × N (no g_t, no t)
  - Loss: ||grasp_pred − g_1||²  (single-grasp regression)
  - 출력: deterministic single grasp (mode collapse 측정용)

Usage (v7 와 동일 hyperparams):
  python scripts/train_direct.py \
      --run runs/yolograsp_v2/v7_direct_mlp_big \
      --epochs 250 --batch_size 16 --lr 1e-3 --weight_decay 1e-4 \
      --warmup_frac 0.04 --ema_decay 0.9998 \
      --n_blocks 12 --hidden 1024 \
      --symmetry_loss --rot_loss_weight 2.0 \
      --marker_boost 1.5 --spam_boost 2.5 \
      --seed 42 --wandb --wandb_project yolograsp-v2
"""
from __future__ import annotations

import argparse
import json
import math
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
from src.flow_model import EMA
from src.direct_model import DirectGraspNet


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=str(ROOT / "datasets/grasp_v2.h5"))
    ap.add_argument("--run", default=str(ROOT / "runs/yolograsp_v2/v7_direct_mlp_big"))
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--aux_mode_weight", type=float, default=0.1)
    ap.add_argument("--cond_dropout", type=float, default=0.0,
                    help="Direct baseline: keep deterministic, no CFG-style dropout")
    ap.add_argument("--n_blocks", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--symmetry_loss", action="store_true",
                    help="min-target loss for lying 180° pairs (same as Flow training)")
    ap.add_argument("--rot_loss_weight", type=float, default=2.0)
    ap.add_argument("--marker_boost", type=float, default=1.5)
    ap.add_argument("--spam_boost", type=float, default=2.5)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="yolograsp-v2")
    ap.add_argument("--warmup_frac", type=float, default=0.04)
    ap.add_argument("--ema_decay", type=float, default=0.9998)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=30,
                    help="early stop: epochs of no val improvement before stopping. 0 = disable")
    ap.add_argument("--min_delta", type=float, default=1e-4,
                    help="minimum val improvement to reset patience counter")
    return ap.parse_args()


def _dim_weights(rot_weight: float, device) -> torch.Tensor:
    w = [1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight, rot_weight, rot_weight]
    return torch.tensor(w, device=device, dtype=torch.float32)


def weighted_grasp_loss(grasp_pred, grasp_tgt, dim_w):
    """per-sample weighted MSE. (B,) shape."""
    return ((grasp_pred - grasp_tgt) ** 2 * dim_w).mean(dim=-1)


def symmetric_min_grasp_loss(grasp_pred, g_1, g_1_alt, dim_w):
    err = weighted_grasp_loss(grasp_pred, g_1, dim_w)
    err_alt = weighted_grasp_loss(grasp_pred, g_1_alt, dim_w)
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
        grasp_pred, mode_logits = model.forward_with_aux(depth, uv)
        if dim_w is None:
            dim_w = torch.ones(8, device=device)
        if symmetry:
            per_sample_loss = torch.minimum(
                weighted_grasp_loss(grasp_pred, g_1, dim_w),
                weighted_grasp_loss(grasp_pred, g_1_alt, dim_w),
            )
        else:
            per_sample_loss = weighted_grasp_loss(grasp_pred, g_1, dim_w)
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

    wandb_run = None
    if args.wandb:
        if not _HAS_WANDB:
            raise ImportError("wandb not installed: pip install wandb")
        wandb_run = wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"direct_mlp_lr{args.lr:g}_nb{args.n_blocks}_h{args.hidden}",
        )
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
    print(f"[direct-train] device={device}")

    train_ds = GraspDataset(args.h5, split="train", augment=True, preload_depth=True)
    val_ds = GraspDataset(args.h5, split="val", augment=False, preload_depth=True)

    train_mode_bins = np.bincount(train_ds.mode, minlength=3)
    val_mode_bins = np.bincount(val_ds.mode, minlength=3)
    print(f"[direct-train] train mode dist: {train_mode_bins}")
    print(f"[direct-train] val   mode dist: {val_mode_bins}")

    class_boost = {}
    if args.marker_boost != 1.0:
        class_boost[3] = args.marker_boost
    if args.spam_boost != 1.0:
        class_boost[4] = args.spam_boost
    sampler = make_weighted_sampler(train_ds, mode_balance=True, power=0.5,
                                     class_boost=(class_boost or None))
    if class_boost:
        print(f"[direct-train] class_boost applied: {class_boost}")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = DirectGraspNet(
        cond_dropout=args.cond_dropout,
        n_blocks=args.n_blocks, hidden=args.hidden,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[direct-train] params: {n_params/1e6:.2f} M")

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
        "model_arch": "DirectGraspNet (encoder + ResMLPBlock × N)",
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
    print(f"[direct-train] cfg saved: {cfg_path}")

    best_val = float("inf")
    best_epoch = 0
    no_improve_count = 0
    global_step = 0
    jsonl_f = open(jsonl_path, "w")

    dim_w = _dim_weights(args.rot_loss_weight, device)
    print(f"[direct-train] dim weights = {dim_w.tolist()}  (rot_weight={args.rot_loss_weight})")
    if args.symmetry_loss:
        print(f"[direct-train] symmetry-aware loss ENABLED (lying 180° pairs)")
    if args.patience > 0:
        print(f"[direct-train] early stop ENABLED: patience={args.patience}, "
              f"min_delta={args.min_delta}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_start = time.time()
        ep_losses = {"grasp": [], "mode_aux": []}
        for bi, batch in enumerate(train_loader):
            if args.smoke and bi >= args.smoke:
                break
            depth = batch["depth"].to(device, non_blocking=True)
            uv = batch["uv"].to(device, non_blocking=True)
            g_1 = batch["g1"].to(device, non_blocking=True)
            g_1_alt = batch["g1_alt"].to(device, non_blocking=True)
            mode = batch["mode"].to(device, non_blocking=True)

            grasp_pred, mode_logits = model.forward_with_aux(depth, uv)
            if args.symmetry_loss:
                loss_grasp = symmetric_min_grasp_loss(grasp_pred, g_1, g_1_alt, dim_w)
            else:
                loss_grasp = weighted_grasp_loss(grasp_pred, g_1, dim_w).mean()
            loss_mode = F.cross_entropy(mode_logits, mode)
            loss = loss_grasp + args.aux_mode_weight * loss_mode

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            ema.update(model)

            ep_losses["grasp"].append(loss_grasp.item())
            ep_losses["mode_aux"].append(loss_mode.item())
            global_step += 1

            if bi % 50 == 0:
                lr = optim.param_groups[0]["lr"]
                print(f"[direct-train] ep{epoch:02d} b{bi:04d}/{batches_per_epoch} "
                      f"loss_grasp={loss_grasp.item():.4f} "
                      f"loss_mode={loss_mode.item():.3f} lr={lr:.2e}", flush=True)

        # eval with EMA
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
            "train_grasp_loss_mean": float(np.mean(ep_losses["grasp"])),
            "train_mode_aux_loss_mean": float(np.mean(ep_losses["mode_aux"])),
            "val_grasp_loss": val_metrics["loss"],
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
        print(f"[direct-train] ep{epoch:02d} DONE  "
              f"train_grasp={rec['train_grasp_loss_mean']:.4f}  "
              f"val_grasp={rec['val_grasp_loss']:.4f} "
              f"val_mode_acc={rec['val_mode_acc']:.3f}  ({ep_time:.1f}s)", flush=True)

        norm_stats = {
            "pos_mean": train_ds.pos_mean.tolist(),
            "pos_std": train_ds.pos_std.tolist(),
            "normalize_pos": bool(train_ds.normalize_pos),
        }
        improved = val_metrics["loss"] < best_val - args.min_delta
        if improved:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            no_improve_count = 0
            torch.save({
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "epoch": epoch,
                "val_loss": best_val,
                "cfg": cfg,
                "norm_stats": norm_stats,
            }, ckpt_dir / "best.pt")
            print(f"[direct-train] ✓ new best (val_grasp={best_val:.4f})", flush=True)
        else:
            no_improve_count += 1
            print(f"[direct-train] no improvement: {no_improve_count}/{args.patience} "
                  f"(best={best_val:.4f} @ ep{best_epoch})", flush=True)
        torch.save({
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "epoch": epoch,
            "cfg": cfg,
            "norm_stats": norm_stats,
        }, ckpt_dir / "last.pt")

        if args.patience > 0 and no_improve_count >= args.patience:
            print(f"[direct-train] EARLY STOP at ep{epoch}: "
                  f"no improvement for {args.patience} epochs. "
                  f"Best val_grasp={best_val:.4f} @ ep{best_epoch}", flush=True)
            break

    jsonl_f.close()
    if wandb_run is not None:
        wandb.run.summary["best_val_grasp"] = best_val
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.finish()
    print(f"[direct-train] done. best val_grasp = {best_val:.4f} @ ep{best_epoch}")


if __name__ == "__main__":
    main()
