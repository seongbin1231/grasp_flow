"""Fig 2 — Conditional Flow Matching architecture (논문용).

방침:
  - 신경망 크기/블록 수 등 구체 수치는 표기하지 않음
  - 수식은 mathtext ($...$) 로 렌더링
  - 단일 행 architecture: Inputs → (Global + Local crop) Encoder → Velocity Network (AdaLN-Zero) → output
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("/home/robotics/Competition/YOLO_Grasp/paper_figs")
OUT.mkdir(parents=True, exist_ok=True)


COL_INPUT  = ("#fff5e0", "#d68910")
COL_GLOBAL = ("#e7f4ea", "#1e8449")
COL_LOCAL  = ("#dff0e0", "#27ae60")
COL_COND   = ("#f3e6f9", "#7d3c98")
COL_VEL    = ("#fde4e1", "#a93226")
COL_ADALN  = ("#fff0ee", "#e74c3c")
COL_OUT    = ("#fff5e0", "#d68910")


def box(ax, xy, w, h, text, fc, ec, fontsize=10, lw=1.5, weight='normal'):
    bx = FancyBboxPatch(xy, w, h,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(bx)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color="#1a1a1a", weight=weight)


def arrow(ax, p1, p2, color="#444", lw=1.4, label=None,
          label_offset=(0.0, 0.10), label_fs=9, style='-|>'):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12,
                        color=color, linewidth=lw,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if label:
        mx = (p1[0] + p2[0]) / 2 + label_offset[0]
        my = (p1[1] + p2[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=label_fs, color="#1a1a1a", style="italic")


def main():
    W, H = 15.0, 3.8
    fig, ax = plt.subplots(figsize=(W, H), dpi=160)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])    # margin 0

    # Inputs
    box(ax, (0.10, 2.60), 1.75, 1.05,
        "Depth image\n$D \\in \\mathbb{R}^{H \\times W}$",
        *COL_INPUT, fontsize=16)
    box(ax, (0.10, 1.35), 1.75, 1.05,
        "Pixel\n$(u, v)$",
        *COL_INPUT, fontsize=16)

    # Encoders: Global + Local
    box(ax, (2.45, 2.60), 2.55, 1.05,
        "Global Depth CNN\n$E_{\\rm global}(D)$",
        *COL_GLOBAL, fontsize=16, weight='bold')
    box(ax, (2.45, 1.35), 2.55, 1.05,
        "Local Crop CNN\n$E_{\\rm local}(D|_{u,v})$",
        *COL_LOCAL, fontsize=16, weight='bold')

    # cond merge
    box(ax, (5.60, 1.95), 1.55, 1.10,
        "$\\mathbf{c}$",
        *COL_COND, fontsize=26, weight='bold')

    # Auxiliary inputs
    box(ax, (5.60, 0.85), 1.55, 0.65,
        "$g_t$  (noisy)",
        "#ffffff", "#888", fontsize=16)
    box(ax, (5.60, 0.10), 1.55, 0.65,
        "time  $t$",
        "#ffffff", "#888", fontsize=16)

    # Velocity Network
    box(ax, (7.75, 0.55), 5.20, 2.85,
        "Velocity Network\n$v_\\theta\\,(g_t,\\, t,\\, \\mathbf{c})$",
        *COL_VEL, fontsize=20, weight='bold')

    # Output
    box(ax, (13.40, 1.65), 1.50, 1.10,
        "Velocity\n$\\hat{v} \\in \\mathbb{R}^8$",
        *COL_OUT, fontsize=17, weight='bold')

    # Arrows: depth → both encoders
    arrow(ax, (1.85, 3.12), (2.45, 3.12), color=COL_GLOBAL[1], lw=1.8)
    arrow(ax, (1.85, 2.85), (2.45, 2.10), color=COL_LOCAL[1], lw=1.8)
    # uv → local crop
    arrow(ax, (1.85, 1.87), (2.45, 1.87), color=COL_LOCAL[1], lw=1.8)

    # encoders → cond
    arrow(ax, (5.00, 3.12), (5.60, 2.78), color="#444", lw=1.6)
    arrow(ax, (5.00, 1.87), (5.60, 2.22), color="#444", lw=1.6)

    # cond, g_t, t → velocity
    arrow(ax, (7.15, 2.50), (7.75, 2.40), color=COL_COND[1], lw=1.8)
    arrow(ax, (7.15, 1.17), (7.75, 1.65), color="#444", lw=1.6)
    arrow(ax, (7.15, 0.42), (7.75, 1.30), color="#444", lw=1.6)

    # velocity → output
    arrow(ax, (12.95, 2.20), (13.40, 2.20), color=COL_VEL[1], lw=2.0)

    out_png = OUT / "fig2_architecture.png"
    out_pdf = OUT / "fig2_architecture.pdf"
    plt.savefig(out_png, dpi=220, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(out_pdf, bbox_inches='tight', pad_inches=0.02)
    print(f"[fig2] {out_png}")
    print(f"[fig2] {out_pdf}")


if __name__ == "__main__":
    main()
