"""Fig 2 — 제안 모델의 구조 블록도 (matplotlib).

논문용 정적 PNG/PDF.
Depth (1×720×1280) → Global CNN + Local Crop CNN → cond(256)
+ uv_norm(2) + sinusoidal time embed(64)
→ Velocity MLP (AdaLN-Zero × N)
→ velocity v(8)
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("/home/robotics/Competition/YOLO_Grasp/paper_figs")
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, xy, w, h, text, color="#e8f4fd", edge="#2c7cd2", fontsize=9, lw=1.2):
    bx = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                        facecolor=color, edgecolor=edge, linewidth=lw)
    ax.add_patch(bx)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center",
            fontsize=fontsize, color="#222")


def arrow(ax, p1, p2, label=None, color="#444", lw=1.2):
    a = FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=10,
                        color=color, linewidth=lw)
    ax.add_patch(a)
    if label:
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        ax.text(mx, my+0.04, label, ha="center", va="bottom",
                fontsize=7, color=color, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(12, 5.0), dpi=150)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # === Inputs (왼쪽) ===
    box(ax, (0.1, 3.6), 1.6, 0.9,
        "Depth\n(1, 720, 1280)\nfloat32 [m]",
        color="#fff4e6", edge="#d68910", fontsize=8)
    box(ax, (0.1, 1.8), 1.6, 0.9,
        "(u, v)\nYOLO mask\ncentroid",
        color="#fff4e6", edge="#d68910", fontsize=8)

    # === Encoders ===
    box(ax, (2.4, 4.0), 2.0, 0.7,
        "Global Depth CNN\n(stride 4-2-2-2 → 23×40)",
        color="#e8f5e9", edge="#27ae60", fontsize=8)
    box(ax, (2.4, 3.0), 2.0, 0.7,
        "Local Crop CNN\n(192×192 around uv)",
        color="#e8f5e9", edge="#27ae60", fontsize=8)

    # === Cond fusion ===
    box(ax, (5.0, 3.4), 1.6, 0.9,
        "cond\n(256)",
        color="#f3e5f5", edge="#9b59b6", fontsize=9)

    # === Auxiliary inputs ===
    box(ax, (5.0, 2.0), 1.6, 0.45,
        "uv_norm (2)", color="#ffffff", edge="#888", fontsize=7)
    box(ax, (5.0, 1.4), 1.6, 0.45,
        "sinusoidal time (64)", color="#ffffff", edge="#888", fontsize=7)
    box(ax, (5.0, 0.8), 1.6, 0.45,
        "g_t  (8)  ← noisy grasp", color="#ffffff", edge="#888", fontsize=7)

    # === Velocity MLP ===
    box(ax, (7.4, 1.6), 2.6, 1.8,
        "Velocity MLP\nAdaLN-Zero × N\n(hidden 1024)",
        color="#ffe0e0", edge="#c0392b", fontsize=10)

    # === Output ===
    box(ax, (10.4, 2.1), 1.4, 0.8,
        "v_θ\n(8)",
        color="#fff4e6", edge="#d68910", fontsize=10)

    # === Arrows ===
    arrow(ax, (1.7, 4.05), (2.4, 4.35), color="#27ae60")
    arrow(ax, (1.7, 4.05), (2.4, 3.35), color="#27ae60")
    arrow(ax, (1.7, 2.25), (2.4, 3.30), color="#d68910")
    arrow(ax, (4.4, 4.35), (5.0, 3.95))
    arrow(ax, (4.4, 3.35), (5.0, 3.65))
    arrow(ax, (6.6, 3.85), (7.4, 3.0), label="cond")
    arrow(ax, (6.6, 2.22), (7.4, 2.7), label="uv")
    arrow(ax, (6.6, 1.62), (7.4, 2.5), label="t")
    arrow(ax, (6.6, 1.02), (7.4, 2.3), label="g_t")
    arrow(ax, (10.0, 2.5), (10.4, 2.5), color="#c0392b", lw=1.5)

    # === Process labels ===
    ax.text(6.0, 4.8, "Encoders 1×/scene", ha="center", fontsize=7,
            color="#27ae60", style="italic")
    ax.text(8.7, 0.85, "Run T_steps × 2 (CFG on/off)", ha="center",
            fontsize=7, color="#c0392b", style="italic")

    # === Title + caption ===
    ax.text(6, 4.95,
            "Fig 2. Proposed Conditional Flow Matching architecture",
            ha="center", fontsize=11, weight="bold")

    plt.tight_layout()
    out_png = OUT / "fig2_architecture.png"
    out_pdf = OUT / "fig2_architecture.pdf"
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"[fig2] {out_png}")
    print(f"[fig2] {out_pdf}")


if __name__ == "__main__":
    main()
