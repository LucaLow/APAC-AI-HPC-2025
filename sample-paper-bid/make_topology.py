"""Parallelism topology schematic: why one node beats two.

Three panels:
  (a) TP=16 spanning two nodes  -- per-layer all-reduce crosses InfiniBand
  (b) TP=8 . PP=2               -- only stage-boundary activations cross
  (c) TP=4.DP4.PP2 + DP-attn    -- one node, NVLink islands, sharded KV

House style matches make_figs.py (serif 8pt, Okabe-Ito palette).
Outputs: topology.pdf (for the paper) + topology.png (preview).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "figure.dpi": 300, "savefig.bbox": "tight",
})
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999", "sky": "#56B4E9"}
SHARD = [C["orange"], C["sky"], C["green"], C["purple"]]
OUT = os.environ.get("FIGDIR", ".")

GPU_FILL, GPU_EDGE = "#ececec", "#666666"
NODE_FILL, NODE_EDGE = "#fbfbfb", "#444444"


def rbox(ax, x0, y0, x1, y1, fc, ec, lw=0.9, ls="-", z=1):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                       boxstyle="round,pad=0,rounding_size=0.12",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def gpu_row(ax, x0, x1, y0, y1, n, kv_colors=None):
    """Draw n GPU cards between x0..x1; optional per-GPU KV bar colors."""
    gap = 0.10
    w = (x1 - x0 - (n - 1) * gap) / n
    for i in range(n):
        gx = x0 + i * (w + gap)
        rbox(ax, gx, y0, gx + w, y1, GPU_FILL, GPU_EDGE, lw=0.5, z=3)
        if kv_colors is not None:
            kc = kv_colors if isinstance(kv_colors, str) else kv_colors[i % len(kv_colors)]
            bw = 0.62 * w
            ax.add_patch(Rectangle((gx + (w - bw) / 2, y0 + 0.16), bw, 0.30,
                                   fc=kc, ec="none", zorder=4))


def two_node_panel(ax, mode):
    """mode: 'tp16' (panel a) or 'pp' (panel b)."""
    n0 = (0.55, 6.10, 9.45, 9.00)   # node 0 box
    n1 = (0.55, 1.70, 9.45, 4.60)   # node 1 box
    stage_ec = C["blue"] if mode == "pp" else NODE_EDGE
    stage_lw = 1.3 if mode == "pp" else 0.9
    for (x0, y0, x1, y1), name, stage in ((n0, "Node 0", "Stage 0"), (n1, "Node 1", "Stage 1")):
        rbox(ax, x0, y0, x1, y1, NODE_FILL, stage_ec, lw=stage_lw, z=1)
        lab = f"{name} · 8×H200 · NVLink" if mode == "tp16" \
            else f"{name} · {stage}: TP=8 inside NVLink"
        ax.text(x0 + 0.22, y1 - 0.42, lab, fontsize=5.8,
                color=C["blue"] if mode == "pp" else "#555555", zorder=5)
        gy1 = y1 - 0.75
        gpu_row(ax, x0 + 0.25, x1 - 0.25, y0 + 0.30, gy1, 8, kv_colors="#bbbbbb")
    # physical link label
    ax.text(0.85, 5.18, "InfiniBand", fontsize=5.8, color="#888888",
            style="italic", zorder=5)
    if mode == "tp16":
        # dashed red group = one TP=16 all-reduce domain
        rbox(ax, 0.28, 1.42, 9.72, 9.28, "none", C["red"], lw=1.0, ls=(0, (4, 3)), z=6)
        ar = FancyArrowPatch((4.4, 4.55), (4.4, 6.15), arrowstyle="<|-|>",
                             mutation_scale=11, lw=2.2, color=C["red"], zorder=7)
        ax.add_patch(ar)
        ax.text(4.75, 5.35, "all-reduce\nevery layer", fontsize=6.0,
                color=C["red"], va="center", linespacing=1.1, zorder=7)
        cap, capc = "11,253 tok/s · 16 GPUs", "#333333"
    else:
        ar = FancyArrowPatch((4.4, 6.10), (4.4, 4.68), arrowstyle="-|>",
                             mutation_scale=9, lw=1.3, color=C["green"], zorder=7)
        ax.add_patch(ar)
        ax.text(4.75, 5.35, "activations\nonce per µ-batch", fontsize=6.0,
                color=C["green"], va="center", linespacing=1.1, zorder=7)
        cap, capc = "14,632 tok/s · 16 GPUs (+30%)", "#333333"
    ax.text(5.0, 0.45, cap, fontsize=7, ha="center", color=capc, fontweight="bold")


def hybrid_gpu_card(ax, gx, gy, w, h, shard, replica_no):
    """GPU card with explicit internals: Attn replica (DP), MoE slice (TP), KV shard."""
    rbox(ax, gx, gy, gx + w, gy + h, GPU_FILL, GPU_EDGE, lw=0.5, z=3)
    pad = 0.14
    bw = w - 2 * pad
    # Attn replica block (distinct colour per rank -> DP=4 is countable)
    ay = gy + 1.04
    ax.add_patch(Rectangle((gx + pad, ay), bw, 0.56, fc=shard, ec="none", zorder=4))
    tc = "white" if shard in (C["green"],) else "#333333"
    ax.text(gx + w / 2, ay + 0.28, f"Attn {replica_no}", fontsize=5.0,
            ha="center", va="center", color=tc, fontweight="bold", zorder=5)
    # MoE slice block (same colour on every rank -> one TP=4 group)
    my = gy + 0.48
    ax.add_patch(Rectangle((gx + pad, my), bw, 0.44, fc="#cfe0f0",
                           ec=C["blue"], lw=0.4, zorder=4))
    ax.text(gx + w / 2, my + 0.22, "MoE · TP", fontsize=4.6,
            ha="center", va="center", color="#26567d", zorder=5)
    # KV shard bar (matches the Attn replica colour)
    ax.add_patch(Rectangle((gx + pad, gy + 0.10), bw, 0.26, fc=shard,
                           ec="none", zorder=4))
    return my + 0.22  # y of MoE band centre, for connectors


def hybrid_panel(ax):
    # single node box
    rbox(ax, 0.55, 2.30, 9.45, 9.30, NODE_FILL, NODE_EDGE, lw=0.9, z=1)
    ax.text(0.77, 8.90, "Node 0", fontsize=5.6,
            color="#555555", zorder=5)
    ax.text(9.23, 8.90, "colour = attn replica + KV shard", fontsize=5.6,
            color="#555555", ha="right", zorder=5)
    stages = (("Stage 0 · TP=4 · DP-attn=4", 5.95, 8.55),
              ("Stage 1 · TP=4 · DP-attn=4", 2.65, 5.25))
    for lab, y0, y1 in stages:
        rbox(ax, 0.85, y0, 9.15, y1, "none", C["blue"], lw=1.3, z=2)
        ax.text(1.05, y1 - 0.38, lab, fontsize=5.8, color=C["blue"], zorder=5)
        # four GPU cards with explicit internals
        x0, x1, gap = 1.10, 8.90, 0.15
        w = (x1 - x0 - 3 * gap) / 4
        gy, h = y0 + 0.18, 1.80
        for i in range(4):
            gx = x0 + i * (w + gap)
            moe_y = hybrid_gpu_card(ax, gx, gy, w, h, SHARD[i], i + 1)
            if i < 3:  # connector: the four MoE slices form one TP group
                ax.plot([gx + w, gx + w + gap], [moe_y, moe_y],
                        color=C["blue"], lw=1.0, zorder=2.5)
    ar = FancyArrowPatch((5.0, 5.90), (5.0, 5.30), arrowstyle="-|>",
                         mutation_scale=9, lw=1.3, color=C["green"], zorder=7)
    ax.add_patch(ar)
    ax.text(5.30, 5.60, "activations · NVLink", fontsize=5.6,
            color=C["green"], va="center", zorder=7)
    # freed second node
    rbox(ax, 0.55, 0.95, 9.45, 1.75, "none", "#aaaaaa", lw=0.8, ls=(0, (3, 3)), z=1)
    ax.text(5.0, 1.35, "Node 1 · freed — serve a second replica",
            fontsize=6.0, color="#888888", ha="center", va="center", style="italic")
    ax.text(5.0, 0.40, "17,417 tok/s · 8 GPUs (+55%)",
            fontsize=7, ha="center", color=C["blue"], fontweight="bold")


fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.30))
titles = ["(a) TP=16 across nodes",
          "(b) TP=8 · PP=2 across nodes",
          "(c) 1 node: TP=4·DP=4·PP=2 + DP-attn"]
for ax, t in zip(axes, titles):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(t, fontsize=7.3, pad=3)
two_node_panel(axes[0], "tp16")
two_node_panel(axes[1], "pp")
hybrid_panel(axes[2])
fig.subplots_adjust(wspace=0.04, left=0.005, right=0.995, top=0.90, bottom=0.02)
fig.savefig(os.path.join(OUT, "topology.pdf"))
fig.savefig(os.path.join(OUT, "topology.png"), dpi=300)
print("wrote", os.path.join(OUT, "topology.pdf"))
