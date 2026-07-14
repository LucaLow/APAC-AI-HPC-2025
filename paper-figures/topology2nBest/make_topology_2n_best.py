"""Standalone figure: the best two-node configuration (log-verified).

two_node_best.sh: --tp 4 --dp 4 --pp 2 --enable-dp-attention
                  --enable-torch-compile --nnodes 2
Gloo log (two-node.out): world size 8 -- ranks 0-3 on node 0, ranks 4-7 on
node 1. So this run used FOUR GPUs per node (eight of sixteen), with the
pipeline boundary placed exactly on the node boundary:
  Stage 0 = node 0, ranks 0-3 (layers 1-30);  Stage 1 = node 1, ranks 4-7
  (layers 31-61). TP=4 + DP-attention=4 inside each node's NVLink island;
  only stage-boundary activations cross InfiniBand, in rank-matched lanes.
Result: 17,032.39 tok/s (vs 17,417.40 for the same layout on one node).

Same visual grammar as topology_c / topology_tp16 / topology_pp2.
Output: topology_2n_best.pdf (+ .png preview).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "figure.dpi": 300, "savefig.bbox": "tight",
})
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999", "sky": "#56B4E9"}
SHARD = [C["orange"], C["sky"], C["green"], C["purple"]]
SHARD_TXT = ["#3d2c00", "#0b3a55", "white", "#4a1f38"]
OUT = os.environ.get("FIGDIR", ".")


def rbox(ax, x0, y0, x1, y1, fc, ec, lw=0.9, ls="-", z=1, r=0.12):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def chip(ax, x0, y0, x1, y1, fc, text, tc, z=4, fs=4.6, bold=False):
    rbox(ax, x0, y0, x1, y1, fc, "none", lw=0, z=z, r=0.07)
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, fontsize=fs, ha="center",
            va="center", color=tc, zorder=z + 1,
            fontweight="bold" if bold else "normal")


fig, ax = plt.subplots(figsize=(3.35, 2.55))
ax.set_xlim(0, 10); ax.set_ylim(-0.05, 9.35)
ax.axis("off")

nodes = ((0.55, 5.60, 9.45, 9.15), (0.55, 0.60, 9.45, 4.15))
lane_x = []
for k, (x0, y0, x1, y1) in enumerate(nodes):
    rbox(ax, x0, y0, x1, y1, "#fbfbfb", "#444444", lw=0.9, z=1)
    ax.text(x0 + 0.24, y1 - 0.42, f"Node {k} · 8×H200", fontsize=5.6,
            color="#555555", zorder=5)
    # ---- stage box: the 4 ACTIVE GPUs of this node ----
    sx0, sx1 = 0.75, 6.45
    sy0, sy1 = y0 + 0.20, y1 - 0.72
    rbox(ax, sx0, sy0, sx1, sy1, "none", C["blue"], lw=1.2, z=2)
    ax.text(sx0 + 0.18, sy1 - 0.32,
            f"Stage {k} · layers {'1–30' if k == 0 else '31–61'} · TP=4 · DP-attn=4",
            fontsize=5.2, color=C["blue"], zorder=5)
    # rail behind the cards (MoE experts, one TP=4 group per stage)
    cx0, cx1, gap = 0.95, 6.25, 0.10
    w = (cx1 - cx0 - 3 * gap) / 4
    cy = sy0 + 0.72                                  # card bottoms
    rbox(ax, cx0 - 0.10, cy - 0.52, cx1 + 0.10, cy + 0.08, "#cfe0f0",
         C["blue"], lw=0.6, z=2.5, r=0.08)
    ax.text((cx0 + cx1) / 2, cy - 0.26, "MoE experts · TP=4 · ¼ per GPU",
            fontsize=4.8, ha="center", va="center", color="#26567d", zorder=7)
    for i in range(4):
        gx = cx0 + i * (w + gap)
        rbox(ax, gx, cy, gx + w, cy + 1.42, "white", "#b3b3b3",
             lw=0.6, z=3, r=0.08)
        ax.text(gx + 0.11, cy + 1.42 - 0.24, f"{4 * k + i}", fontsize=4.2,
                color="#999999", zorder=5)                 # global rank
        chip(ax, gx + 0.11, cy + 0.66, gx + w - 0.11, cy + 1.04,
             SHARD[i], f"shard {i + 1}", SHARD_TXT[i], fs=4.6, bold=True)
        chip(ax, gx + 0.11, cy + 0.16, gx + w - 0.11, cy + 0.54,
             "#e6e6e6", "Attn copy", "#444444", fs=4.4)
        if k == 0:
            lane_x.append(gx + w / 2)
    # ---- the 4 IDLE GPUs of this node ----
    rbox(ax, 6.70, sy0, 9.25, sy1, "none", "#aaaaaa", lw=0.8,
         ls=(0, (3, 3)), z=1)
    ax.text(7.975, (sy0 + sy1) / 2, "4 GPUs idle", fontsize=5.2,
            color="#888888", ha="center", va="center", style="italic")

# legend, top-right of node 0's header line
ax.text(9.21, nodes[0][3] - 0.42, "colour = ¼ of requests + its KV",
        fontsize=5.2, color="#555555", ha="right", zorder=5)

# ---- rank-matched activation lanes across InfiniBand ----
for i, lx in enumerate(lane_x):
    ar = FancyArrowPatch((lx, 5.53), (lx, 4.22), arrowstyle="-|>",
                         mutation_scale=6.5, lw=1.2, color=SHARD[i], zorder=9)
    ax.add_patch(ar)
ax.text(9.30, 5.28, "InfiniBand", fontsize=5.2, color="#888888",
        style="italic", ha="right", zorder=9)
ax.text(9.30, 4.88, "activations at the stage boundary", fontsize=4.5,
        color=C["green"], fontweight="bold", ha="right", zorder=9)
ax.text(9.30, 4.50, "once per micro-batch", fontsize=4.8,
        color=C["green"], ha="right", zorder=9)

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig(os.path.join(OUT, "topology_2n_best.pdf"))
fig.savefig(os.path.join(OUT, "topology_2n_best.png"), dpi=300)
print("wrote", os.path.join(OUT, "topology_2n_best.pdf"))
