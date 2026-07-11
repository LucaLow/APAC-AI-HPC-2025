"""Standalone figure: the winning single-node layout TP=4 . DP-attention=4 . PP=2.

Visual grammar (one glance, three facts):
  - four identical grey chips  = attention weights fully REPLICATED per rank
  - four coloured chips        = requests + KV are DISJOINT shards (DP=4)
  - one continuous blue rail   = MoE experts SPLIT quarter-per-rank (TP=4)

PP splits the model by layers; every layer contains attention, so both
stages hold attention (for their own layers). Coloured lane arrows show
shard i's activations flowing to replica i of the next stage.

House style matches make_figs.py. Output: make_topology_1n_best.pdf (+ .png preview).
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
SHARD_TXT = ["#3d2c00", "#0b3a55", "white", "#4a1f38"]  # readable text per shard
OUT = os.environ.get("FIGDIR", ".")


def rbox(ax, x0, y0, x1, y1, fc, ec, lw=0.9, ls="-", z=1, r=0.12):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def chip(ax, x0, y0, x1, y1, fc, text, tc, ec="none", z=4, fs=5.0, bold=False):
    rbox(ax, x0, y0, x1, y1, fc, ec, lw=0.5, z=z, r=0.07)
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, fontsize=fs, ha="center",
            va="center", color=tc, zorder=z + 1,
            fontweight="bold" if bold else "normal")


fig, ax = plt.subplots(figsize=(3.35, 3.0))
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("One node: TP=4 · DP-attention=4 · PP=2", fontsize=7.8, pad=3)

# ---------- node ----------
rbox(ax, 0.30, 1.95, 9.70, 9.40, "#fbfbfb", "#444444", lw=0.9, z=1)
ax.text(0.58, 9.02, "Node 0", fontsize=6.0, color="#555555", zorder=5)
ax.text(9.42, 9.02, "each shard holds ¼ of the requests + their KV cache",
        fontsize=5.6, color="#555555", ha="right", zorder=5)

# ---------- stages ----------
X0, X1, GAP = 0.85, 9.15, 0.18
W = (X1 - X0 - 3 * GAP) / 4          # card width
CH = 2.20                             # card height
stages = (("Stage 0 · layers 1–30 of 61 · TP=4 · DP-attn=4", 5.90, 8.80, 0),
          ("Stage 1 · layers 31–61 of 61 · TP=4 · DP-attn=4", 2.20, 5.10, 4))
lane_x = []

for lab, y0, y1, gpu0 in stages:
    rbox(ax, 0.60, y0, 9.40, y1, "none", C["blue"], lw=1.2, z=2)
    ax.text(0.82, y1 - 0.32, lab, fontsize=6.0, color=C["blue"], zorder=5)
    cy = y0 + 0.15                    # card bottom
    for i in range(4):
        gx = X0 + i * (W + GAP)
        # GPU card: white, rounded, quiet border
        rbox(ax, gx, cy, gx + W, cy + CH, "white", "#b3b3b3", lw=0.6, z=3, r=0.10)
        ax.text(gx + 0.13, cy + CH - 0.24, f"GPU {gpu0 + i}", fontsize=4.4,
                color="#999999", zorder=5)
        # chip 1 (top): this rank's request shard (+ its KV) -- flows in first
        chip(ax, gx + 0.14, cy + 1.18, gx + W - 0.14, cy + 1.68,
             SHARD[i], f"shard {i + 1}", SHARD_TXT[i], bold=True)
        # chip 2: attention weights -- identical on every rank
        chip(ax, gx + 0.14, cy + 0.56, gx + W - 0.14, cy + 1.06,
             "#e6e6e6", "Attn · full copy", "#444444")
        if y0 == stages[0][1]:
            lane_x.append(gx + W / 2)
    # one continuous rail: the MoE experts, tensor-parallel across the stage
    # (flush with the card bottoms so it reads as part of every GPU)
    rbox(ax, X0, cy, X1, cy + 0.42, "#cfe0f0", C["blue"], lw=0.6, z=6, r=0.10)
    ax.text((X0 + X1) / 2, cy + 0.21, "MoE experts · TP=4 · ¼ per GPU",
            fontsize=5.0, ha="center", va="center", color="#26567d", zorder=7)

# ---------- per-lane activation flow ----------
for i, lx in enumerate(lane_x):
    ar = FancyArrowPatch((lx, 5.85), (lx, 5.20), arrowstyle="-|>",
                         mutation_scale=7, lw=1.2, color=SHARD[i], zorder=7)
    ax.add_patch(ar)
ax.text(5.0, 5.52, "activations\nNVLink", fontsize=5.2, color="#555555",
        ha="center", va="center", linespacing=1.15, zorder=7)

# ---------- freed second node ----------
rbox(ax, 0.30, 0.90, 9.70, 1.62, "none", "#aaaaaa", lw=0.8, ls=(0, (3, 3)), z=1)
ax.text(5.0, 1.26, "Node 1 · freed — serve a second replica",
        fontsize=6.2, color="#888888", ha="center", va="center", style="italic")

ax.text(5.0, 0.32, "17,417 tok/s · 8 GPUs", fontsize=7.5,
        ha="center", color=C["blue"], fontweight="bold")

fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.01)
fig.savefig(os.path.join(OUT, "make_topology_1n_best.pdf"))
fig.savefig(os.path.join(OUT, "make_topology_1n_best.png"), dpi=300)
print("wrote", os.path.join(OUT, "make_topology_1n_best.pdf"))
