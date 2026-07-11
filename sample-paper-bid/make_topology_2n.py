"""Standalone figure: the two-node configurations (companion to topology_c).

Two stacked panels with IDENTICAL geometry -- the only visible difference is
what crosses the InfiniBand gap:
  (a) TP=16: one all-reduce domain spanning both nodes; the collective
      crosses IB at every layer -> 61x per generated token.
  (b) TP=8 . PP=2: each node is one pipeline stage (TP inside NVLink);
      only stage-boundary activations cross -> 1x per micro-batch.

House style matches make_figs.py / make_topology_c.py.
Output: topology_2n.pdf (+ .png preview).
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
OUT = os.environ.get("FIGDIR", ".")


def rbox(ax, x0, y0, x1, y1, fc, ec, lw=0.9, ls="-", z=1, r=0.12):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def gpu_row(ax, x0, x1, y0, y1):
    """Eight plain GPU squares (white cards, quiet borders)."""
    n, gap = 8, 0.09
    w = (x1 - x0 - (n - 1) * gap) / n
    for i in range(n):
        gx = x0 + i * (w + gap)
        rbox(ax, gx, y0, gx + w, y1, "white", "#b3b3b3", lw=0.6, z=3, r=0.08)


def panel(ax, mode):
    """mode: 'tp16' or 'pp'."""
    n0 = (0.55, 5.95, 9.45, 9.35)
    n1 = (0.55, 0.70, 9.45, 4.10)
    if mode == "tp16":
        labels = ("Node 0 · 8×H200 · NVLink", "Node 1 · 8×H200 · NVLink")
        ec, lw = "#444444", 0.9
        lab_c = "#555555"
    else:
        labels = ("Node 0 · Stage 0 · layers 1–30 · TP=8 in NVLink",
                  "Node 1 · Stage 1 · layers 31–61 · TP=8 in NVLink")
        ec, lw = C["blue"], 1.2
        lab_c = C["blue"]
    for (x0, y0, x1, y1), lab in zip((n0, n1), labels):
        rbox(ax, x0, y0, x1, y1, "#fbfbfb", ec, lw=lw, z=1)
        ax.text(x0 + 0.24, y1 - 0.44, lab, fontsize=5.8, color=lab_c, zorder=5)
        gpu_row(ax, x0 + 0.28, x1 - 0.28, y0 + 0.30, y1 - 0.85)
    # the physical link, identical in both panels
    ax.text(0.85, 4.90, "InfiniBand", fontsize=5.8, color="#888888",
            style="italic", va="center", zorder=5)
    if mode == "tp16":
        # one all-reduce domain spanning both nodes
        rbox(ax, 0.30, 0.42, 9.70, 9.62, "none", C["red"], lw=1.0,
             ls=(0, (4, 3)), z=6)
        ar = FancyArrowPatch((3.2, 4.05), (3.2, 6.00), arrowstyle="<|-|>",
                             mutation_scale=11, lw=2.2, color=C["red"], zorder=7)
        ax.add_patch(ar)
        ax.text(3.60, 5.28, "61× per token", fontsize=6.6, color=C["red"],
                va="center", fontweight="bold", zorder=7)
        ax.text(3.60, 4.72, "all-reduce · every layer", fontsize=5.4,
                color=C["red"], va="center", zorder=7)
    else:
        ar = FancyArrowPatch((3.2, 5.95), (3.2, 4.18), arrowstyle="-|>",
                             mutation_scale=9, lw=1.4, color=C["green"], zorder=7)
        ax.add_patch(ar)
        ax.text(3.60, 5.28, "1× per µ-batch", fontsize=6.6, color=C["green"],
                va="center", fontweight="bold", zorder=7)
        ax.text(3.60, 4.72, "stage-boundary activations", fontsize=5.4,
                color=C["green"], va="center", zorder=7)


fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.55))
titles = ("(a) TP=16 spanning both nodes · 11,253 tok/s",
          "(b) TP=8 · PP=2 · 14,632 tok/s (+30%)")
for ax, t in zip(axes, titles):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(t, fontsize=7.3, pad=3)
panel(axes[0], "tp16")
panel(axes[1], "pp")
fig.subplots_adjust(left=0.005, right=0.995, top=0.945, bottom=0.005, hspace=0.22)
fig.savefig(os.path.join(OUT, "topology_2n.pdf"))
fig.savefig(os.path.join(OUT, "topology_2n.png"), dpi=300)
print("wrote", os.path.join(OUT, "topology_2n.pdf"))
