"""Standalone figure: the two-node configurations (companion to topology_c).

Same visual grammar as topology_c:
  - blue rail           = ONE thing split across GPUs (tensor parallelism);
                          drawn passing BEHIND the cards and emerging below
  - identical grey chip = the SAME thing copied on every GPU (replication)

  (a) TP=16: one group spanning both nodes (dashed red domain); its
      per-layer all-reduce crosses InfiniBand >=61 times per generated
      token, and the MLA KV cache is replicated on all 16 ranks.
  (b) TP=8 . PP=2: each node IS one pipeline stage (layers 1-30 / 31-61,
      SGLang's default split of 61). Only stage-boundary activations cross,
      once per micro-batch; KV is still replicated x8 within each stage.

Output: topology_2n_pp2.pdf (+ .png preview).
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


def gpu_rail_block(ax, x0, x1, cy, ch, gpu0, rail_text):
    """The rail runs BEHIND eight GPU cards and emerges below them, so the
    cards are never clipped; the rail label sits in the emerging strip."""
    rbox(ax, x0 - 0.14, cy - 0.58, x1 + 0.14, cy + 0.10, "#cfe0f0", C["blue"],
         lw=0.7, z=2, r=0.08)
    ax.text((x0 + x1) / 2, cy - 0.30, rail_text, fontsize=5.2, ha="center",
            va="center", color="#26567d", zorder=7)
    n, gap = 8, 0.12
    w = (x1 + 0.0 - x0 - (n - 1) * gap) / n
    for i in range(n):
        gx = x0 + i * (w + gap)
        rbox(ax, gx, cy, gx + w, cy + ch, "white", "#b3b3b3", lw=0.6, z=3, r=0.08)
        ax.text(gx + 0.15, cy + ch - 0.30, f"{gpu0 + i}", fontsize=4.4,
                color="#999999", zorder=5)
        rbox(ax, gx + 0.15, cy + 0.40, gx + w - 0.15, cy + 0.94,
             "#e6e6e6", "none", lw=0, z=4, r=0.07)
        ax.text(gx + w / 2, cy + 0.67, "KV", fontsize=4.8, ha="center",
                va="center", color="#444444", zorder=5)


def panel(ax, mode):
    CH = 1.42                                        # card height
    nodes = ((0.55, 5.80, 9.45, 8.95), (0.55, 0.90, 9.45, 4.05))
    for k, (x0, y0, x1, y1) in enumerate(nodes):
        if mode == "tp16":
            rbox(ax, x0, y0, x1, y1, "#fbfbfb", "#444444", lw=0.9, z=1)
            ax.text(x0 + 0.26, y1 - 0.46, f"Node {k} · 8×H200",
                    fontsize=5.8, color="#555555", zorder=5)
            rail_text = "all weights · TP=16 · ¹⁄₁₆ per GPU"
        else:
            # one node IS one stage: a single blue box, no double nesting
            rbox(ax, x0, y0, x1, y1, "#fbfbfb", C["blue"], lw=1.2, z=1)
            ax.text(x0 + 0.26, y1 - 0.46,
                    f"Node {k}  =  Stage {k} · layers {'1–30' if k == 0 else '31–61'}",
                    fontsize=5.8, color=C["blue"], zorder=5)
            rail_text = "stage weights · TP=8 · ⅛ per GPU"
        cy = y0 + 0.82                                # card bottoms
        gpu_rail_block(ax, x0 + 0.34, x1 - 0.34, cy, CH, 8 * k, rail_text)
    # legend, top-right of node 0's header line (same slot as topology_c)
    key = ("grey = full KV-cache copy (×16)" if mode == "tp16"
           else "grey = this stage's KV (×8)")
    ax.text(9.19, nodes[0][3] - 0.46, key, fontsize=5.4, color="#555555",
            ha="right", zorder=5)
    # the physical link, identical in both panels
    ax.text(0.85, 4.93, "InfiniBand", fontsize=5.8, color="#888888",
            style="italic", va="center", zorder=5)
    if mode == "tp16":
        rbox(ax, 0.26, 0.52, 9.74, 9.32, "none", C["red"], lw=1.0,
             ls=(0, (4, 3)), z=8)
        ar = FancyArrowPatch((3.2, 4.15), (3.2, 5.70), arrowstyle="<|-|>",
                             mutation_scale=8, lw=1.8, color=C["red"], zorder=9)
        ax.add_patch(ar)
        ax.text(3.62, 5.18, "all-reduce at every layer", fontsize=6.4,
                color=C["red"], va="center", fontweight="bold", zorder=9)
        ax.text(3.62, 4.62, "≥ 61 crossings per token", fontsize=5.4,
                color=C["red"], va="center", zorder=9)
    else:
        ar = FancyArrowPatch((3.2, 5.70), (3.2, 4.18), arrowstyle="-|>",
                             mutation_scale=8, lw=1.6, color=C["green"], zorder=9)
        ax.add_patch(ar)
        ax.text(3.62, 5.18, "activations at the stage boundary", fontsize=6.4,
                color=C["green"], va="center", fontweight="bold", zorder=9)
        ax.text(3.62, 4.62, "1 crossing per micro batch", fontsize=5.4,
                color=C["green"], va="center", zorder=9)


# two standalone files (no titles -- the LaTeX captions carry those);
# identical figsize + ylim so both print at exactly the same element scale
for mode, fname in (("pp", "topology_2n_pp2"),):
    fig, ax = plt.subplots(figsize=(3.35, 2.20))
    ax.set_xlim(0, 10); ax.set_ylim(-0.25, 9.60)
    ax.axis("off")
    panel(ax, mode)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.savefig(os.path.join(OUT, fname + ".pdf"))
    fig.savefig(os.path.join(OUT, fname + ".png"), dpi=300)
    plt.close(fig)
    print("wrote", os.path.join(OUT, fname + ".pdf"))
