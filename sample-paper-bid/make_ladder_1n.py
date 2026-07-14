"""Single-node optimization ladder — Firmus (System B, 8x H200) only.

Values (see NUMBERS.md):
  - venv 1N TP8:                      11,558.91  (firmus/venv_1node.sh, 2026-07-11;
    NOT pure defaults: --mem-fraction-static 0.87 + expandable_segments required
    because SGLang 0.5.2 auto-sizing OOMs at 1N TP8; container has 0.5.3rc1)
  - Container 1N TP8:                 12,449.13  (singularity-docker/1node.out)
  - 1N TP4*DP4*PP2 + dp-attention:    17,308.39  (xlsx table)
  - + NCCL_NET_GDR_LEVEL=5 (final):   17,417.40  (one_node/one_node.out)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "axes.axisbelow": True,
})
C = {"blue": "#0072B2", "sky": "#56B4E9", "grey": "#999999", "red": "#D55E00"}
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ---------- Single-node ladder (Firmus, 8x H200) ----------
def make_ladder(steps, cols, fname):
    fig, ax = plt.subplots(figsize=(3.35, 0.45 + 0.3 * len(steps)))
    y = np.arange(len(steps))[::-1]
    vals = [s[1] for s in steps]
    ax.barh(y, vals, height=0.62, color=cols)
    for yi, (label, v) in zip(y, steps):
        ax.text(v + 150, yi, f"{v:,.0f}", va="center", fontsize=7)
        ax.text(120, yi, label, va="center", ha="left", fontsize=7,
                color="white" if v > 12000 else "black")
    ax.set_yticks([])
    ax.set_xlim(0, 20200)
    ax.set_xlabel("Total token throughput (tok/s)")
    pct = (vals[-1] / vals[0] - 1) * 100
    ax.annotate(f"+{pct:.0f}%", xy=(vals[-1], y[-1]), xytext=(18500, y[0]),
                fontsize=8, fontweight="bold", ha="center", color=C["red"])
    fig.savefig(os.path.join(FIGDIR, fname + ".pdf"))
    fig.savefig(os.path.join(FIGDIR, fname + ".png"), dpi=300)
    plt.close(fig)
    print("wrote", os.path.join(FIGDIR, fname + ".{pdf,png}"))

steps = [
    ("Baseline: venv, TP8", 11558.91),
    ("Container, TP8", 12449.13),
    ("TP4$\\cdot$DP4$\\cdot$PP2 (DP-attn)", 17308.39),
    ("+ NCCL GDR tuning (final)", 17417.40),
]
make_ladder(steps, [C["grey"], C["sky"], C["blue"], C["blue"]], "ladder_1n")
make_ladder(steps[:-1], [C["grey"], C["sky"], C["blue"]], "ladder_1n_nogdr")
