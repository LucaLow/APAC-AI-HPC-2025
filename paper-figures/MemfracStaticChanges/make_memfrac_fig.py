"""Figure: mem-fraction-static tuning sweep.

Data source (all values verified against logs):
  firmus/optimisation/simple/memfrac_logs/{regular,512,724}_batch/memfrac_0.9*.log
System: Firmus H200, 2 nodes x 8 GPU, TP=16, Singularity container,
sglang.bench_offline_throughput (DeepSeek-R1, 2000 ShareGPT prompts).
Series = --cuda-graph-max-bs (256 / 512 / 724). OOM at 0.93 for bs=724,
OOM at 0.94 for all batch sizes.

Usage: python make_memfrac_fig.py [outdir]
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "."

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.4,
    "axes.axisbelow": True,
})

# --- data (tok/s), NaN = OOM ---
mf = np.array([0.90, 0.91, 0.92, 0.93, 0.94])
series = {  # cuda-graph-max-bs -> throughput
    256: [9695.82, 11057.12, 10957.14, 11130.89, np.nan],
    512: [11460.25, 12366.44, 12465.98, 12311.90, np.nan],
    724: [11822.57, 12989.38, 13059.21, np.nan, np.nan],
}
COL = {256: "#999999", 512: "#E69F00", 724: "#0072B2"}
MRK = {256: "o", 512: "s", 724: "D"}
RED = "#D55E00"

fig, ax = plt.subplots(figsize=(3.4, 2.3))

for bs, vals in series.items():
    v = np.array(vals)
    ax.plot(mf, v, marker=MRK[bs], color=COL[bs], ms=4, lw=1.3,
            label=f"{bs}")
    # mark first OOM point of this series at the level of its last good value
    idx = np.where(np.isnan(v))[0]
    if len(idx):
        i = idx[0]
        ax.plot(mf[i], v[i - 1], marker="x", color=RED, ms=6.5, mew=1.7,
                ls="none", zorder=5, clip_on=False)

# 0.94: OOM for every configuration
ax.axvspan(0.9355, 0.9445, color=RED, alpha=0.07, lw=0)
ax.text(0.94, 10450, "OOM\n(all)", ha="center", va="center",
        fontsize=7, color=RED)
ax.text(0.9312, 13290, "OOM", fontsize=7, color=RED)

# highlight the best point
best = 13059.21
ax.annotate(f"peak {best:,.0f}", xy=(0.92, best), xytext=(0.902, 13450),
            fontsize=7.5, color=COL[724],
            arrowprops=dict(arrowstyle="-", color=COL[724], lw=0.7))

ax.set_xlabel(r"\texttt{--mem-fraction-static}" if plt.rcParams.get("text.usetex")
              else "mem-fraction-static")
ax.set_ylabel("Total token throughput (tok/s)")
ax.set_xticks(mf)
ax.set_xlim(0.8975, 0.9445)
ax.set_ylim(9300, 13900)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(title="cuda-graph-max-bs", frameon=False, loc="lower right",
          ncol=3, title_fontsize=7.5, handlelength=1.6, columnspacing=1.1)

fig.savefig(f"{OUTDIR}/memfrac_tuning.pdf")
fig.savefig(f"{OUTDIR}/memfrac_tuning.png")
print("wrote memfrac_tuning.pdf / .png")
