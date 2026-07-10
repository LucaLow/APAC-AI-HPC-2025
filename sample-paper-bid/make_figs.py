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
C = {"blue":"#0072B2","orange":"#E69F00","green":"#009E73","red":"#D55E00","purple":"#CC79A7","grey":"#999999","sky":"#56B4E9"}
FIGDIR = "/sessions/vibrant-laughing-mayer/mnt/outputs/paper/figures/"

# ---------- Fig 1: optimization ladder ----------
steps = [
    ("Baseline: venv, 2N, TP16", 9610.57, 16),
    ("Container, 2N, TP16", 11253.39, 16),
    ("Container, 1N, TP8", 12449.13, 8),
    ("Container, 2N, TP8$\\cdot$PP2", 14632.44, 16),
    ("1N, TP4$\\cdot$DP4$\\cdot$PP2 (DP-attn)", 17308.39, 8),
    ("+ NCCL GDR tuning (final)", 17417.40, 8),
]
fig, ax = plt.subplots(figsize=(3.35, 2.1))
y = np.arange(len(steps))[::-1]
vals = [s[1] for s in steps]
cols = [C["grey"], C["sky"], C["sky"], C["sky"], C["blue"], C["blue"]]
bars = ax.barh(y, vals, height=0.62, color=cols)
for yi, (label, v, g) in zip(y, steps):
    ax.text(v + 150, yi, f"{v:,.0f}", va="center", fontsize=7)
    ax.text(120, yi, label, va="center", ha="left", fontsize=7, color="white" if v > 11000 else "black",
            path_effects=[])
ax.set_yticks([])
ax.set_xlim(0, 20200)
ax.set_xlabel("Total token throughput (tok/s)")
pct = (17417.40/9610.57 - 1) * 100
ax.annotate(f"+{pct:.0f}%", xy=(17417, y[-1]), xytext=(19000, (y[0]+y[-1])/2),
            fontsize=8, fontweight="bold", ha="center", color=C["red"])
fig.savefig(FIGDIR + "ladder.pdf")
plt.close(fig)

# ---------- Fig 2: mem fraction x batch ----------
mf = [0.90, 0.91, 0.92, 0.93]
b256 = [9695.82, 11057.12, 10957.14, 11130.89]
b512 = [11460.25, 12366.44, 12465.98, 12311.90]
b724 = [11822.57, 12989.38, 13059.21, np.nan]
fig, ax = plt.subplots(figsize=(3.35, 1.95))
ax.plot(mf, b256, "o-", color=C["grey"],  label="256", ms=3.5, lw=1.2)
ax.plot(mf, b512, "s-", color=C["orange"], label="512", ms=3.5, lw=1.2)
ax.plot(mf, b724, "^-", color=C["blue"],  label="724", ms=3.5, lw=1.2)
ax.plot([0.93], [13059.21*1.007], marker="x", color=C["red"], ms=6, mew=1.6, ls="none")
ax.annotate("OOM", xy=(0.93, 13100), xytext=(0.9305, 13600), fontsize=7, color=C["red"])
ax.axvspan(0.935, 0.945, color=C["red"], alpha=0.08)
ax.text(0.94, 10200, "OOM\n(all)", ha="center", fontsize=6.5, color=C["red"])
ax.set_xlabel("mem-fraction-static")
ax.set_ylabel("Throughput (tok/s)")
ax.set_xticks([0.90, 0.91, 0.92, 0.93, 0.94])
ax.set_xlim(0.897, 0.945)
ax.legend(title="Max running requests", ncol=3, frameon=False, loc="lower right", title_fontsize=7)
fig.savefig(FIGDIR + "memfrac.pdf")
plt.close(fig)

# ---------- Fig 3: NCCL sweep ----------
LL     = [2055.89,4820.86,2058.18,4833.10,2035.16,4865.74,2066.51,4841.35,2019.30,4851.10,4864.46,4866.28,4800.07,4793.04,4779.41,4799.24,6066.10,6071.39,6054.20,6344.87,6426.30,6474.61]
LL128  = [9320.93,9445.17,9483.80,9538.34,9306.47,9480.64,9462.84,9308.89,9561.03]
SIMPLE = [7535.79,7476.96,7540.61,8162.72,8133.89,8057.54,8246.53,8251.19,8244.35]
fig, ax = plt.subplots(figsize=(3.35, 2.0))
rng = np.random.default_rng(7)
for i, (name, data, col) in enumerate([("LL", LL, C["grey"]), ("Simple", SIMPLE, C["orange"]), ("LL128", LL128, C["blue"])]):
    x = i + rng.uniform(-0.13, 0.13, len(data))
    ax.plot(x, data, "o", color=col, ms=3, alpha=0.75)
ax.axhline(10053.70, color=C["red"], lw=1.1, ls="--")
ax.text(2.42, 10250, "NCCL defaults: 10,054", fontsize=7, color=C["red"], ha="right")
ax.set_xticks([0,1,2]); ax.set_xticklabels(["LL", "Simple", "LL128"])
ax.set_xlabel("NCCL protocol"); ax.set_ylabel("Throughput (tok/s)")
ax.set_ylim(1500, 11000)
fig.savefig(FIGDIR + "nccl.pdf")
plt.close(fig)

# ---------- Fig 4: per-GPU efficiency ----------
configs = [
    ("2N TP16\n(baseline)", 9610.57, 16),
    ("2N TP16\n(container)", 11253.39, 16),
    ("2N TP8$\\cdot$PP2", 14632.44, 16),
    ("2N hybrid\n(best 2N)", 17032.39, 16),
    ("1N TP8", 12449.13, 8),
    ("1N hybrid\n(final)", 17417.40, 8),
]
fig, ax = plt.subplots(figsize=(3.35, 2.05))
x = np.arange(len(configs))
tot = [c[1] for c in configs]
per = [c[1]/c[2] for c in configs]
cols = [C["grey"]]*4 + [C["blue"]]*2
ax.bar(x, per, 0.58, color=cols)
for xi, p, t in zip(x, per, tot):
    ax.text(xi, p + 25, f"{p:,.0f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([c[0] for c in configs], fontsize=6.3)
ax.set_ylabel("Throughput per GPU (tok/s/GPU)")
ax.set_ylim(0, 2450)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C["grey"], label="16 GPUs (2 nodes)"), Patch(color=C["blue"], label="8 GPUs (1 node)")], frameon=False, loc="upper left")
fig.savefig(FIGDIR + "pergpu.pdf")
plt.close(fig)
print("figures written")
