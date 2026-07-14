"""Two alternative single-figure views of the 40-config NCCL sweep."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 6.8, "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "axes.axisbelow": True,
})
C = {"blue": "#0072B2", "orange": "#E69F00", "grey": "#999999", "red": "#D55E00"}
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "..", "firmus", "optimisation", "network", "throughput_table.csv")
DEFAULT = 10053.70

rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        rows.append(dict(proto=r["proto"], T=int(r["T"]), Cc=int(r["C"]), G=int(r["G"]),
                         thr=float(r["total_token_throughput_tok_s"])))
assert len(rows) == 40
pcol = {"LL128": C["blue"], "Simple": C["orange"], "LL": C["grey"]}
tx = {64: 0, 160: 1, 256: 2}

# ---------- Option A: interaction plot ----------
fig, ax = plt.subplots(figsize=(3.35, 2.4))
rng = np.random.default_rng(7)
for p in ["LL128", "Simple", "LL"]:
    pts = [r for r in rows if r["proto"] == p and r["G"] != 0]
    x = [tx[r["T"]] + rng.uniform(-0.09, 0.09) for r in pts]
    ax.plot(x, [r["thr"] for r in pts], "o", color=pcol[p], ms=3, alpha=0.75, zorder=3)
    means = [np.mean([r["thr"] for r in pts if r["T"] == t]) for t in (64, 160, 256)]
    ax.plot([0, 1, 2], means, "-", color=pcol[p], lw=1.2, alpha=0.9, zorder=2)
    ax.annotate(p, xy=(2.16, means[2]), fontsize=7, color=pcol[p] if p != "LL" else "#666666",
                va="center", fontweight="bold")
g0 = [r for r in rows if r["G"] == 0]
ax.plot([tx[r["T"]] + rng.uniform(-0.09, 0.09) for r in g0], [r["thr"] for r in g0],
        "x", color=C["red"], ms=4.5, mew=1.3, zorder=3)
ax.annotate("GPU-Direct RDMA off\n(2.4$\\times$ loss)", xy=(0.12, 2050), xytext=(0.62, 2900),
            fontsize=6.5, color=C["red"], va="center",
            arrowprops=dict(arrowstyle="-", color=C["red"], lw=0.6))
ax.axhline(DEFAULT, color=C["red"], lw=1.1, ls="--")
ax.text(2.58, 10230, "NCCL defaults: 10,054", fontsize=7, color=C["red"], ha="right")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["64", "160", "256"])
ax.set_xlim(-0.35, 2.62)
ax.set_xlabel("NCCL_NTHREADS")
ax.set_ylabel("Throughput (tok/s)")
ax.set_ylim(1500, 11000)
fig.savefig(os.path.join(HERE, "nccl_interaction.pdf"))
fig.savefig(os.path.join(HERE, "nccl_interaction.png"))
plt.close(fig)

# ---------- Option B: heatmap (27 factorial runs) + note ----------
protos = ["LL128", "Simple", "LL"]
Ts, Cs = [64, 160, 256], [4, 8, 16]
M = np.full((3, 9), np.nan)
for i, p in enumerate(protos):
    for j, (t, c) in enumerate([(t, c) for t in Ts for c in Cs]):
        v = [r["thr"] for r in rows if r["proto"] == p and r["T"] == t and r["Cc"] == c and r["G"] == 2]
        # factorial cell = the 506 run (M2 R1 S1); average duplicates
        if v: M[i, j] = np.mean(v)
fig, ax = plt.subplots(figsize=(3.35, 1.75))
im = ax.imshow(M, cmap="viridis", vmin=2000, vmax=DEFAULT, aspect="auto")
for i in range(3):
    for j in range(9):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i,j]/1000:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] < 7800 else "black")
ax.set_yticks(range(3)); ax.set_yticklabels(protos)
ax.set_xticks(range(9)); ax.set_xticklabels([f"{c}" for t in Ts for c in Cs], fontsize=6.5)
for xpos, t in zip([1, 4, 7], Ts):
    ax.text(xpos, -0.72, f"T={t}", ha="center", fontsize=7)
ax.set_xlabel("NCCL_MIN_NCHANNELS", fontsize=7.5)
ax.grid(False)
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.ax.tick_params(labelsize=6)
cb.ax.axhline(1.0, color=C["red"])
cb.set_label("tok/s ($\\times$1000)", fontsize=6.5)
ax.set_title("All cells below autotuned default (10.1)", fontsize=7, color=C["red"], pad=16)
fig.text(0.02, -0.14, "Not shown: 13 LL/T64/C4 ablations $-$ M/R/S variants 4.8k$\\pm$0.05k; GDR off 2.0k (5 runs).",
         fontsize=5.8, style="italic")
fig.savefig(os.path.join(HERE, "nccl_heatmap.pdf"))
fig.savefig(os.path.join(HERE, "nccl_heatmap.png"))
print("done")
