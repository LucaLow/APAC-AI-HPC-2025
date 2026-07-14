"""All 40 manual NCCL sweep configs in one chart (System B, 2N TP16).

Data: firmus/optimisation/network/throughput_table.csv (parsed from
net_sweep_496/501/506 logs). Default-config reference: 10,053.70 tok/s.
Style matches sample-paper-bid/make_figs.py (serif 8pt, Okabe-Ito).
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 5.4,
    "legend.fontsize": 6.5, "figure.dpi": 300, "savefig.bbox": "tight",
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
        rows.append(dict(proto=r["proto"], T=int(r["T"]), Cc=int(r["C"]), M=int(r["M"]),
                         R=int(r["R"]), S=int(r["S"]), G=int(r["G"]),
                         thr=float(r["total_token_throughput_tok_s"])))
assert len(rows) == 40, f"expected 40 configs, got {len(rows)}"

order = {"LL128": 0, "Simple": 1, "LL": 2}          # best group on top
groups = {p: sorted([r for r in rows if r["proto"] == p], key=lambda r: -r["thr"])
          for p in order}
proto_col = {"LL128": C["blue"], "Simple": C["orange"], "LL": C["grey"]}

fig, ax = plt.subplots(figsize=(3.35, 5.0))
y, ylabels, ypos, gap = 0, [], [], 1.6
for p in ["LL128", "Simple", "LL"]:
    ax.text(150, y + 0.72, p, fontsize=7.5, fontweight="bold", va="bottom",
            color=proto_col[p] if p != "LL" else "#555555")
    for r in groups[p]:
        gdr_off = r["G"] == 0
        ax.barh(y, r["thr"], height=0.78, color=proto_col[p],
                alpha=0.45 if gdr_off else 1.0,
                hatch="/////" if gdr_off else None,
                edgecolor=C["red"] if gdr_off else "none", linewidth=0.4)
        ylabels.append(f'T{r["T"]} C{r["Cc"]} M{r["M"]} R{r["R"]} S{r["S"]} G{r["G"]}')
        ypos.append(y)
        y -= 1
    y -= gap
for p in ["LL128", "Simple", "LL"]:                  # best-in-group value
    best = groups[p][0]
    yb = ypos[[i for i, l in enumerate(ylabels)][sum(len(groups[q]) for q in order if order[q] < order[p])]]
    ax.text(best["thr"] + 130, yb, f'{best["thr"]:,.0f}', va="center", fontsize=6)

ax.axvline(DEFAULT, color=C["red"], lw=1.1, ls="--", zorder=0)
ax.text(DEFAULT - 150, ypos[0] + 1.15, "NCCL defaults: 10,054", fontsize=6.5,
        color=C["red"], ha="right", va="bottom")
ax.set_yticks(ypos)
ax.set_yticklabels(ylabels, fontfamily="monospace")
ax.set_ylim(min(ypos) - 0.9, max(ypos) + 2.1)
ax.set_xlim(0, 11000)
ax.set_xlabel("Total token throughput (tok/s)")
ax.tick_params(axis="y", length=0)
ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#bbbbbb", alpha=0.45, hatch="/////",
                         edgecolor=C["red"], linewidth=0.4, label="GPU-Direct RDMA disabled (G=0)")],
          frameon=False, loc="lower right", handlelength=1.4, handleheight=1.0)

fig.savefig(os.path.join(HERE, "nccl_all.pdf"))
fig.savefig(os.path.join(HERE, "nccl_all.png"))
print("written: nccl_all.pdf / nccl_all.png")
