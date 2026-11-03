"""Figure 8: Kehe et al. (2021) screen — facilitation under tenfold dilution."""
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/claude/ecomdesign")
from ecomdesign.figures import panel

RED, BLUE, GREY = "#c1455a", "#3d6b99", "#9aa0a6"
P = {"Glucose": "Glucose01", "Glycerol": "Glycerol01", "Proline": "Proline01",
     "Pyruvate": "Pyruvate01", "Sucrose": "Sucrose01"}

L = pd.read_csv("kehe_directed.csv")
L = L[L.carbon.isin(list(P) + list(P.values()))].copy()
L["dil"] = L.carbon.str.endswith("01")
L["base"] = L.carbon.str.replace("01", "", regex=False)
raw = pd.read_csv("raw/kehe_data_s1.csv")
raw = raw[(raw["Bug 1"] != raw["Bug 2"]) & raw.Carbon.isin(list(P) + list(P.values()))].copy()
raw["dil"] = raw.Carbon.str.endswith("01")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))

# (A) paired facilitation / inhibition rates
ax = axes[0]
g = L.groupby(["base", "dil"]).agg(fac=("y", "mean"), inh=("inh", "mean"))
srcs = sorted(P)
yy = np.arange(len(srcs))
for k, (col, key, lbl) in enumerate([(BLUE, "fac", "facilitation"),
                                     (GREY, "inh", "inhibition")]):
    off = (0.5 - k) * 0.26
    hi = [100 * g.loc[(s, False), key] for s in srcs]
    lo = [100 * g.loc[(s, True), key] for s in srcs]
    for s, h, l in zip(yy, hi, lo):
        ax.plot([l, h], [s + off, s + off], color=col, lw=1.4, alpha=0.6, zorder=1)
    ax.scatter(hi, yy + off, s=34, color=col, zorder=2, label=f"{lbl}, 1$\\times$")
    ax.scatter(lo, yy + off, s=34, facecolor="white", edgecolor=col, zorder=2,
               label=f"{lbl}, 0.1$\\times$")
ax.set_yticks(yy); ax.set_yticklabels(srcs); ax.invert_yaxis()
ax.set_xlabel("directed outcomes (%)")
ax.legend(fontsize=7.2, frameon=False, ncol=2, loc="upper center",
          bbox_to_anchor=(0.5, 1.17), columnspacing=1.0, handletextpad=0.4)
ax.set_ylim(len(srcs) - 0.4, -0.75)
panel(ax, "A", dx=-0.30, dy=1.14)

# (B) effect-size distributions
ax = axes[1]
bins = np.linspace(-4, 3, 50)
for d, col, lbl in [(False, BLUE, "1$\\times$ carbon"), (True, RED, "0.1$\\times$ carbon")]:
    ax.hist(L[L.dil == d].eff, bins=bins, density=True, histtype="step",
            lw=1.5, color=col, label=lbl)
ax.axvline(0, color="0.4", lw=0.8, ls=":")
ax.set_xlabel("effect of donor on receiver (log$_2$ fold change)")
ax.set_ylabel("density")
ax.legend(fontsize=8, frameon=False)
panel(ax, "B", dx=-0.22, dy=1.14)

# (C) interaction classes
ax = axes[2]
cls = ["Mutualism", "Commensalism", "Neutralism", "Parasitism", "Amensalism",
       "Competition"]
tab = (raw.groupby("dil").Class.value_counts(normalize=True).unstack()
          .reindex(columns=cls) * 100)
x = np.arange(len(cls))
ax.bar(x - 0.19, tab.loc[False], width=0.36, color=BLUE, label="1$\\times$ carbon")
ax.bar(x + 0.19, tab.loc[True], width=0.36, color=RED, label="0.1$\\times$ carbon")
ax.set_xticks(x)
ax.set_xticklabels(["Mut.", "Comm.", "Neut.", "Paras.", "Amens.", "Comp."],
                   rotation=30, ha="right")
ax.set_ylabel("pairs (%)")
ax.legend(fontsize=8, frameon=False)
panel(ax, "C", dx=-0.20, dy=1.14)

fig.tight_layout()
fig.savefig("/home/claude/ecomdesign/figures/fig8_kehe.png", dpi=200,
            bbox_inches="tight")
print("saved")
