"""Figure 7: hypersaline case study (17 residents + 2 invaders)."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/claude/ecomdesign")
from ecomdesign.figures import panel

INV = ["Escherichia", "Mycobacterium"]
SALT_IN = {"Haloarcula", "Halobacterium", "Haloferax", "Halorubrum", "Natronomonas",
           "Halomicrobium", "Haloquadratum", "Salinibacter"}
solo = pd.read_csv("data_solo.csv")
p0 = pd.read_csv("data_pairs_P0.csv"); p3 = pd.read_csv("data_pairs_P3.csv")
import os
inv = {"fba": pd.read_csv("data_invasion_fba.csv"), "strict": pd.read_csv("data_invasion_fba_strict.csv")}
if os.path.exists("data_invasion_tfa.csv"):
    inv["tfa"] = pd.read_csv("data_invasion_tfa.csv")

s1 = solo.query("abund == 1 and thermo and P == 0").set_index("taxon").mu
res = [k for k in s1.sort_values(ascending=False).index if k not in INV]
order = res + INV

fig = plt.figure(figsize=(12, 9.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1.15, 1], hspace=0.62, wspace=0.32)

# (A) directed pair heatmap, TFA, P = 0
ax = fig.add_subplot(gs[0, :])
t = p0.query("thermo").pivot(index="i", columns="j", values="mu_j").reindex(index=order, columns=order)
im = ax.imshow(t.values, cmap="viridis", aspect="auto", vmin=0, vmax=0.7)
lab = [f"{k}{' *' if k in SALT_IN else ''}" for k in order]
ax.set_xticks(range(len(order))); ax.set_xticklabels(lab, rotation=55, ha="right", fontsize=8)
ax.set_yticks(range(len(order))); ax.set_yticklabels(lab, fontsize=8)
for tl in ax.get_xticklabels() + ax.get_yticklabels():
    if tl.get_text() in INV: tl.set_color("#c0392b"); tl.set_fontweight("bold")
n = len(res)
ax.axvline(n - 0.5, color="w", lw=1.5); ax.axhline(n - 0.5, color="w", lw=1.5)
ax.set_xlabel("focal taxon $j$ (growth maximised)"); ax.set_ylabel("selfish partner $i$ ($\\geq$95% of solo growth)")
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("$\\mu_{j\\mid i}$ (h$^{-1}$), TFA, $P=0$")
panel(ax, "A", dx=-0.13, dy=1.02)

# (B) invader growth next to each resident: FBA vs TFA, P = 0 vs 3
ax = fig.add_subplot(gs[1, 0])
y = np.arange(len(res))
for k, (name, col) in enumerate(zip(INV, ["#c0392b", "#2c3e50"])):
    for d, P, mk, fill in [(p0, 0, "o", True), (p3, 3, "s", False)]:
        v = d.query("thermo and j == @name").set_index("i").mu_j.reindex(res)
        ax.scatter(v.values, y + (k - 0.5) * 0.3, marker=mk, s=26, color=col if fill else "none",
                   edgecolors=col, label=f"{name}, $P={P}$")
    ax.axvline(s1[name], color=col, ls=":", lw=1)
ax.set_yticks(y); ax.set_yticklabels(res, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("invader growth next to selfish resident (h$^{-1}$, TFA)")
ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False)
panel(ax, "B", dx=-0.42, dy=1.10)

# (C) invasion into the full resident community vs osmotic pool
ax = fig.add_subplot(gs[1, 1])
for name, col in zip(INV, ["#c0392b", "#2c3e50"]):
    for key, ls, mk, lbl in [("fba", "--", "o", "FBA, residents $\\geq$95%"),
                             ("strict", "-", "s", "FBA, residents $\\geq$99.9%"),
                             ("tfa", ":", "D", "TFA, residents $\\geq$95%")]:
        if key not in inv: continue
        d = inv[key].query("invader == @name").sort_values("P")
        ax.plot(d.P, d.mu_inv / d.mu_res_mean, ls=ls, marker=mk, ms=5, color=col,
                mfc=col if key != "strict" else "none", label=f"{name[:4]}., {lbl}")
ax.set_xscale("symlog", linthresh=1); ax.set_xticks([0, 1, 3, 10]); ax.set_xticklabels(["0", "1", "3", "10"])
ax.set_xlabel("compatible-solute pool $P$ (mmol gDW$^{-1}$)")
ax.set_ylabel("$\\mu_{inv}\\,/\\,\\bar\\mu_{res}$ (growing residents)")
ax.set_ylim(bottom=0)
ax.legend(fontsize=7, frameon=False, loc="upper right")
panel(ax, "C", dx=-0.2, dy=1.10)

fig.savefig("/home/claude/ecomdesign/figures/fig7_hypersaline.png", dpi=200, bbox_inches="tight")
print("saved")
