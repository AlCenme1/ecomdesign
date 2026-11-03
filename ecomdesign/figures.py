"""Publication figures. Panel letters (A), (B), ... sit outside the axes,
top-left, as required for the manuscript."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metabolism import RT

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "figure.dpi": 120,
})
PAL = {"M0: node traits": "#9aa0a6", "M1: + competition": "#c7a46a",
       "M2: + stoichiometric complementarity": "#4c78a8",
       "M3: + thermodynamic window": "#d1495b", "FBA score K (no fit)": "#59a14f"}
SHORT = {"M0: node traits": "M0", "M1: + competition": "M1",
         "M2: + stoichiometric complementarity": "M2",
         "M3: + thermodynamic window": "M3", "FBA score K (no fit)": "FBA-K"}


def panel(ax, letter, dx=-0.16, dy=1.06):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def fig_thermo_filter(res: dict, path):
    """(A) stoichiometric vs thermodynamic facilitation across ceilings;
    (B) classification of FBA-feasible pairs by the exchange window."""
    ceil = sorted(res, reverse=True)
    lab = [f"{c*1e3:g} mM" if c >= 1e-3 else f"{c*1e6:g} µM" for c in ceil]
    ys = [res[c]["pairs"].y_stoich.sum() for c in ceil]
    yt = [res[c]["pairs"].y.sum() for c in ceil]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.38})
    x = np.arange(len(ceil))
    axs[0].bar(x - 0.18, ys, 0.36, color="#4c78a8", label="FBA (stoichiometric)")
    axs[0].bar(x + 0.18, yt, 0.36, color="#d1495b", label="TFA (thermodynamic)")
    for k in range(len(ceil)):
        lost = 100 * (ys[k] - yt[k]) / max(ys[k], 1)
        axs[0].text(x[k] + 0.18, yt[k] + 1, f"−{lost:.0f}%", ha="center", fontsize=7)
    axs[0].set_xticks(x, lab)
    axs[0].set_xlabel("Ceiling of extracellular by-products")
    axs[0].set_ylabel("Facilitative producer–consumer pairs")
    axs[0].set_ylim(0, max(ys) * 1.35)
    axs[0].legend(frameon=False, loc="upper right", ncol=1, fontsize=7)
    panel(axs[0], "A")
    # (B) how the exchange window classifies stoichiometrically feasible pairs
    cats = [("feasible, window ≥ 0", "#4c78a8"), ("blocked, no window", "#d1495b"),
            ("blocked, window ≥ 0", "#f2a6ae"), ("feasible, no window", "#9ecae9")]
    counts = []
    for c in ceil:
        d = res[c]["pairs"]; d = d[d.y_stoich == 1]
        win = d.G.notna() & (d.G >= 0)
        counts.append([((d.y == 1) & win).sum(), ((d.y == 0) & ~win).sum(),
                       ((d.y == 0) & win).sum(), ((d.y == 1) & ~win).sum()])
    counts = np.array(counts, float)
    frac = counts / counts.sum(1, keepdims=True)
    bottom = np.zeros(len(ceil))
    for k, (nm, col) in enumerate(cats):
        axs[1].bar(x, frac[:, k], 0.6, bottom=bottom, color=col, label=nm)
        bottom += frac[:, k]
    axs[1].set_xticks(x, lab); axs[1].set_ylim(0, 1)
    axs[1].set_xlabel("Ceiling of extracellular by-products")
    axs[1].set_ylabel("Fraction of FBA-feasible pairs")
    axs[1].legend(frameon=False, fontsize=7, loc="upper center", ncol=2,
                  bbox_to_anchor=(0.5, -0.22))
    panel(axs[1], "B")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_benchmark(cv_by_ceiling: dict, path):
    ceil = sorted(cv_by_ceiling, reverse=True)
    lab = [f"{c*1e3:g} mM" if c >= 1e-3 else f"{c*1e6:g} µM" for c in ceil]
    models = list(PAL)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.3})
    for a, met, yl in ((0, "auc", "ROC-AUC"), (1, "ap", "Average precision")):
        ax = axs[a]
        w = 0.16
        for m_i, m in enumerate(models):
            vals = [cv_by_ceiling[c].query("model == @m")[met] for c in ceil]
            mu = np.array([v.mean() for v in vals])
            lo = mu - np.array([v.quantile(0.1) for v in vals])
            hi = np.array([v.quantile(0.9) for v in vals]) - mu
            ax.errorbar(np.arange(len(ceil)) + (m_i - 2) * w, mu, yerr=[lo, hi], fmt="o",
                        ms=4, color=PAL[m], label=SHORT[m], capsize=2, lw=1)
        ax.set_xticks(np.arange(len(ceil)), lab)
        ax.set_xlabel("Ceiling of extracellular by-products")
        ax.set_ylabel(yl + " (node-split CV)")
        panel(ax, "AB"[a])
    axs[1].legend(frameon=False, ncol=1, loc="lower left")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_montecarlo(mc: pd.DataFrame, beta, path):
    from matplotlib.ticker import NullFormatter, NullLocator
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.35})
    ns = sorted(mc.n.unique())
    for k, (b, col) in enumerate(zip(beta, ["#4c78a8", "#d1495b"])):
        fe = [(mc[mc.n == n][f"fe{k+1}"] - b).mean() / abs(b) * 100 for n in ns]
        bc = [(mc[mc.n == n][f"bc{k+1}"] - b).mean() / abs(b) * 100 for n in ns]
        axs[0].plot(ns, fe, "-o", color=col, ms=4, label=f"FE-MLE, β{k+1}")
        axs[0].plot(ns, bc, "--s", color=col, ms=4, mfc="white", label=f"bias-corrected, β{k+1}")
    axs[0].axhline(0, color="grey", lw=0.8)
    axs[0].set_xscale("log"); axs[0].set_xticks(ns, [str(n) for n in ns])
    axs[0].set_xlabel("Number of strains $N$"); axs[0].set_ylabel("Relative bias (%)")
    axs[0].legend(frameon=False, fontsize=7)
    panel(axs[0], "A")
    for k, col in enumerate(["#4c78a8", "#d1495b"]):
        rn = [mc[mc.n == n][f"se_naive{k+1}"].mean() / mc[mc.n == n][f"pool{k+1}"].std() for n in ns]
        rd = [mc[mc.n == n][f"se_dyad{k+1}"].mean() / mc[mc.n == n][f"pool{k+1}"].std() for n in ns]
        axs[1].plot(ns, rn, "-o", color=col, ms=4, label=f"naive, β{k+1}")
        axs[1].plot(ns, rd, "--s", color=col, ms=4, mfc="white", label=f"dyadic-robust, β{k+1}")
    axs[1].axhline(1, color="grey", lw=0.8)
    axs[1].set_xscale("log"); axs[1].set_xticks(ns, [str(n) for n in ns])
    axs[1].set_xlabel("Number of strains $N$")
    axs[1].set_ylabel("mean SE / Monte Carlo SD")
    axs[1].legend(frameon=False, fontsize=7)
    panel(axs[1], "B")
    for ax in axs:
        ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_rem(diag, theory, temps, path, eta=None):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.35})
    ax = axs[0]
    ax.plot(temps, diag["PR"], color="#d1495b")
    ax.axvline(diag["Tc"], color="k", ls="--", lw=0.8)
    ax.axvline(1.0, color="#4c78a8", ls=":", lw=1)
    tr = ax.get_xaxis_transform()
    ax.text(diag["Tc"] * 1.08, 0.08, f"$T_c$ = {diag['Tc']:.2f}", fontsize=8, transform=tr)
    ax.text(0.92, 0.92, "fitted $T=1$", fontsize=8, color="#4c78a8", ha="right", transform=tr)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Temperature $T$"); ax.set_ylabel("Effective number of pairs (PR)")
    panel(ax, "A")
    ax = axs[1]
    ax.plot(temps, diag["lnQ"], color="#d1495b", label="empirical")
    ax.plot(temps, theory, color="k", ls="--", lw=1, label="REM approximation")
    ax.axvline(diag["Tc"], color="k", ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("Temperature $T$"); ax.set_ylabel(r"$\ln Q(T)$")
    ax.set_ylim(np.nanmin(diag["lnQ"]) - 1, np.nanmax(diag["lnQ"]) * 1.1 + 1)
    ax.legend(frameon=False)
    panel(ax, "B")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_design(val: pd.DataFrame, path):
    """Success rate of validated consortia by design strategy and ceiling."""
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.35})
    ceil = sorted(val.ceiling.unique(), reverse=True)
    lab = [f"{c*1e3:g} mM" if c >= 1e-3 else f"{c*1e6:g} µM" for c in ceil]
    strat = ["random", "M2 design", "M3 design"]
    cols = {"random": "#9aa0a6", "M2 design": "#4c78a8", "M3 design": "#d1495b"}
    w = 0.25
    for s_i, s in enumerate(strat):
        rates, lo, hi = [], [], []
        for c in ceil:
            v = val[(val.ceiling == c) & (val.strategy == s)].success
            p = v.mean(); n = len(v)
            se = np.sqrt(p * (1 - p) / max(n, 1))
            rates.append(p); lo.append(min(p, 1.96 * se)); hi.append(min(1 - p, 1.96 * se))
        axs[0].bar(np.arange(len(ceil)) + (s_i - 1) * w, rates, w, color=cols[s], label=s,
                   yerr=[lo, hi], capsize=2, error_kw=dict(lw=0.8))
    axs[0].set_xticks(np.arange(len(ceil)), lab); axs[0].set_ylim(0, 1.25)
    axs[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axs[0].set_xlabel("Ceiling of extracellular by-products")
    axs[0].set_ylabel("Feasible 4-member consortia")
    axs[0].legend(frameon=False, loc="upper center", ncol=3, fontsize=7)
    panel(axs[0], "A")
    # (B) design energy separates feasible from infeasible consortia
    ax = axs[1]
    c = ceil[-1]
    v = val[(val.ceiling == c) & np.isfinite(val.H_M3)]
    rng = np.random.default_rng(0)
    for ok, col, nm in ((1, "#4c78a8", "feasible"), (0, "#d1495b", "infeasible")):
        h = v[v.success == ok].H_M3.values
        ax.scatter(h, ok + 0.12 * rng.standard_normal(len(h)), s=10, color=col, alpha=0.7,
                   label=f"{nm} (n={len(h)})")
    ax.set_yticks([0, 1], ["infeasible", "feasible"]); ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel(r"Design energy $H_{\mathrm{M3}}(\mathcal{C})$ (units of $T$)")
    ax.set_title(f"all validated consortia, ceiling {lab[-1]}", fontsize=8)
    panel(ax, "B")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
