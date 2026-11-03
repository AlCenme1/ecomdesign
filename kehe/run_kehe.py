"""Kehe et al. (2021) co-culture screen: the statistical layer on real data.

Outcome: facilitation, i.e. the donor's effect on the receiver is positive
with its interquartile range excluding zero.

Model: logit P(facilitation) = -eta, eta = donor effect + receiver effect
+ environment effect + beta' x, fitted with dyadic-robust standard errors
(pairs sharing a strain are dependent) and parametric-bootstrap bias
correction for the incidental-parameter bias of the strain effects.
"""
import sys
import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, "/home/claude/ecomdesign")
from ecomdesign.stats import logit_fit, dyadic_vcov, naive_vcov, bootstrap_bias_correct

PAIRED = {"Glucose": "Glucose01", "Glycerol": "Glycerol01", "Proline": "Proline01",
          "Pyruvate": "Pyruvate01", "Sucrose": "Sucrose01"}


def load(path="raw/kehe_data_s1.csv"):
    d = pd.read_csv(path)
    d = d[d["Bug 1"] != d["Bug 2"]]
    out = []
    for a, b, e, lo, hi, n in [
            ("Bug 1", "Bug 2", "1 on 2: Effect", "1 on 2: 25th Percentile",
             "1 on 2: 75th Percentile", "1 on 2: Replicates"),
            ("Bug 2", "Bug 1", "2 on 1: Effect", "2 on 1: 25th Percentile",
             "2 on 1: 75th Percentile", "2 on 1: Replicates")]:
        t = d[[a, b, "Carbon", e, lo, hi, n]].copy()
        t.columns = ["donor", "receiver", "carbon", "eff", "q25", "q75", "nrep"]
        out.append(t)
    L = pd.concat(out, ignore_index=True).dropna(subset=["eff"])
    L["y"] = (L.q25 > 0).astype(float)          # facilitation
    L["inh"] = (L.q75 < 0).astype(float)
    return L


def design(L, extra_cols):
    """[donor dummies | receiver dummies | environment dummies | extra]."""
    strains = sorted(set(L.donor) | set(L.receiver))
    si = {s: k for k, s in enumerate(strains)}
    envs = sorted(L.carbon.unique())
    ei = {c: k for k, c in enumerate(envs)}
    P, S, E = len(L), len(strains), len(envs)
    di = L.donor.map(si).to_numpy()
    ri = L.receiver.map(si).to_numpy()
    ev = L.carbon.map(ei).to_numpy()
    D = np.zeros((P, S)); D[np.arange(P), di] = 1.0
    R = np.zeros((P, S)); R[np.arange(P), ri] = 1.0
    V = np.zeros((P, E)); V[np.arange(P), ev] = 1.0
    V = V[:, 1:]                                 # first environment as reference
    W = np.hstack([D, R, V, L[extra_cols].to_numpy(float)])
    names = ([f"donor:{s}" for s in strains] + [f"recv:{s}" for s in strains]
             + [f"env:{c}" for c in envs[1:]] + list(extra_cols))
    return W, names, di, ri, S


def fit_report(L, extra_cols, ridge=0.1, n_boot=0, label=""):
    W, names, di, ri, S = design(L, extra_cols)
    pen = np.ones(W.shape[1], bool)
    theta, H = logit_fit(W, L.y.to_numpy(float), ridge=ridge, pen_mask=pen)
    Vn, Vd = naive_vcov(H), dyadic_vcov(W, L.y.to_numpy(float), theta, di, ri, H)
    sen, sed = np.sqrt(np.maximum(np.diag(Vn), 0)), np.sqrt(np.maximum(np.diag(Vd), 0))
    k = len(names) - len(extra_cols)
    print(f"\n=== {label}  (n = {len(L)}, strains = {S}, "
          f"environments = {L.carbon.nunique()})")
    print(f"{'term':<14}{'beta(energy)':>13}{'SE naive':>10}{'SE dyadic':>11}"
          f"{'ratio':>7}{'z dyadic':>10}")
    for c, nm in enumerate(extra_cols, start=k):
        print(f"{nm:<14}{-theta[c]:13.3f}{sen[c]:10.3f}{sed[c]:11.3f}"
              f"{sed[c]/sen[c]:7.2f}{theta[c]/sed[c]:10.2f}")
    fe = np.arange(2 * S)
    ratio = (sed[fe] / sen[fe])
    print(f"strain effects: dyadic/naive SE ratio  median {np.median(ratio):.2f}, "
          f"max {ratio.max():.2f}")
    if n_boot:
        idx = np.arange(k, len(names))
        bc, draws = bootstrap_bias_correct(W, theta, idx, n_boot=n_boot,
                                           ridge=ridge, pen_mask=pen, rng=11)
        for c, nm in enumerate(extra_cols):
            print(f"  {nm}: raw {-theta[k+c]:.3f} -> bias-corrected {-bc[c]:.3f}")
    return theta, names, sed


if __name__ == "__main__":
    L = load()
    L.to_csv("kehe_directed.csv", index=False)
    print("directed observations:", len(L),
          " facilitation %.3f  inhibition %.3f" % (L.y.mean(), L.inh.mean()))

    # --- paired dilution test: does facilitation become rarer when the
    #     carbon source, and therefore every by-product, is ten times dilute?
    sub = L[L.carbon.isin(list(PAIRED) + list(PAIRED.values()))].copy()
    sub["base"] = sub.carbon.str.replace("01", "", regex=False)
    sub["dilute"] = sub.carbon.str.endswith("01").astype(float)
    sub["carbon"] = sub.base                     # environment FE = base source
    print("\nfacilitation by source and concentration")
    tab = (sub.groupby(["base", "dilute"])
             .agg(n=("y", "size"), fac=("y", "mean"), inh=("inh", "mean"))
             .round(3))
    print(tab.to_string())
    fit_report(sub, ["dilute"], label="paired dilution test", n_boot=100)

    # --- all environments, with dilution as the single covariate
    allc = L.copy()
    allc["dilute"] = allc.carbon.str.endswith("01").astype(float)
    fit_report(allc, ["dilute"], label="all 40 environments")
