"""
Strain library, pairwise features (competition C, complementarity K,
thermodynamic compatibility G) and mechanistic ground truth for pairs.
"""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from .metabolism import (CARBONS, KO_POOL, RT, Strain, community_fba,
                         make_strain, mdf, medium, single_fba, tfa,
                         community_opt)

EPS_MU = 0.01           # minimal growth rate considered "growth" (h^-1)
SCAVENGER_CARBONS = ["ac", "etoh", "lac__D", "acald", "pyr", "succ", "akg",
                     "fum", "mal__L", "glu__L", "gln__L"]


def design_medium(base, glc=10.0, o2=5.0):
    """Shared medium of the design problem: limiting glucose, microaerobic."""
    return medium(base, {"glc__D": glc}, o2)


def build_library(base, n=40, p_fermenter=0.4, p_aerobic=0.7, max_ko=2, seed=1):
    rng = np.random.default_rng(seed)
    lib, seen = [], set()
    while len(lib) < n:
        if rng.random() < p_fermenter:
            carbons = ["glc__D"] + (["fru"] if rng.random() < 0.5 else [])
            extra = rng.choice(SCAVENGER_CARBONS, size=rng.integers(0, 2), replace=False)
            carbons += list(extra)
        else:
            carbons = list(rng.choice(SCAVENGER_CARBONS, size=rng.integers(1, 4), replace=False))
        aerobic = bool(rng.random() < p_aerobic)
        kos = tuple(sorted(rng.choice(KO_POOL, size=rng.integers(0, max_ko + 1), replace=False)))
        key = (tuple(sorted(carbons)), aerobic, kos)
        if key in seen:
            continue
        st = make_strain(base, f"S{len(lib) + 1:02d}", tuple(sorted(carbons)), aerobic, kos)
        rich = medium(base, {c: 10.0 for c in carbons}, 20.0 if aerobic else 0.0)
        sol = single_fba(base, st, rich, pfba=False)
        if sol is None or sol["mu"] < EPS_MU:
            continue                     # non-viable even on its own substrates
        seen.add(key)
        lib.append(st)
    return lib


SELFISH = 0.95          # producer keeps >= 95% of its own optimum
X_MIN = 0.1             # minimal exchange flux (mmol gDW^-1 h^-1) in window problems
ABUND = 0.5             # abundance of each member in the pair problems
O2_GRID = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]   # residual O2 levels (community units)


def node_table(base, lib, D):
    """Per-strain traits computed once per strain (node level).

    mu_D      : growth alone on D (thermodynamics-based FBA, abundance 1)
    grows_D   : 1 if mu_D > EPS_MU (candidate producer)
    For candidate producers, in the pair context (abundance 0.5, >= 95% of
    their own optimum):  sec[k] = max export of each carbon by-product,
    o2_min / glc_min = minimal O2 / glucose uptake (what is left for a partner).
    """
    ext = [base.met_ids[i] for i in base.ext_idx]
    carb_e = [f"{c}_e" for c in CARBONS]
    rows, Ssec = [], []
    for st in lib:
        sol = tfa(base, [st], [1.0], D)
        mu = 0.0 if sol is None else sol["mu"]
        md = mdf(base, sol["v"])[0] if (sol is not None and mu > EPS_MU) else np.nan
        sec = np.zeros(len(carb_e)); o2m = glcm = np.nan; ref = 0.0
        if mu > EPS_MU:
            r0 = community_opt(base, [st], [ABUND], D, thermo=False)
            ref = r0["bio"][0]
            bm = [SELFISH * ref]
            for a, k in enumerate(carb_e):
                if k == "glc__D_e":
                    continue
                r = community_opt(base, [st], [ABUND], D, thermo=False, bio_min=bm,
                                  obj_ext={k: 1.0})
                sec[a] = max(0.0, r["x"][ext.index(k)]) if r else 0.0
            r = community_opt(base, [st], [ABUND], D, thermo=False, bio_min=bm,
                              obj_ext={"o2_e": 1.0})      # max export = min uptake
            o2m = -r["x"][ext.index("o2_e")]
            r = community_opt(base, [st], [ABUND], D, thermo=False, bio_min=bm,
                              obj_ext={"glc__D_e": 1.0})
            glcm = -r["x"][ext.index("glc__D_e")]
        rows.append(dict(strain=st.name, aerobic=st.aerobic, n_carbons=len(st.carbons),
                         n_ko=len(st.knockouts), carbons="+".join(st.carbons),
                         knockouts="+".join(st.knockouts) or "-", mu_D=mu, mdf_D=md,
                         grows_D=float(mu > EPS_MU), mu_pair_ref=ref,
                         o2_min=o2m, glc_min=glcm))
        Ssec.append(sec)
    nodes = pd.DataFrame(rows)
    Ssec = pd.DataFrame(Ssec, columns=carb_e, index=nodes.strain)
    return nodes, Ssec


def exchange_windows(base, lib, nodes, D):
    """Node-level thermodynamic bounds on extracellular concentrations.

    lmax[i,k]   : highest ln[k]_e at which producer i keeps >= 95% of its optimum
                  while exporting k (pair context)        -> producer ceiling.
    lmin[j,k,o] : lowest ln[k]_e at which strain j grows while taking up k with
                  residual O2 level O2_GRID[o]            -> consumer floor.
    Thermodynamics-based FBA (MILP); NaN = infeasible.
    """
    N, Kc = len(lib), len(CARBONS)
    lmax = np.full((N, Kc), np.nan); lmin = np.full((N, Kc, len(O2_GRID)), np.nan)
    for i, st in enumerate(lib):
        if nodes.grows_D.iloc[i]:
            bm = [SELFISH * nodes.mu_pair_ref.iloc[i]]
            for a, c in enumerate(CARBONS):
                k = f"{c}_e"
                if c == "glc__D":
                    continue
                r = community_opt(base, [st], [ABUND], D, thermo=True, bio_min=bm,
                                  ext_target=k, sense=-1, export_min={k: X_MIN})
                if r is not None:
                    lmax[i, a] = r["lnc_ext"][k]
        else:
            for a, c in enumerate(CARBONS):
                if c not in st.carbons:
                    continue
                k = f"{c}_e"
                for o, o2 in enumerate(O2_GRID):
                    med = medium(base, {c: 10.0}, o2)
                    r = community_opt(base, [st], [ABUND], med, thermo=True,
                                      bio_min=[EPS_MU], ext_target=k, sense=+1,
                                      uptake_min={k: X_MIN})
                    if r is not None:
                        lmin[i, a, o] = r["lnc_ext"][k]
    return lmax, lmin


def _o2_level(resid):
    """Index of the largest grid level not exceeding the residual O2."""
    return int(np.searchsorted(O2_GRID, resid + 1e-9) - 1)


def pair_features(base, lib, nodes, Ssec, lmax, lmin, D):
    """Pair features for producer-consumer pairs (others are structural zeros).

    C : resource overlap = consumer's O2 need x producer's O2 share
        + glucose overlap (fraction of D's O2 / glucose the producer takes).
    K : decoupled facilitation potential = growth rate the consumer reaches on
        the producer's maximal by-product supply and the residual O2/glucose
        (plain FBA, no thermodynamics; one round of exchange).
    G : thermodynamic exchange window (kJ/mol) of the best by-product k that
        the consumer uses in that solution: RT (ln c_max^prod - ln c_min^cons).
    """
    ext = [base.met_ids[i] for i in base.ext_idx]
    carb_e = list(Ssec.columns)
    grows = nodes.grows_D.values
    recs = []
    for i, j in itertools.combinations(range(len(lib)), 2):
        if grows[i] + grows[j] != 1:
            continue
        p, c = (i, j) if grows[i] else (j, i)
        o2_left = max(0.0, D["o2_e"] - nodes.o2_min.iloc[p])
        glc_left = max(0.0, D["glc__D_e"] - nodes.glc_min.iloc[p])
        med = medium(base, {}, o2_left)
        for a, k in enumerate(carb_e):
            if Ssec.iloc[p, a] > 1e-6:
                med[k] = Ssec.iloc[p, a]
        if glc_left > 1e-6:
            med["glc__D_e"] = glc_left
        r = community_opt(base, [lib[c]], [ABUND], med, thermo=False)
        K = 0.0 if r is None else r["bio"][0]
        G, k_best = np.nan, "-"
        if r is not None and K > EPS_MU:
            used = [a for a, k in enumerate(carb_e) if r["x"][ext.index(k)] < -1e-6]
            o = _o2_level(o2_left)
            w = [RT * (lmax[p, a] - lmin[c, a, o]) for a in used]
            w = [x for x in w if not np.isnan(x)]
            if w:
                G = max(w); k_best = CARBONS[used[int(np.argmax(w))]] if len(used) else "-"
        o2_share = nodes.o2_min.iloc[p] / D["o2_e"]
        glc_share = nodes.glc_min.iloc[p] / D["glc__D_e"]
        C = float(lib[c].aerobic) * o2_share + float("glc__D" in lib[c].carbons) * glc_share
        recs.append(dict(i=i, j=j, producer=p, consumer=c, C=C, K=K, G=G, k_best=k_best,
                         o2_left=o2_left))
    return pd.DataFrame(recs)


def pair_truth(base, lib, nodes, D, thermo_only=False):
    """Mechanistic labels for every producer-consumer pair (one member grows
    alone on D, the other does not); all other pairs are labelled 0.

    The producer is selfish: it must keep >= 95% of the growth it reaches in
    the same medium share when the partner is absent. The consumer must also
    cover its non-growth ATP maintenance from what it receives. The label is 1 if, under that
    constraint, the consumer can grow (mu_c >= EPS_MU) on the shared medium.
    y_stoich uses plain FBA; y uses thermodynamics-based FBA with a shared
    extracellular concentration pool (ground truth).
    """
    grows = nodes.grows_D.values
    recs = []
    for i, j in itertools.combinations(range(len(lib)), 2):
        mu_f = mu_t = np.nan
        if (grows[i] + grows[j]) == 1:
            p, c = (i, j) if grows[i] else (j, i)
            pair = [lib[p], lib[c]]
            ref = community_opt(base, [lib[p]], [ABUND], D, thermo=True)  # partner absent
            if ref is None:
                recs.append(dict(i=i, j=j, mu_fba=0.0, mu_tfa=0.0, y_stoich=0, y=0))
                continue
            bmin = [SELFISH * ref["bio"][0], 0.0]
            if not thermo_only:
                rf = community_opt(base, pair, [ABUND, ABUND], D, thermo=False,
                                   obj_bio=[0, 1], bio_min=bmin)
                mu_f = 0.0 if rf is None else rf["bio"][1]
            rt = community_opt(base, pair, [ABUND, ABUND], D, thermo=True,
                               obj_bio=[0, 1], bio_min=bmin)
            mu_t = 0.0 if rt is None else rt["bio"][1]
        recs.append(dict(i=i, j=j, mu_fba=mu_f, mu_tfa=mu_t,
                         y_stoich=int(np.nan_to_num(mu_f) > EPS_MU),
                         y=int(np.nan_to_num(mu_t) > EPS_MU)))
    return pd.DataFrame(recs)
