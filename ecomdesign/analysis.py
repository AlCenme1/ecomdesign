"""Benchmark, inference, REM diagnostics and consortium design on a library."""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from .library import ABUND, EPS_MU, SELFISH
from .metabolism import RT, community_opt
from .stats import dyad_design, dyadic_vcov, logit_fit, naive_vcov

FEATURE_SETS = {
    "M0: node traits": [],
    "M1: + competition": ["C"],
    "M2: + stoichiometric complementarity": ["C", "K"],
    "M3: + thermodynamic window": ["C", "K", "thermo_block", "G_margin"],
}
NODE_TRAITS = ["aerobic", "n_carbons", "n_ko"]


def prepare(pairs: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    d = pairs.copy()
    d["thermo_block"] = d.G.isna() | (d.G <= 0)          # no feasible window
    d["thermo_block"] = d.thermo_block.astype(float) * (d.K > EPS_MU)
    d["G_margin"] = (d.G.fillna(0.0).clip(lower=0.0)) / RT  # in units of RT
    d["K"] = d.K / 0.1                                     # per 0.1 h^-1
    return d


def node_matrix(nodes):
    Z = nodes[NODE_TRAITS].astype(float).values
    return (Z - Z.mean(0)) / Z.std(0).clip(1e-9)


def _W(d, feats, Z, n_nodes, node_fe):
    X = d[feats].astype(float).values if feats else np.zeros((len(d), 0))
    # producer and consumer roles differ: separate node-trait blocks
    if node_fe:
        return dyad_design(d.producer.values, d.consumer.values, X, n_nodes, node_fe=True)
    P = len(d)
    return np.hstack([np.ones((P, 1)), Z[d.producer.values], Z[d.consumer.values], X])


def cv_benchmark(d, nodes, n_splits=5, n_repeats=20, ridge=1e-2, seed=0):
    """Node-split cross-validation: train on pairs whose two strains are both in
    the training folds, test on pairs with at least one held-out strain."""
    rng = np.random.default_rng(seed)
    Z = node_matrix(nodes)
    N = len(nodes)
    out = []
    for rep in range(n_repeats):
        folds = rng.permutation(N) % n_splits
        for f in range(n_splits):
            test_nodes = np.where(folds == f)[0]
            te = d.producer.isin(test_nodes) | d.consumer.isin(test_nodes)
            tr = ~te
            if d.y[te].nunique() < 2 or d.y[tr].nunique() < 2:
                continue
            for name, feats in FEATURE_SETS.items():
                Wtr = _W(d[tr], feats, Z, N, False)
                Wte = _W(d[te], feats, Z, N, False)
                pm = np.ones(Wtr.shape[1], bool); pm[0] = False
                th, _ = logit_fit(Wtr, d.y[tr].values.astype(float), ridge=ridge, pen_mask=pm)
                s = Wte @ th
                out.append(dict(rep=rep, fold=f, model=name,
                                auc=roc_auc_score(d.y[te], s),
                                ap=average_precision_score(d.y[te], s)))
            # unsupervised mechanistic baseline: decoupled FBA growth
            out.append(dict(rep=rep, fold=f, model="FBA score K (no fit)",
                            auc=roc_auc_score(d.y[te], d.K[te]),
                            ap=average_precision_score(d.y[te], d.K[te])))
    return pd.DataFrame(out)


def fit_full(d, nodes, feats, ridge=1e-2):
    """Pooled model (node traits) with dyadic-robust SE; returns table + eta."""
    Z = node_matrix(nodes)
    W = _W(d, feats, Z, len(nodes), False)
    pm = np.ones(W.shape[1], bool); pm[0] = False
    y = d.y.values.astype(float)
    th, H = logit_fit(W, y, ridge=ridge, pen_mask=pm)
    Vd = dyadic_vcov(W, y, th, d.producer.values, d.consumer.values, H)
    Vn = naive_vcov(H)
    names = (["intercept"] + [f"prod:{t}" for t in NODE_TRAITS]
             + [f"cons:{t}" for t in NODE_TRAITS] + feats)
    tab = pd.DataFrame({"theta": th, "se_naive": np.sqrt(np.diag(Vn)),
                        "se_dyadic": np.sqrt(np.clip(np.diag(Vd), 0, None))}, index=names)
    tab["beta_energy"] = -tab.theta              # energy coefficient (units of T)
    tab["z_dyadic"] = tab.theta / tab.se_dyadic
    eta = -(W @ th)                              # interaction energy, T = 1
    return tab, eta, th


# --------------------------------------------------------------------------
# consortium design
# --------------------------------------------------------------------------
def design_scores(d, eta, lib, n_cons=3, lam=2.0, top=None):
    # lam: crowding weight. lam=2.0 reproduces the article's benchmark; the
    # estimate from random consortia is 1.2 (SE 2.0), and the validated designs
    # are unchanged for any lam <= 3.6 (run_lambda.py, run_lambda_sensitivity.py).
    """Exhaustive scoring of consortia {1 producer + n_cons consumers}:

        H = sum_c eta(p, c) + lam * sum_{c<c'} overlap_p(c, c'),

    overlap_p(c, c') = 1 if both consumers rely on the same by-product of p
    (k_best of the pair features) + 0.5 if both need O2 left over by p.
    The overlap term is a crowding penalty: pairwise energies are estimated on
    isolated pairs and cannot see consumers competing for the same exchange."""
    E = {(p, c): e for p, c, e in zip(d.producer, d.consumer, eta)}
    kb = {(p, c): k for p, c, k in zip(d.producer, d.consumer, d.k_best)}
    aer = {c: lib[c].aerobic for c in d.consumer.unique()}
    prods = sorted(d.producer.unique()); cons = sorted(d.consumer.unique())
    recs = []
    for p in prods:
        cand = [c for c in cons if (p, c) in E]
        for trio in itertools.combinations(cand, n_cons):
            ov = 0.0
            for a, b in itertools.combinations(trio, 2):
                ov += float(kb[(p, a)] == kb[(p, b)] and kb[(p, a)] != "-")
                ov += 0.5 * float(aer[a] and aer[b])
            recs.append((sum(E[(p, c)] for c in trio) + lam * ov, p) + trio)
    cols = ["H", "producer"] + [f"c{k+1}" for k in range(n_cons)]
    out = pd.DataFrame(recs, columns=cols).sort_values("H").reset_index(drop=True)
    return out if top is None else out.head(top)


def consortium_energy(d, eta, lib, members, lam=2.0):
    """H of eq. (H) for one consortium (producer first); inf if a pair is absent."""
    E = {(p, c): e for p, c, e in zip(d.producer, d.consumer, eta)}
    kb = {(p, c): k for p, c, k in zip(d.producer, d.consumer, d.k_best)}
    p, cons = members[0], members[1:]
    if any((p, c) not in E for c in cons):
        return np.inf
    ov = 0.0
    for a, b in itertools.combinations(cons, 2):
        ov += float(kb[(p, a)] == kb[(p, b)] and kb[(p, a)] != "-")
        ov += 0.5 * float(lib[a].aerobic and lib[b].aerobic)
    return sum(E[(p, c)] for c in cons) + lam * ov


def validate_consortium(base, lib, D, members, time_limit=60.0):
    """4-member thermodynamics-based FBA: producer keeps >= 95% of the growth it
    reaches alone at the same abundance; success = every consumer grows."""
    p, cons = members[0], list(members[1:])
    x = 1.0 / len(members)
    ref = community_opt(base, [lib[p]], [x], D, thermo=True)
    if ref is None:
        return dict(success=0, min_cons_mu=0.0)
    bmin = [SELFISH * ref["bio"][0]] + [EPS_MU] * len(cons)
    r = community_opt(base, [lib[m] for m in members], [x] * len(members), D,
                      thermo=True, obj_bio=[0] + [1] * len(cons), bio_min=bmin,
                      time_limit=time_limit)
    if r is None:
        return dict(success=0, min_cons_mu=0.0)
    return dict(success=1, min_cons_mu=float(min(r["bio"][1:])))
