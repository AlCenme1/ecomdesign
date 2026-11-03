"""Benchmark, inference, REM and design validation on a stored library."""
import sys, pickle, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from ecomdesign.metabolism import load_base, set_ext_ceiling
from ecomdesign.library import design_medium
from ecomdesign.analysis import (FEATURE_SETS, cv_benchmark, design_scores, fit_full,
                                 consortium_energy,
                                 prepare, validate_consortium)
from ecomdesign.stats import rem_diagnostics, rem_lnQ_theory

lib_path = sys.argv[1]; tag = sys.argv[2]
n_rep = int(sys.argv[3]) if len(sys.argv) > 3 else 20
n_val = int(sys.argv[4]) if len(sys.argv) > 4 else 30
o = pickle.load(open(lib_path, "rb"))
nodes, lib = o["nodes"], o["lib"]
b = load_base(); D = design_medium(b)
out = {"cv": {}, "fit": {}, "rem": {}, "val": []}
t0 = time.time()
for c, r in o["res"].items():
    d = prepare(r["pairs"], nodes)
    out["cv"][c] = cv_benchmark(d, nodes, n_repeats=n_rep)
    for name, feats in FEATURE_SETS.items():
        tab, eta, th = fit_full(d, nodes, feats, ridge=0.1)
        out["fit"][(c, name)] = (tab, eta)
    print(c, "cv+fit", round(time.time() - t0), "s", flush=True)
    print(out["cv"][c].groupby("model")[["auc", "ap"]].mean().round(3), flush=True)

# REM diagnostics on the M3 energies at the 0.1 mM ceiling
c_rem = 1e-4
eta = out["fit"][(c_rem, "M3: + thermodynamic window")][1]
temps = np.logspace(-1.5, 1.5, 80)
diag = rem_diagnostics(eta, temps)
theory, Tc = rem_lnQ_theory(eta.mean(), eta.std(), len(eta), temps)
out["rem"] = dict(diag=diag, theory=theory, temps=temps, ceiling=c_rem, eta=eta)
print("REM: sigma", round(diag["sigma"], 2), "Tc", round(diag["Tc"], 3), flush=True)

# consortium design and mechanistic validation
rng = np.random.default_rng(3)
for c in [1e-2, 1e-4]:
    set_ext_ceiling(c)
    d = prepare(o["res"][c]["pairs"], nodes)
    designs = {}
    for lab, name in (("M2 design", "M2: + stoichiometric complementarity"),
                      ("M3 design", "M3: + thermodynamic window")):
        sc = design_scores(d, out["fit"][(c, name)][1], lib, top=None)
        # lowest-energy consortia, greedily diversified: a design is kept only if
        # it shares at most two members with every design already kept
        kept = []
        for x in sc[["producer", "c1", "c2", "c3"]].values:
            if all(len(set(x) & set(k)) <= 2 for k in kept):
                kept.append(tuple(int(v) for v in x))
            if len(kept) == n_val:
                break
        designs[lab] = kept
    prods = sorted(d.producer.unique()); cons = sorted(d.consumer.unique())
    designs["random"] = [(int(rng.choice(prods)),) + tuple(int(x) for x in rng.choice(cons, 3, replace=False))
                         for _ in range(n_val)]
    for lab, mem_list in designs.items():
        for m in mem_list:
            v = validate_consortium(b, lib, D, [int(x) for x in m])
            hs = {f"H_{k[:2]}": consortium_energy(d, out["fit"][(c, k)][1], lib, list(m))
                  for k in FEATURE_SETS if k[:2] in ("M2", "M3")}
            out["val"].append(dict(ceiling=c, strategy=lab, members=m, **v, **hs))
    vv = pd.DataFrame(out["val"]); vv = vv[vv.ceiling == c]
    print(c, vv.groupby("strategy").success.mean().round(3).to_dict(), round(time.time() - t0), "s", flush=True)
out["val"] = pd.DataFrame(out["val"])
pickle.dump(out, open(f"data/analysis_{tag}.pkl", "wb"))
