"""Monte Carlo study of the dyadic Fermi-Dirac (logit) interaction model.

(i)  incidental-parameter bias of the fixed-effect MLE and its parametric
     bootstrap correction, for n = 20, 40, 80 strains;
(ii) coverage of 95% Wald intervals with naive vs dyadic-robust variance in
     the pooled model where node heterogeneity is left in the error term.
"""
import sys, pickle, time
import numpy as np
from ecomdesign.stats import (bootstrap_bias_correct, dyad_design, dyadic_vcov,
                              logit_fit, naive_vcov, simulate_dyadic)

BETA = np.array([1.0, -0.5])
R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
B = int(sys.argv[2]) if len(sys.argv) > 2 else 50
rows = []
t0 = time.time()
for n in [20, 40, 80]:
    rng = np.random.default_rng(1000 + n)
    for r in range(R):
        iu, ju, X, y, a = simulate_dyadic(n, BETA, rng, node_sd=1.0, node_mean=-1.0)
        # --- fixed effects ---
        W = dyad_design(iu, ju, X, n, node_fe=True)
        idx = np.arange(n, n + 2)
        th, H = logit_fit(W, y, ridge=1e-6)
        bc, _ = bootstrap_bias_correct(W, th, idx, n_boot=B, ridge=1e-6, rng=rng)
        se_fe = np.sqrt(np.diag(naive_vcov(H))[idx])
        # --- pooled model (node effects in the error) ---
        W0 = dyad_design(iu, ju, X, n, node_fe=False)
        th0, H0 = logit_fit(W0, y, ridge=0.0)
        Vn, Vd = naive_vcov(H0), dyadic_vcov(W0, y, th0, iu, ju, H0)
        # pseudo-true value of the pooled model: fit on a very large sample is
        # costly; use the attenuation-free target only for coverage of the FE
        rows.append(dict(n=n, rep=r, fe1=th[idx[0]], fe2=th[idx[1]], bc1=bc[0], bc2=bc[1],
                         se_fe1=se_fe[0], se_fe2=se_fe[1],
                         pool1=th0[1], pool2=th0[2],
                         se_naive1=np.sqrt(Vn[1, 1]), se_naive2=np.sqrt(Vn[2, 2]),
                         se_dyad1=np.sqrt(max(Vd[1, 1], 0)), se_dyad2=np.sqrt(max(Vd[2, 2], 0)),
                         prev=y.mean()))
    print(n, "done", round(time.time() - t0), "s", flush=True)
pickle.dump({"rows": rows, "beta": BETA}, open("data/montecarlo.pkl", "wb"))
