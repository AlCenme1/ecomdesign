"""
Statistical layer of the ECOMDESIGN pipeline.

* Fermi-Dirac (logistic) dyadic interaction model
      P(y_ij = 1) = 1 / (1 + exp{(eta_ij - mu)/T}),
      eta_ij = a_i + a_j + x_ij' beta   (energy of the interaction),
  identified up to scale (beta/T); we set T = 1.
* Estimators: fixed-effect MLE (degree heterogeneity), its parametric-bootstrap
  bias correction, and a node-covariate version for out-of-sample prediction.
* Dyadic-robust (Fafchamps-Gubert / Aronow et al. 2015) sandwich variance.
* Random-energy-model (REM) diagnostics of the Boltzmann weights over pairs.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


# --------------------------------------------------------------------------
# design matrices
# --------------------------------------------------------------------------
def dyad_design(pairs_i, pairs_j, X, n_nodes, node_fe=True, Z=None):
    """Columns: [node dummies (a_i + a_j) | X] or [1 | Z_i + Z_j | X].
    With node dummies the intercept is absorbed; one node is the reference."""
    P = len(pairs_i)
    if node_fe:
        D = np.zeros((P, n_nodes))
        D[np.arange(P), pairs_i] += 1.0
        D[np.arange(P), pairs_j] += 1.0
        return np.hstack([D, X])
    blocks = [np.ones((P, 1))]
    if Z is not None:
        blocks.append(Z[pairs_i] + Z[pairs_j])
    blocks.append(X)
    return np.hstack(blocks)


# --------------------------------------------------------------------------
# logistic MLE with optional ridge penalty (separation guard)
# --------------------------------------------------------------------------
def logit_fit(W, y, ridge=0.0, pen_mask=None, theta0=None):
    """Maximise sum_l [y log p + (1-y) log(1-p)] - ridge/2 ||theta[pen_mask]||^2,
    p = expit(W theta) (theta = -beta: higher score = lower interaction energy)."""
    k = W.shape[1]
    pm = np.ones(k, bool) if pen_mask is None else pen_mask

    def f(t):
        z = W @ t
        ll = np.sum(y * z - np.logaddexp(0.0, z))
        return -(ll - 0.5 * ridge * np.sum(t[pm] ** 2))

    def g(t):
        p = expit(W @ t)
        gr = W.T @ (y - p) - ridge * np.where(pm, t, 0.0)
        return -gr

    def h(t):
        p = expit(W @ t)
        H = (W * (p * (1 - p))[:, None]).T @ W + ridge * np.diag(pm.astype(float))
        return H

    t0 = np.zeros(k) if theta0 is None else theta0
    r = minimize(f, t0, jac=g, hess=h, method="trust-exact",
                 options={"gtol": 1e-8, "maxiter": 500})
    return r.x, h(r.x)


def dyadic_vcov(W, y, theta, pairs_i, pairs_j, hess):
    """Dyadic-robust sandwich: V = H^-1 M H^-1 with
    M = sum over dyad pairs (l, m) sharing at least one node of s_l s_m'."""
    p = expit(W @ theta)
    S = W * (y - p)[:, None]                 # scores per dyad
    n = max(pairs_i.max(), pairs_j.max()) + 1
    # sum of scores by node; dyads sharing a node are captured by node sums
    node_sum = np.zeros((n, W.shape[1]))
    np.add.at(node_sum, pairs_i, S)
    np.add.at(node_sum, pairs_j, S)
    # sum_{l,m share node} s_l s_m' = sum_v (sum_{l in v} s_l)(...)'
    #   - sum_l s_l s_l'   (a dyad shares both of its nodes with itself)
    M = node_sum.T @ node_sum - S.T @ S
    Hi = np.linalg.pinv(hess)
    return Hi @ M @ Hi


def naive_vcov(hess):
    return np.linalg.pinv(hess)


def bootstrap_bias_correct(W, theta, idx, n_boot=200, ridge=0.0, pen_mask=None, rng=None):
    """Parametric bootstrap bias correction for the incidental-parameter bias
    of the fixed-effect dyadic logit: theta_bc = 2 theta - mean(theta*)."""
    rng = np.random.default_rng(rng)
    p = expit(W @ theta)
    draws = []
    for _ in range(n_boot):
        yb = (rng.random(len(p)) < p).astype(float)
        tb, _ = logit_fit(W, yb, ridge=ridge, pen_mask=pen_mask, theta0=theta)
        draws.append(tb[idx])
    draws = np.array(draws)
    return 2 * theta[idx] - draws.mean(axis=0), draws


# --------------------------------------------------------------------------
# simulation from the structural model
# --------------------------------------------------------------------------
def simulate_dyadic(n, beta, rng, node_sd=1.0, node_mean=-1.0, x_sd=1.0, corr_node_x=0.0):
    """y_ij ~ Bernoulli(expit(a_i + a_j + x_ij' theta)); x has p columns.
    Covariates contain a node-level component so that dyadic dependence exists."""
    iu, ju = np.triu_indices(n, 1)
    a = node_mean + node_sd * rng.standard_normal(n)
    p = len(beta)
    u = rng.standard_normal((n, p))
    X = x_sd * (rng.standard_normal((len(iu), p)) + u[iu] + u[ju]) / np.sqrt(3)
    if corr_node_x:
        X[:, 0] += corr_node_x * (a[iu] + a[ju])
    z = a[iu] + a[ju] + X @ beta
    y = (rng.random(len(z)) < expit(z)).astype(float)
    return iu, ju, X, y, a


# --------------------------------------------------------------------------
# Random Energy Model diagnostics
# --------------------------------------------------------------------------
def rem_diagnostics(eta, temps):
    """Boltzmann weights w_ij(T) = exp(-eta_ij/T)/Q over pairs.
    Returns participation ratio PR(T) = 1/sum w^2 (effective number of pairs),
    entropy and the REM critical temperature T_c = sigma/sqrt(2 ln n)."""
    eta = np.asarray(eta, float)
    n = len(eta)
    sig = eta.std()
    Tc = sig / np.sqrt(2 * np.log(n))
    PR, H, lnQ = [], [], []
    for T in temps:
        z = -(eta - eta.min()) / T
        w = np.exp(z); Q = w.sum(); w /= Q
        PR.append(1.0 / np.sum(w ** 2))
        H.append(-np.sum(w * np.log(w + 1e-300)))
        lnQ.append(np.log(Q) - eta.min() / T)
    return dict(Tc=Tc, sigma=sig, n=n, PR=np.array(PR), H=np.array(H), lnQ=np.array(lnQ))


def rem_lnQ_theory(mu, sigma, n, temps):
    """Annealed (high-T) and frozen (low-T) REM approximations of ln Q."""
    temps = np.asarray(temps, float)
    Tc = sigma / np.sqrt(2 * np.log(n))
    high = np.log(n) - mu / temps + sigma ** 2 / (2 * temps ** 2)
    low = (-mu + sigma * np.sqrt(2 * np.log(n))) / temps
    return np.where(temps > Tc, high, low), Tc
