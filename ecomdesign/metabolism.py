"""
Metabolic layer of the ECOMDESIGN pipeline.

Builds a library of strains from a genome-scale (here: E. coli core) model,
solves single-strain and fixed-abundance community FBA problems with HiGHS,
and computes Max-min Driving Force (MDF; Noor et al. 2014) for single strains
and for pairs that share an extracellular metabolite pool.

All linear programs are written directly on the stoichiometric matrix so that
community problems (block-diagonal intracellular part + shared extracellular
rows) are fast enough to evaluate every pair of a library.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

R_GAS = 8.314462618e-3  # kJ mol^-1 K^-1
T_K = 298.15
RT = R_GAS * T_K

# carbon source -> uptake transporter (forward direction = uptake)
CARBON_TRANSPORT = {
    "glc__D": "GLCpts", "fru": "FRUpts2", "ac": "ACt2r", "acald": "ACALDt",
    "akg": "AKGt2r", "etoh": "ETOHt2r", "lac__D": "D_LACt2", "pyr": "PYRt2",
    "succ": "SUCCt2_2", "fum": "FUMt2_2", "mal__L": "MALt2_2",
    "glu__L": "GLUt2r", "gln__L": "GLNabc",
}
CARBONS = list(CARBON_TRANSPORT)
INORGANIC = ["co2", "h", "h2o", "nh4", "pi"]
# central-carbon reactions eligible for knockout when building the library
KO_POOL = ["PFK", "PYK", "PDH", "CS", "ICDHyr", "SUCDi", "FUM", "MDH", "PGI",
           "G6PDH2r", "PTAr", "ACKr", "ALCD2x", "LDH_D", "PFL", "ME1", "ME2",
           "PPC", "PPCK", "AKGDH", "ICL", "MALS", "FBP", "PPS", "TKT1", "GND"]
# Membrane- and quinone-coupled reactions: their dG depends on the proton-motive
# force and on the quinone-pool redox state, neither of which is modelled.
# They are left without thermodynamic constraints (standard practice in MDF).
MDF_EXCLUDE = {"ATPS4r", "THD2", "NADH16", "CYTBD", "FRD7", "SUCDi"}
LN_CMIN, LN_CMAX = np.log(1e-6), np.log(1e-2)   # 1 uM .. 10 mM
# Cofactor concentrations fixed at values measured in glucose-fed E. coli
# (Bennett et al. 2009); free cofactor ratios would make MDF meaningless.
FIXED_CONC = {"atp_c": 9.6e-3, "adp_c": 5.6e-4, "amp_c": 2.8e-4,
              "nad_c": 2.6e-3, "nadh_c": 8.3e-5, "nadp_c": 2.1e-6,
              "nadph_c": 1.2e-4, "pi_c": 1.0e-2, "co2_c": 1.0e-5}
# inorganic medium components may exceed 10 mM
EXT_MAX = {"pi_e": np.log(5e-2), "nh4_e": np.log(5e-2), "glc__D_e": np.log(5e-2)}


# Ceiling for extracellular organic metabolites (exchanged by-products).
# Physiological co-culture concentrations are often far below 10 mM; this
# ceiling is the main sensitivity parameter of the thermodynamic layer.
_STATE = {"ext_ceiling": np.log(1e-2)}


def set_ext_ceiling(conc_molar: float):
    _STATE["ext_ceiling"] = float(np.log(conc_molar))


def lnc_bounds(met_id):
    if met_id in FIXED_CONC:
        v = np.log(FIXED_CONC[met_id]); return (v, v)
    if met_id in EXT_MAX:
        return (LN_CMIN, EXT_MAX[met_id])
    if met_id.endswith("_e") and met_id not in ("o2_e", "co2_e"):
        return (LN_CMIN, min(_STATE["ext_ceiling"], LN_CMAX))
    return (LN_CMIN, LN_CMAX)


@dataclass
class Base:
    rxn_ids: list
    met_ids: list
    S: sp.csr_matrix               # mets x rxns (exchanges removed)
    lb: np.ndarray
    ub: np.ndarray
    is_ext: np.ndarray             # bool per metabolite
    bio: int                       # biomass column
    dg0: np.ndarray                # standard dG'0 per reaction (nan = unknown)
    ext_ids: list = field(default_factory=list)

    @property
    def ext_idx(self):
        return np.where(self.is_ext)[0]

    @property
    def int_idx(self):
        return np.where(~self.is_ext)[0]


def load_base(dg_path: str | None = None) -> Base:
    from cobra.io import load_model
    m = load_model("textbook")
    rx = [r for r in m.reactions if not r.boundary]
    mets = list(m.metabolites)
    midx = {mt.id: k for k, mt in enumerate(mets)}
    rows, cols, vals = [], [], []
    for j, r in enumerate(rx):
        for mt, c in r.metabolites.items():
            rows.append(midx[mt.id]); cols.append(j); vals.append(c)
    S = sp.csr_matrix((vals, (rows, cols)), shape=(len(mets), len(rx)))
    lb = np.array([r.lower_bound for r in rx], float)
    ub = np.array([r.upper_bound for r in rx], float)
    is_ext = np.array([mt.compartment == "e" for mt in mets])
    bio = [r.id for r in rx].index("Biomass_Ecoli_core")
    dg_path = dg_path or os.path.join(os.path.dirname(__file__), "dg0_ecoli_core.json")
    dg = json.load(open(dg_path))
    dg0 = np.array([dg[r.id][0] if r.id in dg else np.nan for r in rx])
    dg0[bio] = np.nan
    return Base([r.id for r in rx], [mt.id for mt in mets], S, lb, ub, is_ext,
                bio, dg0, [mt.id for mt in mets if mt.compartment == "e"])


@dataclass
class Strain:
    name: str
    carbons: tuple
    aerobic: bool
    knockouts: tuple
    lb: np.ndarray
    ub: np.ndarray


def make_strain(base: Base, name, carbons, aerobic, knockouts) -> Strain:
    lb, ub = base.lb.copy(), base.ub.copy()
    ridx = {r: k for k, r in enumerate(base.rxn_ids)}
    for c, t in CARBON_TRANSPORT.items():
        if c not in carbons:
            ub[ridx[t]] = 0.0           # block uptake, keep export (if reversible)
    if not aerobic:
        lb[ridx["O2t"]] = ub[ridx["O2t"]] = 0.0
    for k in knockouts:
        lb[ridx[k]] = ub[ridx[k]] = 0.0
    return Strain(name, tuple(carbons), aerobic, tuple(knockouts), lb, ub)


def medium(base: Base, carbon_supply: dict, o2: float) -> dict:
    """Uptake capacity (>0) per extracellular metabolite id, community units."""
    med = {f"{m}_e": 1000.0 for m in INORGANIC}
    med["o2_e"] = o2
    for c, v in carbon_supply.items():
        med[f"{c}_e"] = v
    return med


# --------------------------------------------------------------------------
# community FBA with fixed abundances (SteadyCom-type, balanced growth)
# --------------------------------------------------------------------------
def community_fba(base: Base, strains: list, abund, med: dict,
                  pfba: bool = True, mu_fix: float | None = None):
    """Maximise the common growth rate mu of a community with fixed abundances.

    Variables: per-capita fluxes v_i (n each), exchange x (n_ext, >0 = export to
    the environment), mu. Returns dict or None if infeasible.
    If mu_fix is given, growth is fixed and only the pFBA stage is solved.
    """
    n, ne, ns = len(base.rxn_ids), len(base.ext_idx), len(strains)
    Sint = base.S[base.int_idx]
    Sext = base.S[base.ext_idx]
    nv = n * ns
    # variable order: [v_1..v_ns, x, mu]
    blocks_int = sp.block_diag([Sint] * ns, format="csr")
    Aext = sp.hstack([abund[i] * Sext for i in range(ns)] + [-sp.eye(ne), sp.csr_matrix((ne, 1))])
    Aint = sp.hstack([blocks_int, sp.csr_matrix((blocks_int.shape[0], ne + 1))])
    # biomass_i - mu = 0
    rows = []
    for i in range(ns):
        r = sp.lil_matrix((1, nv + ne + 1))
        r[0, i * n + base.bio] = 1.0
        r[0, nv + ne] = -1.0
        rows.append(r.tocsr())
    Abio = sp.vstack(rows)
    Aeq = sp.vstack([Aint, Aext, Abio], format="csr")
    beq = np.zeros(Aeq.shape[0])
    lb = np.concatenate([s.lb for s in strains] + [np.zeros(ne), [0.0]])
    ub = np.concatenate([s.ub for s in strains] + [np.full(ne, 1000.0), [1000.0]])
    for k, mi in enumerate(base.ext_idx):
        lb[nv + k] = -med.get(base.met_ids[mi], 0.0)
    c = np.zeros(nv + ne + 1); c[-1] = -1.0
    if mu_fix is None:
        res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=list(zip(lb, ub)), method="highs")
        if res.status != 0:
            return None
        mu = res.x[-1]
    else:
        mu = mu_fix
    out = {"mu": float(mu)}
    if not pfba or mu < 1e-6:
        if mu_fix is None:
            out["v"] = [res.x[i * n:(i + 1) * n] for i in range(ns)]
            out["x"] = res.x[nv:nv + ne]
        return out if "v" in out else None
    # pFBA: fix mu, minimise sum_i abund_i * |v_i|
    lb2, ub2 = lb.copy(), ub.copy()
    lb2[-1] = ub2[-1] = mu * 0.999
    nt = nv
    Aeq2 = sp.hstack([Aeq, sp.csr_matrix((Aeq.shape[0], nt))], format="csr")
    I = sp.eye(nv); Z = sp.csr_matrix((nv, ne + 1))
    Aub = sp.vstack([sp.hstack([I, Z, -I]), sp.hstack([-I, Z, -I])], format="csr")
    bub = np.zeros(2 * nv)
    w = np.concatenate([np.full(n, abund[i]) for i in range(ns)])
    c2 = np.concatenate([np.zeros(nv + ne + 1), w])
    bnds = list(zip(lb2, ub2)) + [(0, None)] * nt
    res2 = linprog(c2, A_ub=Aub, b_ub=bub, A_eq=Aeq2, b_eq=beq, bounds=bnds, method="highs")
    if res2.status != 0:
        return None
    xx = res2.x
    out["v"] = [xx[i * n:(i + 1) * n] for i in range(ns)]
    out["x"] = xx[nv:nv + ne]
    return out


def single_fba(base, strain, med, pfba=True):
    return community_fba(base, [strain], [1.0], med, pfba=pfba)


def fva_exchange(base, strain, med, frac=0.9):
    """Max secretion of each extracellular metabolite at >= frac * mu_max."""
    sol = single_fba(base, strain, med, pfba=False)
    n, ne = len(base.rxn_ids), len(base.ext_idx)
    out = np.zeros(ne)
    if sol is None or sol["mu"] < 1e-3:
        return out, 0.0
    mu = sol["mu"]
    Sint, Sext = base.S[base.int_idx], base.S[base.ext_idx]
    Aeq = sp.vstack([sp.hstack([Sint, sp.csr_matrix((Sint.shape[0], ne))]),
                     sp.hstack([Sext, -sp.eye(ne)])], format="csr")
    beq = np.zeros(Aeq.shape[0])
    lb = np.concatenate([strain.lb, np.zeros(ne)])
    ub = np.concatenate([strain.ub, np.full(ne, 1000.0)])
    for k, mi in enumerate(base.ext_idx):
        lb[n + k] = -med.get(base.met_ids[mi], 0.0)
    lb[base.bio] = frac * mu
    for k in range(ne):
        c = np.zeros(n + ne); c[n + k] = -1.0
        r = linprog(c, A_eq=Aeq, b_eq=beq, bounds=list(zip(lb, ub)), method="highs")
        out[k] = max(0.0, r.x[n + k]) if r.status == 0 else 0.0
    return out, mu


# --------------------------------------------------------------------------
# Thermodynamic layer: shared concentration variables, MDF and TFA
# --------------------------------------------------------------------------
_SKIP = {"h_c", "h_e", "h2o_c", "h2o_e"}


class ThermoLayout:
    """Log-concentration variables for `ns` strains sharing one extracellular
    pool. Intracellular metabolites are strain specific."""

    def __init__(self, base: Base, ns: int):
        mets = base.met_ids
        keep = [k for k, m in enumerate(mets) if m not in _SKIP]
        self.ext_keep = [k for k in keep if base.is_ext[k]]
        self.int_keep = [k for k in keep if not base.is_ext[k]]
        self.col, self.bounds = {}, []
        for a, k in enumerate(self.ext_keep):
            for i in range(ns):
                self.col[(i, k)] = a
            self.bounds.append(lnc_bounds(mets[k]))
        off = len(self.ext_keep)
        for i in range(ns):
            for a, k in enumerate(self.int_keep):
                self.col[(i, k)] = off + a
            self.bounds += [lnc_bounds(mets[k]) for k in self.int_keep]
            off += len(self.int_keep)
        self.n = off
        self.ext_col = {base.met_ids[k]: a for a, k in enumerate(self.ext_keep)}
        S = base.S.tocsc()
        # thermodynamic rows: (strain, reaction) -> dense coefficient vector RT*S
        self.rows = {}
        for i in range(ns):
            for j, rid in enumerate(base.rxn_ids):
                if np.isnan(base.dg0[j]) or rid in MDF_EXCLUDE:
                    continue
                row = np.zeros(self.n)
                sl = S[:, j]
                for k, c in zip(sl.indices, sl.data):
                    if (i, k) in self.col:
                        row[self.col[(i, k)]] += RT * c
                if np.any(row):
                    self.rows[(i, j)] = row


def mdf(base: Base, fluxes: list, tol: float = 1e-6):
    """Max-min driving force (kJ/mol) of given flux patterns (one per strain)
    with a shared extracellular pool. Returns (B, n_constraints)."""
    L = ThermoLayout(base, len(fluxes))
    A, b = [], []
    for (i, j), row in L.rows.items():
        v = fluxes[i][j]
        if abs(v) < tol:
            continue
        d = np.sign(v)
        A.append(np.append(d * row, 1.0)); b.append(-d * base.dg0[j])
    if not A:
        return np.nan, 0
    c = np.zeros(L.n + 1); c[-1] = -1.0
    r = linprog(c, A_ub=np.array(A), b_ub=np.array(b),
                bounds=L.bounds + [(None, None)], method="highs")
    return (float(r.x[-1]) if r.status == 0 else np.nan), len(A)


def community_opt(base: Base, strains: list, abund, med: dict, thermo: bool = True,
                  balanced: bool = False, obj_bio=None, bio_min=None,
                  ext_target: str | None = None, sense: int = -1,
                  export_min: dict | None = None, uptake_min: dict | None = None,
                  delta: float = 0.1, time_limit: float = 30.0,
                  obj_ext: dict | None = None):
    """(Thermodynamics-based) FBA for a fixed-abundance community sharing one
    extracellular pool (Henry et al. 2007 formulation, big-M MILP).

    balanced=True  : all strains share one growth rate (biomass_i = mu).
    obj_bio        : weights on strain biomass fluxes to maximise (default: all 1).
    bio_min        : per-strain lower bounds on biomass flux.
    ext_target     : if given, optimise ln[ext_target] instead (sense -1 max, +1 min).
    obj_ext        : {ext_met_id: weight}; if given, maximise the weighted community
                     export fluxes instead (used for FVA of exchanges).
    thermo=False   : plain FBA (no concentration variables / binaries).
    Active thermodynamically constrained reactions need dG <= -delta.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    n, ne, ns = len(base.rxn_ids), len(base.ext_idx), len(strains)
    L = ThermoLayout(base, ns) if thermo else None
    keys = list(L.rows) if thermo else []
    nb, nc = len(keys), (L.n if thermo else 0)
    nv = n * ns
    o_x, o_mu, o_c = nv, nv + ne, nv + ne + 1
    o_zf = o_c + nc
    o_zr = o_zf + nb
    nvar = o_zr + nb
    Sint, Sext = base.S[base.int_idx], base.S[base.ext_idx]
    blocks = sp.block_diag([Sint] * ns, format="csr")
    A1 = sp.hstack([blocks, sp.csr_matrix((blocks.shape[0], nvar - nv))])
    A2 = sp.hstack([sp.hstack([abund[i] * Sext for i in range(ns)]), -sp.eye(ne),
                    sp.csr_matrix((ne, nvar - nv - ne))])
    eqs = [A1, A2]
    if balanced:
        Ab = sp.lil_matrix((ns, nvar))
        for i in range(ns):
            Ab[i, i * n + base.bio] = 1.0; Ab[i, o_mu] = -1.0
        eqs.append(Ab.tocsr())
    cons = [LinearConstraint(sp.vstack(eqs, format="csr"), 0.0, 0.0)]
    if thermo and nb:
        M, K = 1000.0, 1000.0
        T = sp.lil_matrix((5 * nb, nvar)); lo = np.full(5 * nb, -np.inf); hi = np.zeros(5 * nb)
        for q, (i, j) in enumerate(keys):
            vcol = i * n + j
            row = L.rows[(i, j)]
            nz = np.nonzero(row)[0]
            T[5*q, vcol] = 1.0; T[5*q, o_zf + q] = -M
            T[5*q+1, vcol] = -1.0; T[5*q+1, o_zr + q] = -M
            T[5*q+2, o_zf + q] = 1.0; T[5*q+2, o_zr + q] = 1.0; hi[5*q+2] = 1.0
            for c in nz:
                T[5*q+3, o_c + c] = row[c]
                T[5*q+4, o_c + c] = -row[c]
            T[5*q+3, o_zf + q] = K; hi[5*q+3] = K - delta - base.dg0[j]
            T[5*q+4, o_zr + q] = K; hi[5*q+4] = K - delta + base.dg0[j]
        cons.append(LinearConstraint(T.tocsr(), lo, hi))
    lb = np.concatenate([s.lb for s in strains] + [np.zeros(ne), [0.0],
                        [b[0] for b in (L.bounds if thermo else [])], np.zeros(2 * nb)])
    ub = np.concatenate([s.ub for s in strains] + [np.full(ne, 1000.0), [1000.0],
                        [b[1] for b in (L.bounds if thermo else [])], np.ones(2 * nb)])
    ext_ids = [base.met_ids[k] for k in base.ext_idx]
    for k, mid in enumerate(ext_ids):
        lb[o_x + k] = -med.get(mid, 0.0)
    for mid, f in (export_min or {}).items():
        lb[o_x + ext_ids.index(mid)] = max(lb[o_x + ext_ids.index(mid)], f)
    for mid, f in (uptake_min or {}).items():
        ub[o_x + ext_ids.index(mid)] = min(ub[o_x + ext_ids.index(mid)], -f)
    for i, bm in enumerate(bio_min or [0.0] * ns):
        lb[i * n + base.bio] = max(lb[i * n + base.bio], bm)
    integ = np.zeros(nvar); integ[o_zf:] = 1
    c = np.zeros(nvar)
    if ext_target is not None:
        c[o_c + L.ext_col[ext_target]] = float(sense)
    elif obj_ext is not None:
        for mid, w in obj_ext.items():
            c[o_x + ext_ids.index(mid)] = -w
    else:
        w = obj_bio if obj_bio is not None else [1.0] * ns
        for i in range(ns):
            c[i * n + base.bio] = -w[i]
    res = milp(c, constraints=cons, integrality=integ if nb else None,
               bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-4, "disp": False})
    if res.x is None:
        return None
    x = res.x
    out = {"bio": [float(x[i * n + base.bio]) for i in range(ns)],
           "v": [x[i * n:(i + 1) * n] for i in range(ns)],
           "x": x[o_x:o_x + ne], "status": res.status}
    out["mu"] = float(x[o_mu]) if balanced else out["bio"][0]
    if thermo:
        out["lnc_ext"] = {m: x[o_c + a] for m, a in L.ext_col.items()}
    return out


def tfa(base, strains, abund, med, objective="mu", mu_min=0.0, ext_target=None,
        sense=-1, export_min=None, uptake_min=None, delta=0.1, time_limit=30.0):
    """Balanced-growth TFA (thin wrapper kept for single-strain use)."""
    r = community_opt(base, strains, abund, med, thermo=True, balanced=True,
                      bio_min=[mu_min] * len(strains),
                      ext_target=ext_target if objective == "lnc" else None,
                      sense=sense, export_min=export_min, uptake_min=uptake_min,
                      delta=delta, time_limit=time_limit)
    return r
