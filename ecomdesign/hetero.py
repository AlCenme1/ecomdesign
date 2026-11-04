"""
Heterogeneous-taxon layer: real GSMMs (different stoichiometric matrices)
sharing one extracellular pool.

* split_community(): turns a merged community SBML (one compartment set per
  taxon, one shared extracellular compartment) into per-taxon networks.
* TaxonNet: arrays for one taxon (S over its own metabolites, bounds, biomass,
  standard Gibbs energies, osmotic-solute demand).
* community_opt_h(): FBA / thermodynamics-based FBA for any set of TaxonNets
  with fixed abundances (generalises metabolism.community_opt).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .metabolism import LN_CMAX, LN_CMIN, RT, _STATE, EXT_MAX

# compatible solutes a salt-out strategist may accumulate (BiGG ids, cytosol)
OSMOLYTES = ["ectoine_c", "glyb_c", "pro__L_c", "tre_c", "glu__L_c"]
# fixed cofactor pools (Bennett et al. 2009), cytosol only
FIXED_C = {"atp_c": 9.6e-3, "adp_c": 5.6e-4, "amp_c": 2.8e-4, "nad_c": 2.6e-3,
           "nadh_c": 8.3e-5, "nadp_c": 2.1e-6, "nadph_c": 1.2e-4, "pi_c": 1.0e-2,
           "co2_c": 1.0e-5}
SKIP = {"h_c", "h_p", "h_e", "h2o_c", "h2o_p", "h2o_e"}
QUINONES = re.compile(r"^(q8|q8h2|mqn8|mql8|2dmmq8|2dmmql8|mqn7|mql7|mqn9|mql9|q9|q9h2|mk\w*)_")


@dataclass
class TaxonNet:
    name: str
    rxn_ids: list
    met_ids: list
    S: sp.csr_matrix
    lb: np.ndarray
    ub: np.ndarray
    bio: int
    is_ext: np.ndarray
    dg0: np.ndarray = None
    dg0_se: np.ndarray = None
    thermo_ok: np.ndarray = None     # reactions that receive dG constraints
    osm_cols: list = field(default_factory=list)   # demand columns of osmolytes
    group: str = "resident"
    strategy: str = "salt-out"       # salt-in / salt-out (osmoadaptation)

    @property
    def n(self):
        return len(self.rxn_ids)


def _strip(s, suffix):
    return s[: -len(suffix) - 1] if s.endswith("_" + suffix) else s


def split_community(model):
    """Return {taxon_short_name: cobra.Model} from a merged community model."""
    import cobra
    suffixes = [r.id[len("Growth_"):] for r in model.reactions if r.id.startswith("Growth_")]
    out = {}
    for suf in suffixes:
        name = suf.split("_")[0]
        m = cobra.Model(name)
        rxs = [r for r in model.reactions if r.id.endswith("_" + suf)]
        new = []
        for r in rxs:
            nr = cobra.Reaction(_strip(r.id, suf), name=r.name,
                                lower_bound=r.lower_bound, upper_bound=r.upper_bound)
            mets = {}
            for met, c in r.metabolites.items():
                if met.id.startswith("community_biomass"):
                    continue
                mid = _strip(met.id, suf)
                if met.compartment == "ext":
                    comp = "e"
                elif met.compartment.startswith("C_p"):
                    comp = "p"
                else:
                    comp = "c"
                if mid not in m.metabolites:
                    mm = cobra.Metabolite(mid, name=met.name, compartment=comp,
                                          formula=met.formula, charge=met.charge)
                    mm.annotation = dict(met.annotation)
                    m.add_metabolites([mm])
                mets[m.metabolites.get_by_id(mid)] = c
            nr.add_metabolites(mets)
            new.append(nr)
        m.add_reactions(new)
        m.objective = "Growth"
        out[name] = m
    return out


def to_net(model, name, bio_id, group="resident", strategy="salt-out", bound=100.0):
    """Arrays for one taxon. Boundary reactions are dropped (the environment is
    modelled by the community-level exchange variables). One demand reaction per
    available osmolyte is appended (bounds 0..1000, linked to growth later)."""
    # keep demand/sink reactions on internal metabolites (e.g. DM_4crsol_c in
    # iJO1366, needed to remove biomass by-products); drop true exchanges only
    rx = [r for r in model.reactions
          if not r.boundary or all(m.compartment != "e" for m in r.metabolites)]
    mets = list(model.metabolites)
    idx = {m.id: k for k, m in enumerate(mets)}
    rows, cols, vals = [], [], []
    for j, r in enumerate(rx):
        for mt, c in r.metabolites.items():
            rows.append(idx[mt.id]); cols.append(j); vals.append(c)
    rxn_ids = [r.id for r in rx]
    lb = np.clip([r.lower_bound for r in rx], -bound, bound).astype(float)
    ub = np.clip([r.upper_bound for r in rx], -bound, bound).astype(float)
    osm = [o for o in OSMOLYTES if o in idx]
    for o in osm:
        j = len(rxn_ids)
        rxn_ids.append(f"DM_OSM_{o}")
        rows.append(idx[o]); cols.append(j); vals.append(-1.0)
        lb = np.append(lb, 0.0); ub = np.append(ub, 1000.0)
    S = sp.csr_matrix((vals, (rows, cols)), shape=(len(mets), len(rxn_ids)))
    is_ext = np.array([m.compartment == "e" for m in mets])
    net = TaxonNet(name, rxn_ids, [m.id for m in mets], S, lb, ub,
                   rxn_ids.index(bio_id), is_ext, group=group, strategy=strategy)
    net.osm_cols = [rxn_ids.index(f"DM_OSM_{o}") for o in osm]
    return net


def reaction_signature(model, rid):
    """Compartment-collapsed stoichiometry used for dG'0 look-up."""
    r = model.reactions.get_by_id(rid)
    st = {}
    for met, c in r.metabolites.items():
        base = met.id.rsplit("_", 1)[0]
        st[base] = st.get(base, 0) + c
    return tuple(sorted((k, round(v, 6)) for k, v in st.items() if abs(v) > 1e-9))


def is_energy_coupled(model, rid):
    """Membrane- or quinone-coupled reactions: dG depends on pmf / quinone redox
    state (not modelled) -> no thermodynamic constraint."""
    r = model.reactions.get_by_id(rid)
    ids = [m.id for m in r.metabolites]
    if any(QUINONES.match(i) for i in ids):
        return True
    hcomp = {i.rsplit("_", 1)[1] for i in ids if i.rsplit("_", 1)[0] == "h"}
    chem = [k for k, v in reaction_signature(model, rid) if k not in ("h", "h2o")]
    # translocates protons AND transforms chemistry (not a pure symport)
    if len(hcomp) >= 2 and len(chem) > 0:
        sig = dict(reaction_signature(model, rid))
        if any(abs(v) > 1e-9 for k, v in sig.items() if k not in ("h", "h2o")):
            return True
    return False


# --------------------------------------------------------------------------
# thermodynamic layout and community optimisation for heterogeneous taxa
# --------------------------------------------------------------------------
LN_CMIN_INT = np.log(1e-7)        # intracellular/periplasmic floor (0.1 uM)
INORGANIC_MAX = np.log(1e-1)      # ions and inorganic nutrients up to 100 mM
INORGANIC = {"pi", "so4", "nh4", "k", "na1", "cl", "mg2", "ca2", "fe2", "fe3", "mn2", "zn2",
             "cu2", "cobalt2", "mobd", "ni2", "sel", "slnt", "tungs", "h2co3"}
NSIGMA = 2.0                      # dG'0 allowed to move by +/- NSIGMA standard errors


def _bounds(met_id):
    base, comp = met_id.rsplit("_", 1)
    if met_id in FIXED_C:
        v = np.log(FIXED_C[met_id]); return (v, v)
    if met_id in EXT_MAX:
        return (LN_CMIN, EXT_MAX[met_id])
    if base in INORGANIC:
        return (LN_CMIN, INORGANIC_MAX)
    if comp == "e" and met_id not in ("o2_e", "co2_e"):
        return (LN_CMIN, min(_STATE["ext_ceiling"], LN_CMAX))
    return (LN_CMIN_INT, LN_CMAX)


class HLayout:
    def __init__(self, nets, ext_ids):
        self.ext_col = {m: a for a, m in enumerate([e for e in ext_ids if e not in SKIP])}
        self.bounds = [None] * len(self.ext_col)
        for m, a in self.ext_col.items():
            self.bounds[a] = _bounds(m)
        self.col = []          # per taxon: {met index -> column}
        off = len(self.ext_col)
        for t in nets:
            d = {}
            for k, m in enumerate(t.met_ids):
                if m in SKIP:
                    continue
                if t.is_ext[k]:
                    if m in self.ext_col:
                        d[k] = self.ext_col[m]
                else:
                    d[k] = off; self.bounds.append(_bounds(m)); off += 1
            self.col.append(d)
        self.n = off
        self.rows = {}
        for i, t in enumerate(nets):
            if t.dg0 is None:
                continue
            Sc = t.S.tocsc()
            for j in range(t.n):
                if not t.thermo_ok[j]:
                    continue
                sl = Sc[:, j]
                coef = {}
                for k, c in zip(sl.indices, sl.data):
                    if k in self.col[i]:
                        coef[self.col[i][k]] = coef.get(self.col[i][k], 0.0) + RT * c
                coef = {k: v for k, v in coef.items() if abs(v) > 1e-12}
                if coef:
                    self.rows[(i, j)] = coef


MU_PROBE = 1e-4


def _scip_solve(c, cons, lb, ub, integ, time_limit):
    """Solve min c'x s.t. lo <= A x <= hi (list of LinearConstraint), bounds,
    integrality with SCIP (HiGHS returned inconsistent results on these
    big-M thermodynamic MILPs)."""
    from pyscipopt import Model, quicksum
    m = Model(); m.hideOutput()
    m.setParam("limits/time", time_limit)
    m.setParam("limits/gap", 1e-4)
    m.setParam("numerics/feastol", 1e-7)
    xs = []
    for k in range(len(c)):
        l = None if not np.isfinite(lb[k]) else float(lb[k])
        u = None if not np.isfinite(ub[k]) else float(ub[k])
        xs.append(m.addVar(lb=l, ub=u, vtype="B" if integ[k] else "C",
                           obj=float(c[k])))
    for lc in cons:
        A = sp.csr_matrix(lc.A)
        lo = np.broadcast_to(lc.lb, (A.shape[0],)); hi = np.broadcast_to(lc.ub, (A.shape[0],))
        for r in range(A.shape[0]):
            st, en = A.indptr[r], A.indptr[r + 1]
            if st == en:
                continue
            expr = quicksum(float(v) * xs[j] for j, v in zip(A.indices[st:en], A.data[st:en]))
            if np.isfinite(lo[r]) and np.isfinite(hi[r]) and lo[r] == hi[r]:
                m.addCons(expr == float(lo[r]))
            else:
                if np.isfinite(lo[r]):
                    m.addCons(expr >= float(lo[r]))
                if np.isfinite(hi[r]):
                    m.addCons(expr <= float(hi[r]))
    m.optimize()
    if m.getNSols() == 0:
        if m.getStatus() != "infeasible":
            # a time or node limit without an incumbent is NOT a proof of
            # infeasibility; the caller must not report it as one
            raise TimeoutError(f"SCIP stopped with status {m.getStatus()!r} "
                               f"after {time_limit}s without a feasible solution")
        return None
    sol = m.getBestSol()
    return np.array([sol[v] for v in xs])


def community_opt_h(nets, abund, med, ext_ids, thermo=False, obj_bio=None, bio_min=None,
                    **kw):
    """Wrapper: with thermodynamic binaries HiGHS occasionally returns the
    zero-growth vertex as 'optimal' for growth maximisation. We therefore first
    require a probe growth of MU_PROBE for every maximised taxon and fall back
    to the unforced problem only if the probe is infeasible."""
    if thermo and kw.get("ext_target") is None and kw.get("obj_ext") is None:
        w = obj_bio if obj_bio is not None else [1.0] * len(nets)
        bm = list(bio_min or [0.0] * len(nets))
        probe = [max(b, MU_PROBE) if wi > 0 else b for b, wi in zip(bm, w)]
        if probe != bm:
            r = _community_opt_h(nets, abund, med, ext_ids, thermo, obj_bio, probe, **kw)
            if r is not None:
                return r
    return _community_opt_h(nets, abund, med, ext_ids, thermo, obj_bio, bio_min, **kw)


def _community_opt_h(nets, abund, med, ext_ids, thermo=False, obj_bio=None, bio_min=None,
                     osm_pool=0.0, delta=0.1, time_limit=120.0, active_only=None,
                     ext_target=None, sense=-1, export_min=None, uptake_min=None, obj_ext=None,
                     presolve=False):
    """FBA / TFA for taxa with different networks and one shared extracellular pool.

    osm_pool : compatible-solute pool (mmol gDW^-1) of salt-out strategists; at
               balanced growth its synthesis must equal dilution, sum_s v_DM,s =
               osm_pool * mu_i (Section 2.x of the manuscript).
    active_only : optional list (per taxon) of boolean masks; thermodynamic
               binaries are created only for those reactions (others free).
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    ne, ns = len(ext_ids), len(nets)
    eidx = {m: a for a, m in enumerate(ext_ids)}
    offs = np.cumsum([0] + [t.n for t in nets])
    nv = offs[-1]
    L = HLayout(nets, ext_ids) if thermo else None
    keys = []
    if thermo:
        for (i, j) in L.rows:
            if active_only is None or active_only[i][j]:
                keys.append((i, j))
    nb, nc = len(keys), (L.n if thermo else 0)
    o_x = nv; o_c = nv + ne; o_zf = o_c + nc; o_zr = o_zf + nb; nvar = o_zr + nb
    blocks, ext_rows = [], []
    for i, t in enumerate(nets):
        Si = t.S[np.where(~t.is_ext)[0]]
        blocks.append(Si)
    A1 = sp.hstack([sp.block_diag(blocks, format="csr"),
                    sp.csr_matrix((sum(b.shape[0] for b in blocks), nvar - nv))])
    # shared extracellular balance
    E = sp.lil_matrix((ne, nvar))
    for i, t in enumerate(nets):
        ek = np.where(t.is_ext)[0]
        Se = t.S[ek].tocoo()
        for r, c, v in zip(Se.row, Se.col, Se.data):
            E[eidx[t.met_ids[ek[r]]], offs[i] + c] += abund[i] * v
    for a in range(ne):
        E[a, o_x + a] = -1.0
    eqs = [A1, E.tocsr()]
    # osmolyte synthesis = pool * growth for salt-out strategists
    if osm_pool > 0:
        O = sp.lil_matrix((ns, nvar))
        for i, t in enumerate(nets):
            if t.strategy == "salt-out" and t.osm_cols:
                for j in t.osm_cols:
                    O[i, offs[i] + j] = 1.0
                O[i, offs[i] + t.bio] = -osm_pool
        eqs.append(O.tocsr())
    cons = [LinearConstraint(sp.vstack(eqs, format="csr"), 0.0, 0.0)]
    if thermo and nb:
        T = sp.lil_matrix((5 * nb, nvar)); lo = np.full(5 * nb, -np.inf); hi = np.zeros(5 * nb)
        for q, (i, j) in enumerate(keys):
            vcol = offs[i] + j
            coef = L.rows[(i, j)]
            # tight big-M constants: M = flux bound, K = largest attainable |dG| + slack
            M = max(abs(nets[i].lb[j]), abs(nets[i].ub[j]), 1.0)
            span = sum(abs(v) * max(abs(L.bounds[cc][0]), abs(L.bounds[cc][1]))
                       for cc, v in coef.items())   # bound on |RT sum S ln c|
            K = abs(nets[i].dg0[j]) + NSIGMA * nets[i].dg0_se[j] + span + 10.0
            T[5*q, vcol] = 1.0; T[5*q, o_zf + q] = -M
            T[5*q+1, vcol] = -1.0; T[5*q+1, o_zr + q] = -M
            T[5*q+2, o_zf + q] = 1.0; T[5*q+2, o_zr + q] = 1.0; hi[5*q+2] = 1.0
            for cc, v in coef.items():
                T[5*q+3, o_c + cc] = v
                T[5*q+4, o_c + cc] = -v
            g0, sg = nets[i].dg0[j], NSIGMA * nets[i].dg0_se[j]
            T[5*q+3, o_zf + q] = K; hi[5*q+3] = K - delta - (g0 - sg)
            T[5*q+4, o_zr + q] = K; hi[5*q+4] = K - delta + (g0 + sg)
        cons.append(LinearConstraint(T.tocsr(), lo, hi))
    lb = np.concatenate([t.lb for t in nets] + [np.zeros(ne)]
                        + ([np.array([b[0] for b in L.bounds])] if thermo else [])
                        + [np.zeros(2 * nb)])
    ub = np.concatenate([t.ub for t in nets] + [np.full(ne, 1000.0)]
                        + ([np.array([b[1] for b in L.bounds])] if thermo else [])
                        + [np.ones(2 * nb)])
    lb[o_x:o_x + ne] = 0.0
    ub[o_x:o_x + ne] = 1000.0
    for m, a in eidx.items():
        lb[o_x + a] = -min(med.get(m, 0.0), 100.0)
    for m, f in (export_min or {}).items():
        lb[o_x + eidx[m]] = max(lb[o_x + eidx[m]], f)
    for m, f in (uptake_min or {}).items():
        ub[o_x + eidx[m]] = min(ub[o_x + eidx[m]], -f)
    for i, bm in enumerate(bio_min or [0.0] * ns):
        lb[offs[i] + nets[i].bio] = max(lb[offs[i] + nets[i].bio], bm)
    c = np.zeros(nvar)
    if ext_target is not None:
        c[o_c + L.ext_col[ext_target]] = float(sense)
    elif obj_ext is not None:
        for m, w in obj_ext.items():
            c[o_x + eidx[m]] = -w
    else:
        w = obj_bio if obj_bio is not None else [1.0] * ns
        for i, t in enumerate(nets):
            c[offs[i] + t.bio] = -w[i]
    integ = np.zeros(nvar); integ[o_zf:] = 1
    if nb:
        x = _scip_solve(c, cons, lb, ub, integ, time_limit)
        if x is None:
            return None
        res_status = 0
    else:
        res = milp(c, constraints=cons, bounds=Bounds(lb, ub),
                   options={"time_limit": time_limit, "disp": False})
        if res.x is None:
            return None
        x, res_status = res.x, res.status
    out = {"bio": [float(x[offs[i] + t.bio]) for i, t in enumerate(nets)],
           "v": [x[offs[i]:offs[i + 1]] for i in range(ns)],
           "x": x[o_x:o_x + ne], "status": res_status}
    if thermo:
        out["lnc_ext"] = {m: x[o_c + a] for m, a in L.ext_col.items()}
    return out


# Thermodynamic scope: central carbon and energy metabolism, defined as the
# metabolite set of the E. coli core model plus glycerol/DHA metabolism. Only
# reactions whose metabolites all lie in this set (transport included) receive
# dG constraints; peripheral biosynthesis, where automated reconstructions often
# carry cofactor or directionality artefacts, is left unconstrained.
EXTRA_CORE = {"glyc", "dha", "glyc3p", "glyald", "glyc__R", "13dpg", "dhap", "mthgxl",
              "lac__L", "ppi", "o2", "h2o2", "acald", "for", "etoh", "but", "ppa"}


def core_metabolites():
    from cobra.io import load_model
    m = load_model("textbook")
    return {x.id.rsplit("_", 1)[0] for x in m.metabolites} | EXTRA_CORE


def attach_thermo(nets: dict, models: dict, cache: dict, scope: str = "core"):
    """Attach standard dG'0 (NaN when eQuilibrator could not estimate it) and
    the mask of reactions that receive thermodynamic constraints."""
    core = core_metabolites() if scope == "core" else None
    for k, t in nets.items():
        m = models[k]
        dg = np.full(t.n, np.nan); se = np.zeros(t.n)
        ok = np.zeros(t.n, bool)
        for j, rid in enumerate(t.rxn_ids):
            if rid.startswith("DM_OSM_") or j == t.bio or rid not in m.reactions:
                continue
            r = m.reactions.get_by_id(rid)
            if r.boundary:
                continue
            v = cache.get(repr(reaction_signature(m, rid)))
            if v is None:
                continue
            dg[j], se[j] = v[0], v[1]
            in_scope = core is None or all(x.id.rsplit("_", 1)[0] in core for x in r.metabolites)
            ok[j] = in_scope and not is_energy_coupled(m, rid)
        t.dg0, t.dg0_se, t.thermo_ok = dg, se, ok


def mdf_h(nets, ext_ids, fluxes, tol=1e-6):
    """Max-min driving force of given flux patterns with a shared ext pool.
    Returns (B, binding) where binding lists (taxon, reaction) constraints with
    non-zero dual (the thermodynamic bottlenecks)."""
    from scipy.optimize import linprog
    L = HLayout(nets, ext_ids)
    A, b, keys = [], [], []
    for (i, j), coef in L.rows.items():
        v = fluxes[i][j]
        if abs(v) < tol:
            continue
        d = np.sign(v)
        row = np.zeros(L.n + 1)
        for cc, val in coef.items():
            row[cc] = d * val
        row[-1] = 1.0
        g_fav = nets[i].dg0[j] - d * NSIGMA * nets[i].dg0_se[j]   # most favourable value
        A.append(row); b.append(-d * g_fav); keys.append((i, j))
    if not A:
        return np.inf, []
    c = np.zeros(L.n + 1); c[-1] = -1.0
    r = linprog(c, A_ub=sp.csr_matrix(np.array(A)), b_ub=np.array(b),
                bounds=L.bounds + [(None, None)], method="highs")
    if r.status != 0:
        return -np.inf, keys
    binding = [keys[q] for q, mval in enumerate(r.ineqlin.marginals) if abs(mval) > 1e-9]
    return float(r.x[-1]), binding


def tfa_lazy(nets, abund, med, ext_ids, delta=0.1, max_iter=40, verbose=False,
             init_masks=None, per_taxon=True, **kw):
    """Thermodynamics-based FBA by lazy constraint generation.

    Solve FBA; compute the MDF of the solution; if some active reaction cannot
    satisfy dG <= -delta, add binary thermodynamic constraints for the
    bottleneck reactions and re-solve. Terminates with (i) a solution whose MDF
    >= delta, which is feasible and optimal for the full TFA problem because it
    is optimal for a relaxation, or (ii) infeasibility of a relaxation, which
    proves TFA infeasibility. Returns (solution | None, info)."""
    masks = [np.zeros(t.n, bool) if init_masks is None or init_masks[i] is None
             else init_masks[i].copy() for i, t in enumerate(nets)]
    for it in range(max_iter):
        n_bin = int(sum(m.sum() for m in masks))
        try:
            sol = community_opt_h(nets, abund, med, ext_ids, thermo=n_bin > 0,
                                  active_only=masks, delta=delta, **kw)
        except TimeoutError as e:
            return None, {"iter": it, "n_bin": n_bin, "status": "timeout", "msg": str(e)}
        if sol is None:
            return None, {"iter": it, "n_bin": n_bin, "status": "infeasible"}
        B, binding = mdf_h(nets, ext_ids, sol["v"])
        if verbose:
            print(it, "bio", np.round(sol["bio"], 4), "MDF", round(B, 3), "binding", len(binding))
        if B >= delta - 1e-6:
            return sol, {"iter": it, "n_bin": n_bin, "status": "ok", "mdf": B, "masks": masks}
        new = [(i, j) for (i, j) in binding if not masks[i][j]]
        if per_taxon and len(nets) > 1:
            # also add each member's own bottleneck (necessary condition,
            # extracellular potentials free): speeds convergence in large communities
            for i, t in enumerate(nets):
                vi = [np.zeros(u.n) for u in nets]
                vi[i] = sol["v"][i]
                Bi, bi = mdf_h(nets, ext_ids, vi)
                if Bi < delta - 1e-6:
                    new += [(a, b) for (a, b) in bi if not masks[a][b] and (a, b) not in new]
        if not new:          # MDF<delta but nothing new to add: add all active
            new = [(i, j) for i, t in enumerate(nets) for j in range(t.n)
                   if t.thermo_ok[j] and abs(sol["v"][i][j]) > 1e-6 and not masks[i][j]]
            if not new:
                return sol, {"iter": it, "n_bin": n_bin, "status": "unresolved", "mdf": B}
        for i, j in new:
            masks[i][j] = True
    return sol, {"iter": max_iter, "n_bin": int(sum(m.sum() for m in masks)),
                 "status": "max_iter", "mdf": B}
