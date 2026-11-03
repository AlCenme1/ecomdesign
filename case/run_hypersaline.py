"""Hypersaline case study: 17 curated resident taxa + 2 invaders
(E. coli iJO1366, M. tuberculosis iNJ661) on a Dunaliella-derived medium.

Stage 1  solo growth (FBA, TFA) under an osmotic-cost sweep
Stage 2  directed pairwise outcomes with a selfish partner (FBA, TFA)
Stage 3  invasion from rare into the full resident community
Stage 4  validation of invader-containing 4-member consortia
"""
import sys, time, pickle, warnings, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/ecomdesign")
from ecomdesign.hetero import community_opt_h, tfa_lazy, mdf_h

EPS = 0.01
import os
SELFISH = float(os.environ.get("SELFISH", 0.95))
SALT_IN = {"Haloarcula", "Halobacterium", "Haloferax", "Halorubrum", "Natronomonas",
           "Halomicrobium", "Haloquadratum", "Salinibacter"}
INVADERS = ["Escherichia", "Mycobacterium"]
P_GRID = [0.0, 1.0, 3.0, 10.0]          # compatible-solute pool, mmol gDW^-1

nets, EXT = pickle.load(open("nets.pkl", "rb"))
D = pickle.load(open("medium.pkl", "rb"))
for k, t in nets.items():
    t.strategy = "salt-in" if k in SALT_IN else "salt-out"
    t.group = "invader" if k in INVADERS else "resident"
NAMES = list(nets)
RES = [k for k in NAMES if k not in INVADERS]


def solve(members, abund, thermo, P, obj=None, bmin=None):
    T = [nets[m] for m in members]
    if thermo:
        sol, info = tfa_lazy(T, abund, D, EXT, osm_pool=P, obj_bio=obj, bio_min=bmin)
        return sol
    return community_opt_h(T, abund, D, EXT, osm_pool=P, obj_bio=obj, bio_min=bmin)


def growth(sol, k):
    return 0.0 if sol is None else max(0.0, sol["bio"][k])


# ---------------- stage 1 ----------------
def solo(args):
    k, P, thermo, x = args
    s = solve([k], [x], thermo, P)
    return dict(taxon=k, P=P, thermo=thermo, abund=x, mu=growth(s, 0))


# ---------------- stage 2 ----------------
def pair(args):
    i, j, P, thermo, ref_i = args
    # maximise growth of j next to a selfish i (keeps >= 95% of its solo growth)
    bmin = [SELFISH * ref_i if ref_i > EPS else 0.0, 0.0]
    s = solve([i, j], [0.5, 0.5], thermo, P, obj=[0.0, 1.0], bmin=bmin)
    out = dict(i=i, j=j, P=P, thermo=thermo, mu_j=growth(s, 1), feasible=s is not None)
    if s is not None and thermo:
        # exchanged carbon: what j takes up that i secretes (community export < 0)
        out["mdf"] = mdf_h([nets[i], nets[j]], EXT, s["v"])[0]
    return out


if __name__ == "__main__":
    stage = sys.argv[1]
    t0 = time.time()
    if stage == "solo":
        jobs = [(k, P, th, x) for k in NAMES for P in P_GRID for th in (False, True) for x in (1.0, 0.5)]
        with Pool(2) as pool:
            rows = pool.map(solo, jobs)
        pd.DataFrame(rows).to_csv("data_solo.csv", index=False)
        print("solo done", round(time.time() - t0), "s", flush=True)
    elif stage == "pairs":
        P = float(sys.argv[2])
        solo_df = pd.read_csv("data_solo.csv")
        ref = {(r.taxon, r.thermo): r.mu for r in solo_df.query("P == @P and abund == 0.5").itertuples()}
        jobs = [(i, j, P, th, ref[(i, th)]) for i in NAMES for j in NAMES if i != j for th in (False, True)]
        with Pool(2) as pool:
            rows = pool.map(pair, jobs, chunksize=4)
        pd.DataFrame(rows).to_csv(f"data_pairs_P{P:g}.csv", index=False)
        print("pairs done", P, round(time.time() - t0), "s", flush=True)
    elif stage == "invasion":
        # residents at equal abundance maximise total growth; invader enters at the
        # same abundance and maximises its own growth while every resident keeps
        # >= 95% of its growth in the resident-only optimum.
        th = sys.argv[2] == "tfa"
        grid = [float(p) for p in sys.argv[3].split(",")] if len(sys.argv) > 3 else P_GRID
        x = 1.0 / (len(RES) + 1)
        seeds = pickle.load(open("seed_masks_P0.pkl", "rb")) if th else None
        rows = []
        for P in grid:
            if th:
                base, info = tfa_lazy([nets[k] for k in RES], [x] * len(RES), D, EXT, osm_pool=P,
                                      init_masks=[seeds[k] for k in RES], time_limit=300)
                bm = info.get("masks")
            else:
                base = solve(RES, [x] * len(RES), False, P)
            mu_res = [growth(base, k) for k in range(len(RES))]
            grow = [m for m in mu_res if m > EPS]
            print("base", P, th, round(sum(mu_res), 3), len(grow), round(time.time() - t0), flush=True)
            for inv in INVADERS:
                members = RES + [inv]
                bmin = [SELFISH * m for m in mu_res] + [0.0]
                if th:
                    s, _ = tfa_lazy([nets[m] for m in members], [x] * len(members), D, EXT, osm_pool=P,
                                    obj_bio=[0.0] * len(RES) + [1.0], bio_min=bmin, time_limit=300,
                                    init_masks=(bm if bm else [seeds[k] for k in RES]) + [seeds[inv]])
                else:
                    s = solve(members, [x] * len(members), False, P,
                              obj=[0.0] * len(RES) + [1.0], bmin=bmin)
                rows.append(dict(P=P, thermo=th, invader=inv, mu_inv=growth(s, len(RES)),
                                 feasible=s is not None, n_res_growing=len(grow),
                                 res_total=float(sum(mu_res)),
                                 mu_res_mean=float(np.mean(grow)) if grow else 0.0,
                                 mu_res_max=float(max(mu_res))))
                print(rows[-1], round(time.time() - t0), "s", flush=True)
                pd.DataFrame(rows).to_csv(f"data_invasion_{'tfa' if th else 'fba'}{os.environ.get('TAG', '')}.csv", index=False)
