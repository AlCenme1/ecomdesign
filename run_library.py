"""Build library + features + labels across extracellular concentration ceilings."""
import sys, pickle, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from ecomdesign.metabolism import load_base, set_ext_ceiling
from ecomdesign.library import *
seed = int(sys.argv[1]); N = int(sys.argv[2])
ceilings = [1e-2, 1e-3, 1e-4, 1e-5]
b = load_base(); D = design_medium(b)
set_ext_ceiling(1e-2)
t0 = time.time()
lib = build_library(b, n=N, seed=seed)
nodes, Ssec = node_table(b, lib, D)
print("nodes done", round(time.time() - t0), "s; producers", int(nodes.grows_D.sum()), flush=True)
out = {"lib": lib, "nodes": nodes, "Ssec": Ssec, "res": {}}
for cmax in ceilings:
    set_ext_ceiling(cmax)
    lmax, lmin = exchange_windows(b, lib, nodes, D)
    F = pair_features(b, lib, nodes, Ssec, lmax, lmin, D)
    Y = pair_truth(b, lib, nodes, D, thermo_only=(cmax != ceilings[0]))
    df = F.merge(Y, on=["i", "j"])
    if cmax != ceilings[0]:
        df["y_stoich"] = out["res"][ceilings[0]]["pairs"]["y_stoich"].values
        df["mu_fba"] = out["res"][ceilings[0]]["pairs"]["mu_fba"].values
    out["res"][cmax] = {"lmax": lmax, "lmin": lmin, "pairs": df}
    print(cmax, "pairs", len(df), "y", int(df.y.sum()), "y_stoich", int(df.y_stoich.sum()),
          round(time.time() - t0), "s", flush=True)
    pickle.dump(out, open(f"data/library_seed{seed}_N{N}.pkl", "wb"))
