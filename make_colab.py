"""Write the Colab notebook: ecomdesign running in a browser, no installation.

    python make_colab.py
"""
import nbformat as nbf

REPO = "https://github.com/AlCenme1/ecomdesign"
nb = nbf.v4.new_notebook()
C = []
md = lambda t: C.append(nbf.v4.new_markdown_cell(t))
code = lambda t: C.append(nbf.v4.new_code_cell(t))

md(f"""# ecomdesign in Colab

Designing microbial consortia on an explicit energy scale. This notebook runs
the whole pipeline in the browser, with nothing installed on your machine.

Run the cells in order. The first one takes two or three minutes; the rest are
seconds apiece, except where a cell says otherwise.

Source and issues: [{REPO}]({REPO})""")

md("## 1. Install")
code(f"""!pip -q install pyscipopt cobra openpyxl
!git clone -q {REPO} ecomdesign_repo || echo "already cloned"
%cd ecomdesign_repo
import sys; sys.path.insert(0, ".")
print("ready")""")

md("""## 2. Check an input workbook

The pipeline takes one spreadsheet and a folder of metabolic models. Start from
the template, fill it in, and check it before anything is computed: the report
lists every problem in plain words.""")
code("""from ecomdesign.io_excel import read_inputs, report
print(report(read_inputs("templates/ecomdesign_example.xlsx")))""")

md("""Upload your own workbook with the cell below, then check it the same way.
The metabolic models go in a folder of their own; pass that folder as
`models_dir` so the checker can confirm each file is there.""")
code("""# from google.colab import files
# up = files.upload()                       # choose your .xlsx
# print(report(read_inputs(next(iter(up)), models_dir="models")))""")

md("""## 3. The thermodynamic exchange window

For one strain and one metabolite, thermodynamics-based FBA with a shared
extracellular pool gives the highest concentration at which the strain can
still secrete the metabolite and the lowest at which it can still consume it.
The difference is the exchange window, and its sign decides whether
cross-feeding is possible at all. The cell below reads one real pair out of the
benchmark; section 4 loads the file it comes from.""")
code("""import numpy as np, pickle
from ecomdesign.library import CARBONS, O2_GRID

L = pickle.load(open("data/library_seed11_N80.pkl", "rb"))
res = L["res"][1e-4]                       # by-product ceiling 0.1 mM
lmax, lmin, pairs = res["lmax"], res["lmin"], res["pairs"]
mM = lambda ln: np.exp(ln) * 1e3

# the widest window among pairs mediated by a by-product the producer makes
cand = pairs[pairs.G.notna() & (pairs.G > 0) & ~pairs.k_best.isin(["glc__D", "-"])]
best = None
for r in cand.itertuples():
    a = CARBONS.index(r.k_best)
    p, c = int(r.producer), int(r.consumer)
    if np.isfinite(lmax[p, a]) and np.isfinite(lmin[c, a]).any():
        if best is None or r.G > best[0]:
            best = (r.G, p, c, r.k_best, a)

G, p, c, k, a = best
o = int(np.nanargmin(np.where(np.isnan(lmin[c, a]), np.inf, lmin[c, a])))
print(f"strain {p} feeds strain {c} on {k}")
print(f"  producer can still secrete it up to {mM(lmax[p, a]):>9.3g} mM")
print(f"  consumer needs at least             {mM(lmin[c, a, o]):>9.3g} mM"
      f"   (residual O2 {O2_GRID[o]})")
print(f"  window  G = RT ln(ceiling/floor)    {G:>+9.2f} kJ/mol")
blocked = (pairs.G.isna() | (pairs.G <= 0)).mean()
print(f"\\nat this ceiling, {blocked:.0%} of candidate pairs have no window at all")""")

md('''A positive window means some concentration satisfies both cells at once.
A negative one means no concentration does, and the exchange cannot happen
however favourable the stoichiometry looks.''')

md("""## 4. The benchmark

Eighty strains derived from the *E. coli* core model, 1,596 producer-consumer
pairs, ground truth from multi-strain thermodynamic FBA. Building the library
from scratch takes about an hour, so the notebook loads the one shipped with
the repository; set `REBUILD = True` to recompute it.""")
code("""import pickle, pandas as pd
from pathlib import Path

REBUILD = False
lib_file = Path("data/library_seed11_N80.pkl")
if REBUILD or not lib_file.exists():
    !python run_library.py 80 11
L = pickle.load(open(lib_file, "rb"))
rows = []
for c in [1e-2, 1e-3, 1e-4, 1e-5]:
    p = L["res"][c]["pairs"]
    rows.append({"ceiling (M)": c, "feasible by stoichiometry": int(p.y_stoich.sum()),
                 "feasible with thermodynamics": int(p.y.sum())})
d = pd.DataFrame(rows)
d["removed (%)"] = (100 * (1 - d["feasible with thermodynamics"] / d["feasible by stoichiometry"])).round(1)
d""")

md("""The filter bites only when by-products stay dilute. That is the claim the
whole pipeline rests on, and the next cell shows what it costs in prediction.""")
code("""A = pickle.load(open("data/analysis_N80.pkl", "rb"))
cv = A["cv"][1e-4].groupby("model")[["auc", "ap"]].mean().round(3)
cv.loc[["M2: + stoichiometric complementarity", "M3: + thermodynamic window"]]""")

md("""## 5. Interaction energies

The probability that two strains interact follows a Fermi-Dirac law in an
energy. Because the noise temperature is fixed to one by the logit, every
coefficient below *is* an energy, and they can be added and compared. The two
standard-error columns are the point of the statistical layer: pairs that share
a strain are not independent observations.""")
code("""A["fit"][(1e-4, "M3: + thermodynamic window")][0].round(3)""")

md("""## 6. Figures

Each figure of the article is one function call on the stored results.""")
code("""from IPython.display import Image
import ecomdesign.figures as F
F.fig_benchmark(A["cv"], "fig3.png")
Image("fig3.png")""")

md("""## 7. Your own study

With a workbook and a folder of models, the same three stages run on your own
strains. Growth alone first, then every ordered pair. A genome-scale strain
takes seconds; a pair takes tens of seconds, so start small.""")
code("""# from ecomdesign.study import Study
# inp = read_inputs("my_study.xlsx", models_dir="models")
# assert inp.ok, report(inp)
# st = Study(inp, "models"); st.load()
# solo = st.solo(); solo""")

md(f"""## What this does not do

The cost of making compatible solutes is in the model; the stability of a
protein at molar ionic strength is not. A strain can look perfectly viable here
on metabolic grounds and still be impossible in the real pond, which is why the
workbook has a tolerance sheet and why the article argues that physiological
limits must enter design as an explicit filter rather than be expected to
emerge from stoichiometry and thermodynamics.

MIT licence. If this contributed to your work, cite the article and the
archived release at [{REPO}]({REPO}).""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
nbf.write(nb, "notebooks/ecomdesign_colab.ipynb")
print("wrote notebooks/ecomdesign_colab.ipynb")
