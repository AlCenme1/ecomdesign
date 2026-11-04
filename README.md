<p align="center">
  <img src="https://raw.githubusercontent.com/AlCenme1/ecomdesign/main/app/assets/mfcb_logo.png" alt="MFCB" height="150">
</p>

# ecomdesign

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlCenme1/ecomdesign/blob/main/notebooks/ecomdesign_colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)


Energy-aware design of microbial consortia: thermodynamic exchange windows + Fermi–Dirac dyadic interaction model.

- `ECOMDESIGN_pipeline.ipynb` — reproduces every result and figure (uses cached results in `data/`; set `REBUILD = True` to recompute, ~1 h on two cores).
- `ecomdesign/` — package: `metabolism.py` (shared-pool TFA, MDF), `library.py` (strain library, exchange windows, pair features, ground truth), `stats.py` (Fermi–Dirac/dyadic logit, bias correction, dyadic-robust SE, REM), `analysis.py` (benchmark, design), `figures.py`.
- `run_library.py`, `run_analysis.py`, `run_montecarlo.py` — the heavy computations called by the notebook.
- `manuscript/` — LaTeX source, bibliography, figures and compiled PDF.

Requirements: `pip install cobra equilibrator-api scikit-learn scipy pandas matplotlib`.


## Case study: hypersaline community (`case/`)

`case/run_hypersaline.py` runs the real-data analyses on the curated 17-taxon
SBML model plus iJO1366 and iNJ661 (model files are not redistributed here):

| stage | command | output |
|---|---|---|
| solo growth | `python run_hypersaline.py solo` | `data_solo.csv` |
| directed pairs | `python run_hypersaline.py pairs 0` / `pairs 3` | `data_pairs_P*.csv` |
| invasion from rare | `python run_hypersaline.py invasion fba` | `data_invasion_fba.csv` |
| strict invasion | `SELFISH=0.999 TAG=_strict python run_hypersaline.py invasion fba` | `data_invasion_fba_strict.csv` |

`python fig_case.py` rebuilds Figure 7. The preprocessing step
(`ecomdesign.hetero.split_community`, `to_net`, `attach_thermo`) writes
`nets.pkl`, `medium.pkl` and `dg0_cache.pkl`, which the stages above consume.


## Experimental validation: Kehe et al. (2021) screen (`kehe/`)

`kehe/run_kehe.py` reads the published data file of Kehe et al. (2021,
*Sci. Adv.* 7:eabi7159, data S1) and fits the dyadic facilitation model with
donor, receiver and carbon-source effects, dyadic-robust standard errors and
bootstrap bias correction. Put the data file at `kehe/raw/kehe_data_s1.csv`
(the published file is distributed with an `.xlsx` extension but is a CSV).
`kehe/fig_kehe.py` rebuilds Figure 8.


## Input workbook (`templates/`)

A user supplies one spreadsheet and a folder of metabolic models:

| sheet | required | holds |
|---|---|---|
| `strains` | yes | one row per organism: `strain`, `model_file`, optional `group`, `notes` |
| `medium` | yes | one row per nutrient: `metabolite`, `max_uptake`, optional `notes` |
| `abundances` | no | `strain`, `abundance` |
| `osmoadaptation` | no | `strain`, `strategy` (salt-in / salt-out), `pool_P` |
| `tolerance` | no | `strain`, `salinity_min/max`, `temperature_min/max`, `pH_min/max` |
| `settings` | no | `setting`, `value` (ceiling, delta, pH, ionic strength, oxygen, consortium size, scope) |
| `labels` | no | `strain_i`, `strain_j`, `outcome`, optional `environment`, `effect_size` |

`templates/ecomdesign_input_template.xlsx` is empty with a note on every
header; `templates/ecomdesign_example.xlsx` is the same workbook filled in.
Check a workbook before running anything:

```
python -m ecomdesign.io_excel my_study.xlsx models/
```

It lists every problem in plain words and computes nothing until there are
none. The metabolic models themselves cannot be given as a spreadsheet: they
are SBML (.xml) or COBRA JSON files.


## The browser application (`app/`)

```
pip install -e ".[all]"
streamlit run app/streamlit_app.py
```

Four steps in the browser: upload the workbook and the model files, read the
check report, run a stage, read the results and download them as CSV. Nothing
is computed until the check comes up clean, so a user never meets a traceback.
Deploy it free on Streamlit Community Cloud or Hugging Face Spaces by pointing
either at this repository and at `app/streamlit_app.py`; the free tiers suit
small libraries, while a 17-taxon study belongs on your own machine.

## The results dashboard (`dashboard/`)

```
python dashboard/build_dashboard.py      # reads dashboard/data/*.csv
```

Writes one self-contained `dashboard/index.html`: the tables and d3 are
embedded, so it opens by double-clicking, runs offline and needs no server.
`.github/workflows/pages.yml` publishes it at
`https://<user>.github.io/<repo>/` on every push that touches `dashboard/`.
To refresh it after a new run, overwrite the CSVs in `dashboard/data/` and
build again.

## Making it citable

The repository carries an MIT licence and a `CITATION.cff`. Connecting it to
Zenodo and cutting a release mints a DOI, which is what the manuscript's
*Code and data availability* section should point to.


## The manuscript (`manuscript/`)

`ecomdesign_manuscript.tex` is the working single-column file and the source of
truth. `manuscript/cas/` holds the Elsevier CAS version for *Computational
Biology and Chemistry*, generated from it with `build_cas.py` (see the README
in that folder) — edit the working file, then rebuild.


## Putting the app online

The app runs unchanged on Streamlit Community Cloud or Hugging Face Spaces.
Point the service at this repository and at `app/streamlit_app.py`; the
root `requirements.txt` is what it installs. `pyscipopt` ships SCIP inside its
wheel, so no system packages are needed.

Two limits of a free tier are worth knowing before you share the link:

- **Memory.** A free instance has about 1 GB. Genome-scale models are large,
  so keep a hosted study to a few strains; the 17-taxon case study belongs on
  your own machine, where it takes hours of solver time.
- **Gibbs energies.** `equilibrator-api` is not installed online, because its
  first run downloads a compound cache far larger than the tier allows. The
  app therefore reads the energies from the workbook's `gibbs` sheet. Without
  that sheet it still runs, but no reaction receives a thermodynamic
  constraint and the results are ordinary flux balance analysis — the app says
  so before you press Run.


## Running it in Colab

`notebooks/ecomdesign_colab.ipynb` runs the pipeline in a browser with nothing
installed locally: it checks an input workbook, reproduces the benchmark, shows
the fitted interaction energies with both standard errors, and redraws the
figures. Click the badge at the top of this file. The first cell installs the
dependencies and takes a few minutes; the rest are quick.
