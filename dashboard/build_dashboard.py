"""Build the standalone results dashboard.

Reads the CSV tables in `data/` and writes a single self-contained
`index.html`: the data are embedded, so the file works by double-clicking it,
from a USB stick, or published on GitHub Pages. Only d3 is loaded from a CDN.

    python dashboard/build_dashboard.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "index.html"

TABLES = ["benchmark_filter", "benchmark_cv", "benchmark_coefficients",
          "benchmark_designs", "hypersaline_solo", "hypersaline_pairs",
          "hypersaline_invasion", "kehe_paired", "kehe_classes", "kehe_model"]

SOURCES = {
    "benchmark_filter": "Feasible producer–consumer pairs with and without thermodynamics, by concentration ceiling.",
    "benchmark_cv": "Node-split cross-validated ROC-AUC and average precision, mean of 100 splits.",
    "benchmark_coefficients": "Energies of model M3 at a 0.1 mM ceiling, with naive and dyadic-robust standard errors.",
    "benchmark_designs": "Four-member consortia that passed multi-strain thermodynamic validation.",
    "hypersaline_solo": "Growth of each taxon alone on glycerol and dihydroxyacetone, across the compatible-solute pool.",
    "hypersaline_pairs": "Growth of a focal taxon beside a selfish partner, for every ordered pair.",
    "hypersaline_invasion": "Growth of each introduced taxon entering the resident community.",
    "kehe_paired": "Facilitation and inhibition in Kehe et al. (2021), at full and one-tenth carbon.",
    "kehe_classes": "Share of pairs in each interaction class at the two carbon concentrations.",
    "kehe_model": "Fitted energy of tenfold dilution in the dyadic facilitation model.",
}


def read_tables() -> dict:
    out = {}
    for name in TABLES:
        path = DATA / f"{name}.csv"
        if not path.exists():
            raise SystemExit(f"missing data file: {path}")
        df = pd.read_csv(path)
        out[name] = json.loads(df.to_json(orient="records"))
    return out


def build() -> None:
    data = read_tables()
    page = (HERE / "template.html").read_text()
    page = page.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    page = page.replace("/*__SOURCES__*/", json.dumps(SOURCES, separators=(",", ":")))
    d3 = HERE / "vendor" / "d3.v7.min.js"
    page = page.replace("/*__D3__*/", d3.read_text() if d3.exists() else
                        "document.write('<script src=\\'https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js\\'><\\/script>')")
    logo = HERE.parent / "app" / "assets" / "mfcb_logo.svg"
    mark = ""
    if logo.exists():
        # show the emblem only: beside the page title the wordmark would be illegible
        mark = logo.read_text().replace('width="400" height="470" viewBox="0 0 400 470"',
                                        'viewBox="0 0 400 330" role="img" aria-label="MFCB"')
    page = page.replace("<!--__LOGO__-->", mark)
    OUT.write_text(page)
    size = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT} ({size:.0f} KB, {sum(len(v) for v in data.values())} rows embedded)")


if __name__ == "__main__":
    build()
