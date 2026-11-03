"""Reading the user's input workbook, and saying clearly what is wrong with it.

The pipeline needs one thing that a spreadsheet cannot hold: the metabolic
networks themselves, which are SBML or JSON files. Everything else the user
supplies is a table, and this module reads those tables from one .xlsx (or a
folder of CSVs with the same names) and validates them before any solver runs.

    from ecomdesign.io_excel import read_inputs, report
    inp = read_inputs("my_study.xlsx", models_dir="models/")
    print(report(inp))          # human-readable list of problems, if any
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import pandas as pd

# sheet name -> (required columns, optional columns)
SCHEMA = {
    "strains": (["strain", "model_file"], ["group", "notes"]),
    "medium": (["metabolite", "max_uptake"], ["notes"]),
    "abundances": (["strain", "abundance"], []),
    "osmoadaptation": (["strain", "strategy", "pool_p"], []),
    "tolerance": (["strain"], ["salinity_min", "salinity_max", "temperature_min",
                               "temperature_max", "pH_min", "pH_max"]),
    "settings": (["setting", "value"], []),
    "labels": (["strain_i", "strain_j", "outcome"], ["environment", "effect_size"]),
}
REQUIRED_SHEETS = ["strains", "medium"]
STRATEGIES = {"salt-in", "salt-out"}
SETTINGS = {"ceiling_mM", "delta_kJ_per_mol", "pH", "ionic_strength_M",
            "oxygen_max", "consortium_size", "thermodynamic_scope"}


@dataclass
class Inputs:
    tables: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _read_any(path: str) -> dict:
    """One .xlsx (a sheet per table) or a folder of CSVs named after the tables."""
    if os.path.isdir(path):
        out = {}
        for fn in os.listdir(path):
            stem, ext = os.path.splitext(fn)
            if ext.lower() in (".csv", ".tsv"):
                sep = "\t" if ext.lower() == ".tsv" else ","
                out[_norm(stem)] = pd.read_csv(os.path.join(path, fn), sep=sep)
        return out
    book = pd.read_excel(path, sheet_name=None)
    return {_norm(k): v for k, v in book.items()}


def read_inputs(path: str, models_dir: str | None = None) -> Inputs:
    """Read and validate the workbook. Never raises on bad user input: the
    problems are collected so that all of them can be shown at once."""
    inp = Inputs()
    try:
        raw = _read_any(path)
    except FileNotFoundError:
        inp.problems.append(f"No file or folder at '{path}'.")
        return inp
    except Exception as exc:                      # unreadable workbook
        inp.problems.append(f"'{path}' could not be read as a workbook ({exc}).")
        return inp

    for ignored in ("readme", "guide", "notes"):
        raw.pop(ignored, None)
    for sheet in REQUIRED_SHEETS:
        if sheet not in raw:
            inp.problems.append(
                f"The sheet '{sheet}' is missing. The workbook needs at least "
                f"'strains' and 'medium'.")

    for name, df in raw.items():
        if name not in SCHEMA:
            inp.warnings.append(f"Sheet '{name}' is not used by the pipeline; it was ignored.")
            continue
        df = df.rename(columns={c: _norm(c) for c in df.columns})
        df = df.dropna(how="all")
        need, _ = SCHEMA[name]
        shown = {"pool_p": "pool_P"}
        missing = [shown.get(c, c) for c in need if c not in df.columns]
        if missing:
            inp.problems.append(
                f"Sheet '{name}' is missing the column(s) {', '.join(missing)}. "
                f"It must have: {', '.join(shown.get(c, c) for c in need)}.")
            continue
        inp.tables[name] = df

    if "strains" in inp.tables:
        s = inp.tables["strains"]
        if s.strain.duplicated().any():
            dup = sorted(set(s.strain[s.strain.duplicated()]))
            inp.problems.append(f"Sheet 'strains': repeated strain name(s): {', '.join(map(str, dup))}.")
        if len(s) < 2:
            inp.problems.append("Sheet 'strains': at least two strains are needed to have a pair.")
        if models_dir:
            for r in s.itertuples():
                f = str(r.model_file)
                if not os.path.exists(os.path.join(models_dir, f)):
                    inp.problems.append(
                        f"Sheet 'strains': the model file '{f}' for strain "
                        f"'{r.strain}' was not found in '{models_dir}'.")
                elif not f.lower().endswith((".xml", ".sbml", ".json", ".xml.gz", ".mat")):
                    inp.problems.append(
                        f"Sheet 'strains': '{f}' is not a metabolic model. A model is an "
                        f"SBML (.xml) or COBRA JSON file, not a spreadsheet.")

    if "medium" in inp.tables:
        m = inp.tables["medium"]
        bad = m[pd.to_numeric(m.max_uptake, errors="coerce").isna()]
        if len(bad):
            inp.problems.append(
                f"Sheet 'medium': max_uptake must be a number; "
                f"check row(s) for {', '.join(map(str, bad.metabolite.head(5)))}.")
        neg = pd.to_numeric(m.max_uptake, errors="coerce") < 0
        if neg.any():
            inp.problems.append("Sheet 'medium': max_uptake is an uptake limit, so it cannot be negative.")
        if not len(m):
            inp.problems.append("Sheet 'medium' is empty: the community has nothing to grow on.")

    known = set(inp.tables.get("strains", pd.DataFrame({"strain": []})).strain.astype(str))
    for sheet, col in [("abundances", "strain"), ("osmoadaptation", "strain"),
                       ("tolerance", "strain")]:
        if sheet in inp.tables and known:
            unknown = sorted(set(inp.tables[sheet][col].astype(str)) - known)
            if unknown:
                inp.problems.append(
                    f"Sheet '{sheet}': {', '.join(unknown[:5])} "
                    f"{'is' if len(unknown) == 1 else 'are'} not listed in 'strains'.")

    if "osmoadaptation" in inp.tables:
        o = inp.tables["osmoadaptation"]
        bad = sorted(set(o.strategy.astype(str).str.strip()) - STRATEGIES)
        if bad:
            inp.problems.append(
                f"Sheet 'osmoadaptation': strategy must be 'salt-in' or 'salt-out'; "
                f"found {', '.join(bad)}.")
        p = pd.to_numeric(o.pool_p, errors="coerce")
        if p.isna().any() or (p < 0).any():
            inp.problems.append("Sheet 'osmoadaptation': pool_P must be a number of 0 or more.")

    if "abundances" in inp.tables:
        a = pd.to_numeric(inp.tables["abundances"].abundance, errors="coerce")
        if a.isna().any() or (a <= 0).any():
            inp.problems.append("Sheet 'abundances': abundance must be a positive number.")

    if "settings" in inp.tables:
        st = inp.tables["settings"]
        unknown = sorted(set(st.setting.astype(str)) - SETTINGS)
        if unknown:
            inp.warnings.append(
                f"Sheet 'settings': {', '.join(unknown)} "
                f"{'is' if len(unknown) == 1 else 'are'} not a setting the pipeline reads; "
                f"the default was used. Known settings: {', '.join(sorted(SETTINGS))}.")

    if "labels" in inp.tables:
        lb = inp.tables["labels"]
        vals = set(pd.to_numeric(lb.outcome, errors="coerce").dropna().unique())
        if not vals <= {0.0, 1.0}:
            inp.problems.append(
                "Sheet 'labels': outcome must be 1 (the pair interacts) or 0 (it does not).")
    return inp


def settings_dict(inp: Inputs) -> dict:
    if "settings" not in inp.tables:
        return {}
    st = inp.tables["settings"]
    out = {}
    for r in st.itertuples():
        v = r.value
        try:
            v = float(v)
        except (TypeError, ValueError):
            pass
        out[str(r.setting)] = v
    return out


def report(inp: Inputs) -> str:
    """One readable block: what is wrong, what was ignored, what was read."""
    lines = []
    if inp.problems:
        lines.append(f"{len(inp.problems)} problem(s) to fix before the pipeline can run:")
        lines += [f"  - {p}" for p in inp.problems]
    else:
        lines.append("The input is complete; the pipeline can run.")
    if inp.warnings:
        lines.append("")
        lines.append("Notes:")
        lines += [f"  - {w}" for w in inp.warnings]
    if inp.tables:
        lines.append("")
        lines.append("Read:")
        for k, v in inp.tables.items():
            lines.append(f"  - {k}: {len(v)} row(s)")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "ecomdesign_input_template.xlsx"
    models = sys.argv[2] if len(sys.argv) > 2 else None
    print(report(read_inputs(path, models_dir=models)))
