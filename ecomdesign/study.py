"""Running a user's own study, described by the input workbook.

This is the driver the application calls. It takes the validated tables from
`io_excel`, loads the metabolic models they name, and runs the pipeline stage
by stage, reporting progress so that a user interface can show it.

    from ecomdesign.io_excel import read_inputs
    from ecomdesign.study import Study
    st = Study(read_inputs("my_study.xlsx", "models/"), "models/")
    st.load()                       # read models, split, attach energies
    solo = st.solo()                # growth of each strain alone
    pairs = st.pairs()              # directed pairwise outcomes
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .hetero import attach_thermo, community_opt_h, tfa_lazy, to_net
from .io_excel import Inputs, settings_dict

EPS = 1e-2
SELFISH = 0.95


def _noop(frac: float, message: str) -> None:
    pass


@dataclass
class Study:
    inputs: Inputs
    models_dir: str
    cache_path: str | None = None
    nets: dict = field(default_factory=dict)
    ext: list = field(default_factory=list)
    medium: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    # ------------------------------------------------------------------ setup
    @property
    def settings(self) -> dict:
        s = {"ceiling_mM": 0.1, "delta_kJ_per_mol": 0.1, "pH": 7.0,
             "ionic_strength_M": 0.25, "oxygen_max": 5.0, "consortium_size": 4,
             "thermodynamic_scope": "core"}
        s.update(settings_dict(self.inputs))
        return s

    @property
    def strains(self) -> pd.DataFrame:
        return self.inputs.tables["strains"]

    def _strategy(self, name: str) -> tuple[str, float]:
        o = self.inputs.tables.get("osmoadaptation")
        if o is None:
            return "salt-out", 0.0
        row = o[o.strain.astype(str) == name]
        if not len(row):
            return "salt-out", 0.0
        return str(row.strategy.iloc[0]).strip(), float(row.pool_p.iloc[0])

    def abundances(self, names: list) -> list:
        a = self.inputs.tables.get("abundances")
        if a is None:
            return [1.0 / len(names)] * len(names)
        m = {str(r.strain): float(r.abundance) for r in a.itertuples()}
        v = np.array([m.get(n, 0.0) for n in names], float)
        if v.sum() <= 0:
            return [1.0 / len(names)] * len(names)
        return list(v / v.sum())

    def _gibbs_from_workbook(self, models: dict) -> dict:
        """Turn the user's table of reaction energies into the signature-keyed
        cache that attach_thermo expects. A row without a strain applies to
        every model that has a reaction of that identifier."""
        g = self.inputs.tables.get("gibbs")
        if g is None:
            return {}
        from .hetero import reaction_signature
        out = {}
        for r in g.itertuples():
            rid = str(r.reaction)
            se = float(getattr(r, "std_error", 0.0) or 0.0)
            only = str(getattr(r, "strain", "") or "")
            for name, m in models.items():
                if only and only != name:
                    continue
                if rid in m.reactions:
                    out[repr(reaction_signature(m, rid))] = (float(r.dg0), se)
        if not out:
            self.notes.append(
                "None of the reaction identifiers in the 'gibbs' sheet were found in "
                "the models, so no energy was attached.")
        return out

    def load(self, progress=_noop) -> None:
        """Read every model, convert it to arrays and attach standard energies."""
        import cobra

        rows = list(self.strains.itertuples())
        models, bios = {}, {}
        for i, r in enumerate(rows):
            progress(i / (len(rows) + 1), f"Reading the model of {r.strain}")
            path = os.path.join(self.models_dir, str(r.model_file))
            m = (cobra.io.load_json_model(path) if path.lower().endswith(".json")
                 else cobra.io.read_sbml_model(path))
            m.id = str(r.strain)
            models[str(r.strain)] = m
            obj = [rx.id for rx in m.reactions if rx.objective_coefficient]
            bio = obj[0] if obj else next(
                (rx.id for rx in m.reactions if "biomass" in rx.id.lower()), None)
            if bio is None:
                self.notes.append(
                    f"{r.strain}: no biomass reaction was found, so this strain "
                    f"cannot grow in the model. Set one as the objective.")
                bio = m.reactions[0].id
            bios[str(r.strain)] = bio

        for name, m in models.items():
            strat, _pool = self._strategy(name)
            group = "resident"
            if "group" in self.strains.columns:
                g = self.strains[self.strains.strain.astype(str) == name].group
                if len(g) and pd.notna(g.iloc[0]):
                    group = str(g.iloc[0])
            self.nets[name] = to_net(m, name, bios[name], group=group, strategy=strat)

        progress(0.8, "Collecting standard Gibbs energies")
        cache = {}
        if self.cache_path and os.path.exists(self.cache_path):
            cache = pickle.load(open(self.cache_path, "rb"))
        cache.update(self._gibbs_from_workbook(models))
        if not cache:
            self.notes.append(
                "No standard Gibbs energies were available, so no reaction received a "
                "thermodynamic constraint and the thermodynamic solutions are identical "
                "to the plain ones. Supply them in the 'gibbs' sheet of the workbook, or "
                "install equilibrator-api and precompute them, before reading any result "
                "as thermodynamic.")
        attach_thermo(self.nets, models, cache, scope=str(self.settings["thermodynamic_scope"]))
        if self.cache_path:
            pickle.dump(cache, open(self.cache_path, "wb"))

        ext = []
        for t in self.nets.values():
            ext += [m for m, e in zip(t.met_ids, t.is_ext) if e]
        self.ext = sorted(set(ext))
        med = self.inputs.tables["medium"]
        self.medium = {str(r.metabolite): float(r.max_uptake) for r in med.itertuples()}
        unknown = [k for k in self.medium if k not in self.ext]
        if unknown:
            self.notes.append(
                "These medium components are not exchanged by any model, so they were "
                "ignored: " + ", ".join(unknown[:8]) +
                ("…" if len(unknown) > 8 else "") +
                ". Check that the identifiers match the ones your models use.")
        progress(1.0, "Models ready")

    # ------------------------------------------------------------- the stages
    def _solve(self, names, abund, thermo, pool, **kw):
        nets = [self.nets[n] for n in names]
        delta = float(self.settings["delta_kJ_per_mol"])
        if thermo:
            sol, _info = tfa_lazy(nets, abund, self.medium, self.ext,
                                  delta=delta, osm_pool=pool, **kw)
            return sol
        return community_opt_h(nets, abund, self.medium, self.ext, osm_pool=pool, **kw)

    def solo(self, thermo=(False, True), progress=_noop) -> pd.DataFrame:
        """Growth of each strain on its own, with and without thermodynamics."""
        rows, names = [], [str(s) for s in self.strains.strain]
        for i, n in enumerate(names):
            strat, pool = self._strategy(n)
            for th in thermo:
                progress(i / len(names), f"Growing {n} alone"
                                         f"{' with thermodynamics' if th else ''}")
                s = self._solve([n], [1.0], th, pool if strat == "salt-out" else 0.0)
                rows.append({"strain": n, "strategy": strat, "pool_P": pool,
                             "method": "TFA" if th else "FBA",
                             "growth": 0.0 if s is None else max(0.0, s["bio"][0])})
        progress(1.0, "Done")
        return pd.DataFrame(rows)

    def pairs(self, solo: pd.DataFrame | None = None, thermo=True,
              progress=_noop) -> pd.DataFrame:
        """For every ordered pair, the growth of the focal strain beside a
        partner that keeps at least 95% of its own solo growth."""
        names = [str(s) for s in self.strains.strain]
        if solo is None:
            solo = self.solo(thermo=(thermo,))
        ref = {r.strain: r.growth for r in
               solo[solo.method == ("TFA" if thermo else "FBA")].itertuples()}
        jobs = [(i, j) for i in names for j in names if i != j]
        rows = []
        for k, (i, j) in enumerate(jobs):
            progress(k / len(jobs), f"{j} beside {i}")
            _si, pi = self._strategy(i)
            _sj, pj = self._strategy(j)
            bmin = [SELFISH * ref.get(i, 0.0) if ref.get(i, 0.0) > EPS else 0.0, 0.0]
            s = self._solve([i, j], [0.5, 0.5], thermo, max(pi, pj),
                            obj_bio=[0.0, 1.0], bio_min=bmin)
            rows.append({"partner": i, "focal": j,
                         "method": "TFA" if thermo else "FBA",
                         "growth_focal": 0.0 if s is None else max(0.0, s["bio"][1]),
                         "feasible": s is not None})
        progress(1.0, "Done")
        return pd.DataFrame(rows)

    def tolerance_filter(self, salinity=None, temperature=None, pH=None) -> pd.DataFrame:
        """Which strains the user's own tolerance ranges admit in a given
        condition. The models say nothing about this; the table does."""
        t = self.inputs.tables.get("tolerance")
        names = [str(s) for s in self.strains.strain]
        if t is None:
            return pd.DataFrame({"strain": names, "admitted": True,
                                 "reason": "no tolerance data supplied"})
        rows = []
        for n in names:
            r = t[t.strain.astype(str) == n]
            ok, why = True, []
            if len(r):
                r = r.iloc[0]
                for value, lo, hi, label, unit in [
                        (salinity, "salinity_min", "salinity_max", "salinity", "% NaCl"),
                        (temperature, "temperature_min", "temperature_max", "temperature", "°C"),
                        (pH, "ph_min", "ph_max", "pH", "")]:
                    if value is None or lo not in r.index:
                        continue
                    a, b = r.get(lo), r.get(hi)
                    if pd.notna(a) and value < float(a):
                        ok = False
                        why.append(f"{label} below its range ({a}–{b} {unit})".strip())
                    elif pd.notna(b) and value > float(b):
                        ok = False
                        why.append(f"{label} above its range ({a}–{b} {unit})".strip())
            else:
                why.append("no tolerance data for this strain")
            rows.append({"strain": n, "admitted": ok,
                         "reason": "; ".join(why) if why else "within every range"})
        return pd.DataFrame(rows)
