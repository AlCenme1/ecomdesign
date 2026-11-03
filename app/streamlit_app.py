"""ecomdesign — a browser interface to the pipeline.

Run locally:      streamlit run app/streamlit_app.py
Or deploy it free on Streamlit Community Cloud or Hugging Face Spaces.

The app asks for one workbook and a folder of metabolic models, checks them,
and runs the pipeline stage by stage. Nothing is computed until the input
passes validation, so the user never meets a Python traceback.
"""
from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecomdesign.io_excel import read_inputs, report  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE.parent / "templates"
INK, RED, CREAM = "#1B3A4B", "#8C1D1D", "#F4EDE4"

st.set_page_config(page_title="ecomdesign", page_icon="🧫", layout="wide")


# ---------------------------------------------------------------- appearance
def brand_header() -> None:
    logo = HERE / "assets" / "mfcb_logo.svg"
    svg = logo.read_text() if logo.exists() else ""
    b64 = base64.b64encode(svg.encode()).decode()
    st.markdown(
        f"""
        <style>
          .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
          .mf-head {{ display: flex; align-items: center; gap: 1.1rem;
                      border-bottom: 2px solid {RED}; padding-bottom: 0.9rem;
                      margin-bottom: 1.4rem; }}
          .mf-head img {{ height: 72px; }}
          .mf-title {{ font-family: Georgia, 'Times New Roman', serif; color: {INK};
                       font-size: 1.85rem; font-weight: 700; line-height: 1.1; margin: 0; }}
          .mf-sub {{ font-family: Georgia, serif; color: {RED}; font-size: 0.92rem;
                     letter-spacing: 0.04em; margin: 0.25rem 0 0; }}
          .mf-note {{ background: {CREAM}; border-left: 3px solid {RED};
                      padding: 0.75rem 1rem; font-size: 0.9rem; color: {INK};
                      border-radius: 3px; }}
          .stTabs [data-baseweb="tab"] {{ font-family: Georgia, serif; }}
          h2, h3 {{ font-family: Georgia, serif; color: {INK}; }}
        </style>
        <div class="mf-head">
          <img src="data:image/svg+xml;base64,{b64}" alt="MFCB">
          <div>
            <p class="mf-title">ecomdesign</p>
            <p class="mf-sub">Designing microbial consortia on an explicit energy scale</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f'<div class="mf-note">{text}</div>', unsafe_allow_html=True)


def download_button(df: pd.DataFrame, label: str, name: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode(), file_name=name,
                       mime="text/csv", use_container_width=True)


# ------------------------------------------------------------------- session
for key, default in [("inputs", None), ("models_dir", None), ("study", None),
                     ("solo", None), ("pairs", None), ("workdir", None)]:
    st.session_state.setdefault(key, default)


def workdir() -> str:
    if st.session_state.workdir is None:
        st.session_state.workdir = tempfile.mkdtemp(prefix="ecomdesign_")
    return st.session_state.workdir


brand_header()

tab_in, tab_check, tab_run, tab_res, tab_help = st.tabs(
    ["1 · Your data", "2 · Check", "3 · Run", "4 · Results", "How it works"])

# --------------------------------------------------------------- 1. the data
with tab_in:
    st.subheader("What to upload")
    note(
        "Two things: <b>one workbook</b> describing the study, and <b>one metabolic model "
        "per strain</b> as an SBML (.xml) or COBRA JSON file. The models are the only input "
        "a spreadsheet cannot replace — they are the reaction networks themselves. "
        "If you do not have them, they can be built from genomes with CarveMe or gapseq, "
        "or downloaded from BiGG, AGORA or ModelSEED.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Start from the template")
        for fn, label in [("ecomdesign_input_template.xlsx", "Empty template"),
                          ("ecomdesign_example.xlsx", "Filled example")]:
            p = TEMPLATES / fn
            if p.exists():
                st.download_button(f"Download the {label.lower()}", p.read_bytes(),
                                   file_name=fn, use_container_width=True,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.caption("Only two sheets are required: strains and medium. Every column header "
                   "carries a note explaining what goes in it.")

    with c2:
        st.markdown("#### Upload")
        book = st.file_uploader("The workbook (.xlsx)", type=["xlsx", "xls"])
        models = st.file_uploader(
            "The metabolic models (.xml, .json, or one .zip holding them)",
            type=["xml", "sbml", "json", "zip"], accept_multiple_files=True)

    if book is not None:
        d = workdir()
        book_path = os.path.join(d, "study.xlsx")
        with open(book_path, "wb") as f:
            f.write(book.getbuffer())
        mdir = os.path.join(d, "models")
        os.makedirs(mdir, exist_ok=True)
        for up in models or []:
            target = os.path.join(mdir, up.name)
            with open(target, "wb") as f:
                f.write(up.getbuffer())
            if up.name.lower().endswith(".zip"):
                with zipfile.ZipFile(target) as z:
                    for member in z.namelist():
                        if member.endswith("/") or member.startswith(("/", "..")):
                            continue
                        with z.open(member) as src, open(
                                os.path.join(mdir, os.path.basename(member)), "wb") as dst:
                            dst.write(src.read())
        st.session_state.inputs = read_inputs(book_path, models_dir=mdir if models else None)
        st.session_state.models_dir = mdir
        st.success("Workbook received. Open the Check tab.")

# ------------------------------------------------------------------ 2. check
with tab_check:
    st.subheader("Is the input complete?")
    inp = st.session_state.inputs
    if inp is None:
        st.info("Upload a workbook first.")
    else:
        if inp.ok:
            st.success("The input is complete. Nothing to fix.")
        else:
            st.error(f"{len(inp.problems)} thing(s) to fix before anything can be computed.")
            for p in inp.problems:
                st.markdown(f"- {p}")
        for w in inp.warnings:
            st.warning(w)
        if inp.tables:
            st.markdown("#### What was read")
            st.dataframe(pd.DataFrame(
                [{"sheet": k, "rows": len(v), "columns": ", ".join(v.columns)}
                 for k, v in inp.tables.items()]),
                hide_index=True, use_container_width=True)
            with st.expander("Look at a sheet"):
                which = st.selectbox("Sheet", list(inp.tables))
                st.dataframe(inp.tables[which], hide_index=True, use_container_width=True)

# -------------------------------------------------------------------- 3. run
with tab_run:
    st.subheader("Run the pipeline")
    inp = st.session_state.inputs
    if inp is None or not inp.ok:
        st.info("The Check tab has to come up clean first.")
    else:
        note(
            "Each stage is a separate optimisation, and thermodynamic solutions are "
            "slower than plain ones: a genome-scale strain takes seconds, a pair takes "
            "tens of seconds. For a library of more than about ten strains, run the "
            "pipeline from the notebook on your own machine rather than here.")
        use_thermo = st.toggle("Apply thermodynamics", value=True,
                               help="Off, this is ordinary flux balance analysis; on, every "
                                    "active reaction must keep a positive driving force.")
        stage = st.radio("Stage", ["Growth of each strain alone",
                                   "Pairwise outcomes (every ordered pair)"],
                         horizontal=False)
        if st.button("Run", type="primary"):
            from ecomdesign.study import Study
            bar = st.progress(0.0, text="Starting")

            def up(frac, msg):
                bar.progress(min(1.0, max(0.0, frac)), text=msg)

            try:
                study = st.session_state.study
                if study is None:
                    study = Study(inp, st.session_state.models_dir,
                                  cache_path=os.path.join(workdir(), "dg0_cache.pkl"))
                    study.load(progress=up)
                    st.session_state.study = study
                for n in study.notes:
                    st.warning(n)
                if stage.startswith("Growth"):
                    st.session_state.solo = study.solo(
                        thermo=(False, True) if use_thermo else (False,), progress=up)
                else:
                    st.session_state.pairs = study.pairs(
                        solo=st.session_state.solo, thermo=use_thermo, progress=up)
                bar.progress(1.0, text="Finished")
                st.success("Done. Open the Results tab.")
            except Exception as exc:                      # keep the UI honest
                bar.empty()
                st.error(f"The run stopped: {exc}")
                st.caption("If this names a reaction or metabolite identifier, it usually "
                           "means the medium in the workbook does not match the identifiers "
                           "your models use.")

# ---------------------------------------------------------------- 4. results
with tab_res:
    st.subheader("Results")
    solo, pairs = st.session_state.solo, st.session_state.pairs
    if solo is None and pairs is None:
        st.info("Nothing has been run yet.")
    if solo is not None:
        st.markdown("#### Growth alone")
        wide = solo.pivot_table(index=["strain", "strategy"], columns="method",
                                values="growth").reset_index()
        st.dataframe(wide, hide_index=True, use_container_width=True)
        if {"FBA", "TFA"} <= set(solo.method):
            blocked = wide[(wide.FBA > 0.01) & (wide.TFA <= 0.01)]
            if len(blocked):
                note("Thermodynamically blocked alone, although stoichiometry allows growth: "
                     + ", ".join(blocked.strain) +
                     ". A partner that supplies the right intermediate can lift this.")
        st.bar_chart(solo.pivot_table(index="strain", columns="method", values="growth"),
                     height=320)
        download_button(solo, "Download growth table (CSV)", "solo_growth.csv")

    if pairs is not None:
        st.markdown("#### Who grows beside whom")
        st.caption("Growth of the focal strain while its partner keeps 95% of its own "
                   "solo growth. Rows are partners, columns are focal strains.")
        mat = pairs.pivot_table(index="partner", columns="focal", values="growth_focal")
        st.dataframe(mat.style.background_gradient(cmap="Blues", axis=None)
                     .format("{:.3f}", na_rep="—"), use_container_width=True)
        download_button(pairs, "Download pairwise table (CSV)", "pairwise.csv")

    study = st.session_state.study
    if study is not None and "tolerance" in study.inputs.tables:
        st.markdown("#### Does the condition admit each strain?")
        note("This comes from your tolerance sheet, not from the models. A metabolic model "
             "has no notion of what a protein survives, so a strain can look perfectly "
             "viable here on metabolic grounds and still be impossible in the real pond.")
        c1, c2, c3 = st.columns(3)
        sal = c1.number_input("Salinity (% NaCl)", value=20.0, step=1.0)
        tmp = c2.number_input("Temperature (°C)", value=30.0, step=1.0)
        ph = c3.number_input("pH", value=7.5, step=0.1)
        tol = study.tolerance_filter(salinity=sal, temperature=tmp, pH=ph)
        st.dataframe(tol, hide_index=True, use_container_width=True)

# --------------------------------------------------------------------- help
with tab_help:
    st.subheader("What the pipeline does")
    st.markdown(
        """
Two cells can only feed each other within a window of concentrations: the
producer must keep the shared metabolite **low enough to release it**, and the
consumer needs it **high enough to use it**. When those two requirements do not
overlap, an exchange that looks perfect on paper cannot happen. The pipeline
computes that window once per strain, combines it with competition for
resources and with what each strain can make for the other, and turns the
result into a single number — an interaction energy — that ranks candidate
consortia.

**What it tells you**

- which strains grow alone, and which need a partner for thermodynamic reasons;
- for every ordered pair, how well one strain does beside a selfish partner;
- what osmoadaptation costs a strain that has to make its own compatible solutes;
- which strains a given salinity, temperature and pH admit — from your table,
  since no metabolic model contains that.

**What it cannot tell you.** Whether an organism survives the conditions. The
cost of making compatible solutes is in the model; the stability of its proteins
at molar ionic strength is not. That is why the tolerance sheet exists.
        """)
    st.markdown("#### Citing and reusing")
    st.caption("ecomdesign is open source under the MIT licence. "
               "If it contributed to your work, cite the paper and the archived "
               "release of the code.")
