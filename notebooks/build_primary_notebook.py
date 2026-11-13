"""
build_primary_notebook.py — builds notebooks/primary_analysis.ipynb: every analysis AND every figure.

The notebook starts from the saved derivatives (no preprocessing) and contains, in order:
  0     configuration and shared functions (inline)
  1–3   the statistics behind Figures 1–3, then the code that draws each figure
  4     DCM: reads the SPM outputs, tables, and draws the DCM figures (fitting stays in MATLAB)
  5     validation of every method on simulated data
  6     every Results number vs this notebook
  7     inventory of the figure files written

The analysis cells are taken from build_results_notebook.py (same folder). The figure cells are built
from the CURRENT figure scripts in scripts/: each script's code is placed inline in the notebook,
inside one function per figure, so its settings and helper functions cannot overwrite the notebook's
variables. Rebuild after editing a figure script:

    python notebooks/build_primary_notebook.py

Output: notebooks/primary_analysis.ipynb
"""
import ast
import re
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPTS = ROOT / "scripts"

# ----------------------------------------------------------------------------- analysis cells
# Run the results-notebook builder up to (not including) its write step to obtain its cell list.
src = (HERE / "build_results_notebook.py").read_text()
src = src[: src.index("nb = nbf.v4.new_notebook()")] if "nb = nbf.v4.new_notebook()" in src else src
ns = {}
exec(compile(src, "build_results_notebook.py", "exec"), ns)
cells = ns["cells"]
md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s: nbf.v4.new_code_cell(s.strip("\n"))


def heading_index(prefix):
    for i, c in enumerate(cells):
        if c.cell_type == "markdown" and c.source.lstrip().startswith(prefix):
            return i
    raise KeyError(prefix)


def figure_cells(script, fn_name, calls, title, outputs, edits=(), inner="main()"):
    """Markdown (the script's own docstring) + one code cell holding the script inline.

    The script body is wrapped in `def fn_name(...)`, so its CONFIG constants and helpers are local to
    the figure. `ROOT` is set to the notebook's project root, the Agg backend line is removed, the
    `if __name__ == "__main__"` block is replaced by `calls`, and the saved PNGs are displayed.
    """
    text = (SCRIPTS / script).read_text()
    tree = ast.parse(text)
    doc = ast.get_docstring(tree) or ""
    first = tree.body[0]
    body = text.splitlines()[first.end_lineno:] if isinstance(first, ast.Expr) else text.splitlines()
    body = "\n".join(body)
    body = body[: body.index('if __name__ == "__main__":')].rstrip()
    body = re.sub(r"^matplotlib\.use\([^)]*\)\n", "", body, flags=re.M)
    body = re.sub(r"^ROOT = Path\(__file__\)\.resolve\(\)\.parents\[1\]", "ROOT = PROJECT_ROOT", body, flags=re.M)
    for old, new in edits:
        assert old in body, f"{script}: '{old}' not found"
        body = body.replace(old, new)
    indented = "\n".join(("    " + ln) if ln.strip() else "" for ln in body.splitlines())
    args, call_lines = calls
    cell = (f"# ---- {script} (inline; rebuild the notebook after editing the script)\n"
            f"def {fn_name}({args}):\n{indented}\n\n    return {inner}\n\n\n{call_lines}\n\n"
            "plt.close('all')\n"
            f"for f in {outputs!r}:\n"
            "    p = PROJECT_ROOT / 'figures' / f\n"
            "    display(Image(filename=str(p), width=900) if p.exists() else f'missing: {p}')")
    note = f"### {title}\n\n`scripts/{script}`. From the script's header:\n\n```text\n{doc}\n```"
    return [md(note), code(cell)]


SETUP = code(r"""
# ---------------------------------------------------------------- figures: shared settings
from IPython.display import Image, display
import sys
PROJECT_ROOT = ROOT                      # every figure function reads its inputs relative to this
(PROJECT_ROOT / "figures").mkdir(exist_ok=True)
FIG_LOG = []                             # figures made in this session (section 7)

def run_figure(name, fn, *args, **kwargs):
    '''Run one figure function; report (rather than stop the notebook on) a missing input.'''
    t0 = time.time()
    try:
        fn(*args, **kwargs)
        FIG_LOG.append((name, "ok", round(time.time() - t0, 1)))
    except Exception as e:                # e.g. template head or DCM fits not available on this computer
        FIG_LOG.append((name, f"FAILED: {type(e).__name__}: {e}", round(time.time() - t0, 1)))
        print(f"{name} failed: {type(e).__name__}: {e}")
""")

FIG1 = figure_cells(
    "14_figure1.py", "make_figure1", ("lean=False, erp_only=False", "\n".join([
        "run_figure('Figure 1', make_figure1)                         # full figure (A–F)",
        "run_figure('Figure 1 lean', make_figure1, lean=True)         # lean version (fewer topography bins)",
        "run_figure('Figure 1 D–F print', make_figure1, erp_only=True)  # D–F at print size, editable text"])),
    "Figure 1 — design, topographies, ERPs and participant effects",
    ["Figure1.png", "Figure1_lean.png", "Figure1_DE.png"],
    edits=[('LEAN = "--lean" in sys.argv', "LEAN = lean"),
           ('ERP_ONLY = "--erp" in sys.argv', "ERP_ONLY = erp_only")])
FIG2A = figure_cells("15_figure2_schematic.py", "make_figure2_schematic",
                     ("", "run_figure('Figure 2A–B', make_figure2_schematic)"),
                     "Figure 2A–B — model schematic and regressors", ["Figure2_schematic.png"],
                     inner="main(SUBJECT)")
FIG2 = figure_cells("16_figure2_results.py", "make_figure2_results",
                    ("", "run_figure('Figure 2C–F', make_figure2_results)"),
                    "Figure 2C–F — model comparison results", ["Figure2_results.png"])
FIG2R = figure_cells("26_figure2_recovery.py", "make_figure2_recovery",
                     ("", "run_figure('Figure 2 recovery', make_figure2_recovery)"),
                     "Figure 2B (alternative) — model recovery", ["Figure2_recovery.png"])
FIG3 = figure_cells("19_figure3.py", "make_figure3", ("", "run_figure('Figure 3', make_figure3)"),
                    "Figure 3 — auditory-cortex localisation of the local effect", ["Figure3.png"])
DCMW = figure_cells("28_figure_dcm_windows.py", "make_figure_dcm_windows",
                    ("", "run_figure('DCM windows', make_figure_dcm_windows)"),
                    "DCM figure — full response (0–600 ms) vs MMN window (0–250 ms)", ["Figure_DCM_windows.png"])
DCMR = figure_cells("24_figure_dcm_robust.py", "make_figure_dcm_robust",
                    ("", "run_figure('DCM robustness', make_figure_dcm_robust)"),
                    "Supplementary — DCM robustness across five analyses (0/1 coding)", ["Figure_DCM_robust.png"])
DCMM = figure_cells("22_figure_dcm.py", "make_figure_dcm_main",
                    ("", "run_figure('DCM main model', make_figure_dcm_main)"),
                    "Supplementary — main DCM (0/1 coding, 0–600 ms)", ["Figure_DCM.png"])
DCMP = figure_cells("23_dcm_predicted_erps.py", "make_figure_dcm_predicted",
                    ("", "run_figure('DCM predicted ERPs', make_figure_dcm_predicted)"),
                    "Supplementary — ERPs predicted by the DCM vs observed", ["DCM_predicted_ERPs.png"])

DCM_INTRO = [md(r"""
## 4. DCM — effective connectivity (results read from SPM)

The DCMs are fitted in MATLAB/SPM12 and cannot run here. Run order (from `scripts/dcm/`):
`dcm_01_prepare.m` → `dcm_02_fit.m` → `dcm_03_peb.m` → `dcm_04_robustness.m` → (after
`python scripts/25_dcm_surprise_export.py`) `dcm_05_surprise.m` (0–600 ms) → `dcm_06_early.m` (0–250 ms) →
`dcm_07_search_full.m`. This section reads their outputs.

**Codings.** *Surprise*: each participant's condition-mean learner surprise (M2 local, M3 global), relative to
LS/GS, scaled to a maximum of 1. *Interaction*: 0/1 labels plus a local × global term. **Credible** = posterior
probability of being present > .95 after per-connection Bayesian model reduction (`Pp_search`; falls back to
the sign probability `Pp` where that column is missing). **Robust** = credible with the same sign in both codings.
"""), code(r"""
# ---------------------------------------------------------------- DCM: connection tables per window x coding
DCM_DIR = ROOT / "derivatives" / "dcm"
dcm_tabs = []
for win, sub in [("0–600 ms", "surprise_coded"), ("0–250 ms", "early_window")]:
    for cod in ["surprise", "interaction"]:
        f = DCM_DIR / sub / cod / "connections.csv"
        if not f.exists():
            print("missing:", f); continue
        d = pd.read_csv(f)
        pp = "Pp_search" if "Pp_search" in d else "Pp"
        ep = "Ep_search" if "Ep_search" in d else "Ep"
        dcm_tabs.append(d.assign(window=win, coding=cod, Ep_use=d[ep], credible=d[pp] > 0.95, stat=pp))
dcm = pd.concat(dcm_tabs, ignore_index=True)
print("credibility statistic used:", dcm.groupby(["window", "coding"])["stat"].first().to_dict())

# robust = credible with the same sign in both codings
rob = (dcm[dcm.effect.isin(["local", "global"])]
       .groupby(["window", "effect", "connection"])
       .apply(lambda g: pd.Series(dict(robust=bool(g.credible.all() and np.sign(g.Ep_use).nunique() == 1),
                                       mean_log_gain=g.Ep_use.mean(),
                                       **{f"Ep_{c}": float(g.loc[g.coding == c, "Ep_use"].iloc[0]) for c in g.coding})),
              include_groups=False)
       .reset_index())
for (win, eff), g in rob[rob.robust].groupby(["window", "effect"], sort=False):
    print(f"\n{win}, {eff}: robust connection changes (log gain)")
    print(g[["connection", "mean_log_gain", "Ep_surprise", "Ep_interaction"]].round(2).to_string(index=False))
""")]

DCM_SURPRISE = [md(r"""
### 4.1 Main DCM results: surprise coding (Table X)

The Results report the surprise coding. The table gives the log gain of every modulatory parameter after
per-connection Bayesian model reduction. `*` = credible (posterior probability of presence > .95), `—` = pruned
(estimate 0), and the family with the highest posterior probability is shown for each effect. The interaction coding
(above) is the sensitivity check.
"""), code(r"""
# ---------------------------------------------------------------- DCM: surprise-coded table (Table X)
CONN_ORDER = ["lA1->lSTG", "rA1->rSTG", "lSTG->lIFG", "rSTG->rIFG", "lSTG->lIPL", "rSTG->rIPL",      # forward
              "lSTG->lA1", "rSTG->rA1", "lIFG->lSTG", "rIFG->rSTG", "lIPL->lSTG", "rIPL->rSTG"]      # backward
FAMS = ["none", "forward", "backward", "both"]

def fmt(ep, cred):
    return "—" if abs(ep) < 0.005 else f"{ep:+.2f}{'*' if cred else ''}"

sur = dcm[dcm.coding == "surprise"]
tableX = pd.DataFrame(index=CONN_ORDER)
for eff in ["local", "global"]:
    for win in ["0–250 ms", "0–600 ms"]:
        g = sur[(sur.effect == eff) & (sur.window == win)].set_index("connection").reindex(CONN_ORDER)
        tableX[f"{eff} {win}"] = [fmt(e, c) for e, c in zip(g.Ep_use, g.credible)]
tableX.insert(0, "type", ["forward"] * 6 + ["backward"] * 6)

# family posteriors (none / forward / backward / both) from the saved PEB/BMR results
fam_row = {}
for win, sub in [("0–250 ms", "early_window"), ("0–600 ms", "surprise_coded")]:
    fpeb = DCM_DIR / sub / "surprise" / "PEB_BMR.mat"
    if not fpeb.exists():
        print("missing:", fpeb); fam_row.update({f"{e} {win}": "n/a" for e in ["local", "global"]}); continue
    zz = loadmat(fpeb, squeeze_me=True, struct_as_record=False)
    for eff, f in zip(["local", "global"], np.atleast_1d(zz["fam"])):
        post = np.atleast_1d(f.family.post).astype(float)
        fam_row[f"{eff} {win}"] = f"{FAMS[int(np.argmax(post))]} {post.max():.2f}"
tableX.loc["family"] = [""] + [fam_row[c] for c in tableX.columns[1:]]
print(tableX.to_string())
tableX.to_csv(DCM_DIR / "table_dcm_surprise.csv")
print("\n->", DCM_DIR / "table_dcm_surprise.csv")
""")]

FIG4 = figure_cells("29_figure_dcm_main.py", "make_figure4_dcm",
                    ("", "run_figure('Figure 4 DCM', make_figure4_dcm)"),
                    "Figure 4 — DCM networks: local, global and interaction, both windows", ["Figure4_DCM.png"])
FIGS = figure_cells("30_figure_dcm_supplement.py", "make_figure_dcm_supplement",
                    ("", "run_figure('DCM supplement (fit, estimates)', make_figure_dcm_supplement)"),
                    "Supplementary — DCM model fit and all parameter estimates",
                    ["FigureS_DCM_fit.png", "FigureS_DCM_estimates.png"])

CROSSCHECKS = [md(r"""
## 7. Cross-checks raised in external review (under discussion with collaborators)

Two analyses run in response to a review of the global learner. Neither changes the Results unless the team adopts it.

**7.1 Alternative global-learner coding.** The global learner (M3) observes whether the fifth tone changed. Within a
block this identifies which of the block's two quintlets was played, so it is equivalent to coding "quintlet differs
from the block standard" whenever counts reset at block starts (identical surprise). Without a reset the two differ:
the alternative needs the block's standard, which the listener can only know by having learned the rule. Here every
model is refitted with the alternative coding to check that the model comparison does not depend on it.

**7.2 Joint model.** Per participant, single-trial amplitude ~ drift + LD + GD + LD×GD + local surprise + global
surprise (all z-scored), then a t-test on each surprise slope across participants. This asks whether each learner
explains amplitude beyond the condition labels, slow drift and the other learner at the same time.
"""), code(r"""
# ---------------------------------------------------------------- 7.1 alternative global coding: equivalence + refit
RUN_CROSSCHECKS = True          # the refit takes a few minutes (all participants x 6 models x 2 windows)

def global_alt_outcomes(qs):
    # 1 = quintlet differs from its block's standard (the most frequent quintlet in the block)
    std = qs.groupby("block")["quintlet"].agg(lambda x: x.mode()[0])
    return (qs["quintlet"] != qs["block"].map(std)).astype(int).to_numpy()

# equivalence: with a block reset, both codings give identical surprise at every half-life
eq = []
for s in subs:
    qs, _ = load_sequence(s)
    start = (qs["trial_in_block"] == 0).to_numpy()
    ours = (qs["quintlet"].str[4] != qs["quintlet"].str[0]).astype(int).to_numpy()
    alt = global_alt_outcomes(qs)
    for h in H_GLOBAL:
        for r in RESETS:
            a, b = leaky_surprise(ours, h, start if r else None), leaky_surprise(alt, h, start if r else None)
            eq.append(dict(reset=r, h=h, max_abs_diff=np.abs(a - b).max(), r=np.corrcoef(a, b)[0, 1]))
print(pd.DataFrame(eq).groupby(["reset", "h"])[["max_abs_diff", "r"]].agg(["min", "max"]).round(3))

if RUN_CROSSCHECKS:
    rows = []
    for s in subs:
        tr, bank = load_trials(s), regressor_bank(s)
        qs, _ = load_sequence(s)
        start = (qs["trial_in_block"] == 0).to_numpy()
        alt = global_alt_outcomes(qs)
        bank_alt = dict(bank)
        for h in H_GLOBAL:
            for r in RESETS:
                bank_alt[("glob", h, r)] = leaky_surprise(alt, h, start if r else None)
        for comp in ["local_80-140ms", "global_300-700ms"]:
            for m in MODELS:
                rows.append(dict(subject=s, component=comp, model=m,
                                 ll_alt=nested_cv(tr, tr[comp].to_numpy(), bank_alt, m)[0]))
    alt_cv = pd.DataFrame(rows)
    cv = pd.read_csv(MC_DIR / "cv_loglik.tsv", sep="\t").merge(alt_cv, on=["subject", "component", "model"])
    print("\nlargest change in held-out log-likelihood (0 expected for M0-M2):")
    print((cv.ll_alt - cv.cv_ll).abs().groupby([cv.component, cv.model]).max().round(2).unstack(0))
    for comp in ["local_80-140ms", "global_300-700ms"]:
        for col, lab in [("cv_ll", "original"), ("ll_alt", "alternative")]:
            L = cv[cv.component == comp].pivot(index="subject", columns="model", values=col)[MODELS].to_numpy()
            b = rfx_bms(L, n_samples=100000, seed=0)
            print(f"{comp:17s} {lab:11s}: PXP " + ", ".join(f"{m.split()[0]} {p:.2f}" for m, p in zip(MODELS, b['pxp']))
                  + f"; BOR {b['bor']:.3f}")
"""), code(r"""
# ---------------------------------------------------------------- 7.2 joint model: both learners + condition + drift
clocks = pd.read_csv(MC_DIR / "clocks.tsv", sep="\t")
zsc = lambda x: (x - x.mean()) / x.std()
out = []
for s in subs:
    d = load_trials(s).merge(learner_surprise(s, clocks)[["seq_idx", "S_loc", "S_glob"]], on="seq_idx")
    ld = (d["local"] == "LD").astype(float) - 0.5
    gd = (d["global"] == "GD").astype(float) - 0.5
    X = np.column_stack([np.ones(len(d)), zsc(d.seq_idx), ld, gd, 4 * ld * gd, zsc(d.S_loc), zsc(d.S_glob)])
    for comp in ["local_80-140ms", "global_300-700ms"]:
        b = np.linalg.lstsq(X, d[comp].to_numpy(), rcond=None)[0]
        out.append(dict(subject=s, window=comp, local_surprise=b[5], global_surprise=b[6]))
o = pd.DataFrame(out)
for comp, g in o.groupby("window"):
    for k in ["local_surprise", "global_surprise"]:
        t = stats.ttest_1samp(g[k], 0)
        n_same = int((np.sign(g[k]) == np.sign(g[k].mean())).sum())
        print(f"{comp:17s} {k:15s}: {g[k].mean():+.3f} µV/SD, t(14) = {t.statistic:+.2f}, p = {t.pvalue:.3g}, {n_same}/15")
""")]

INVENTORY = [md(r"""
## 8. Figure inventory

Every figure function run in this session, whether it succeeded, and the files now in `figures/`.
"""), code(r"""
print(pd.DataFrame(FIG_LOG, columns=["figure", "status", "seconds"]).to_string(index=False))
inv = sorted((PROJECT_ROOT / "figures").glob("*.pdf"))
print(f"\n{len(inv)} PDF figures in figures/ (each also as .png, most as .svg):")
for p in inv:
    print(f"  {p.name:34s} {time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))}")
""")]

# ----------------------------------------------------------------------------- assemble
cells[0] = md(r"""
# Local–global EEG: primary analysis notebook

Every analysis in the Results and every figure, in one place, starting from the saved derivatives.

| Section | What it does | Figure files |
|---|---|---|
| 0 | configuration, shared functions (inline) | — |
| 1 | MMN / P3b window tests, clusters, 2 × 2 ANOVA, reference control | `Figure1`, `Figure1_lean`, `Figure1_DE` |
| 2 | learners, cross-validated model comparison, recovery, surprise–amplitude relations, reset | `Figure2_schematic`, `Figure2_results`, `Figure2_recovery` |
| 3 | eLORETA cluster, auditory-cortex time course, spread checks, global null | `Figure3` |
| 4 | DCM: interaction-coding check, surprise-coded table (Table X), DCM figures | `Figure4_DCM`, `FigureS_DCM_fit`, `FigureS_DCM_estimates`, `Figure_DCM_windows`, `Figure_DCM_robust`, `Figure_DCM`, `DCM_predicted_ERPs` |
| 5 | validation of every method on simulated data | — |
| 6 | every Results number vs this notebook | — |
| 7 | cross-checks from external review (alternative global coding; joint model) | — |
| 8 | figure inventory | — |

**Not in this notebook:** preprocessing (`scripts/01`–`02`), the single-trial exports (`07`), the full
model-comparison and recovery fits (`06`; loaded here unless `RECOMPUTE_ALL_CV = True`), source extraction
(`11`, `18 --extract`) and the DCM fits (MATLAB, `scripts/dcm/`).

**How the figure cells work.** Each figure's code is the current figure script, placed inline inside one function
(`make_figure1()`, …) so its settings cannot change the notebook's variables. Figures are written to `figures/`
(PDF/SVG with editable text, PNG) and shown below each cell. If an input is missing on this computer (e.g. the
template head for Figure 3, or the DCM fits), that figure reports the problem and the notebook continues.
To change a figure, edit its script and rebuild this notebook (`python notebooks/build_primary_notebook.py`).

Run top to bottom with the `lg-eeg` environment. Total time: about 20–30 minutes (set `RUN_CROSSCHECKS = False` in Section 7 to skip the refit).
""")

i2, i3, i4, i5 = (heading_index(p) for p in ("## 2.", "## 3.", "## 4.", "## 5."))
cells[i4].source = cells[i4].source.replace("## 4.", "## 5.", 1)
cells[i5].source = cells[i5].source.replace("## 5.", "## 6.", 1)
new = (cells[:i2] + [SETUP] + FIG1 + cells[i2:i3] + FIG2A + FIG2 + FIG2R + cells[i3:i4] + FIG3
       + DCM_INTRO + DCM_SURPRISE + FIG4 + FIGS + DCMW + DCMR + DCMM + DCMP + cells[i4:] + CROSSCHECKS + INVENTORY)
nb = nbf.v4.new_notebook()
nb.cells = new
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (lg-eeg)", "language": "python"}
out = HERE / "primary_analysis.ipynb"
nbf.write(nb, out)
print(f"{out}  ({len(new)} cells)")
