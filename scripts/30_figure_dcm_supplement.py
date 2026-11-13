"""
30_figure_dcm_supplement.py — supplementary DCM figures for the surprise- and interaction-coded models
(0–600 ms: dcm_05_surprise.m + dcm_07_search_full.m; 0–250 ms: dcm_06_early.m). Each figure 173.695 mm wide.

FigureS_DCM_fit — how well the models reproduce the data
  Columns: the four analyses (window x coding).
  A  Local effect (LD − LS) at the frontocentral ROI: observed (black, ± SEM) vs model (coloured, dashed).
  B  Global effect (GD − GS) at the parietal ROI, same layout.
  C  Variance explained per participant (dots; bar = median).
  Data and predictions are in the DCM's channel space after projection onto its 8 spatial modes (a.u.).
  The text in each panel gives the correlation between the group-mean observed and predicted waves.

FigureS_DCM_estimates — every modulatory parameter
  A  Family posteriors: probability that each effect modulates no / forward / backward / both connections.
  B–D  Log gain (posterior mean ± 90% credible interval, Bayesian model average after per-connection
     reduction) for every connection, for local, global and local x global effects. Filled = credible
     (posterior probability of presence > .95); open = not credible; pruned parameters sit at 0.
     Connections are ordered along the hierarchy (A1 -> STG -> IFG / IPL, then the backward connections).

Inputs : derivatives/dcm/{surprise_coded,early_window}/<coding>/{DCM_sub-*.mat, PEB_BMR.mat, connections.csv}
Outputs: figures/FigureS_DCM_fit.*, figures/FigureS_DCM_estimates.*, derivatives/dcm/supplement_fit_stats.json

Usage:  python scripts/30_figure_dcm_supplement.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy import stats

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
DCM_DIR = ROOT / "derivatives" / "dcm"
FIG_DIR = ROOT / "figures"
WIDTH_MM = 173.695
ANALYSES = [("surprise_coded", "surprise", "0–600 ms", "Surprise"), ("surprise_coded", "interaction", "0–600 ms", "Interaction"),
            ("early_window", "surprise", "0–250 ms", "Surprise"), ("early_window", "interaction", "0–250 ms", "Interaction")]
CONDS = ["LSGS", "LDGS", "LSGD", "LDGD"]                                  # trial order in the DCM
LEVELS = {"LS": [0, 2], "LD": [1, 3], "GS": [0, 1], "GD": [2, 3]}
EFFECT_ROWS = [("local", ["Fz", "FC1", "FC2", "F1", "F2", "Cz"], ("LS", "LD"), "Local effect, LD − LS (a.u.)\nfrontocentral ROI"),
               ("global", ["Pz", "P1", "P2", "CPz", "POz", "CP1", "CP2"], ("GS", "GD"), "Global effect, GD − GS (a.u.)\nparietal ROI")]
# connections along the hierarchy: forward (A1->STG, STG->IFG, STG->IPL), then backward; left then right
CONN_ORDER = ["lA1->lSTG", "rA1->rSTG", "lSTG->lIFG", "rSTG->rIFG", "lSTG->lIPL", "rSTG->rIPL",
              "lSTG->lA1", "rSTG->rA1", "lIFG->lSTG", "rIFG->rSTG", "lIPL->lSTG", "rIPL->rSTG"]
N_FWD = 6
FAMILIES = ["none", "forward", "backward", "both"]
C_EFF = {"local": "#C44E52", "global": "#8172B3", "interaction": "0.3"}
C_ANALYSIS = {("0–600 ms", "Surprise"): "#4C72B0", ("0–600 ms", "Interaction"): "#9EB9D9",
              ("0–250 ms", "Surprise"): "#DD8452", ("0–250 ms", "Interaction"): "#F2C29B"}
MARK = {"Surprise": "o", "Interaction": "s"}
Z90 = stats.norm.ppf(0.95)
TEXT_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
           "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
           "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
           "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
           "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
           "axes.labelpad": 2}
# ------------------------------------------------------------------------------------


def dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x, float)


def load_fit(f):
    """Observed and predicted channel waves (cond x chan x time) and variance explained for one DCM."""
    D = loadmat(f, squeeze_me=True, struct_as_record=False, variable_names=["DCM"])["DCM"]
    U = dense(D.M.U)
    scale = float(D.xY.scale)                     # SPM rescales the data before fitting; undo it
    H = [dense(h) / scale for h in np.atleast_1d(D.H)]
    R = [dense(r) / scale for r in np.atleast_1d(D.R)]
    r2 = 1 - sum((x ** 2).sum() for x in R) / sum(((h + x) ** 2).sum() for h, x in zip(H, R))
    return dict(pred=np.stack([(h @ U.T).T for h in H]), obs=np.stack([((h + r) @ U.T).T for h, r in zip(H, R)]),
                chans=list(D.xY.name), t=np.asarray(D.xY.pst, float), r2=float(r2))


def level(X, lv):
    return X[:, LEVELS[lv]].mean(1)


def figure_fit(stats_out):
    mm = 1 / 25.4
    fig, axes = plt.subplots(3, 4, figsize=(WIDTH_MM * mm, 118 * mm),
                             gridspec_kw=dict(left=0.1, right=0.99, top=0.9, bottom=0.08, hspace=0.55, wspace=0.32,
                                              height_ratios=[1, 1, 0.8]))
    for j, (sub, cod, win, clab) in enumerate(ANALYSES):
        fits = [load_fit(f) for f in sorted((DCM_DIR / sub / cod).glob("DCM_sub-*.mat"))]
        t, chans = fits[0]["t"], fits[0]["chans"]
        col = C_ANALYSIS[(win, clab)]
        key = f"{win} {clab}"
        stats_out[key] = dict(r2_median=float(np.median([f["r2"] for f in fits])),
                              r2_range=[float(min(f["r2"] for f in fits)), float(max(f["r2"] for f in fits))])
        for i, (eff, roi, (std, dev), rlab) in enumerate(EFFECT_ROWS):
            ci = [chans.index(c) for c in roi]
            O = np.stack([f["obs"][:, ci].mean(1) for f in fits])        # subj x cond x time
            P = np.stack([f["pred"][:, ci].mean(1) for f in fits])
            dO, dP = level(O, dev) - level(O, std), level(P, dev) - level(P, std)
            ax = axes[i, j]
            m, se = dO.mean(0), dO.std(0, ddof=1) / np.sqrt(len(dO))
            ax.fill_between(t, m - se, m + se, color="0.75", lw=0)
            ax.plot(t, m, color="k", lw=0.9, label="observed")
            ax.plot(t, dP.mean(0), color=col, lw=1.1, ls="--", label="model")
            r = float(np.corrcoef(m, dP.mean(0))[0, 1])
            stats_out[key][f"{eff}_group_r"] = r
            ax.text(0.97, 0.05 if eff == "local" else 0.92, f"r = {r:.2f}", transform=ax.transAxes, ha="right",
                    va="bottom" if eff == "local" else "top", fontsize=6)
            ax.axhline(0, color="k", lw=0.4)
            ax.axvline(0, color="k", lw=0.4)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_fontsize(5.5)
            ax.set_xlim(0, 600 if win.startswith("0–600") else 250)
            ax.set_xticks([0, 200, 400, 600] if win.startswith("0–600") else [0, 100, 200])
            sns.despine(ax=ax)
            if j == 0:
                ax.set_ylabel(rlab)
            if i == 0:
                ax.set_title(f"{win}\n{clab} coding", color=col if clab == "Surprise" else "0.25")
            if i == 1:
                ax.set_xlabel("Time from 5th sound (ms)")
            if i == 0 and j == 0:
                fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="upper left", ncol=2,
                           bbox_to_anchor=(0.005, 1.0), handlelength=1.8)
        ax = axes[2, j]
        v = np.array([f["r2"] for f in fits]) * 100
        ax.scatter(np.random.default_rng(j).uniform(-0.18, 0.18, len(v)), v, s=6, color=col, lw=0)
        ax.plot([-0.3, 0.3], [np.median(v)] * 2, color="k", lw=1.1)
        ax.text(0.42, np.median(v), f"{np.median(v):.0f}%", va="center", fontsize=6)
        ax.set_xlim(-0.5, 0.9)
        ax.set_ylim(75, 100)
        ax.set_xticks([])
        sns.despine(ax=ax, bottom=True)
        if j == 0:
            ax.set_ylabel("Variance explained (%)")
    for i, L in enumerate("ABC"):
        fig.text(0.01, [0.92, 0.6, 0.29][i], L, fontsize=9, weight="bold", va="top")
    return fig


def load_estimates():
    rows, fams = [], []
    for sub, cod, win, clab in ANALYSES:
        d = pd.read_csv(DCM_DIR / sub / cod / "connections.csv")
        z = loadmat(DCM_DIR / sub / cod / "PEB_BMR.mat", squeeze_me=True, struct_as_record=False)
        B = z["BMAs"]
        Cp = dense(B.Cp)
        sd = np.sqrt(np.clip(np.diag(Cp) if Cp.ndim == 2 and Cp.shape[0] == Cp.shape[1] and Cp.shape[0] > 1
                             else Cp.ravel(), 0, None))
        assert len(sd) == len(d), "BMA parameters do not match connections.csv"
        rows.append(d.assign(window=win, coding=clab, sd=sd, credible=d["Pp_search"] > 0.95))
        effs = list(dict.fromkeys(d["effect"]))
        for k, f in enumerate(np.atleast_1d(z["fam"])):
            post = np.atleast_1d(f.family.post).astype(float)
            fams.append(dict(window=win, coding=clab, effect=effs[k], **dict(zip(FAMILIES, post))))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(fams)


def figure_estimates():
    est, fam = load_estimates()
    mm = 1 / 25.4
    fig = plt.figure(figsize=(WIDTH_MM * mm, 112 * mm))
    # ---- A: family posteriors (stacked bars)
    axA = fig.add_axes([0.17, 0.6, 0.2, 0.33])
    fam["label"] = (fam["effect"].map({"local": "Local", "global": "Global", "interaction": "L × G"}) + ", "
                    + fam["window"] + ", " + fam["coding"].str.lower().str[:4] + ".")
    fam = fam.sort_values(["effect", "window", "coding"], key=lambda s: s.map(
        {"local": 0, "global": 1, "interaction": 2, "0–600 ms": 0, "0–250 ms": 1, "Surprise": 0, "Interaction": 1})).reset_index(drop=True)
    left = np.zeros(len(fam))
    fcol = {"none": "0.85", "forward": "#55A868", "backward": "#DD8452", "both": "#4C72B0"}
    for fname in FAMILIES:
        axA.barh(np.arange(len(fam)), fam[fname], left=left, color=fcol[fname], height=0.75, label=fname)
        left += fam[fname].to_numpy()
    axA.set_yticks(np.arange(len(fam)), fam["label"], fontsize=5.4)
    for tl, eff in zip(axA.get_yticklabels(), fam["effect"]):
        tl.set_color(C_EFF[eff])
    axA.invert_yaxis()
    axA.set_xlim(0, 1)
    axA.set_xlabel("Family posterior probability")
    axA.legend(frameon=False, ncol=4, loc="lower right", bbox_to_anchor=(1.0, 1.0), handlelength=0.9, columnspacing=0.6, fontsize=5.5)
    sns.despine(ax=axA)
    fig.text(0.01, 0.985, "A", fontsize=9, weight="bold", va="top")
    # ---- B–D: forest plots per effect
    panels = [("local", [0.47, 0.6, 0.25, 0.33], "B"), ("global", [0.745, 0.6, 0.25, 0.33], "C"),
              ("interaction", [0.47, 0.09, 0.25, 0.33], "D")]
    ypos = {c: i for i, c in enumerate(CONN_ORDER)}
    offsets = {k: o for k, o in zip(C_ANALYSIS, [-0.27, -0.09, 0.09, 0.27])}
    for eff, rect, L in panels:
        ax = fig.add_axes(rect)
        ax.axvline(0, color="k", lw=0.5)
        ax.axhspan(N_FWD - 0.5, len(CONN_ORDER) - 0.5, color="0.95", lw=0, zorder=0)
        g = est[est["effect"] == eff]
        for (win, clab), col in C_ANALYSIS.items():
            q = g[(g.window == win) & (g.coding == clab)].set_index("connection").reindex(CONN_ORDER).dropna(subset=["Ep_search"])
            if q.empty:
                continue
            y = np.array([ypos[c] for c in q.index]) + offsets[(win, clab)]
            ax.errorbar(q["Ep_search"], y, xerr=Z90 * q["sd"], fmt="none", ecolor=col, elinewidth=0.8, capsize=0)
            ax.scatter(q["Ep_search"], y, s=12, marker=MARK[clab], zorder=3,
                       facecolor=np.where(q["credible"], col, "white"), edgecolor=col, lw=0.8)
        ax.set_yticks(range(len(CONN_ORDER)),
                      [f"{c[0].upper()} {c.split('->')[0][1:]} → {c.split('->')[1][1:]}" for c in CONN_ORDER])
        ax.invert_yaxis()
        ax.set_ylim(len(CONN_ORDER) - 0.5, -0.5)
        ax.set_xlabel("Log gain (± 90% CI)")
        ax.set_title({"local": "Local", "global": "Global", "interaction": "Local × global (interaction coding)"}[eff],
                     color=C_EFF[eff], weight="bold")
        if eff == "global":
            ax.set_yticklabels([])
        sns.despine(ax=ax)
        fig.text(rect[0] - 0.105 if eff != "global" else rect[0] - 0.02, rect[1] + rect[3] + 0.05, L, fontsize=9,
                 weight="bold", va="top")
    # legend for analyses
    axL = fig.add_axes([0.06, 0.06, 0.3, 0.36])
    axL.set_axis_off()
    for k, ((win, clab), col) in enumerate(C_ANALYSIS.items()):
        axL.scatter([0.05], [0.9 - k * 0.16], s=16, marker=MARK[clab], facecolor=col, edgecolor=col)
        axL.text(0.13, 0.9 - k * 0.16, f"{win}, {clab.lower()} coding", va="center", fontsize=6)
    axL.scatter([0.05], [0.9 - 4 * 0.16 - 0.04], s=16, marker="o", facecolor="white", edgecolor="0.4")
    axL.text(0.13, 0.9 - 4 * 0.16 - 0.04, "open = not credible\n(P(present) ≤ .95)", va="center", fontsize=6)
    axL.add_patch(plt.Rectangle((0.0, 0.9 - 5 * 0.16 - 0.1), 0.1, 0.07, color="0.95"))
    axL.text(0.13, 0.9 - 5 * 0.16 - 0.065, "shaded rows = backward connections", va="center", fontsize=6)
    axL.set_xlim(0, 1)
    axL.set_ylim(0, 1)
    return fig


def save(fig, stem):
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}")
    svg = FIG_DIR / f"{stem}.svg"                        # SVG names fonts: Liberation Sans -> Arial (same metrics)
    svg.write_text(svg.read_text().replace("Liberation Sans", "Arial"))
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=500)
    print(f"Figure -> {FIG_DIR / stem}.pdf/.svg/.png")


def main():
    sns.set_theme(style="ticks", rc=TEXT_RC)
    stats_out = {}
    save(figure_fit(stats_out), "FigureS_DCM_fit")
    save(figure_estimates(), "FigureS_DCM_estimates")
    (DCM_DIR / "supplement_fit_stats.json").write_text(json.dumps(stats_out, indent=2))
    print(json.dumps(stats_out, indent=2))


if __name__ == "__main__":
    main()
