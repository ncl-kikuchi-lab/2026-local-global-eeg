"""
29_figure_dcm_main.py — main-text DCM figure (173.695 mm wide): robust connection changes by effect and window.

Rows:    local deviance, global deviance, local x global interaction
Columns: full response (0–600 ms), MMN window (0–250 ms)

Local and global rows: an arrow is drawn in colour when the change is ROBUST, i.e. credible (posterior
probability of presence > .95 after per-connection Bayesian model reduction, `Pp_search`) with the same sign
under both codings (surprise and interaction). Its label is the mean log gain of the two codings and its
width scales with |log gain|. A thin grey arrow = credible under one coding only (or opposite signs).
Interaction row: the local x global term exists only in the interaction coding, so arrows are drawn when
credible under that coding; label = its log gain.

Orange = stronger, blue = weaker than for standards (LS/GS). Dashed grey lines = lateral IFG–IPL connections
(not modulated). Grey arrows below A1 = driving input.

The estimate matrix and model fit are in the supplementary figure (28_figure_dcm_windows.py).

Inputs : derivatives/dcm/surprise_coded/<coding>/connections.csv  (0–600 ms; needs dcm_07_search_full.m)
         derivatives/dcm/early_window/<coding>/connections.csv    (0–250 ms)
Outputs: figures/Figure4_DCM.pdf / .svg / .png

Usage:  python scripts/29_figure_dcm_main.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import seaborn as sns

# ----------------------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[1]
DCM_DIR = ROOT / "derivatives" / "dcm"
FIG_DIR = ROOT / "figures"
SIZE_MM = (173.695, 150.0)
WINDOWS = [("surprise_coded", "Full response (0–600 ms)"), ("early_window", "MMN window (0–250 ms)")]
CODINGS = ["surprise", "interaction"]
ROWS = [("local", "Local deviance"), ("global", "Global deviance"), ("interaction", "Local × global")]
PP_THR = 0.95
# sources ordered along the hierarchy: primary auditory -> secondary auditory -> frontal / parietal
POS = {"lA1": (-1.0, 0.0), "lSTG": (-1.0, 1.0), "lIFG": (-1.55, 2.05), "lIPL": (-0.45, 2.05),
       "rA1": (1.0, 0.0), "rSTG": (1.0, 1.0), "rIFG": (1.55, 2.05), "rIPL": (0.45, 2.05)}
FWD = [("lA1", "lSTG"), ("rA1", "rSTG"), ("lSTG", "lIFG"), ("rSTG", "rIFG"), ("lSTG", "lIPL"), ("rSTG", "rIPL")]
LAT = [("lIFG", "lIPL"), ("rIFG", "rIPL")]
RAD = 0.28                                       # curvature, so forward and backward arrows separate
C_UP, C_DOWN, C_NS = "#D55E00", "#0072B2", "0.75"
C_ROW = {"local": "#C44E52", "global": "#8172B3", "interaction": "0.25"}
NODE_EC = {"A1": "#9DB4C0", "STG": "#4C72B0", "IFG": "#DD8452", "IPL": "#55A868"}   # region colours
TEXT_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
           "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
           "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5}
# ------------------------------------------------------------------------------------


def load(sub, coding):
    d = pd.read_csv(DCM_DIR / sub / coding / "connections.csv")
    if "Pp_search" not in d:
        raise SystemExit(f"{sub}/{coding}: no Pp_search column — run scripts/dcm/dcm_07_search_full.m first")
    return d.assign(credible=d["Pp_search"] > PP_THR, Ep=d["Ep_search"])[["effect", "connection", "Ep", "credible"]]


def arrows_for(sub, effect):
    """connection -> (status, log gain): status 'robust', 'single' (one coding / opposite signs), or None."""
    tabs = {c: load(sub, c).set_index(["effect", "connection"]) for c in CODINGS}
    out = {}
    for lo, hi in FWD:
        for a, b in ((lo, hi), (hi, lo)):
            conn = f"{a}->{b}"
            if effect == "interaction":
                r = tabs["interaction"].loc[(effect, conn)]
                out[conn] = ("robust", r.Ep) if r.credible else (None, 0.0)
                continue
            rs = [tabs[c].loc[(effect, conn)] for c in CODINGS]
            cred = [r.credible for r in rs]
            same = np.sign(rs[0].Ep) == np.sign(rs[1].Ep)
            if all(cred) and same:
                out[conn] = ("robust", float(np.mean([r.Ep for r in rs])))
            elif any(cred):
                out[conn] = ("single", 0.0)
            else:
                out[conn] = (None, 0.0)
    return out


def label_pos(p0, p1, rad, t=0.38, push=None):
    """Point at fraction t (from the source) along matplotlib's arc3 Bezier, pushed outward from the chord.
    Placing labels nearer the source keeps the labels of two arrows that share a target from colliding."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    n = np.hypot(*d)
    normal = np.array([d[1], -d[0]]) / n
    c = (p0 + p1) / 2 + rad * n * normal
    b = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1
    if push is None:                              # labels are wide: push further from near-vertical arrows
        push = 0.12 + 0.16 * abs(d[1]) / n
    return b + np.sign(rad) * push * normal


def draw(ax, arrows):
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-0.55, 2.4)
    ax.set_aspect("equal")
    ax.set_axis_off()
    for a, b in LAT:
        (x0, y0), (x1, y1) = POS[a], POS[b]
        ax.plot([x0, x1], [y0, y1], ls=(0, (2, 2)), color="0.8", lw=0.6, zorder=1)
    for side in ("l", "r"):
        x, y = POS[f"{side}A1"]
        ax.add_patch(FancyArrowPatch((x, y - 0.5), (x, y - 0.17), arrowstyle="-|>", mutation_scale=7,
                                     color="0.55", lw=0.8))
    for lo, hi in FWD:
        for a, b in ((lo, hi), (hi, lo)):
            status, ep = arrows[f"{a}->{b}"]
            if status is None:
                continue
            if status == "robust":
                col = C_UP if ep > 0 else C_DOWN
                lw = 0.8 + 1.5 * min(abs(ep), 1.3)
            else:
                col, lw = C_NS, 0.6
            ax.add_patch(FancyArrowPatch(POS[a], POS[b], connectionstyle=f"arc3,rad={RAD}", arrowstyle="-|>",
                                         mutation_scale=5 + 2.2 * lw, lw=lw, color=col, shrinkA=9, shrinkB=9,
                                         zorder=3 if status == "robust" else 2))
            if status == "robust":
                lx, ly = label_pos(POS[a], POS[b], RAD)
                ax.text(lx, ly, f"{ep:+.2f}", ha="center", va="center", fontsize=5.6, color=col, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    for name, (x, y) in POS.items():
        ax.add_patch(FancyBboxPatch((x - 0.23, y - 0.13), 0.46, 0.26, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="white", ec=NODE_EC[name[1:]], lw=1.0, zorder=4))
        ax.text(x, y, name[1:], ha="center", va="center", fontsize=6.2, zorder=5)
    ax.text(-1.5, -0.48, "L", ha="center", va="bottom", fontsize=6, color="0.45")
    ax.text(1.5, -0.48, "R", ha="center", va="bottom", fontsize=6, color="0.45")


def main():
    sns.set_theme(style="white", rc=TEXT_RC)
    mm = 1 / 25.4
    fig = plt.figure(figsize=(SIZE_MM[0] * mm, SIZE_MM[1] * mm))
    left, right, top, bottom = 0.075, 0.995, 0.955, 0.075
    W = (right - left) / 2
    H = (top - bottom) / 3
    letters = iter("ABCDEF")
    for i, (eff, rlab) in enumerate(ROWS):
        for j, (sub, wlab) in enumerate(WINDOWS):
            x0, y0 = left + j * W, top - (i + 1) * H
            ax = fig.add_axes([x0 + 0.01, y0 + 0.005, W - 0.02, H - 0.01])
            draw(ax, arrows_for(sub, eff))
            fig.text(x0, y0 + H - 0.005, next(letters), fontsize=10, weight="bold", va="top")
            if i == 0:
                fig.text(x0 + W / 2, top + 0.01, wlab, ha="center", va="bottom", fontsize=7.5)
        fig.text(0.02, top - (i + 0.5) * H, rlab, rotation=90, ha="center", va="center", fontsize=7.5,
                 weight="bold", color=C_ROW[eff])
    fig.text(0.04, top - 2.5 * H, "(interaction coding)", rotation=90, ha="center", va="center", fontsize=6,
             color="0.4")
    # legend
    y, x = 0.03, 0.10
    for col, txt in ((C_UP, "stronger"), (C_DOWN, "weaker")):
        fig.add_artist(plt.Line2D([x, x + 0.03], [y, y], color=col, lw=2, transform=fig.transFigure))
        fig.text(x + 0.036, y, txt, va="center", fontsize=6.5)
        x += 0.11
    fig.add_artist(plt.Line2D([x, x + 0.03], [y, y], color=C_NS, lw=0.7, transform=fig.transFigure))
    fig.text(x + 0.036, y, "credible under one coding only", va="center", fontsize=6.5)
    x += 0.25
    fig.add_artist(plt.Line2D([x, x + 0.03], [y, y], color="0.8", lw=0.7, ls=(0, (2, 2)),
                              transform=fig.transFigure))
    fig.text(x + 0.036, y, "lateral (not modulated)", va="center", fontsize=6.5)
    x += 0.18
    fig.text(x, y, "labels: log gain", va="center", fontsize=6.5, color="0.35")

    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(FIG_DIR / f"Figure4_DCM.{ext}")
    svg = FIG_DIR / "Figure4_DCM.svg"                    # SVG names fonts: Liberation Sans -> Arial (same metrics)
    svg.write_text(svg.read_text().replace("Liberation Sans", "Arial"))
    fig.savefig(FIG_DIR / "Figure4_DCM.png", dpi=600)
    print(f"Figure -> {FIG_DIR / 'Figure4_DCM'}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
