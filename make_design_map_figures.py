"""The three design-space figures of the sparging design chapter.

All read the 300-sample design study and split it on the three governing groups. The Venn and
the violin answer two different questions and are deliberately built differently:

    venn     "given the region I operate in, which model may I use, and how far off is the
              analytical solution?"      -> signed error of tau_ana per region
    violin   "which region do I want to operate in for the fastest extraction?"
                                         -> tau_num against the mass-transfer-limited ideal
    parity   "may I trust the analytical solution here?"
                                         -> tau_num vs tau_ana, valid groups highlighted

The Venn circles are the *violated* assumptions, so its centre is the region where none of them
holds, and each region carries the case number of the decision table (tab:design_map).

Run from the repository root: python3 make_design_map_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import morethemes as mt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib_venn import venn3
from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm

matplotlib.use("Agg")

RUN_DIR = Path("data/analytical_validity2")
FIG_DIR = Path("figures")
THESIS_FIG = Path("../masters_thesis/fig")

RMSE_THRESHOLD = 1e-4      # above this the inventory decay is not a single exponential
NONEXP_HATCH_FRAC = 0.5    # hatch a region when this fraction of its samples are non-exponential
THRESH = 0.1               # common threshold on the three groups
PARITY_BAND = 0.05         # +-5 % band on the parity plot

# --- notation -------------------------------------------------------------- #
# single point of truth for the two group symbols, so switching G -> Gamma in the thesis
# is a one-line change here plus a rerun. Currently matches \Gmix and \Gp in thesis.tex.
SYM_MIX = r"G_\mathrm{mix}"
SYM_P = r"G_P"
SYM_PI = r"\langle\Pi\rangle"

# regions keyed as (Pi, G_mix, G_P) VIOLATED flags -> case number in tab:design_map,
# which is indexed on the SATISFIED conditions (G_mix<0.1, G_P<0.1, Pi<0.1)
CASE_OF = {
    (0, 0, 0): 1, (1, 0, 0): 2, (0, 0, 1): 3, (1, 0, 1): 4,
    (0, 1, 0): 5, (1, 1, 0): 6, (0, 1, 1): 7, (1, 1, 1): 8,
}
REGION_NAME = {
    (0, 0, 0): "none", (1, 0, 0): rf"${SYM_PI}$", (0, 1, 0): rf"${SYM_MIX}$",
    (0, 0, 1): rf"${SYM_P}$", (1, 1, 0): rf"${SYM_PI}+{SYM_MIX}$",
    (1, 0, 1): rf"${SYM_PI}+{SYM_P}$", (0, 1, 1): rf"${SYM_MIX}+{SYM_P}$",
    (1, 1, 1): "all three",
}

mt.set_theme("minimal")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "hatch.linewidth": 0.3,          # discreet: the hatch is secondary information
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})


def load():
    df = pd.read_csv(RUN_DIR / "analytical_validity2_data.csv", comment="#")
    # the fitted tau carries the backward Euler +dt/2 bias; removing it leaves the model error
    df["tau_fit_nodisc_s"] = df.tau_fitted_s - df.dt_s / 2
    df["e_pred_nodisc"] = (df.tau_fit_nodisc_s - df.tau_pred_s) / df.tau_pred_s
    df["nonexp"] = df.fit_rmse_norm >= RMSE_THRESHOLD
    df["vPi"] = (df.Pi > THRESH).astype(int)
    df["vmix"] = (df.G_mix > THRESH).astype(int)
    df["vP"] = (df.G_P > THRESH).astype(int)
    return df


def region_mask(df, key):
    return (df.vPi == key[0]) & (df.vmix == key[1]) & (df.vP == key[2])


def region_stats(df):
    out = {}
    for key in REGION_NAME:
        s = df[region_mask(df, key)]
        out[key] = dict(name=REGION_NAME[key], case=CASE_OF[key], n=len(s),
                        median_e=s.e_pred_nodisc.median() if len(s) else np.nan,
                        p10=s.e_pred_nodisc.quantile(0.10) if len(s) else np.nan,
                        p90=s.e_pred_nodisc.quantile(0.90) if len(s) else np.nan,
                        nonexp=s.nonexp.mean() if len(s) else 0.0)
    return out


def text_on(rgba):
    """Black or white, whichever stays readable on `rgba` (ITU-R BT.601 luminance)."""
    r, g, b = rgba[:3]
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "0.1"


def pct(value):
    """Signed percentage, without the '-0.0%' that rounding to one decimal produces."""
    v = value * 100
    return f"{0.0:+.1f}%".replace("+", " ") if abs(v) < 0.05 else f"{v:+.1f}%"


# --------------------------------------------------------------------------- #
# Venn: which model may I use, and how wrong is the analytical one
# --------------------------------------------------------------------------- #

def make_venn(stats):
    lim = max(abs(v["median_e"]) for v in stats.values())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    cmap = plt.get_cmap("RdBu_r")        # blue = faster than predicted, red = slower

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    v = venn3(
        subsets=(1, 1, 1, 1, 1, 1, 1),   # equal areas: a design map, size carries no meaning
        set_labels=(rf"${SYM_PI}>0.1$", rf"${SYM_MIX}>0.1$", rf"${SYM_P}>0.1$"),
        ax=ax,
        layout_algorithm=DefaultLayoutAlgorithm(fixed_subset_sizes=(1,) * 7),
    )

    outside = stats[(0, 0, 0)]
    for key, st in stats.items():
        if key == (0, 0, 0):
            continue
        pid = "".join(str(b) for b in key)
        patch, label = v.get_patch_by_id(pid), v.get_label_by_id(pid)
        if patch is None:
            continue
        patch.set_color(cmap(norm(st["median_e"])))
        patch.set_alpha(1.0)
        patch.set_edgecolor("0.25")
        patch.set_linewidth(0.7)
        if st["nonexp"] >= NONEXP_HATCH_FRAC:
            patch.set_hatch("/////")
        if label is not None:
            label.set_text(f"{st['case']}\n{pct(st['median_e'])}")
            label.set_fontsize(8)
            label.set_color(text_on(cmap(norm(st["median_e"]))))
    for t in v.set_labels:
        if t is not None:
            t.set_fontsize(9)

    # "design space": everything a sparging system can be, case 1 being outside every circle
    ax.set_xlim(-0.80, 0.80)
    ax.set_ylim(-0.68, 0.72)
    box = FancyBboxPatch((-0.78, -0.66), 1.56, 1.36,
                         boxstyle="round,pad=0.005,rounding_size=0.03",
                         linewidth=0.9, edgecolor="0.35",
                         facecolor=cmap(norm(outside["median_e"])), zorder=-5)
    ax.add_patch(box)
    ax.annotate(f"1\n{pct(outside['median_e'])}", xy=(-0.71, -0.57), fontsize=8,
                ha="center", color=text_on(cmap(norm(outside["median_e"]))))
    ax.annotate("design space", xy=(-0.76, 0.66), fontsize=8, style="italic",
                va="top", color="0.35")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r"median $(\tau_\mathrm{num}-\tau_\mathrm{ana})/\tau_\mathrm{ana}$")
    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t * 100:+.0f}%" for t in ticks])
    cbar.ax.text(0.5, 1.03, "slower", transform=cbar.ax.transAxes, ha="center", fontsize=7)
    cbar.ax.text(0.5, -0.055, "faster", transform=cbar.ax.transAxes, ha="center", va="top",
                 fontsize=7)
    ax.annotate("numbers are the cases of the decision table; hatched = decay is not a single "
                "exponential", xy=(0.5, -0.02), xycoords="axes fraction", ha="center",
                va="top", fontsize=7, color="0.35")
    return fig


# --------------------------------------------------------------------------- #
# Violin: which region extracts fastest, against the mass-transfer-limited ideal
# --------------------------------------------------------------------------- #

def make_violin(df, stats):
    order = sorted(REGION_NAME, key=lambda k: CASE_OF[k])
    data, labels, frac = [], [], []
    for key in order:
        s = df[region_mask(df, key)]
        if len(s) == 0:
            continue
        # tau_SPP is the mass-transfer-limited ideal: the extraction time the same system would
        # reach if the bubbles never saturated. It depends on the same a and h_l as the sample,
        # so the ratio isolates the penalty of the regime rather than the parameter scaling.
        data.append(np.log10(s.tau_fit_nodisc_s / s.tau_pred_SPP_s))
        labels.append(f"case {CASE_OF[key]}\n{REGION_NAME[key]}\n$n={len(s)}$")
        frac.append(s.nonexp.mean())

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    pos = np.arange(len(data))
    parts = ax.violinplot(data, positions=pos, widths=0.82, showextrema=False)
    for body, f in zip(parts["bodies"], frac):
        body.set_facecolor("#EE7733" if f >= NONEXP_HATCH_FRAC else "#BBBBBB")
        body.set_alpha(0.45)
        body.set_edgecolor("0.3")
        body.set_linewidth(0.6)
    ax.boxplot(data, positions=pos, widths=0.15, showfliers=False,
               medianprops=dict(color="0.1", lw=1.2),
               boxprops=dict(color="0.25", lw=0.7),
               whiskerprops=dict(color="0.25", lw=0.7),
               capprops=dict(color="0.25", lw=0.7))

    ax.axhline(0, color="#CC3311", lw=0.9, ls="--", zorder=0)
    ax.annotate(r"mass-transfer-limited ideal $\tau_\mathrm{SPP}$",
                xy=(len(data) - 0.45, 0), xytext=(0, 3), textcoords="offset points",
                ha="right", fontsize=7, color="#CC3311")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"$\log_{10}\left(\tau_\mathrm{num}/\tau_\mathrm{SPP}\right)$")
    ax.grid(False)
    # second axis in plain multiples, which is how a designer reads the penalty
    sec = ax.secondary_yaxis("right", functions=(lambda v: 10 ** v, np.log10))
    sec.set_yticks([1, 1.5, 2, 3, 5, 10, 30, 100])
    sec.set_yticklabels([r"$\times$1", r"$\times$1.5", r"$\times$2", r"$\times$3",
                         r"$\times$5", r"$\times$10", r"$\times$30", r"$\times$100"],
                        fontsize=7)
    sec.set_ylabel("penalty against the ideal", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.45, ec="0.3", lw=0.6)
               for c in ("#BBBBBB", "#EE7733")]
    ax.legend(handles, ["exponential decay", "mostly non-exponential"],
              loc="upper left", fontsize=7.5)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Parity: may I trust the analytical solution here
# --------------------------------------------------------------------------- #

def make_parity(df):
    HOUR = 3600
    # the two groups for which the analytical solution was shown to hold
    both = (df.G_mix < THRESH) & (df.G_P < THRESH)                 # any Pi
    mtl = (df.Pi < THRESH) & (df.G_P < THRESH)                     # any G_mix
    rest = ~(both | mtl)

    x = df.tau_pred_s / HOUR
    y = df.tau_fit_nodisc_s / HOUR
    ratio = df.tau_fit_nodisc_s / df.tau_pred_s

    LBL_MTL = rf"${SYM_PI}<0.1$, ${SYM_P}<0.1$, any ${SYM_MIX}$"
    LBL_BOTH = rf"${SYM_MIX}<0.1$, ${SYM_P}<0.1$, any ${SYM_PI}$"

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(7.4, 3.5))

    # (a) parity. Over four decades a +-5 % band is thinner than the line, so the band lives
    # in panel (b); here the y = x line only shows that nothing is grossly wrong.
    lo, hi = min(x.min(), y.min()) * 0.5, max(x.max(), y.max()) * 2
    line = np.array([lo, hi])
    ax.plot(line, line, color="0.35", lw=0.9, zorder=1)
    ax.scatter(x[rest], y[rest], s=11, c="0.65", alpha=0.35, linewidths=0, zorder=2,
               label="outside both")
    # a ring, so samples belonging to both groups stay visible under the filled marker
    ax.scatter(x[mtl], y[mtl], s=46, facecolors="none", edgecolors="#EE7733", linewidths=0.9,
               zorder=3, label=LBL_MTL)
    ax.scatter(x[both], y[both], s=13, c="#0077BB", linewidths=0, zorder=4, label=LBL_BOTH)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\tau_\mathrm{ana}$ [h]")
    ax.set_ylabel(r"$\tau_\mathrm{num}$ [h]")
    ax.set_title("(a) parity", fontsize=8.5)
    ax.grid(False)

    # (b) the same points as a ratio against the saturation number: this is where the +-5 %
    # claim is legible, and it shows the blue group holding at every Pi
    axr.axhspan(1 - PARITY_BAND, 1 + PARITY_BAND, color="0.5", alpha=0.18, lw=0, zorder=0)
    axr.axhline(1.0, color="0.35", lw=0.9, zorder=1)
    axr.scatter(df.Pi[rest], ratio[rest], s=11, c="0.65", alpha=0.35, linewidths=0, zorder=2)
    axr.scatter(df.Pi[mtl], ratio[mtl], s=46, facecolors="none", edgecolors="#EE7733",
                linewidths=0.9, zorder=3)
    axr.scatter(df.Pi[both], ratio[both], s=13, c="#0077BB", linewidths=0, zorder=4)
    axr.axvline(THRESH, color="0.6", lw=0.6, ls=":", zorder=1)
    axr.set_xscale("log")
    axr.set_xlabel(rf"${SYM_PI}$")
    axr.set_ylabel(r"$\tau_\mathrm{num}/\tau_\mathrm{ana}$")
    axr.set_ylim(0.55, 1.25)
    axr.set_title(rf"(b) residual, shaded band $\pm{PARITY_BAND * 100:.0f}\,\%$", fontsize=8.5)
    axr.grid(False)

    handles = [Line2D([], [], ls="none", marker="o", ms=3.5, mfc="0.65", mec="none",
                      label="outside both"),
               Line2D([], [], ls="none", marker="o", ms=6, mfc="none", mec="#EE7733",
                      mew=0.9, label=LBL_MTL),
               Line2D([], [], ls="none", marker="o", ms=3.8, mfc="#0077BB", mec="none",
                      label=LBL_BOTH)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7,
               handletextpad=0.4, columnspacing=1.2, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()

    for name, sel in (("G_mix<0.1 & G_P<0.1 (any Pi)", both),
                      ("Pi<0.1 & G_P<0.1 (any G_mix)", mtl),
                      ("union", both | mtl)):
        e = df.loc[sel, "e_pred_nodisc"].abs()
        print(f"  parity {name:34s} n={sel.sum():3d}  within +-{PARITY_BAND:.0%}: "
              f"{(e < PARITY_BAND).sum():3d} ({100 * (e < PARITY_BAND).mean():5.1f}%)  "
              f"max|e|={e.max():.3f}")
    return fig


def main():
    df = load()
    if "tau_pred_SPP_s" not in df.columns:      # dataset predates the rename
        df["tau_pred_SPP_s"] = df["tau_pred_SPP_s"]
    stats = region_stats(df)

    for key in sorted(REGION_NAME, key=lambda k: CASE_OF[k]):
        st = stats[key]
        print(f"  case {st['case']}  {st['name']:24s} n={st['n']:3d}  "
              f"median e={st['median_e']:+.4f}  nonexp={st['nonexp']:.0%}")

    FIG_DIR.mkdir(exist_ok=True, parents=True)
    figs = {"design_venn": make_venn(stats),
            "design_violin": make_violin(df, stats),
            "validity_parity_nodisc": make_parity(df)}
    for name, fig in figs.items():
        for out in (FIG_DIR / f"{name}.pdf", THESIS_FIG / f"{name}.pdf"):
            fig.savefig(out)
        print(f"wrote {name}.pdf")


if __name__ == "__main__":
    main()
