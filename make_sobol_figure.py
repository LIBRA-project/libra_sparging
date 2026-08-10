"""Sobol sensitivity figure for the LIBRA Pi chapter.

Dot-and-whisker rather than bars: a Sobol index is a share of variance, not an extensive
quantity, so filled area would misrepresent it. One panel per transport-property scenario,
each parameter carrying its first order and total order index side by side, so the reader can
see the two intervals overlap and conclude that higher order interactions are weak.

Run from the repository root: python3 make_sobol_figure.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import morethemes as mt

matplotlib.use("Agg")   # headless: this script only writes files
from scipy.stats import sobol_indices

RUNS_DIR = Path("data")
FIG_DIR = Path("figures")
THESIS_FIG = Path("../masters_thesis/fig")

SCENARIOS = ["pessimistic", "optimistic"]          # panel order: pessimistic left
QOI = "t99_nodisc_s"
N_RESAMPLES = 999

SCEN_LABEL = {"pessimistic": "pessimistic (Calderoni $K_s$, $D_l$)",
              "optimistic": "optimistic (Malinauskas $K_s$, Fukada $D_l$)"}
PARAM_LABEL = {"temperature": r"$T$", "gas_flow": r"$\dot{n}_g$",
               "top_pressure": r"$P_\mathrm{top}$", "nozzle_diameter": r"$d_\mathrm{noz}$"}

# first order filled, total order open: same hue, so the eye pairs them per parameter
ORDER_STYLE = {
    "first_order": dict(label=r"first order $S_i$", marker="o", mfc="#0077BB", mec="#0077BB"),
    "total_order": dict(label=r"total order $S_{T_i}$", marker="s", mfc="white", mec="#CC3311"),
}
OFFSET = 0.17

mt.set_theme("minimal")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "ytick.minor.visible": True,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})


def sobol_matrices(df, d, n, qoi):
    """Regroup the tagged rows into the {f_A, f_B, f_AB} dict scipy expects.

    The scrambled Sobol prefixes are nested, so the first `n` base samples are a valid
    design on their own. The assertion is the design-completeness check.
    """
    yA, yB = np.full(n, np.nan), np.full(n, np.nan)
    yAB = np.full((d, n), np.nan)
    for role, j, i, val in zip(df.design_role, df.base_index, df.var_index, df[qoi]):
        j = int(j)
        if j >= n:
            continue
        if role == "A":
            yA[j] = val
        elif role == "B":
            yB[j] = val
        else:
            yAB[int(i), j] = val
    assert np.isfinite(yA).all() and np.isfinite(yB).all() and np.isfinite(yAB).all(), (
        f"incomplete Saltelli design for n={n}"
    )
    return {"f_A": yA[None, :], "f_B": yB[None, :], "f_AB": yAB[:, None, :]}


def load(scenario):
    d = RUNS_DIR / f"sobol_{scenario}"
    meta = json.load(open(d / "metadata.json"))
    df = pd.read_csv(d / "sobol_data.csv")
    # t99 carries the same backward-Euler bias as tau_fitted; the decay is exponential,
    # so rescaling by tau_nodisc/tau_fitted removes it exactly
    df[QOI] = df.t_extract_99_s * df.tau_fit_nodisc_s / df.tau_fitted_s
    res = sobol_indices(
        func=sobol_matrices(df, meta["d"], meta["n_base_samples"], QOI),
        n=meta["n_base_samples"],
    )
    return dict(
        res=res,
        ci=res.bootstrap(n_resamples=N_RESAMPLES),
        names=[p["name"] for p in meta["param_space"]],
        n=meta["n_base_samples"],
        n_runs=len(df),
    )


def whiskers(value, ci_low, ci_high):
    """Asymmetric error bars, clipped at zero: near-zero indices get NaN bootstrap bounds."""
    err = np.vstack([value - ci_low, ci_high - value])
    return np.nan_to_num(np.clip(err, 0, None))


def main():
    data = {s: load(s) for s in SCENARIOS}
    names = data[SCENARIOS[0]]["names"]

    # one ordering for both panels, by mean total order, so they stay comparable
    mean_total = np.mean([data[s]["res"].total_order.ravel() for s in SCENARIOS], axis=0)
    order = np.argsort(mean_total)[::-1]
    xpos = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        d = data[scenario]
        for sign, kind in ((-1, "first_order"), (+1, "total_order")):
            value = getattr(d["res"], kind).ravel()[order]
            interval = getattr(d["ci"], kind).confidence_interval
            err = whiskers(value, interval.low.ravel()[order], interval.high.ravel()[order])
            style = ORDER_STYLE[kind]
            ax.errorbar(
                xpos + sign * OFFSET, value, yerr=err,
                fmt=style["marker"], ms=4.5, mfc=style["mfc"], mec=style["mec"], mew=0.9,
                ecolor=style["mec"], elinewidth=0.9, capsize=2.5, capthick=0.9,
                linestyle="none", label=style["label"], zorder=3,
            )
        # faint separators between parameters, so the pairing is unambiguous
        for x in xpos[:-1]:
            ax.axvline(x + 0.5, color="0.85", lw=0.5, zorder=0)
        ax.axhline(0, color="0.6", lw=0.6, zorder=1)
        ax.set_xticks(xpos)
        ax.set_xticklabels([PARAM_LABEL[names[k]] for k in order])
        ax.tick_params(axis="x", which="minor", bottom=False)   # categorical axis
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.grid(False)
        ax.set_title(SCEN_LABEL[scenario], fontsize=8.5)

        total_first = getattr(d["res"], "first_order").ravel().sum()
        print(f"{scenario:12s} N={d['n']} ({d['n_runs']} runs)  sum S_i = {total_first:.3f}")
        for k in order:
            s1 = d["res"].first_order.ravel()[k]
            st = d["res"].total_order.ravel()[k]
            print(f"   {names[k]:16s} S_1={s1:6.3f}  S_T={st:6.3f}  S_T-S_1={st - s1:+.3f}")

    axes[0].set_ylabel("Sobol index")
    axes[0].legend(loc="upper right", handletextpad=0.4)
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True, parents=True)
    for out in (FIG_DIR / "sobol_indices.pdf", THESIS_FIG / "sobol_indices.pdf"):
        fig.savefig(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
