"""Operating-point map for the LIBRA Pi chapter.

Time to 99 % extraction over the (gas flow, temperature) plane at the design pressure and
nozzle diameter, one panel per transport-property scenario on a shared colour scale. Read as a
topographic map: equally spaced isolines whose spacing is the local gradient.

Same content as CELL B4 of exploration.ipynb, with the colour mesh rasterised so the file stays
small; isolines, labels and axes remain vector.

Run from the repository root: python3 make_operating_map.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import morethemes as mt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy.interpolate import RBFInterpolator

matplotlib.use("Agg")

RUNS_DIR = Path("data")
FIG_DIR = Path("figures")
THESIS_FIG = Path("../masters_thesis/fig")

SCENARIOS = ["pessimistic", "optimistic"]      # panel order: pessimistic left
QOI = "t99_nodisc_s"
HOURS_TO_SEC = 3600

SCEN_LABEL = {"pessimistic": "pessimistic (Calderoni $K_s$, $D_l$)",
              "optimistic": "optimistic (Malinauskas $K_s$, Fukada $D_l$)"}

P_NOM, D_NOM = 1.2, 2.0                 # LIBRA Pi design pressure [atm] and nozzle [mm]
NOMINAL = dict(flow=500, T=550)
N_GRID = 300
RBF_KERNEL, RBF_SMOOTHING = "quintic", 0.0
MAX_ISOLINES = 16
N_SUBDIV = 5                            # intermediate isolines per main interval
LABEL_MARGIN = 0.07                     # keep labels this far (axes fraction) from the edges
RASTER_DPI = 300

mt.set_theme("minimal")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})


def design_matrix(T_C, flow, P_top, d_noz):
    """Surrogate features: log10 on the input whose effect is a power law over a decade."""
    return np.column_stack([np.asarray(T_C, float), np.log10(flow),
                            np.asarray(P_top, float), np.asarray(d_noz, float)])


def fit_tau_surface(df, qoi=QOI):
    """4-D RBF surrogate of log10(qoi in hours). The model is deterministic, so there is no
    noise to average out: a pure interpolant (smoothing = 0) beats a smoothed spline."""
    X = design_matrix(df.temperature_C, df.gas_flow_sccm, df.top_pressure_atm,
                      df.nozzle_diameter_mm)
    mu, sd = X.mean(0), X.std(0)
    rbf = RBFInterpolator((X - mu) / sd, np.log10(df[qoi].to_numpy() / HOURS_TO_SEC),
                          kernel=RBF_KERNEL, smoothing=len(X) * RBF_SMOOTHING)
    return lambda Xn: 10 ** rbf((Xn - mu) / sd)


def contour_interval(vmin, vmax, max_lines=MAX_ISOLINES):
    """First interval of a 1/2/2.5/5/10 ladder that keeps the panel readable."""
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000):
        if (vmax - vmin) / step <= max_lines:
            return step
    return 5000


def inner_label_positions(cs, ax, wanted, margin=LABEL_MARGIN):
    """One label position per wanted level, taken on the contour itself and kept `margin`
    (in axes fraction) away from the four edges, so no label collides with an axis."""
    pos, kept = [], []
    for lev, segs in zip(cs.levels, cs.allsegs):
        if lev not in wanted:
            continue
        best, best_d = None, -np.inf
        for seg in segs:
            if len(seg) < 2:
                continue
            f = ax.transLimits.transform(seg)            # data -> axes fraction
            d = np.minimum(f, 1 - f).min(axis=1)         # distance to nearest edge
            k = int(np.argmax(d))
            if d[k] > best_d:
                best, best_d = seg[k], d[k]
        if best is not None and best_d >= margin:
            pos.append(tuple(best))
            kept.append(lev)
    return pos, kept


def load(scenario):
    d = RUNS_DIR / f"sobol_{scenario}"
    df = pd.read_csv(d / "sobol_data.csv")
    df[QOI] = df.t_extract_99_s * df.tau_fit_nodisc_s / df.tau_fitted_s
    return df


def main():
    data = {s: load(s) for s in SCENARIOS}

    surf = {}
    for name, df in data.items():
        Tg = np.linspace(df.temperature_C.min(), df.temperature_C.max(), N_GRID)
        Fg = np.linspace(df.gas_flow_sccm.min(), df.gas_flow_sccm.max(), N_GRID)
        TT, FF = np.meshgrid(Tg, Fg, indexing="ij")
        grid = design_matrix(TT.ravel(), FF.ravel(), np.full(TT.size, P_NOM),
                             np.full(TT.size, D_NOM))
        surf[name] = dict(TT=TT, FF=FF, Z=fit_tau_surface(df)(grid).reshape(TT.shape))

    norm = LogNorm(vmin=min(v["Z"].min() for v in surf.values()),
                   vmax=max(v["Z"].max() for v in surf.values()))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True, constrained_layout=True)
    for ax, name in zip(axes, SCENARIOS):
        v, df = surf[name], data[name]
        # rasterised: a 300x300 gouraud mesh as vectors is ~4 MB on its own
        pcm = ax.pcolormesh(v["FF"], v["TT"], v["Z"], cmap="YlOrRd", norm=norm,
                            shading="gouraud", zorder=0, rasterized=True)
        ax.scatter(df.gas_flow_sccm, df.temperature_C, s=2.6, c="0.15", alpha=0.35,
                   linewidths=0, zorder=1, rasterized=True)
        ax.set_axisbelow(False)
        ax.grid(True, color="0.25", lw=0.3, alpha=0.18, zorder=1.5)

        step = contour_interval(v["Z"].min(), v["Z"].max())
        lo = np.ceil(v["Z"].min() / step) * step
        levels = np.arange(lo, v["Z"].max(), step)
        minor = np.arange(lo - step, v["Z"].max(), step / N_SUBDIV)
        minor = np.array([m for m in minor if m > v["Z"].min()
                          and not np.isclose(m % step, 0, atol=step * 1e-6)])
        ax.contour(v["FF"], v["TT"], v["Z"], levels=minor, colors="0.25", linewidths=0.2,
                   zorder=2)
        cs = ax.contour(v["FF"], v["TT"], v["Z"], levels=levels, colors="0.1",
                        linewidths=0.85, zorder=3)

        # label every main line, but only every second one where they crowd
        crowd = np.median(levels)
        wanted = {lev for i, lev in enumerate(levels) if lev <= crowd or i % 2 == 0}
        pos, kept = inner_label_positions(cs, ax, wanted)
        ax.clabel(cs, manual=pos, fmt=lambda x: f"{x:g}", fontsize=5, inline=True,
                  inline_spacing=1)

        ax.plot(NOMINAL["flow"], NOMINAL["T"], marker="*", ms=13, mfc="white", mec="0.1",
                mew=0.9, zorder=6)
        ax.set_xticks([100, 250, 400, 550, 700, 850, 1000])
        ax.set_xlim(df.gas_flow_sccm.min(), df.gas_flow_sccm.max())
        ax.set_xlabel(r"gas flow rate $\dot{n}_g$ [sccm]")
        ax.set_title(f"{SCEN_LABEL[name]}\ncontour interval {step:g} h "
                     f"({step / N_SUBDIV:g} h intermediate)", fontsize=8)
        print(f"{name:12s} t99 {v['Z'].min():.1f}-{v['Z'].max():.1f} h | interval {step:g} h "
              f"({len(levels)} main + {len(minor)} intermediate) | {len(kept)} labelled "
              f"| {len(df)} design points")

    axes[0].set_ylabel(r"temperature [$^\circ$C]")
    cbar = fig.colorbar(pcm, ax=list(axes), pad=0.02,
                        ticks=[20, 30, 50, 80, 120, 200, 300, 500, 800])
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    cbar.set_label(r"time to 99 % extraction $t_{99}$ [h]")

    FIG_DIR.mkdir(exist_ok=True, parents=True)
    for out in (FIG_DIR / "operating_point_map.pdf", THESIS_FIG / "operating_point_map.pdf"):
        fig.savefig(out, dpi=RASTER_DPI)
        print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
