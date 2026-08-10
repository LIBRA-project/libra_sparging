"""Generate the data underlying the paper's and the thesis' figures.

Exposes `convergence_study` (mesh/time-step convergence), `verification_case`
(comparison with the analytical solution), `generate_validity_data` (validity of the
analytical extraction time) and `sobol_study` (input sensitivity). Outputs go to
subfolders of ``data/``. Run the convergence study with
``python paper/generate_paper_data.py``; the others via their functions.
"""

from __future__ import annotations

import os

# single-thread BLAS/OpenMP (set before numpy import) to avoid oversubscription
# when the studies fan out one solve per process
for _v in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import hashlib
import json
import logging
import multiprocessing as mp
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sparging import get_sim_input_LIBRA_Pi, Simulation, ureg
from sparging import config as helpers

logger = logging.getLogger(__name__)

OUT_DIR = Path("data/convergence_study_2")

# ---------------------------------------------------------------------------
# study configuration
# ---------------------------------------------------------------------------
C_T2_INIT = 3e-11 * ureg.molT2 / ureg.m**3  # initial concentration

# transport-parameter scenarios bracketing the tritium properties of the salt, shared by the
# Sobol study and libra_pi_scenarios(). No correlation exists for ClLiF, so the two available
# sets of measurements are used as bounds: high solubility + low diffusivity (the salt retains
# tritium and releases it slowly) against low solubility + high diffusivity.
SCEN_CORRELATIONS = {
    "pessimistic": {"K_s": "K_s_calderoni", "D_l": "D_l_calderoni"},
    "optimistic": {"K_s": "K_s_malinauskas", "D_l": "D_l_fukada"},
}

CONV_T_FINAL_IN_TAU = 4  # simulated duration in units of tau

# temporal sweep dt = t_final / n_steps at a fixed fine mesh; the finest 3 keep
# ratio r=2 for Richardson extrapolation
N_STEPS_SWEEP = [3, 6, 12, 25, 50, 100, 200, 400, 800, 1600]
N_CELLS_FIXED = 160

# spatial sweep dx = H / n_cells at a fixed fine time step
N_CELLS_SWEEP = [14, 28, 56, 112, 224, 448]
N_STEPS_FIXED = 800

REFINEMENT_RATIO = 2.0


def _run(sim: Simulation, dt: "ureg.Quantity", dx: "ureg.Quantity") -> dict:
    """Solve once; return the convergence record plus the raw inventory series."""
    out = sim.solve(dt=dt, dx=dx)
    rec = out.convergence_record()
    rec["_times_s"] = out.times.to("s").magnitude.tolist()
    rec["_inv_mol"] = out.n_T2_salt_series.to("molT2").magnitude.tolist()
    return rec


def convergence_study(out_dir: Path = OUT_DIR) -> Path:
    """Run the {temporal, spatial} convergence sweeps on the nominal LIBRA-Pi input
    and write a consolidated JSON plus one sanity CSV.

    Returns the path to the consolidated JSON.
    """
    warnings.filterwarnings("ignore")  # silence curve_fit / pint fit warnings
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = get_sim_input_LIBRA_Pi()
    inp.c_T2_init = C_T2_INIT
    tau = inp.get_tau()
    Pi = inp.get_Pi_ave()
    Bo = inp.get_Bo()
    H = inp.height
    t_final = CONV_T_FINAL_IN_TAU * tau

    dx_fixed = (H / N_CELLS_FIXED).to("m")
    dt_fixed = (t_final / N_STEPS_FIXED).to("s")

    logger.info(
        "tau=%.3f h, Pi=%.3f, Bo=%.2f, t_final=%.3f h, dx_fixed=%.2e m, dt_fixed=%.2e s",
        tau.to("hour").magnitude,
        Pi.magnitude,
        Bo.magnitude,
        t_final.to("hour").magnitude,
        dx_fixed.magnitude,
        dt_fixed.magnitude,
    )

    sim = Simulation(inp, t_final=t_final, dispersion_on=True, constant_profiles=False)

    t_start = time.time()

    # ---- temporal sweep (refine dt, fixed fine mesh) ----
    dt_records = []
    for n_steps in N_STEPS_SWEEP:
        dt = (t_final / n_steps).to("s")
        rec = _run(sim, dt, dx_fixed)
        logger.info(
            "  [dt] n_steps=%4d dt=%.2e s -> tau_fit=%.4f h (resid=%.3f)",
            n_steps,
            dt.magnitude,
            rec["tau_fitted_s"] / 3600,
            rec["residual_fraction"],
        )
        dt_records.append(rec)

    # ---- spatial sweep (refine dx, fixed fine time step) ----
    dx_records = []
    for n_cells in N_CELLS_SWEEP:
        dx = (H / n_cells).to("m")
        rec = _run(sim, dt_fixed, dx)
        logger.info(
            "  [dx] n_cells=%4d dx=%.2e m -> tau_fit=%.4f h (resid=%.3f)",
            n_cells,
            dx.magnitude,
            rec["tau_fitted_s"] / 3600,
            rec["residual_fraction"],
        )
        dx_records.append(rec)

    # ---- sanity CSV: inventory decay of the finest temporal run ----
    finest = dt_records[-1]
    sanity = np.column_stack([finest.pop("_times_s"), finest.pop("_inv_mol")])
    header = (
        f"# Pi={Pi.magnitude:.4f}  "
        f"tau_fitted_h={finest['tau_fitted_s'] / 3600:.4f}  "
        f"n0_fitted_mol={finest['n0_fitted_mol']:.4e}\n"
        "t_seconds,n_T2_salt_mol"
    )
    np.savetxt(
        out_dir / "finest_inventory.csv",
        sanity,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.6e",
    )

    # drop the bulky raw series from every record before archiving
    for rec in (*dt_records, *dx_records):
        rec.pop("_times_s", None)
        rec.pop("_inv_mol", None)

    consolidated = {
        "metadata": {
            "git_commit": helpers.get_git_hash(),
            "date": datetime.now().isoformat(),
            "description": (
                "Mesh/time-step convergence study on the nominal LIBRA-Pi input. "
                "Metric: fitted inventory decay time tau_fitted."
            ),
            "t_final_s": float(t_final.to("s").magnitude),
            "t_final_in_tau": CONV_T_FINAL_IN_TAU,
            "tau_s": float(tau.to("s").magnitude),
            "Pi": float(Pi.magnitude),
            "Bo": float(Bo.magnitude),
            "H_m": float(H.to("m").magnitude),
            "K_s_mol_m3_Pa": float(inp.K_s.to("mol/m**3/Pa").magnitude),
            "refinement_ratio": REFINEMENT_RATIO,
            "temporal_sweep": {
                "n_steps": N_STEPS_SWEEP,
                "n_cells_fixed": N_CELLS_FIXED,
                "dx_fixed_m": float(dx_fixed.magnitude),
            },
            "spatial_sweep": {
                "n_cells": N_CELLS_SWEEP,
                "n_steps_fixed": N_STEPS_FIXED,
                "dt_fixed_s": float(dt_fixed.magnitude),
            },
        },
        "dt_sweep": dt_records,
        "dx_sweep": dx_records,
    }

    out_path = out_dir / "convergence_data.json"
    with open(out_path, "w") as f:
        json.dump(consolidated, f, indent=2)

    logger.info(
        "convergence study done in %.1f s -> %s", time.time() - t_start, out_path
    )
    return out_path


# ---------------------------------------------------------------------------
# Analytical (0D) extraction time validity study
# ---------------------------------------------------------------------------
"""
Validity of the 0D analytical extraction time (get_tau / get_tau_SPP / get_tau0) against the
full 1D ARD model over a sampled space, correlated with the dimensionless groups
of each simplifying assumption (Pi, G_mix_pred, G_P). Samples vary the product
h_l*K_s and the tank height H log-uniformly over +-DECADES around nominal; sampled
values may be unphysical (numerical-validity, not operating, study).

Run with::

    python -c "from paper.generate_paper_data import generate_validity_data; generate_validity_data()"

Output is written to ``data/analytical_validity/``.
"""

OUT_DIR_VALIDITY = Path("data/analytical_validity")

N_SAMPLES = 300
SEED = 42
DECADES = 2.0  # +- log-uniform sampling range for the h_l*K_s multiplier and H
T_FINAL_IN_TAU = 2  # simulated duration in units of tau_pred
DT_FRACTION_OF_TAU = 0.01  # dt = tau_pred * this
MESH_PE = 2  # mesh Peclet number setting dx
MIN_CELLS = 20  # floor on n_cells
# NB: the non-exponential (fit RMSE) flag and the "other groups are small" rule are
# postprocessing choices, not run conditions -- they live in the notebook CONTROLS cell.


def _solve_and_record(
    inp,
    sample: dict,
    *,
    dt_fraction: float,
    t_final_in_tau: float,
    mesh_pe: float,
    min_cells: int,
    extraction_levels: tuple[float, ...] = (),
    extra_fit_in_tau: tuple[float, ...] = (),
) -> dict:
    """Solve once with dt = dt_fraction*tau, t_final = t_final_in_tau*tau
    and dx from mesh Peclet `mesh_pe` (>= `min_cells` cells); return one
    validity_record row. Shared by the analytical-validity and Sobol studies.

    `extraction_levels` adds one `t_extract_<pct>_s` column per extracted fraction, measured on
    the inventory curve. `extra_fit_in_tau` adds `tau_fitted_<w>tau_s` / `fit_rmse_norm_<w>tau`
    for each shorter fit window, to check the fit is insensitive to it. Both default to empty,
    so the studies that do not ask for them are unaffected."""
    from sparging.postprocess import fit_exp, get_exp_fit_rmse, get_time_to_fraction

    tau = inp.get_tau()
    dt = (tau * dt_fraction).to("s")
    t_final = (t_final_in_tau * tau).to("s")
    n_cells = max(
        int(round((inp.height / inp.dx_from_Pe(mesh_pe)).to("dimensionless").magnitude)),
        min_cells,
    )
    dx = (inp.height / n_cells).to("m")

    sim = Simulation(inp, t_final=t_final, dispersion_on=True, constant_profiles=False)
    out = sim.solve(dt=dt, dx=dx)
    rec = out.validity_record(sample=sample, t_0=0 * ureg.s, t_end=t_final)

    for level in extraction_levels:
        t_x = get_time_to_fraction(
            out.n_T2_salt_series, out.times, 1.0 - level, t_0=0 * ureg.s
        )
        rec[f"t_extract_{round(level * 100)}_s"] = float(t_x.to("s").magnitude)
    for window in extra_fit_in_tau:
        t_w = (window * tau).to("s")
        (tau_w, n0_w), _ = fit_exp(
            out.n_T2_salt_series, out.times, 0 * ureg.s, t_w, "decay", tau_guess=tau
        )
        rec[f"tau_fitted_{window:g}tau_s"] = float(tau_w.to("s").magnitude)
        rec[f"fit_rmse_norm_{window:g}tau"] = float(
            get_exp_fit_rmse(
                out.n_T2_salt_series, out.times, 0 * ureg.s, t_w, tau_w, n0_w
            ).magnitude
        )
    return rec


def _set_scaled(inp, name: str, value):
    """Override an input parameter after the graph has been resolved, keeping the graph node in
    sync. Without this the exported `intermediate_params` still report the pre-scaling value,
    since that section is serialised from the graph and not from the attributes."""
    setattr(inp, name, value)
    if getattr(inp, "graph", None) is not None and name in inp.graph.nodes:
        inp.graph.nodes[name]["value"] = value


def _make_scaled_input(
    height: "ureg.Quantity",
    factor: str,
    multiplier: float,
    e_l_scale: float = 1.0,
    p_top_scale: float = 1.0,
):
    """LIBRA-Pi input with tank height `height`, h_l or K_s scaled by `multiplier`, the liquid
    dispersion coefficient scaled by `e_l_scale` and the top pressure by `p_top_scale` (area
    and molar gas flow fixed). E_l enters neither Pi, G_P nor tau, so it moves G_mix alone;
    P_top sets G_P = rho*g*H/P_top but also rescales the whole gas phase (eps_g, u_g, d_b, a)."""
    from sparging import (
        SimulationInput,
        LIBRA_PI_GEOM,
        LIBRA_PI_MAT,
        LIBRA_PI_OPERATING_PARAMS,
        LIBRA_PI_SPARGING_PARAMS,
    )

    geom = LIBRA_PI_GEOM.copy()
    geom.height = height
    op = LIBRA_PI_OPERATING_PARAMS.copy()
    op.P_top = op.P_top * p_top_scale
    inp = SimulationInput.from_parameters(
        geom,
        LIBRA_PI_MAT.copy(),
        op,
        LIBRA_PI_SPARGING_PARAMS.copy(),
    )
    inp.c_T2_init = C_T2_INIT
    if factor == "h_l":
        base_h_l = inp.h_l
        _set_scaled(inp, "h_l", lambda z, _f=base_h_l, _m=multiplier: _f(z) * _m)
    else:
        _set_scaled(inp, "K_s", inp.K_s * multiplier)
    _set_scaled(inp, "E_l", inp.E_l * e_l_scale)
    return inp


def _draw_sample_specs(n_samples: int, seed: int) -> list[dict]:
    """Draw the (multiplier, factor, height) samples as picklable dict specs."""
    rng = np.random.default_rng(seed)
    H_nominal = get_sim_input_LIBRA_Pi().height.to("m").magnitude
    return [
        {
            "sample_id": i,
            "multiplier": float(10 ** rng.uniform(-DECADES, DECADES)),
            "factor": str(rng.choice(["h_l", "K_s"])),
            "height_m": float(H_nominal * 10 ** rng.uniform(-DECADES, DECADES)),
        }
        for i in range(n_samples)
    ]


def _run_one_sample(spec: dict) -> dict:
    """Rebuild the input from `spec` and solve once; return one validity row.
    (Spawned workers rebuild the input because profile closures aren't picklable.)"""
    warnings.filterwarnings("ignore")

    inp = _make_scaled_input(
        spec["height_m"] * ureg.m, spec["factor"], spec["multiplier"]
    )
    return _solve_and_record(
        inp,
        spec,
        dt_fraction=DT_FRACTION_OF_TAU,
        t_final_in_tau=T_FINAL_IN_TAU,
        mesh_pe=MESH_PE,
        min_cells=MIN_CELLS,
    )


def generate_validity_data(
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
    n_workers: int = 4,
    out_dir: Path = OUT_DIR_VALIDITY,
) -> Path:
    """Run the analytical-validity study (one 1D ARD solve per sample, spawn-parallel) and
    write analytical_validity_data.csv + metadata.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _draw_sample_specs(n_samples, seed)

    t_start = time.time()
    with mp.get_context("spawn").Pool(n_workers) as pool:
        rows = list(pool.imap_unordered(_run_one_sample, specs))
    elapsed = time.time() - t_start
    logger.info("0D validity study: %d samples in %.1f s", len(rows), elapsed)

    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    csv_path = out_dir / "analytical_validity_data.csv"
    df.to_csv(csv_path, index=False)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "git_commit": helpers.get_git_hash(),
                "date": datetime.now().isoformat(),
                "n_samples": len(df),
                "decades": DECADES,
                "t_final_in_tau": T_FINAL_IN_TAU,
                "dt_fraction_of_tau": DT_FRACTION_OF_TAU,
                "mesh_Pe": MESH_PE,
            },
            f,
            indent=2,
        )
    return csv_path


# ---------------------------------------------------------------------------
# Factorial study: isolate the three governing groups
# ---------------------------------------------------------------------------
"""
The log-uniform sampling of `generate_validity_data` cannot separate G_mix from G_P: G_P is a
pure function of H, G_mix goes as H^2/(E_l*tau) with E_l left to its correlation, and tau is
bounded below by tau_SPP, so G_mix has a ceiling set by H alone. This study instead puts the
three groups on a factorial grid, which is possible because each has its own exact knob:

    G_P     <- H          (G_P = rho*g*H / P_top)
    Pi      <- K_s        (Pi is exactly linear in K_s at fixed H)
    G_mix   <- E_l        (E_l enters neither Pi, G_P nor tau, so G_mix goes as 1/E_l)

The mesh is deliberately conservative: the gas Peclet criterion is tightened fourfold and at
least CELLS_PER_SCALE cells are placed across both the gas saturation length H/Pi and the liquid
Thiele length H/sqrt(G_mix), so that discretisation cannot be the limiting factor.

Run with::

    python -c "from paper.generate_paper_data import factorial_study; factorial_study()"

Output is written to ``data/factorial_study/``.
"""

OUT_DIR_FACT = Path("data/factorial_study")

FACT_G_P_LEVELS = [0.01, 0.1, 1.0, 3.0]
FACT_PI_LEVELS = [0.01, 0.1, 1.0, 10.0]
FACT_E_L_SCALES = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
FACT_DT_FRACTION = 0.005
FACT_T_FINAL_IN_TAU = 2
FACT_MESH_PE = 0.5  # 4x finer than the sampled study
FACT_MIN_CELLS = 40
FACT_CELLS_PER_SCALE = 20  # across H/Pi and H/sqrt(G_mix)
FACT_MAX_CELLS = 2000


def _height_for_G_P(g_p: float) -> "ureg.Quantity":
    """Tank height giving the target hydrostatic ratio G_P = rho*g*H / P_top."""
    base = get_sim_input_LIBRA_Pi()
    P_top = base.P_l(base.height)
    return (g_p * P_top / (base.rho_l * 9.81 * ureg.m / ureg.s**2)).to("m")


def _factorial_specs() -> list[dict]:
    """One spec per grid point; the K_s scale is solved from the Pi target (Pi ~ K_s)."""
    specs = []
    for g_p in FACT_G_P_LEVELS:
        height = _height_for_G_P(g_p)
        pi_unit = _make_scaled_input(height, "K_s", 1.0).get_Pi_ave().magnitude
        for pi in FACT_PI_LEVELS:
            for e_l_scale in FACT_E_L_SCALES:
                specs.append({
                    "sample_id": len(specs),
                    "G_P_target": g_p,
                    "Pi_target": pi,
                    "e_l_scale": e_l_scale,
                    "k_s_scale": pi / pi_unit,
                    "height_m": float(height.magnitude),
                })
    return specs


def _run_one_factorial(spec: dict) -> dict:
    """Rebuild the input from `spec`, pick a mesh fine enough for both phases, solve once."""
    warnings.filterwarnings("ignore")

    inp = _make_scaled_input(
        spec["height_m"] * ureg.m, "K_s", spec["k_s_scale"], e_l_scale=spec["e_l_scale"]
    )
    min_cells = min(
        FACT_MAX_CELLS,
        max(
            FACT_MIN_CELLS,
            int(np.ceil(FACT_CELLS_PER_SCALE * inp.get_Pi_ave().magnitude)),
            int(np.ceil(FACT_CELLS_PER_SCALE * np.sqrt(inp.get_G_mix_pred().magnitude))),
        ),
    )
    return _solve_and_record(
        inp,
        spec,
        dt_fraction=FACT_DT_FRACTION,
        t_final_in_tau=FACT_T_FINAL_IN_TAU,
        mesh_pe=FACT_MESH_PE,
        min_cells=min_cells,
    )


def factorial_study(n_workers: int = 6, out_dir: Path = OUT_DIR_FACT) -> Path:
    """Run the (G_P x Pi x G_mix) factorial grid and write factorial_data.csv + metadata.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _factorial_specs()

    t_start = time.time()
    with mp.get_context("spawn").Pool(n_workers) as pool:
        rows = list(pool.imap_unordered(_run_one_factorial, specs))
    elapsed = time.time() - t_start
    logger.info("factorial study: %d runs in %.1f s", len(rows), elapsed)

    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    csv_path = out_dir / "factorial_data.csv"
    df.to_csv(csv_path, index=False)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "git_commit": helpers.get_git_hash(),
                "date": datetime.now().isoformat(),
                "n_runs": len(df),
                "G_P_levels": FACT_G_P_LEVELS,
                "Pi_levels": FACT_PI_LEVELS,
                "E_l_scales": FACT_E_L_SCALES,
                "t_final_in_tau": FACT_T_FINAL_IN_TAU,
                "dt_fraction_of_tau": FACT_DT_FRACTION,
                "mesh_Pe": FACT_MESH_PE,
                "cells_per_scale": FACT_CELLS_PER_SCALE,
                "min_cells": FACT_MIN_CELLS,
                "max_cells": FACT_MAX_CELLS,
                "elapsed_s": elapsed,
            },
            f,
            indent=2,
        )
    return csv_path


# ---------------------------------------------------------------------------
# Analytical validity, second design: one independent knob per group
# ---------------------------------------------------------------------------
"""
Random log-uniform sampling, but with each dimensionless group driven by a quantity of its own,
starting from the nominal LIBRA-Pi input at fixed geometry and fixed molar gas flow:

    Pi     <- K_s     (Pi is exactly linear in K_s; K_s touches neither G_P nor the hydrodynamics)
    G_P    <- P_top   (G_P = rho*g*H/P_top; keeping H fixed also keeps Bo, hence the Peclet cell
                       count, constant -- the mesh no longer jumps between the two rules)
    G_mix  <- E_l     (E_l enters neither Pi, G_P nor tau, so G_mix goes as 1/E_l)

P_top is not orthogonal: at fixed molar flow it also rescales eps_g, u_g, d_b and a, which is
exactly the design question this study is meant to answer (does a lower top pressure extract
faster?). It drags Pi over ~1.3 decades against 4 from K_s, and the induced G_P-Pi correlation is
negative, which populates the (high G_P, low Pi) corner the earlier designs left empty.

Two closure caveats are recorded rather than avoided: eps_g(H) grows as ~(1+G_P) and crosses the
1% no-coalescence limit at the lowest pressures, and the bubble diameter correlation is
extrapolated at both ends of the P_top range. `eps_gH` and `d_b0_m` are in the record so the
physically defensible sub-range can be delimited from the data.

Run with::

    python -c "from paper.generate_paper_data import analytical_validity2; analytical_validity2()"

Output is written to ``data/analytical_validity2/``.
"""

OUT_DIR_AV2 = Path("data/analytical_validity2")

AV2_N_SAMPLES = 300
AV2_SEED = 7
AV2_K_S_DECADES = (-2.0, 2.0)  # -> Pi
AV2_P_TOP_DECADES = (-2.0, 2.0)  # -> G_P
AV2_E_L_DECADES = (-6.0, 0.0)  # -> G_mix
AV2_DT_FRACTION = 0.01
AV2_T_FINAL_IN_TAU = 2


def _av2_specs(n_samples: int, seed: int) -> list[dict]:
    """Draw the three independent log-uniform scalings as picklable dict specs."""
    rng = np.random.default_rng(seed)
    return [
        {
            "sample_id": i,
            "k_s_scale": float(10 ** rng.uniform(*AV2_K_S_DECADES)),
            "p_top_scale": float(10 ** rng.uniform(*AV2_P_TOP_DECADES)),
            "e_l_scale": float(10 ** rng.uniform(*AV2_E_L_DECADES)),
        }
        for i in range(n_samples)
    ]


def _make_av2_input(spec: dict):
    """Nominal LIBRA-Pi geometry with the three sampled scalings applied."""
    return _make_scaled_input(
        get_sim_input_LIBRA_Pi().height,
        "K_s",
        spec["k_s_scale"],
        e_l_scale=spec["e_l_scale"],
        p_top_scale=spec["p_top_scale"],
    )


def _run_one_av2(spec: dict) -> dict:
    """Rebuild the input from `spec` and solve once on the conservative mesh."""
    warnings.filterwarnings("ignore")

    inp = _make_av2_input(spec)
    min_cells = min(
        FACT_MAX_CELLS,
        max(
            FACT_MIN_CELLS,
            int(np.ceil(FACT_CELLS_PER_SCALE * inp.get_Pi_ave().magnitude)),
            int(np.ceil(FACT_CELLS_PER_SCALE * np.sqrt(inp.get_G_mix_pred().magnitude))),
        ),
    )
    return _solve_and_record(
        inp,
        spec,
        dt_fraction=AV2_DT_FRACTION,
        t_final_in_tau=AV2_T_FINAL_IN_TAU,
        mesh_pe=FACT_MESH_PE,
        min_cells=min_cells,
    )


def analytical_validity2(
    n_samples: int = AV2_N_SAMPLES,
    seed: int = AV2_SEED,
    n_workers: int = 6,
    out_dir: Path = OUT_DIR_AV2,
) -> Path:
    """Run the second validity design and write analytical_validity2_data.csv + metadata.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _av2_specs(n_samples, seed)

    t_start = time.time()
    with mp.get_context("spawn").Pool(n_workers) as pool:
        rows = list(pool.imap_unordered(_run_one_av2, specs))
    elapsed = time.time() - t_start
    logger.info("analytical validity 2: %d samples in %.1f s", len(rows), elapsed)

    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    csv_path = out_dir / "analytical_validity2_data.csv"
    df.to_csv(csv_path, index=False)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "git_commit": helpers.get_git_hash(),
                "date": datetime.now().isoformat(),
                "description": (
                    "Random log-uniform sampling from the nominal LIBRA-Pi input, one independent "
                    "knob per dimensionless group, at fixed geometry and fixed molar gas flow."
                ),
                "n_samples": len(df),
                "seed": seed,
                "sampling": {
                    "K_s": {
                        "law": "K_s_nominal * 10**U(a, b)",
                        "decades": list(AV2_K_S_DECADES),
                        "targets": "Pi (exactly linear in K_s)",
                    },
                    "P_top": {
                        "law": "P_top_nominal * 10**U(a, b)",
                        "decades": list(AV2_P_TOP_DECADES),
                        "targets": "G_P = rho*g*H/P_top; also rescales eps_g, u_g, d_b, a",
                    },
                    "E_l": {
                        "law": "E_l_nominal * 10**U(a, b)",
                        "decades": list(AV2_E_L_DECADES),
                        "targets": "G_mix (E_l enters neither Pi, G_P nor tau)",
                    },
                },
                "fixed": "tank geometry (H, A), temperature, molar gas flow, nozzle diameter",
                "t_final_in_tau": AV2_T_FINAL_IN_TAU,
                "dt_fraction_of_tau": AV2_DT_FRACTION,
                "mesh_Pe": FACT_MESH_PE,
                "cells_per_scale": FACT_CELLS_PER_SCALE,
                "min_cells": FACT_MIN_CELLS,
                "max_cells": FACT_MAX_CELLS,
                "mesh_rule": (
                    "n_cells = max(H/dx_from_Pe(mesh_Pe), min_cells, "
                    "cells_per_scale*Pi, cells_per_scale*sqrt(G_mix)), capped at max_cells"
                ),
                "no_coalescence_eps_g_limit": 0.01,
                "elapsed_s": elapsed,
            },
            f,
            indent=2,
        )
    return csv_path


# ---------------------------------------------------------------------------
# Non-exponential case (poor mixing, but Pi and G_P small)
# ---------------------------------------------------------------------------
"""
Re-run of a single analytical_validity2 sample chosen for being flagged as non-exponential while
Pi and G_P are both small: the liquid is badly mixed (G_mix >> 1) but the extraction rate is
almost uniform along z. It is the case where the decay stops being a single exponential although
the value of tau is still right, so it is worth looking at the fields.

Run with::

    python -c "from paper.generate_paper_data import non_exponential_case; non_exponential_case()"

Output is written to ``data/non_exponential/``.
"""

OUT_DIR_NONEXP = Path("data/non_exponential")
NONEXP_SAMPLE_ID = 139  # highest fit RMSE among the flagged samples with Pi < 0.1 and G_P < 0.1


def non_exponential_case(
    sample_id: int = NONEXP_SAMPLE_ID,
    out_dir: Path = OUT_DIR_NONEXP,
    av2_csv: Path = OUT_DIR_AV2 / "analytical_validity2_data.csv",
) -> Path:
    """Re-run one analytical_validity2 sample with every field exported, same discretisation as
    the study so the fit diagnostics reproduce."""
    warnings.filterwarnings("ignore")
    row = pd.read_csv(av2_csv).set_index("sample_id").loc[sample_id]
    spec = {
        "sample_id": int(sample_id),
        "k_s_scale": float(row.k_s_scale),
        "p_top_scale": float(row.p_top_scale),
        "e_l_scale": float(row.e_l_scale),
    }

    inp = _make_av2_input(spec)
    tau = inp.get_tau()
    dt = (tau * AV2_DT_FRACTION).to("s")
    t_final = (AV2_T_FINAL_IN_TAU * tau).to("s")
    n_cells = int(row.n_cells)  # identical mesh to the study
    dx = (inp.height / n_cells).to("m")

    logger.info(
        "=== sample %d: Pi=%.3g, G_P=%.3g, G_mix=%.3g, tau=%.2f h, n_cells=%d ===",
        sample_id, inp.get_Pi_ave().magnitude, inp.get_G_P().magnitude,
        inp.get_G_mix_pred().magnitude, tau.to("hour").magnitude, n_cells,
    )
    sim = Simulation(inp, t_final=t_final, dispersion_on=True, constant_profiles=False)
    sim.exports = VERIF_EXPORTS
    out = sim.solve(dt=dt, dx=dx, verbose=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    out.exports_to_csv(out_dir)
    out.to_json(
        out_dir / "summary.json",
        ["analytical_quantities", "fit_summary", "intermediate_params"],
    )
    record = out.validity_record(sample=spec)
    _ard_inputs_table(inp, dt, dx, out.n_cells, out.n_steps).to_csv(
        out_dir / "ard_inputs.csv", index=False
    )
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "git_commit": helpers.get_git_hash(),
                "date": datetime.now().isoformat(),
                "source_study": str(av2_csv.parent),
                "sample_id": int(sample_id),
                "selected_because": (
                    "flagged non-exponential (fit RMSE > threshold) while Pi < 0.1 and G_P < 0.1: "
                    "poor mixing alone, with an almost uniform extraction rate along z"
                ),
                "spec": spec,
                "dt_fraction_of_tau": AV2_DT_FRACTION,
                "t_final_in_tau": AV2_T_FINAL_IN_TAU,
                "record": record,
            },
            f,
            indent=2,
        )
    logger.info(
        "fit RMSE (normalized) = %.3e, tau_fitted = %.4f h, tau_pred = %.4f h",
        record["fit_rmse_norm"], record["tau_fitted_s"] / 3600, record["tau_pred_s"] / 3600,
    )
    return out_dir


# ---------------------------------------------------------------------------
# Verification case (comparison with the analytical solution)
# ---------------------------------------------------------------------------
"""
Reference case for the verification of the ARD implementation against the analytical
solution. The tank height is divided by VERIF_HEIGHT_SCALE so that the assumptions
behind the analytical solution hold (uniform hydrodynamics G_P << 1, uniform liquid
concentration G_mix << 1); two K_s scalings place it in the small partial pressure
and in the partial pressure limited regime. All fields are exported, together with
the ARD inputs (scalars, bottom-evaluated profiles and dimensionless groups).

Run with::

    python -c "from paper.generate_paper_data import verification_case; verification_case()"

Output is written to ``data/verification_case/<variant>/``.
"""

OUT_DIR_VERIF = Path("data/verification_case")

VERIF_HEIGHT_SCALE = 0.1  # short column -> G_P ~ 1.5e-2, G_mix ~ 7e-6
VERIF_VARIANTS = {"spp": 1.0, "ppl": 45.0}  # K_s scalings -> Pi ~ 0.02 and ~1
VERIF_DT_FRACTION = 0.005  # dt = tau * this
VERIF_T_FINAL_IN_TAU = 5
VERIF_EXPORTS = [
    "c_T2", "y_T2", "P_T2", "aJ_T2", "J_T2",       # time-varying profiles
    "P_g", "h_l", "eps_g", "eps_l", "a", "u_g",     # static profiles
    "n_T2_salt", "ndot_T2",                          # scalar time series
]


def _ard_inputs_table(inp, dt, dx, n_cells, n_steps) -> pd.DataFrame:
    """Flat table of the ARD inputs of `inp`: scalars, profiles evaluated at the tank
    bottom (they vary little over a short column) and dimensionless groups."""
    rows = [
        ("H", r"$H$", "tank height", inp.height, "m"),
        ("A", r"$A$", "tank base area", inp.area, "m**2"),
        ("T", r"$T$", "temperature", inp.temperature, "degC"),
        ("K_s", r"$K_s$", "solubility constant", inp.K_s, "mol/m**3/Pa"),
        ("rho_l", r"$\rho_l$", "liquid density", inp.rho_l, "kg/m**3"),
        ("E_l", r"$E_l$", "liquid dispersion coeff.", inp.E_l, "m**2/s"),
        ("E_g", r"$E_g$", "gas dispersion coeff.", inp.E_g, "m**2/s"),
        ("c_T2_init", r"$c_{T_2}(t=0)$", "initial concentration", inp.c_T2_init, "molT2/m**3"),
        ("P_l0", r"$P_l(0)$", "liquid pressure at bottom", inp.P_l0, "Pa"),
        ("P_g0", r"$P_g(0)$", "gas pressure at bottom", inp.P_g0, "Pa"),
        ("eps_g0", r"$\varepsilon_g(0)$", "gas fraction at bottom", inp.eps_g0, "dimensionless"),
        ("a_0", r"$a(0)$", "specific area at bottom", inp.a_0, "1/m"),
        ("u_g0", r"$u_g(0)$", "gas velocity at bottom", inp.u_g0, "m/s"),
        ("h_l0", r"$h_l(0)$", "mass transfer coeff. at bottom", inp.h_l0, "m/s"),
        ("Pi_ave", r"$\langle\Pi\rangle$", "partial pressure number", inp.get_Pi_ave(), "dimensionless"),
        ("Pi_0", r"$\Pi(0)$", "partial pressure number at bottom", inp.get_Pi0(), "dimensionless"),
        ("G_P", r"$G_P$", "relative hydrostatic variation", inp.get_G_P(), "dimensionless"),
        ("G_mix", r"$G_\mathrm{mix}$", "mixing number", inp.get_G_mix_pred(), "dimensionless"),
        ("Bo", r"$\mathrm{Bo}$", "gas Bodenstein number", inp.get_Bo(), "dimensionless"),
        ("tau", r"$\tau$", "extraction time", inp.get_tau(), "hour"),
        ("tau_SPP", r"$\tau_\mathrm{SPP}$", "SPP extraction time", inp.get_tau_SPP(), "hour"),
        ("dt", r"$\Delta t$", "time step", dt, "s"),
        ("dx", r"$\Delta x$", "mesh size", dx, "m"),
    ]
    df = pd.DataFrame(
        [(k, sym, name, float(q.to(u).magnitude), u) for k, sym, name, q, u in rows],
        columns=["key", "latex_symbol", "description", "value", "unit"],
    )
    return pd.concat([df, pd.DataFrame([
        {"key": "n_cells", "latex_symbol": r"$N_x$", "description": "mesh cells",
         "value": n_cells, "unit": "-"},
        {"key": "n_steps", "latex_symbol": r"$N_t$", "description": "time steps",
         "value": n_steps, "unit": "-"},
    ])], ignore_index=True)


def verification_case(out_dir: Path = OUT_DIR_VERIF) -> Path:
    """Run the verification variants, export every field plus the ARD inputs table."""
    warnings.filterwarnings("ignore")
    H = get_sim_input_LIBRA_Pi().height * VERIF_HEIGHT_SCALE

    for variant, k_s_scale in VERIF_VARIANTS.items():
        inp = _make_scaled_input(H, "K_s", k_s_scale)
        tau = inp.get_tau()
        dt = (tau * VERIF_DT_FRACTION).to("s")
        t_final = (VERIF_T_FINAL_IN_TAU * tau).to("s")
        n_cells = max(int(round((inp.height / inp.dx_from_Pe(MESH_PE)).to("").magnitude)), MIN_CELLS)
        dx = (inp.height / n_cells).to("m")

        sim = Simulation(inp, t_final=t_final, dispersion_on=True, constant_profiles=False)
        sim.exports = VERIF_EXPORTS
        logger.info(
            "=== %s: K_s x %g -> Pi=%.3g, G_P=%.3g, G_mix=%.2g, tau=%.2f h ===",
            variant, k_s_scale, inp.get_Pi_ave().magnitude, inp.get_G_P().magnitude,
            inp.get_G_mix_pred().magnitude, tau.to("hour").magnitude,
        )
        out = sim.solve(dt=dt, dx=dx, verbose=False)

        run_dir = out_dir / variant
        run_dir.mkdir(parents=True, exist_ok=True)
        out.exports_to_csv(run_dir)
        out.to_json(
            run_dir / "summary.json",
            ["analytical_quantities", "fit_summary", "intermediate_params"],
        )
        _ard_inputs_table(inp, dt, dx, out.n_cells, out.n_steps).to_csv(
            run_dir / "ard_inputs.csv", index=False
        )
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(
                {
                    "git_commit": helpers.get_git_hash(),
                    "date": datetime.now().isoformat(),
                    "variant": variant,
                    "height_scale": VERIF_HEIGHT_SCALE,
                    "k_s_scale": k_s_scale,
                    "dt_fraction_of_tau": VERIF_DT_FRACTION,
                    "t_final_in_tau": VERIF_T_FINAL_IN_TAU,
                    "mesh_Pe": MESH_PE,
                },
                f,
                indent=2,
            )
    return out_dir


# ---------------------------------------------------------------------------
# Sobol sensitivity study (operating / design inputs -> fitted tau)
# ---------------------------------------------------------------------------
"""
Total-order Sobol sensitivity of the extraction time to the four controllable inputs
(temperature, gas flow, top pressure, nozzle diameter) over the LIBRA-Pi operating range,
via the Saltelli scheme (scipy.stats.sobol_indices). The same design is run once per
transport-parameter scenario (SCEN_CORRELATIONS), so the two datasets are paired row by row.
This module only generates the design and evaluates the 1D ARD model (spawn-parallel),
writing one CSV row per run (validity_record schema + design tags + the extraction times);
the indices and the operating map are computed downstream in the notebooks. Height, area and
number of nozzles are fixed; the rest follows from the sampled inputs through the correlation
graph.

Run with::

    python -c "from paper.generate_paper_data import sobol_scenario_studies; sobol_scenario_studies()"

Output is written to ``data/sobol_<scenario>/``. Earlier single-scenario runs were produced with an older
PARAM_SPACE and an older tau, are no longer reproducible from HEAD, and each records its own
git commit and param_space in its metadata.json.
"""

OUT_DIR_SOBOL = Path("data/sobol_input_params")  # legacy single-scenario default
SOBOL_SCEN_DIR = {s: Path(f"data/sobol_{s}") for s in SCEN_CORRELATIONS}

SOBOL_N_BASE = 256  # base samples N (power of 2); total runs = N * (d + 2)
SOBOL_SEED = 2024  # same seed for both scenarios -> identical, paired design
SOBOL_DT_FRACTION = 0.02  # dt = tau_pred * this
SOBOL_T_FINAL_IN_TAU = 5  # simulated duration in units of tau_pred; t99 needs 4.66 tau
SOBOL_EXTRACTION_LEVELS = (0.50, 0.90, 0.99)  # extracted fractions timed on the inventory curve
SOBOL_FIT_WINDOWS_IN_TAU = (2.0,)  # extra fit windows, for comparability with the older studies
SOBOL_CELLS_PER_PI = 20  # cells across the gas saturation length H/Pi
SOBOL_CONV_SUBSETS = [8, 16, 32, 64, 128, 256, 512]  # prefix sizes for the convergence check
KANAI_RANGE_CM3_S = (3.0, 10.0)  # validated nozzle flow range of the d_b0 correlation

# sampled inputs in Sobol-column order; each mapped from u in [0,1] by _transform.
# ranges are the attainable operating range of tab:model_input; temperature sampled uniformly
# in KELVIN (723.15-923.15 = 450-650 degC). Only the gas flow gets a log measure: it alone spans
# a decade and tau goes roughly as its inverse. A Sobol index is defined relative to the input
# measure, but log vs uniform over the *same* range was checked to leave it unchanged (report.md).
PARAM_SPACE = [
    {"name": "temperature", "csv": "temperature_K", "unit": "K",
     "min": 723.15, "max": 923.15, "scale": "uniform"},
    {"name": "gas_flow", "csv": "gas_flow_sccm", "unit": "sccm",
     "min": 100.0, "max": 1000.0, "scale": "log"},
    {"name": "top_pressure", "csv": "top_pressure_atm", "unit": "atm",
     "min": 1.0, "max": 2.0, "scale": "uniform"},
    {"name": "nozzle_diameter", "csv": "nozzle_diameter_mm", "unit": "mm",
     "min": 1.0, "max": 4.0, "scale": "uniform"},
]


def _transform(u: float, spec: dict) -> float:
    """Map u in [0,1] to a physical value, uniform or log-uniform over [min, max]."""
    lo, hi = spec["min"], spec["max"]
    if spec["scale"] == "log":
        return float(10 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo))))
    return float(lo + u * (hi - lo))


def param_space_with(**overrides) -> list[dict]:
    """Copy PARAM_SPACE, updating fields of named parameters. e.g.
    param_space_with(gas_flow=dict(min=300, max=1000, scale="uniform"))."""
    out = []
    for sp in PARAM_SPACE:
        sp = dict(sp)
        if sp["name"] in overrides:
            sp.update(overrides[sp["name"]])
        out.append(sp)
    return out


def _saltelli_design(
    n_base: int, seed: int, param_space: list[dict], scenario: str | None = None
) -> list[dict]:
    """Build the Saltelli sample (matrices A, B and the d hybrids AB_i) from a
    scrambled Sobol sequence. Returns one picklable, role-tagged spec per run.
    The scenario travels as a plain string: Correlation objects hold lambdas and would not
    pickle to the spawned workers, which resolve them by identifier instead."""
    from scipy.stats import qmc

    d = len(param_space)
    pts = qmc.Sobol(d=2 * d, scramble=True, seed=seed).random(n_base)
    A, B = pts[:, :d], pts[:, d:]

    specs = []

    def _spec(role: str, base_j: int, var_i: int, u_row) -> None:
        params = {sp["csv"]: _transform(u_row[k], sp) for k, sp in enumerate(param_space)}
        specs.append({
            "sample_id": len(specs),
            "design_role": role,   # "A", "B" or "AB"
            "base_index": base_j,  # which base sample j (for A/B/AB_i pairing)
            "var_index": var_i,    # AB: column swapped from B; A/B: -1
            "scenario": scenario,
            **params,
        })

    for j in range(n_base):
        _spec("A", j, -1, A[j])
    for j in range(n_base):
        _spec("B", j, -1, B[j])
    for i in range(d):
        for j in range(n_base):
            u = A[j].copy()
            u[i] = B[j, i]
            _spec("AB", j, i, u)
    return specs


def _make_sobol_input(spec: dict):
    """LIBRA-Pi input with the four sampled inputs from `spec` applied, and the (K_s, D_l)
    correlations of its scenario. A spec without a scenario keeps the nominal material."""
    from sparging import (
        SimulationInput,
        LIBRA_PI_GEOM,
        LIBRA_PI_MAT,
        LIBRA_PI_OPERATING_PARAMS,
        LIBRA_PI_SPARGING_PARAMS,
        all_correlations,
    )

    geom = LIBRA_PI_GEOM.copy()
    geom.nozzle_diameter = spec["nozzle_diameter_mm"] * ureg.mm
    mat = LIBRA_PI_MAT.copy()
    for name, identifier in SCEN_CORRELATIONS.get(spec.get("scenario"), {}).items():
        setattr(mat, name, all_correlations(identifier))
    op = LIBRA_PI_OPERATING_PARAMS.copy()
    op.temperature = spec["temperature_K"] * ureg.K
    op.ndot_g0 = spec["gas_flow_sccm"] * ureg.sccm
    op.P_top = spec["top_pressure_atm"] * ureg.atm

    inp = SimulationInput.from_parameters(
        geom, mat, op, LIBRA_PI_SPARGING_PARAMS.copy()
    )
    inp.c_T2_init = C_T2_INIT
    return inp


def _run_one_sample_sobol(spec: dict) -> dict:
    """Rebuild the input from `spec`, solve once; return one CSV row
    (validity_record + extraction times + design tags + solve time)."""
    warnings.filterwarnings("ignore")  # the Kanai range is flagged in the record, not warned about
    t0 = time.time()
    inp = _make_sobol_input(spec)
    rec = _solve_and_record(
        inp,
        spec,
        dt_fraction=SOBOL_DT_FRACTION,
        t_final_in_tau=SOBOL_T_FINAL_IN_TAU,
        mesh_pe=MESH_PE,
        min_cells=spec["n_cells"],
        extraction_levels=SOBOL_EXTRACTION_LEVELS,
        extra_fit_in_tau=SOBOL_FIT_WINDOWS_IN_TAU,
    )
    # backward Euler gives tau_num = tau + dt/2 (measured 0.501 in the convergence study)
    rec["tau_fit_nodisc_s"] = rec["tau_fitted_s"] - rec["dt_s"] / 2
    # read back from the graph rather than from SCEN_CORRELATIONS: catches a silent fallback
    # to the default correlation if a name is ever wrong
    rec["K_s_correlation"] = inp.graph.nodes["K_s"]["origin"]
    rec["D_l_correlation"] = inp.graph.nodes["D_l"]["origin"]
    lo, hi = KANAI_RANGE_CM3_S
    flow = rec["nozzle_flow_cm3_s"]
    rec["kanai_in_range"] = bool(flow is not None and lo <= flow <= hi)
    rec["solve_time_s"] = time.time() - t0
    return rec


def _sobol_n_cells(param_space: list[dict], scenario: str | None) -> tuple[int, float]:
    """Cell count for the whole design, sized on the Pi-maximising corner of the box.
    Pi grows with T and P_top and decreases with the gas flow and the nozzle diameter, so the
    corner is known analytically. A per-sample rule would make n_cells a step function of the
    sampled inputs and leak that discontinuity into the Sobol indices."""
    bounds = {sp["name"]: sp for sp in param_space}
    corner = {
        "temperature_K": bounds["temperature"]["max"],
        "gas_flow_sccm": bounds["gas_flow"]["min"],
        "top_pressure_atm": bounds["top_pressure"]["max"],
        "nozzle_diameter_mm": bounds["nozzle_diameter"]["min"],
        "scenario": scenario,
    }
    pi_max = float(_make_sobol_input(corner).get_Pi_ave().magnitude)
    n_cells = min(
        FACT_MAX_CELLS,
        max(MIN_CELLS, int(np.ceil(SOBOL_CELLS_PER_PI * pi_max))),
    )
    return n_cells, pi_max


def sobol_study(
    n_base: int = SOBOL_N_BASE,
    seed: int = SOBOL_SEED,
    n_workers: int = 8,
    out_dir: Path | None = None,
    param_space: list[dict] = PARAM_SPACE,
    scenario: str | None = None,
    resume: bool = True,
) -> Path:
    """Generate the Saltelli design and evaluate the 1D ARD model at every point
    (spawn-parallel), writing sobol_data.csv + metadata.json and logging per-sample
    time + ETA. Pass a `param_space` (see `param_space_with`) and a distinct
    `out_dir` for a sampling-space variant, a `scenario` (see SCEN_CORRELATIONS) to
    override the (K_s, D_l) correlations. With `resume`, runs already present in the
    CSV are skipped, so an interrupted study can be continued and a design can be
    extended to a larger n_base (the scrambled Sobol prefixes are nested)."""
    import scipy

    out_dir = out_dir or (SOBOL_SCEN_DIR[scenario] if scenario else OUT_DIR_SOBOL)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _saltelli_design(n_base, seed, param_space, scenario=scenario)
    n_total = len(specs)
    d = len(param_space)
    conv_subsets = [s for s in SOBOL_CONV_SUBSETS if s <= n_base]
    n_cells, pi_max = _sobol_n_cells(param_space, scenario)
    for spec in specs:
        spec["n_cells"] = n_cells

    rows = []
    csv_path = out_dir / "sobol_data.csv"
    if resume and csv_path.exists():
        done = pd.read_csv(csv_path)
        rows = done.to_dict("records")
        keys = set(zip(done.design_role, done.base_index, done.var_index))
        specs = [
            s for s in specs
            if (s["design_role"], s["base_index"], s["var_index"]) not in keys
        ]
        logger.info("resuming: %d runs already in %s, %d to go", len(rows), csv_path, len(specs))

    logger.info(
        "Sobol study [%s]: d=%d inputs, N_base=%d -> %d runs on %d workers, "
        "n_cells=%d (Pi_max=%.3g)",
        scenario, d, n_base, n_total, n_workers, n_cells, pi_max,
    )

    times = []
    t_start = time.time()
    with mp.get_context("spawn").Pool(n_workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(_run_one_sample_sobol, specs), 1):
            rows.append(rec)
            times.append(rec["solve_time_s"])
            avg = float(np.mean(times))
            eta_min = (len(specs) - k) * avg / n_workers / 60
            logger.info(
                "[%4d/%d] id=%4d %-2s var=%+d : %5.1fs  (avg %.1fs, ETA %.0f min)",
                k, len(specs), rec["sample_id"], rec["design_role"],
                rec["var_index"], rec["solve_time_s"], avg, eta_min,
            )
    elapsed = time.time() - t_start
    logger.info("Sobol study: %d runs in %.0f s (%.1f min)", len(specs), elapsed, elapsed / 60)

    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    df.to_csv(csv_path, index=False)

    from sparging import LIBRA_PI_GEOM

    base = get_sim_input_LIBRA_Pi()
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "git_commit": helpers.get_git_hash(),
                "date": datetime.now().isoformat(),
                "description": (
                    "Total-order Sobol sensitivity of the extraction time to 4 operating/"
                    "design inputs; Saltelli design, scipy.stats.sobol_indices."
                ),
                "scenario": scenario,
                "correlations": SCEN_CORRELATIONS.get(scenario, {}),
                "qoi": "tau_fit_nodisc_s",
                "qoi_note": (
                    "tau_fitted_s minus the backward-Euler bias dt/2; the bias is a purely "
                    "multiplicative dt_fraction/2 here, so it leaves the indices unchanged"
                ),
                "qoi_alternatives": ["tau_fitted_s", "tau_fitted_2tau_s", "t_extract_99_s"],
                "estimator": "scipy.stats.sobol_indices (saltelli_2010)",
                "scipy_version": scipy.__version__,
                "n_base_samples": n_base,
                "d": d,
                "n_total_runs": n_total,
                "seed": seed,
                "design_hash": hashlib.sha256(
                    json.dumps(
                        {"n_base": n_base, "seed": seed, "param_space": param_space},
                        sort_keys=True,
                    ).encode()
                ).hexdigest()[:12],
                "convergence_subsets": conv_subsets,
                "param_space": [{**sp, "sobol_column": i} for i, sp in enumerate(param_space)],
                "sampling_measure": (
                    "uniform in temperature, top pressure and nozzle diameter; "
                    "log-uniform in gas flow"
                ),
                "fixed_params": {
                    "height_m": float(base.height.to("m").magnitude),
                    "area_m2": float(base.area.to("m**2").magnitude),
                    "nb_nozzle": float(LIBRA_PI_GEOM.nb_nozzle.magnitude),
                    "c_T2_init_mol_m3": float(C_T2_INIT.magnitude),
                },
                "discretization": {
                    "dt_fraction_of_tau": SOBOL_DT_FRACTION,
                    "t_final_in_tau": SOBOL_T_FINAL_IN_TAU,
                    "mesh_Pe": MESH_PE,
                    "min_cells": MIN_CELLS,
                    "n_cells": n_cells,
                    "n_cells_rule": (
                        "max(MIN_CELLS, ceil(SOBOL_CELLS_PER_PI * Pi at the Pi-maximising "
                        "corner of the box); fixed over the design so the mesh is not a "
                        "function of the sampled inputs"
                    ),
                    "cells_per_Pi": SOBOL_CELLS_PER_PI,
                    "Pi_max_corner": pi_max,
                },
                "extraction_levels": list(SOBOL_EXTRACTION_LEVELS),
                "extra_fit_windows_in_tau": list(SOBOL_FIT_WINDOWS_IN_TAU),
                "kanai_validated_range_cm3_s": list(KANAI_RANGE_CM3_S),
                "design_note": (
                    "rows tagged design_role in {A,B,AB} + base_index + var_index; "
                    "f_AB[i] = A with column i replaced by B's column i"
                ),
                "n_workers": n_workers,
                "solve_time_s": {
                    "median": float(np.median(df.solve_time_s)),
                    "max": float(df.solve_time_s.max()),
                    "total": float(df.solve_time_s.sum()),
                },
                "elapsed_s": elapsed,
            },
            f,
            indent=2,
        )
    logger.info("wrote %s and metadata.json", csv_path)
    return csv_path


def sobol_scenario_studies(
    n_base: int = SOBOL_N_BASE,
    seed: int = SOBOL_SEED,
    n_workers: int = 8,
    scenarios: tuple[str, ...] = tuple(SCEN_CORRELATIONS),
) -> dict[str, Path]:
    """Run the same Saltelli design once per transport-parameter scenario. Same seed, so the
    two datasets are paired row by row and can be differenced directly."""
    return {
        s: sobol_study(n_base=n_base, seed=seed, n_workers=n_workers, scenario=s)
        for s in scenarios
    }


# ---------------------------------------------------------------------------
# LIBRA-Pi transport-parameter scenarios (bounds on Pi)
# ---------------------------------------------------------------------------
"""
Bounds on the LIBRA-Pi extraction time under the uncertainty on the tritium transport
properties of the salt. Two scenarios bracket the available correlations:

    pessimistic : K_s and D_l both from Calderoni 2008 (T2 in FLiBe)
                  -> high solubility (the salt retains tritium) and low diffusivity
    optimistic  : K_s from Malinauskas 1974 (D2 in FLiBe), D_l from Fukada 2006 (H2 in FLiNaK)
                  -> low solubility and high diffusivity

Each scenario is run at the two corners of the attainable operating range that maximise and
minimise Pi (temperature, gas flow, headspace pressure and nozzle diameter each set to the end
of their range that pushes Pi in the wanted direction; checked one input at a time on the
resolved closures). G_P and G_mix stay within a factor ~2 and ~30 over the whole box, so Pi
alone carries the spread. All fields are exported.

Run with::

    python -c "from paper.generate_paper_data import libra_pi_scenarios; libra_pi_scenarios()"

Output is written to ``data/<scenario>_<corner>/``.
"""

OUT_DIR_SCEN = Path("data")

# corners of the attainable operating range (tab:model_input): Pi grows with T, with P_top and
# with 1/ndot_g, and decreases with the nozzle diameter (both through the bubble diameter).
SCEN_CORNERS = {
    "high_Pi": {
        "temperature_degC": 650.0,
        "gas_flow_sccm": 100.0,
        "top_pressure_atm": 2.0,
        "nozzle_diameter_mm": 1.0,
    },
    "low_Pi": {
        "temperature_degC": 450.0,
        "gas_flow_sccm": 1000.0,
        "top_pressure_atm": 1.0,
        "nozzle_diameter_mm": 4.0,
    },
}

SCEN_DT_FRACTION = 0.01  # dt = tau * this
SCEN_T_FINAL_IN_TAU = 5


def _make_scenario_input(scenario: str, corner: str):
    """LIBRA-Pi input with the scenario's (K_s, D_l) correlations and the corner's
    operating point applied; geometry and tritium source stay nominal."""
    from sparging import (
        SimulationInput,
        LIBRA_PI_GEOM,
        LIBRA_PI_MAT,
        LIBRA_PI_OPERATING_PARAMS,
        LIBRA_PI_SPARGING_PARAMS,
        all_correlations,
    )

    spec = SCEN_CORNERS[corner]
    geom = LIBRA_PI_GEOM.copy()
    geom.nozzle_diameter = spec["nozzle_diameter_mm"] * ureg.mm
    mat = LIBRA_PI_MAT.copy()
    for name, identifier in SCEN_CORRELATIONS[scenario].items():
        setattr(mat, name, all_correlations(identifier))
    op = LIBRA_PI_OPERATING_PARAMS.copy()
    op.temperature = spec["temperature_degC"] * ureg.celsius
    op.ndot_g0 = spec["gas_flow_sccm"] * ureg.sccm
    op.P_top = spec["top_pressure_atm"] * ureg.atm

    inp = SimulationInput.from_parameters(
        geom, mat, op, LIBRA_PI_SPARGING_PARAMS.copy()
    )
    inp.c_T2_init = C_T2_INIT
    return inp


def libra_pi_scenarios(out_dir: Path = OUT_DIR_SCEN) -> Path:
    """Run the four (scenario, corner) combinations and export every field, the parameter
    summary and the ARD inputs table for each."""
    warnings.filterwarnings("ignore")

    for scenario in SCEN_CORRELATIONS:
        for corner in SCEN_CORNERS:
            inp = _make_scenario_input(scenario, corner)
            tau = inp.get_tau()
            dt = (tau * SCEN_DT_FRACTION).to("s")
            t_final = (SCEN_T_FINAL_IN_TAU * tau).to("s")
            n_cells = min(
                FACT_MAX_CELLS,
                max(
                    FACT_MIN_CELLS,
                    int(round((inp.height / inp.dx_from_Pe(FACT_MESH_PE)).to("").magnitude)),
                    int(np.ceil(FACT_CELLS_PER_SCALE * inp.get_Pi_ave().magnitude)),
                ),
            )
            dx = (inp.height / n_cells).to("m")

            logger.info(
                "=== %s / %s: Pi=%.3g, G_P=%.3g, G_mix=%.3g, tau=%.2f h, n_cells=%d ===",
                scenario, corner, inp.get_Pi_ave().magnitude, inp.get_G_P().magnitude,
                inp.get_G_mix_pred().magnitude, tau.to("hour").magnitude, n_cells,
            )
            sim = Simulation(inp, t_final=t_final, dispersion_on=True, constant_profiles=False)
            sim.exports = VERIF_EXPORTS
            out = sim.solve(dt=dt, dx=dx, verbose=False)

            run_dir = out_dir / f"{scenario}_{corner}"
            run_dir.mkdir(parents=True, exist_ok=True)
            out.exports_to_csv(run_dir)
            out.to_json(
                run_dir / "summary.json",
                ["analytical_quantities", "fit_summary", "intermediate_params"],
            )
            record = out.validity_record(
                sample={"scenario": scenario, "corner": corner, **SCEN_CORNERS[corner]}
            )
            _ard_inputs_table(inp, dt, dx, out.n_cells, out.n_steps).to_csv(
                run_dir / "ard_inputs.csv", index=False
            )
            with open(run_dir / "metadata.json", "w") as f:
                json.dump(
                    {
                        "git_commit": helpers.get_git_hash(),
                        "date": datetime.now().isoformat(),
                        "scenario": scenario,
                        "corner": corner,
                        "correlations": SCEN_CORRELATIONS[scenario],
                        "operating_point": SCEN_CORNERS[corner],
                        "c_T2_init_mol_m3": float(C_T2_INIT.magnitude),
                        "dt_fraction_of_tau": SCEN_DT_FRACTION,
                        "t_final_in_tau": SCEN_T_FINAL_IN_TAU,
                        "record": record,
                    },
                    f,
                    indent=2,
                )
            logger.info(
                "    tau_fitted = %.4f h (analytical %.4f h), fit RMSE = %.3e",
                record["tau_fitted_s"] / 3600, record["tau_pred_s"] / 3600,
                record["fit_rmse_norm"],
            )
    return out_dir


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("sparging").setLevel(logging.WARNING)  # mute solver chatter
    convergence_study()
