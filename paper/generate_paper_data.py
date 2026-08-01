"""Generate the data underlying the paper's figures.

Exposes `convergence_study` (mesh/time-step convergence), `generate_validity_data`
(0D-approximation validity) and `sobol_study` (input sensitivity). Outputs go to
subfolders of ``paper/runs/``. Run the convergence study with
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
from sparging import helpers

logger = logging.getLogger(__name__)

OUT_DIR = Path("paper/runs/convergence_study")

# ---------------------------------------------------------------------------
# study configuration
# ---------------------------------------------------------------------------
C_T2_INIT = 3e-11 * ureg.molT2 / ureg.m**3  # initial concentration

# Pi is linear in K_s; scaling K_s alone moves Pi without changing tau, Bo or the
# fluid-dynamic closures. Multipliers -> Pi regimes {SPP, nominal, PPL}.
CASES = {"low": 0.1, "mid": 1.0, "high": 10.0}

T_FINAL_IN_TAU = 4  # simulated duration in units of the SPP tau

# temporal sweep dt = t_final / n_steps at a fixed fine mesh; the finest 3 keep
# ratio r=2 for Richardson extrapolation
N_STEPS_SWEEP = [3, 6, 12, 25, 50, 100, 200, 400, 800, 1600]
N_CELLS_FIXED = 160

# spatial sweep dx = H / n_cells at a fixed fine time step
N_CELLS_SWEEP = [14, 28, 56, 112, 224, 448]
N_STEPS_FIXED = 800

REFINEMENT_RATIO = 2.0


def _make_input(k_s_scale: float):
    """LIBRA-Pi input with K_s scaled by `k_s_scale`."""
    inp = get_sim_input_LIBRA_Pi()
    inp.c_T2_init = C_T2_INIT
    inp.K_s = inp.K_s * k_s_scale
    return inp


def _run(sim: Simulation, dt: "ureg.Quantity", dx: "ureg.Quantity") -> dict:
    """Solve once; return the convergence record plus the raw inventory series."""
    out = sim.solve(dt=dt, dx=dx)
    rec = out.convergence_record()
    rec["_times_s"] = out.times.to("s").magnitude.tolist()
    rec["_inv_mol"] = out.n_T2_salt_series.to("molT2").magnitude.tolist()
    return rec


def convergence_study(out_dir: Path = OUT_DIR) -> Path:
    """Run the full 3-regime x {temporal, spatial} convergence study and write
    a single consolidated JSON plus one sanity CSV per regime.

    Returns the path to the consolidated JSON.
    """
    warnings.filterwarnings("ignore")  # silence curve_fit / pint fit warnings
    out_dir.mkdir(parents=True, exist_ok=True)

    # tau and Bo are K_s-independent -> take them from the nominal input
    base = get_sim_input_LIBRA_Pi()
    tau_spp = base.get_tau_SPP()
    Bo = base.get_Bo()
    H = base.height
    t_final = T_FINAL_IN_TAU * tau_spp

    dx_fixed = (H / N_CELLS_FIXED).to("m")
    dt_fixed = (t_final / N_STEPS_FIXED).to("s")

    logger.info(
        "tau_SPP=%.3f h, Bo=%.2f, t_final=%.3f h, dx_fixed=%.2e m, dt_fixed=%.2e s",
        tau_spp.to("hour").magnitude,
        Bo.magnitude,
        t_final.to("hour").magnitude,
        dx_fixed.magnitude,
        dt_fixed.magnitude,
    )

    consolidated = {
        "metadata": {
            "git_commit": helpers.get_git_hash(),
            "date": datetime.now().isoformat(),
            "description": (
                "Mesh/time-step convergence study. Metric: fitted inventory "
                "decay time tau_fitted. Pi varied via K_s only."
            ),
            "t_final_s": float(t_final.to("s").magnitude),
            "t_final_in_tau_SPP": T_FINAL_IN_TAU,
            "tau_SPP_s": float(tau_spp.to("s").magnitude),
            "Bo": float(Bo.magnitude),
            "H_m": float(H.to("m").magnitude),
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
        "cases": {},
    }

    t_start = time.time()
    for label, scale in CASES.items():
        inp = _make_input(scale)
        Pi = inp.get_Pi_ave()
        logger.info("=== case %s: K_s x %g -> Pi=%.3f ===", label, scale, Pi.magnitude)

        sim = Simulation(
            inp, t_final=t_final, dispersion_on=True, constant_profiles=False
        )

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
            f"# case={label}  Pi={Pi.magnitude:.4f}  "
            f"tau_fitted_h={finest['tau_fitted_s'] / 3600:.4f}  "
            f"n0_fitted_mol={finest['n0_fitted_mol']:.4e}\n"
            "t_seconds,n_T2_salt_mol"
        )
        np.savetxt(
            out_dir / f"{label}_finest_inventory.csv",
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

        consolidated["cases"][label] = {
            "k_s_scale": scale,
            "Pi": float(Pi.magnitude),
            "Bo": float(Bo.magnitude),
            "tau_SPP_s": float(tau_spp.to("s").magnitude),
            "K_s_mol_m3_Pa": float(inp.K_s.to("mol/m**3/Pa").magnitude),
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
# 0D approximation validity study
# ---------------------------------------------------------------------------
"""
Validity of the 0D analytical extraction time (get_tau / get_tau_SPP / get_tau0) against the
full 1D ARD model over a sampled space, correlated with the dimensionless groups
of each simplifying assumption (Pi, G_mix_pred, G_P). Samples vary the product
h_l*K_s and the tank height H log-uniformly over +-DECADES around nominal; sampled
values may be unphysical (numerical-validity, not operating, study).

Run with::

    python -c "from paper.generate_paper_data import generate_validity_data; generate_validity_data()"

Output is written to ``paper/runs/0D_validity_study/``.
"""

OUT_DIR_VALIDITY = Path("paper/runs/0D_validity_study")

N_SAMPLES = 300
SEED = 42
DECADES = 2.0  # +- log-uniform sampling range for the h_l*K_s multiplier and H
T_FINAL_IN_TAU = 2  # simulated duration in units of tau_pred
DT_FRACTION_OF_TAU = 0.01  # dt = tau_pred * this
MESH_PE = 2  # mesh Peclet number setting dx
MIN_CELLS = 20  # floor on n_cells
OTHER_GROUP_THRESHOLD = 0.1  # plot highlighting rule: other groups < this
RMSE_FLAG_THRESHOLD = 1e-5  # normalized-RMSE above this flags non-exponential decay


def _solve_and_record(
    inp,
    sample: dict,
    *,
    dt_fraction: float,
    t_final_in_tau: float,
    mesh_pe: float,
    min_cells: int,
) -> dict:
    """Solve once with dt = dt_fraction*tau, t_final = t_final_in_tau*tau
    and dx from mesh Peclet `mesh_pe` (>= `min_cells` cells); return one
    validity_record row. Shared by the 0D-validity and Sobol studies."""
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
    return out.validity_record(sample=sample, t_0=0 * ureg.s, t_end=t_final)


def _make_validity_input(height: "ureg.Quantity", factor: str, multiplier: float):
    """LIBRA-Pi input with tank height `height` and h_l or K_s scaled by
    `multiplier` (area fixed)."""
    from sparging import (
        SimulationInput,
        LIBRA_PI_GEOM,
        LIBRA_PI_MAT,
        LIBRA_PI_OPERATING_PARAMS,
        LIBRA_PI_SPARGING_PARAMS,
    )

    geom = LIBRA_PI_GEOM.copy()
    geom.height = height
    inp = SimulationInput.from_parameters(
        geom,
        LIBRA_PI_MAT.copy(),
        LIBRA_PI_OPERATING_PARAMS.copy(),
        LIBRA_PI_SPARGING_PARAMS.copy(),
    )
    inp.c_T2_init = C_T2_INIT
    if factor == "h_l":
        base_h_l = inp.h_l
        inp.h_l = lambda z, _f=base_h_l, _m=multiplier: _f(z) * _m
    else:
        inp.K_s = inp.K_s * multiplier
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

    inp = _make_validity_input(
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
    """Run the 0D-validity study (one 1D ARD solve per sample, spawn-parallel) and
    write validity_data.csv + metadata.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _draw_sample_specs(n_samples, seed)

    t_start = time.time()
    with mp.get_context("spawn").Pool(n_workers) as pool:
        rows = list(pool.imap_unordered(_run_one_sample, specs))
    elapsed = time.time() - t_start
    logger.info("0D validity study: %d samples in %.1f s", len(rows), elapsed)

    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    csv_path = out_dir / "validity_data.csv"
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
                "other_group_threshold": OTHER_GROUP_THRESHOLD,
                "rmse_flag_threshold": RMSE_FLAG_THRESHOLD,
            },
            f,
            indent=2,
        )
    return csv_path


# ---------------------------------------------------------------------------
# Sobol sensitivity study (operating / design inputs -> fitted tau)
# ---------------------------------------------------------------------------
"""
Total-order Sobol sensitivity of tau_fitted to the four controllable inputs
(temperature, gas flow, top pressure, nozzle diameter) over the LIBRA-Pi design
envelope, via the Saltelli scheme (scipy.stats.sobol_indices). This module only
generates the design and evaluates the 1D ARD model (spawn-parallel), writing one
CSV row per run (validity_record schema + design tags); the indices are computed
downstream in paper_plots.ipynb. Height/area fixed; the rest follows from the
sampled inputs through the correlation graph.

Run with::

    python -c "from paper.generate_paper_data import sobol_study; sobol_study()"

Output is written to ``paper/runs/sobol_input_params/``.
"""

OUT_DIR_SOBOL = Path("paper/runs/sobol_input_params")

SOBOL_N_BASE = 512  # base samples N (power of 2); total runs = N * (d + 2)
SOBOL_SEED = 2024
SOBOL_DT_FRACTION = 0.02  # dt = tau_pred * this
SOBOL_T_FINAL_IN_TAU = 2  # simulated duration in units of tau_pred
SOBOL_CONV_SUBSETS = [8, 16, 32, 64, 128, 256, 512]  # prefix sizes for the convergence check

# sampled inputs in Sobol-column order; each mapped from u in [0,1] by _transform.
# temperature sampled uniformly in KELVIN (723.15-923.15 = 450-650 degC)
PARAM_SPACE = [
    {"name": "temperature", "csv": "temperature_K", "unit": "K",
     "min": 723.15, "max": 923.15, "scale": "uniform"},
    {"name": "gas_flow", "csv": "gas_flow_sccm", "unit": "sccm",
     "min": 50.0, "max": 1000.0, "scale": "log"},
    {"name": "top_pressure", "csv": "top_pressure_atm", "unit": "atm",
     "min": 1.0, "max": 2.0, "scale": "uniform"},
    {"name": "nozzle_diameter", "csv": "nozzle_diameter_mm", "unit": "mm",
     "min": 0.5, "max": 5.0, "scale": "log"},
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


def _saltelli_design(n_base: int, seed: int, param_space: list[dict]) -> list[dict]:
    """Build the Saltelli sample (matrices A, B and the d hybrids AB_i) from a
    scrambled Sobol sequence. Returns one picklable, role-tagged spec per run."""
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
    """LIBRA-Pi input with the four sampled inputs from `spec` applied."""
    from sparging import (
        SimulationInput,
        LIBRA_PI_GEOM,
        LIBRA_PI_MAT,
        LIBRA_PI_OPERATING_PARAMS,
        LIBRA_PI_SPARGING_PARAMS,
    )

    geom = LIBRA_PI_GEOM.copy()
    geom.nozzle_diameter = spec["nozzle_diameter_mm"] * ureg.mm
    op = LIBRA_PI_OPERATING_PARAMS.copy()
    op.temperature = spec["temperature_K"] * ureg.K
    op.ndot_g0 = spec["gas_flow_sccm"] * ureg.sccm
    op.P_top = spec["top_pressure_atm"] * ureg.atm

    inp = SimulationInput.from_parameters(
        geom, LIBRA_PI_MAT.copy(), op, LIBRA_PI_SPARGING_PARAMS.copy()
    )
    inp.c_T2_init = C_T2_INIT
    return inp


def _run_one_sample_sobol(spec: dict) -> dict:
    """Rebuild the input from `spec`, solve once; return one CSV row
    (validity_record + design tags + solve time)."""
    warnings.filterwarnings("ignore")
    t0 = time.time()
    inp = _make_sobol_input(spec)
    rec = _solve_and_record(
        inp,
        spec,
        dt_fraction=SOBOL_DT_FRACTION,
        t_final_in_tau=SOBOL_T_FINAL_IN_TAU,
        mesh_pe=MESH_PE,
        min_cells=MIN_CELLS,
    )
    rec["solve_time_s"] = time.time() - t0
    return rec


def sobol_study(
    n_base: int = SOBOL_N_BASE,
    seed: int = SOBOL_SEED,
    n_workers: int = 8,
    out_dir: Path = OUT_DIR_SOBOL,
    param_space: list[dict] = PARAM_SPACE,
) -> Path:
    """Generate the Saltelli design and evaluate the 1D ARD model at every point
    (spawn-parallel), writing sobol_data.csv + metadata.json and logging per-sample
    time + ETA. Pass a `param_space` (see `param_space_with`) and a distinct
    `out_dir` for a sampling-space variant."""
    import scipy

    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _saltelli_design(n_base, seed, param_space)
    n_total = len(specs)
    d = len(param_space)
    conv_subsets = [s for s in SOBOL_CONV_SUBSETS if s <= n_base]
    logger.info(
        "Sobol study: d=%d inputs, N_base=%d -> %d runs on %d workers",
        d, n_base, n_total, n_workers,
    )

    rows, times = [], []
    t_start = time.time()
    with mp.get_context("spawn").Pool(n_workers) as pool:
        for k, rec in enumerate(pool.imap_unordered(_run_one_sample_sobol, specs), 1):
            rows.append(rec)
            times.append(rec["solve_time_s"])
            avg = float(np.mean(times))
            eta_min = (n_total - k) * avg / n_workers / 60
            logger.info(
                "[%4d/%d] id=%4d %-2s var=%+d : %5.1fs  (avg %.1fs, ETA %.0f min)",
                k, n_total, rec["sample_id"], rec["design_role"],
                rec["var_index"], rec["solve_time_s"], avg, eta_min,
            )
    elapsed = time.time() - t_start
    logger.info("Sobol study: %d runs in %.0f s (%.1f min)", n_total, elapsed, elapsed / 60)

    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    csv_path = out_dir / "sobol_data.csv"
    df.to_csv(csv_path, index=False)

    from sparging import LIBRA_PI_GEOM

    base = get_sim_input_LIBRA_Pi()
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "git_commit": helpers.get_git_hash(),
                "date": datetime.now().isoformat(),
                "description": (
                    "Total-order Sobol sensitivity of tau_fitted to 4 operating/"
                    "design inputs; Saltelli design, scipy.stats.sobol_indices."
                ),
                "qoi": "tau_fitted_s",
                "estimator": "scipy.stats.sobol_indices (saltelli_2010)",
                "scipy_version": scipy.__version__,
                "n_base_samples": n_base,
                "d": d,
                "n_total_runs": n_total,
                "seed": seed,
                "convergence_subsets": conv_subsets,
                "param_space": [{**sp, "sobol_column": i} for i, sp in enumerate(param_space)],
                "fixed_params": {
                    "height_m": float(base.height.to("m").magnitude),
                    "area_m2": float(base.area.to("m**2").magnitude),
                    "nb_nozzle": float(LIBRA_PI_GEOM.nb_nozzle.magnitude),
                },
                "discretization": {
                    "dt_fraction_of_tau": SOBOL_DT_FRACTION,
                    "t_final_in_tau": SOBOL_T_FINAL_IN_TAU,
                    "mesh_Pe": MESH_PE,
                    "min_cells": MIN_CELLS,
                },
                "design_note": (
                    "rows tagged design_role in {A,B,AB} + base_index + var_index; "
                    "f_AB[i] = A with column i replaced by B's column i"
                ),
                "elapsed_s": elapsed,
            },
            f,
            indent=2,
        )
    logger.info("wrote %s and metadata.json", csv_path)
    return csv_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("sparging").setLevel(logging.WARNING)  # mute solver chatter
    convergence_study()
