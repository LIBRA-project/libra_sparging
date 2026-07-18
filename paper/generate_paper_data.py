"""Generate the data underlying the paper's figures.

Currently exposes `convergence_study`, which produces the mesh / time-step
convergence data used by the "Numerical convergence" section. It starts from a
LIBRA-Pi-like input (same geometry, initial condition and *non-constant* closure
profiles as ``examples/sparging_standard.py``) and varies **only the solubility
K_s** to sweep the partial-pressure number Pi through three regimes:

    - low  Pi (<< 0.1) : small-partial-pressure (SPP) regime
    - mid  Pi (~ 0.2)  : the nominal LIBRA-Pi operating point
    - high Pi (>> 0.1) : partial-pressure-limited (PPL) regime

For each regime two decoupled 1-D refinement studies are run:

    - a **temporal** sweep: refine dt at a fixed, fine mesh
    - a **spatial**  sweep: refine dx at a fixed, fine time step

The derived scalar tracked for convergence is the fitted inventory decay time
``tau_fitted`` (see ``postprocess.summarize_decay``). Only the compact
per-run record from ``SimulationResults.convergence_record`` is stored (no
spatial profiles), plus, for a manual post-run sanity check, the inventory decay
series of the finest temporal run of each regime.

Also exposes `generate_validity_data`, which produces the data for the 0D
approximation validity study (see its docstring below).

Run with::

    python paper/generate_paper_data.py

Output is written to ``paper/runs/convergence_study/`` and
``paper/runs/0D_validity_study/``.
"""

from __future__ import annotations

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
C_T2_INIT = 3e-11 * ureg.molT2 / ureg.m**3  # same IC as sparging_standard.py

# Pi is linear in K_s (see SimulationInput.get_Pi_number); K_s enters neither
# get_tau nor get_Bo nor the fluid-dynamic closures, so scaling it isolates the
# partial-pressure effect without perturbing the discretisation-relevant scales.
CASES = {
    "low": 0.1,   # Pi ~ 0.02   (SPP)
    "mid": 1.0,   # Pi ~ 0.23   (nominal LIBRA-Pi)
    "high": 10.0,  # Pi ~ 2.3    (PPL)
}

T_FINAL_IN_TAU = 4  # simulate 4 SPP-tau (enough decay for a clean exp fit)

# temporal sweep: refine dt (= t_final / n_steps) at a fixed fine mesh.
# The 4 coarsest entries (few steps -> large dt) probe the coarse-timestep regime
# useful for fast parametric sweeps; the finest 3 (400/800/1600) keep ratio r=2
# for Richardson extrapolation.
N_STEPS_SWEEP = [3, 6, 12, 25, 50, 100, 200, 400, 800, 1600]  # ratio r ~ 2
N_CELLS_FIXED = 160  # fine mesh held during the temporal sweep

# spatial sweep: refine dx (= H / n_cells) at a fixed fine time step
N_CELLS_SWEEP = [14, 28, 56, 112, 224, 448]  # ratio r = 2
N_STEPS_FIXED = 800  # fine time step held during the spatial sweep

REFINEMENT_RATIO = 2.0  # geometric ratio r, used later for Richardson/GCI


def _make_input(k_s_scale: float):
    """LIBRA-Pi input with K_s scaled to move Pi (everything else untouched)."""
    inp = get_sim_input_LIBRA_Pi()
    inp.c_T2_init = C_T2_INIT
    inp.K_s = inp.K_s * k_s_scale
    return inp


def _run(sim: Simulation, dt: "ureg.Quantity", dx: "ureg.Quantity") -> dict:
    """Solve once and return the compact convergence record."""
    out = sim.solve(dt=dt, dx=dx)
    rec = out.convergence_record()  # t_0 defaults to the inventory peak
    # keep the raw inventory series of this run around for the caller (sanity)
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
    tau_spp = base.get_tau()
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
        Pi = inp.get_Pi_number()
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
Quantifies when the 0D analytical approximation of the extraction time
(SimulationInput.get_tau / get_tau_ave) is valid, by comparing it against the
full 1D ARD model over a sampled space, and correlating the disagreement with
the dimensionless groups governing each simplifying assumption: Pi (negligible
interfacial partial pressure / solubility), G_mix_pred (perfectly mixed
liquid) and G_P (constant hydrodynamic parameters along z).

Two things are varied, log-uniformly over +-DECADES around nominal LIBRA-Pi
values: the product h_l*K_s (via a multiplier applied to a coin-flip choice of
h_l or K_s -- Pi depends on the product, tau_pred depends only on h_l, so
comparing the two populations tells apart Pi-controlled behaviour from a pure
h_l effect), and the tank height H (area held fixed). This is a numerical-
validity study, not a LIBRA-Pi operating study: sampled values may be
unphysical for the real experiment, that is fine and intended.

Run with::

    python -c "from paper.generate_paper_data import generate_validity_data; generate_validity_data()"

Output is written to ``paper/runs/0D_validity_study/``.
"""

OUT_DIR_VALIDITY = Path("paper/runs/0D_validity_study")

N_SAMPLES = 300
SEED = 42
DECADES = 2.0  # log-uniform sampling range (+-) for both the h_l*K_s multiplier and H
T_FINAL_IN_TAU = 2  # simulate 2x tau_pred_ave -- enough to see the decay
DT_FRACTION_OF_TAU = 0.01  # dt = tau_pred_ave * this
MESH_PE = 2  # mesh Peclet number for dx (matches examples/sparging_standard.py)
MIN_CELLS = 20  # floor on n_cells: dx_from_Pe(MESH_PE) does not scale with H, so
# small-H samples would otherwise plan for very few mesh cells
OTHER_GROUP_THRESHOLD = 0.1  # "other groups < 0.1" highlighting rule used in the plots
RMSE_FLAG_THRESHOLD = 1e-5  # normalized-RMSE above this -> non-exponential decay


def _make_validity_input(height: "ureg.Quantity", factor: str, multiplier: float):
    """LIBRA-Pi-like input with height `height` (area fixed at the nominal
    value) and h_l or K_s scaled by `multiplier`. Both the profile (h_l) and
    scalar (K_s) overrides are read fresh by every downstream getter and by
    Simulation.solve(), so the change is automatically consistent everywhere
    (ARD coefficients, tau_pred, Pi, ...)."""
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
    """Draw the (multiplier, factor, height) sample space. Plain-dict specs
    (only picklable primitives) so they can be sent to a worker process."""
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
    """Build the input from `spec` (not a pre-built SimulationInput: profile
    closures aren't picklable, so a spawned worker process must rebuild the
    input itself), run the 1D ARD model once, and return one row of the
    validity dataset (see SimulationResults.validity_record)."""
    warnings.filterwarnings("ignore")  # silence curve_fit / pint fit warnings

    inp = _make_validity_input(
        spec["height_m"] * ureg.m, spec["factor"], spec["multiplier"]
    )

    tau_pred_ave = inp.get_tau_ave()
    dt = (tau_pred_ave * DT_FRACTION_OF_TAU).to("s")
    t_final = (T_FINAL_IN_TAU * tau_pred_ave).to("s")
    n_cells = max(
        int(round((inp.height / inp.dx_from_Pe(MESH_PE)).to("dimensionless").magnitude)),
        MIN_CELLS,
    )
    dx = (inp.height / n_cells).to("m")

    sim = Simulation(inp, t_final=t_final, dispersion_on=True, constant_profiles=False)
    out = sim.solve(dt=dt, dx=dx)
    return out.validity_record(sample=spec, t_0=0 * ureg.s, t_end=t_final)


def generate_validity_data(
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
    n_workers: int = 4,
    out_dir: Path = OUT_DIR_VALIDITY,
) -> Path:
    """Run the 0D-approximation validity study: one 1D ARD solve per sampled
    (h_l*K_s multiplier, height) point, in parallel, and write the results to
    a single CSV. Spawn-based multiprocessing: each worker rebuilds its own
    SimulationInput from a picklable spec, which sidesteps both the
    unpicklable profile closures and dolfinx/mpi4py process-fork hazards.
    """
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("sparging").setLevel(logging.WARNING)  # mute solver chatter
    convergence_study()
