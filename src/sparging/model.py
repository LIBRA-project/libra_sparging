from __future__ import annotations
from mpi4py import MPI
import dolfinx
import basix
import ufl
import numpy as np
import scipy.constants as const
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc
from dataclasses import dataclass

from datetime import datetime

import yaml
import sparging.helpers as helpers
import json
from pathlib import Path
import pandas as pd

from sparging.config import ureg, const_g

from sparging.inputs import SimulationInput
import pint
from collections.abc import Callable
import logging
from dataclasses import dataclass, field
import warnings
from enum import Enum

logger = logging.getLogger(__name__)

EPS = 1e-26

# log.set_log_level(log.LogLevel.INFO)


class ExportKind(Enum):
    PROFILE = "profile"  # time-varying spatial profile -> 2D (n_time, n_x)
    STATIC = "static"  # time-invariant spatial profile -> 1D (n_x,)
    SERIES = "series"  # scalar per timestep -> 1D (n_time,)


@dataclass
class Export:
    kind: ExportKind
    data: pint.Quantity
    """PROFILE: (n_time, n_x) | STATIC: (n_x,) | SERIES: (n_time,)"""


@dataclass
class SimulationResults:
    times: np.ndarray[pint.Quantity]
    c_T2_profiles: np.ndarray[pint.Quantity]
    """profiles c_T2(z) stacked over time. axis 0: time step, axis 1: position z"""
    y_T2_profiles: np.ndarray[pint.Quantity]
    aJ_T2_profiles: np.ndarray[pint.Quantity]
    x_ct: np.ndarray[pint.Quantity]
    x_y: np.ndarray[pint.Quantity]
    n_T2_salt_series: np.ndarray[pint.Quantity]
    """liquid tritium inventory n_T2(t) [molT2] over time"""
    sources_T2_series: np.ndarray[pint.Quantity]
    ndot_T2_series: np.ndarray[pint.Quantity]
    dt: pint.Quantity = None
    dx: pint.Quantity = None
    sim_input: SimulationInput = None
    exports: dict[str, pint.Quantity] = None
    """name -> 2D Quantity, axis 0: time step, axis 1: position (x_export)"""
    x_export: np.ndarray[pint.Quantity] = None

    keys_to_ignore_results = [
        "sim_input",
        "dt",
        "dx",
        "exports",  # dict of Quantities: exported via exports_to_csv, not JSON/YAML
    ]

    # Backward-compatibility: old field names -> new (axis-named) fields.
    # Applied when deserializing legacy JSON/pickle files.
    _legacy_key_map = {
        "c_T2_solutions": "c_T2_profiles",
        "y_T2_solutions": "y_T2_profiles",
        "aJ_T2_solutions": "aJ_T2_profiles",
        "inventories_T2_salt": "n_T2_salt_series",
        "sources_T2": "sources_T2_series",
        "ndot_T2": "ndot_T2_series",
    }

    def to_yaml(self, output_path: Path):
        sim_dict = self.sim_input.__dict__.copy()
        helpers.setup_yaml()
        output = {
            "metadata": {
                "git_commit": helpers.get_git_hash(),
                "date": datetime.now().isoformat(),
            },
        }
        output["simulation parameters"] = {}
        for key, value in sim_dict.items():
            output["simulation parameters"][key] = str(value)
        output["results"] = self.__dict__.copy()
        for key in self.keys_to_ignore_results:
            output["results"].pop(key, None)
        with open(output_path, "w") as f:
            yaml.dump(output, f, sort_keys=False)

    def serialize_output(self):
        sim_dict = self.sim_input.__dict__.copy()
        output = {
            "metadata": {
                "git_commit": helpers.get_git_hash(),
                "date": datetime.now().isoformat(),
            },
        }
        output["simulation parameters"] = {}
        for key, value in sim_dict.items():
            output["simulation parameters"][key] = str(value)
        output["results"] = self.__dict__.copy()
        for key in self.keys_to_ignore_results:
            output["results"].pop(key, None)

        for key, value in output.items():
            if isinstance(value, np.ndarray):
                output[key] = value.tolist()
                logger.verbose(
                    "found list in results, converting to list for JSON serialization"
                )
            if isinstance(value, pint.Quantity):
                output[key] = value.to_base_units().magnitude
                logger.verbose(
                    "found pint.Quantity in results, converting to magnitude for JSON serialization"
                )
            else:
                logger.verbose(
                    f"{key} is of type {type(value)}, no conversion needed for JSON serialization"
                )

        for k, v in output["results"].items():
            if isinstance(v, pint.Quantity):
                units = str(v.units)
                output["results"][k] = {"value": v.magnitude, "units": units}
                if isinstance(v.magnitude, np.ndarray):
                    logger.verbose(
                        f"found pint.Quantity with numpy array magnitude in results[{k}], converting to list for JSON serialization"
                    )
                    output["results"][k]["value"] = v.magnitude.tolist()
        return output

    # ---- scalar summary export ---------------------------------------------

    @staticmethod
    def _serialize_scalar(v):
        """pint.Quantity -> {value, units}; tuples -> list; else passthrough."""
        if isinstance(v, pint.Quantity):
            mag = v.magnitude
            return {
                "value": float(mag) if np.ndim(mag) == 0 else np.asarray(mag).tolist(),
                "units": f"{v.units:~}",
            }
        if isinstance(v, tuple):
            return [SimulationResults._serialize_scalar(x) for x in v]
        return v

    def _section_intermediate_params(self, **_) -> dict:
        return self.sim_input.intermediate_params_dict()

    def _section_fit_summary(self, **kwargs) -> dict:
        from sparging.postprocess import summarize_decay

        summary = summarize_decay(
            self,
            t_0=kwargs.get("t_0"),
            t_end=kwargs.get("t_end"),
        )
        return {k: self._serialize_scalar(v) for k, v in summary.items()}

    def _section_analytical_quantities(self, **_) -> dict:
        si = self.sim_input
        # each entry wrapped so one failing property doesn't kill the whole export
        candidates = {
            "tau_predicted": si.get_tau,
            "c_T2_steady_state": si.get_c_T2_SS,
            "Pi_number": si.get_Pi_number,
            "Bodenstein_number": si.get_Bo,
            "S_T": si.get_S_T,
            "dP_dx_over_c": si.get_dP_dx,
        }
        out = {}
        for name, getter in candidates.items():
            try:
                out[name] = self._serialize_scalar(getter())
            except Exception as e:  # e.g. graph missing v_g0 node
                out[name] = {"error": str(e)}
        return out

    def to_json(self, output_path: Path, sections: list[str], **kwargs):
        """Write a SCALAR summary of the run.

        sections: any of
            "intermediate_params"    -> resolved closure-relation values (from graph)
            "fit_summary"            -> exponential decay fit of n_T2_salt vs get_tau()
            "analytical_quantities"  -> Pi, predicted tau, c_T2 steady state, Bo, ...

        Extra kwargs (e.g. t_0, t_end) are forwarded to the section builders.
        """
        import json

        builders = {
            "intermediate_params": self._section_intermediate_params,
            "fit_summary": self._section_fit_summary,
            "analytical_quantities": self._section_analytical_quantities,
        }
        unknown = [s for s in sections if s not in builders]
        if unknown:
            raise ValueError(
                f"Unknown section(s) {unknown}. Available: {sorted(builders)}"
            )

        output = {
            "metadata": {
                "git_commit": helpers.get_git_hash(),
                "date": datetime.now().isoformat(),
            }
        }
        for section in sections:
            output[section] = builders[section](**kwargs)

        with open(output_path, "w") as f:
            json.dump(output, f, indent=3)

    def to_pickle(self, output_path: Path):
        """Use to_pickle / from_pickle for full serialization."""
        import pickle

        output = self.serialize_output()
        with open(output_path, "wb") as f:
            pickle.dump(output, f)

    def profiles_to_csv(self, output_directory: Path):
        """Save c_T2 and y_T2 profiles at all time steps as CSV files.
        replaced by exports_to_csv, but kept for legacy"""
        times_s = np.array([t.to("seconds").magnitude for t in self.times])
        col_names = [f"t={t:.1f}s" for t in times_s]

        df_c_T2 = pd.DataFrame(
            np.column_stack([self.x_ct.magnitude, self.c_T2_profiles.magnitude.T]),
            columns=["x_metres", *col_names],
        )
        df_y_T2 = pd.DataFrame(
            np.column_stack([self.x_y.magnitude, self.y_T2_profiles.magnitude.T]),
            columns=["x_metres", *col_names],
        )
        df_aJ_T2 = pd.DataFrame(
            np.column_stack([self.x_y.magnitude, self.aJ_T2_profiles.magnitude.T]),
            columns=["x_metres", *col_names],
        )

        df_c_T2.to_csv(output_directory / "c_T2.csv", index=False, float_format="%.6e")
        df_y_T2.to_csv(output_directory / "y_T2.csv", index=False, float_format="%.6e")
        df_aJ_T2.to_csv(
            output_directory / "aJ_T2.csv", index=False, float_format="%.6e"
        )

    def profiles_to_cdf(self, output_directory: Path):
        """Export profiles to a self-describing NetCDF file, preserving units. Use for archiving"""
        import xarray as xr

        def split(q, target_unit=None):
            if target_unit is not None:
                q = q.to(target_unit)
            return np.asarray(q.magnitude), f"{q.units:~P}"

        t_mag, t_unit = split(self.times, "s")
        x_ct_mag, x_ct_unit = split(self.x_ct, "m")
        x_y_mag, x_y_unit = split(self.x_y, "m")

        aJ_T2_mag, aJ_T2_unit = split(self.aJ_T2_profiles, "molT2/m^3/s")
        c_mag, c_unit = split(self.c_T2_profiles, "molT2/m^3")
        y_mag, y_unit = split(self.y_T2_profiles)

        ds = xr.Dataset(
            data_vars={
                "c_T2": (
                    ("time", "x_ct"),
                    c_mag,
                    {"units": c_unit, "long_name": "T2 concentration"},
                ),
                "y_T2": (
                    ("time", "x_y"),
                    y_mag,
                    {"units": y_unit, "long_name": "T2 molar fraction"},
                ),
                "aJ_T2": (
                    ("time", "x_y"),
                    aJ_T2_mag,
                    {"units": aJ_T2_unit, "long_name": "T2 reaction rate"},
                ),
            },
            coords={
                "time": ("time", t_mag, {"units": t_unit, "long_name": "time"}),
                "x_ct": (
                    "x_ct",
                    x_ct_mag,
                    {"units": x_ct_unit, "long_name": "position"},
                ),
                "x_y": ("x_y", x_y_mag, {"units": x_y_unit, "long_name": "position"}),
            },
            attrs={
                "description": "T2 concentration and molar fraction profiles",
                "source": f"{type(self).__name__} simulation results",
            },
        )

        output_directory.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output_directory / "profiles.nc")

    def exports_to_csv(self, output_directory: Path):
        """Write each export to a self-describing CSV. The export KIND is implied
        by the column layout, and the physical UNIT is stored as a leading
        '# units: <unit>' comment line.

            PROFILE : col 0 'x_metres', then one column per time step 't=<s>s'
            STATIC  : columns 'x_metres', 'value'
            SERIES  : columns 't_seconds', 'value'
        """
        if not self.exports:
            warnings.warn(
                "No exports to write. Set `simulation.exports = [...]` before solving."
            )
            return

        output_directory.mkdir(parents=True, exist_ok=True)
        times_s = self.times.to("seconds").magnitude
        x_m = self.x_export.magnitude

        for name, exp in self.exports.items():
            if exp.kind == ExportKind.PROFILE:
                col_names = [f"t={t:.1f}s" for t in times_s]
                df = pd.DataFrame(
                    np.column_stack([x_m, exp.data.magnitude.T]),
                    columns=["x_metres", *col_names],
                )
            elif exp.kind == ExportKind.STATIC:
                df = pd.DataFrame({"x_metres": x_m, "value": exp.data.magnitude})
            elif exp.kind == ExportKind.SERIES:
                df = pd.DataFrame({"t_seconds": times_s, "value": exp.data.magnitude})

            with open(output_directory / f"{name}.csv", "w", newline="") as f:
                f.write(f"# units: {exp.data.units:~P}\n")
                df.to_csv(f, index=False, float_format="%.6e")

    @classmethod
    def deserialize_output(cls, data: dict) -> SimulationResults:
        results = data.get("results", {})

        # backward compatibility: remap legacy field names to axis-named fields
        for old_key, new_key in cls._legacy_key_map.items():
            if old_key in results and new_key not in results:
                results[new_key] = results.pop(old_key)

        for k, v in results.items():
            if isinstance(v, dict) and "value" in v and "units" in v:
                results[k] = ureg.Quantity(v["value"], v["units"])

        return cls(**results)

    @classmethod
    def from_json(cls, input_path: Path) -> SimulationResults:
        with open(input_path, "r") as f:
            data = json.load(f)
        return cls.deserialize_output(data)

    @classmethod
    def from_pickle(cls, input_path: Path) -> SimulationResults:
        import pickle

        with open(input_path, "rb") as f:
            data = pickle.load(f)
        return cls.deserialize_output(data)


@dataclass
class Simulation:
    sim_input: SimulationInput
    t_final: pint.Quantity
    dispersion_on: bool = True
    constant_profiles: bool = False
    exports: list[str] = field(default_factory=list)
    """Names of quantities to export (must be keys of the export registry built
    in `solve`). Each is written to '<name>.csv' by `SimulationResults.exports_to_csv`."""

    def normalize_profile(
        self, profile: Callable[[float], float] | None, length: float, mesh, func_space
    ):
        "helper: make sure a profile is normalized (integral over the dimensionless length is 1)"
        if profile is not None:  # spatially varying profile is provided
            arbitrary_profile = dolfinx.fem.Function(func_space)
            arbitrary_profile.interpolate(lambda x: x[0] * 0 + profile(x[0] / length))
            profile_mean = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(
                    arbitrary_profile * 1 / length * ufl.dx  # TODO
                )  # dimensionless integral
            )
            normalized_profile = arbitrary_profile / profile_mean
        else:  # of no profile provided, assume homogeneous
            normalized_profile = dolfinx.fem.Constant(mesh, PETSc.ScalarType(1.0))
        return normalized_profile

    def solve(
        self,
        dt: pint.Quantity | None = None,
        dx: pint.Quantity | None = None,
        fast_solve: bool = False,
    ) -> SimulationResults:
        """Input:
        - dt: time step, 1000 equal time steps by default
        - dx: spatial step, 1000 equal spatial steps by default
        - fast_solve: if True, use only 50 equal time and spatial steps
        """
        # unpack pint.Quantities
        t_final = self.t_final.to("seconds").magnitude
        tank_height = self.sim_input.height.to("m").magnitude
        tank_area = self.sim_input.area.to("m**2").magnitude
        tank_volume = self.sim_input.volume.to("m**3").magnitude
        h_l = self.sim_input.h_l.to("m/s").magnitude
        K_s = self.sim_input.K_s.to("mol/m**3/Pa").magnitude  # convert to molT2 ?
        T = self.sim_input.temperature.to("K").magnitude
        E_g = self.sim_input.E_g.to("m**2/s").magnitude
        E_l = self.sim_input.E_l.to("m**2/s").magnitude
        Q_T2 = self.sim_input.Q_T.to("molT2/s").magnitude

        dt = (
            dt.to("seconds").magnitude
            if dt is not None
            else (t_final / 1000 if not fast_solve else t_final / 50)
        )
        dx = (
            dx.to("m").magnitude
            if dx is not None
            else (tank_height / 1000 if not fast_solve else tank_height / 50)
        )
        # eps_l = 1 - eps_g

        # MESH AND FUNCTION SPACES
        mesh = dolfinx.mesh.create_interval(
            MPI.COMM_WORLD, int(tank_height / dx), points=[0, tank_height]
        )
        fdim = mesh.topology.dim - 1
        cg_el = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(2,))
        profile_el = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1)

        V = dolfinx.fem.functionspace(mesh, cg_el)
        V_profile = dolfinx.fem.functionspace(mesh, profile_el)

        u = dolfinx.fem.Function(V)
        u_n = dolfinx.fem.Function(V)
        v_c, v_y = ufl.TestFunctions(V)

        # define spatially varying profiles
        if not self.constant_profiles:
            eps_g = dolfinx.fem.Function(V_profile)
            eps_l = dolfinx.fem.Function(V_profile)
            a = dolfinx.fem.Function(V_profile)
            P_g = dolfinx.fem.Function(V_profile)
            u_g = dolfinx.fem.Function(V_profile)

            eps_g.interpolate(
                lambda x: (
                    self.sim_input.eps_g(x[0] * ureg.m).to("dimensionless").magnitude
                )
            )
            eps_l.interpolate(
                lambda x: (
                    1.0
                    - self.sim_input.eps_g(x[0] * ureg.m).to("dimensionless").magnitude
                )
            )
            a.interpolate(lambda x: self.sim_input.a(x[0] * ureg.m).to("1/m").magnitude)
            P_g.interpolate(
                lambda x: self.sim_input.P_g(x[0] * ureg.m).to("Pa").magnitude
            )
            u_g.interpolate(
                lambda x: self.sim_input.u_g(x[0] * ureg.m).to("m/s").magnitude
            )
        else:
            # use values at z=0 for constant profiles
            eps_g0 = self.sim_input.eps_g0.to("dimensionless").magnitude
            a_0 = self.sim_input.a_0.to("1/m").magnitude
            P_g0 = self.sim_input.P_g0.to("Pa").magnitude
            u_g0 = self.sim_input.u_g0.to("m/s").magnitude

            eps_g = dolfinx.fem.Constant(mesh, PETSc.ScalarType(eps_g0))
            eps_l = dolfinx.fem.Constant(mesh, PETSc.ScalarType(1 - eps_g0))
            a = dolfinx.fem.Constant(mesh, PETSc.ScalarType(a_0))
            P_g = dolfinx.fem.Constant(mesh, PETSc.ScalarType(P_g0))
            u_g = dolfinx.fem.Constant(mesh, PETSc.ScalarType(u_g0))

        # set initial concentration
        c_T2_init_ufl_expr = dolfinx.fem.Constant(
            mesh, self.sim_input.c_T2_init.to("molT2/m**3").magnitude
        ) * self.normalize_profile(
            self.sim_input.profile_c_T2_init, tank_height, mesh, V_profile
        )
        u_n.sub(0).interpolate(
            dolfinx.fem.Expression(
                c_T2_init_ufl_expr, V.sub(0).element.interpolation_points
            )
        )
        # make u match u_n at t=0 so interpolated exports are correct at the first step
        u.x.array[:] = u_n.x.array[:]

        c_T2, y_T2 = ufl.split(u)
        c_T2_n, y_T2_n = ufl.split(u_n)

        h_l_const = dolfinx.fem.Constant(mesh, PETSc.ScalarType(h_l))

        gen_T2_ave = dolfinx.fem.Constant(
            mesh, Q_T2 / tank_volume * self.sim_input.signal_irr(0 * ureg.s)
        )  # magnitude of the generation term

        gen_T2 = gen_T2_ave * self.normalize_profile(
            self.sim_input.profile_source_T, tank_height, mesh, V_profile
        )

        # VARIATIONAL FORMULATION

        # mass transfer rate
        aJ_T2 = a * h_l_const * (c_T2 - K_s * (P_g * y_T2 + EPS))

        F = 0  # variational formulation

        # transient terms: implicit (backward) euler scheme: [u_n+1 - u_n)] / dt = f(u_n+1) -> new state appears in both derivative and function it is equal to
        F += eps_l * ((c_T2 - c_T2_n) / dt) * v_c * ufl.dx
        F += eps_g * 1 / (const.R * T) * (P_g * (y_T2 - y_T2_n) / dt) * v_y * ufl.dx

        # dispersive terms
        if self.dispersion_on is True:
            F += eps_l * E_l * ufl.dot(ufl.grad(c_T2), ufl.grad(v_c)) * ufl.dx
            F += (
                eps_g
                * E_g
                * 1
                / (const.R * T)
                * ufl.dot(ufl.grad(P_g * y_T2), ufl.grad(v_y))
                * ufl.dx
            )

        # mass exchange (coupling term)
        F += aJ_T2 * v_c * ufl.dx - aJ_T2 * v_y * ufl.dx

        # Generation term in the breeder
        F += -gen_T2 * v_c * ufl.dx

        # advection of gas
        F += 1 / (const.R * T) * ufl.grad(u_g * P_g * y_T2)[0] * v_y * ufl.dx

        # BOUNDARY CONDITIONS
        gas_inlet_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.isclose(x[0], 0.0)
        )
        gas_outlet_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.isclose(x[0], tank_height)
        )
        # bc1 = dolfinx.fem.dirichletbc(
        #     dolfinx.fem.Constant(mesh, 0.0),
        #     dolfinx.fem.locate_dofs_topological(V.sub(1), fdim, gas_inlet_facets),
        #     V.sub(1),
        # )  # Dirichlet BC y_T2 = 0 at gas inlet

        # Custom measure
        all_facets = np.concatenate((gas_inlet_facets, gas_outlet_facets))
        all_tags = np.concatenate(
            (np.full_like(gas_inlet_facets, 1), np.full_like(gas_outlet_facets, 2))
        )
        facet_markers = dolfinx.mesh.meshtags(mesh, fdim, all_facets, all_tags)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_markers)

        # Danckwert BC at gas inlet
        P_T2_inlet = 0
        F += 1 / (const.R * T) * u_g * ufl.inner((P_g * y_T2 - P_T2_inlet), v_y) * ds(1)

        # n = ufl.FacetNormal(mesh)

        # set up problem
        problem = NonlinearProblem(
            F,
            u,
            # bcs=[bc1],  # Neumann BCs on c_T2 at inlet and outlet are naturally enforced
            petsc_options_prefix="librasparge",
            # petsc_options={"snes_monitor": None},
        )

        # initialise post processing

        # we define a function and an expression for aJ_T2 for use in post processing
        aJ_T2_func = dolfinx.fem.Function(V_profile)
        aJ_T2_expr = dolfinx.fem.Expression(
            aJ_T2, V_profile.element.interpolation_points
        )

        V0_ct, ct_dofs = u.function_space.sub(0).collapse()
        coords = V0_ct.tabulate_dof_coordinates()[:, 0]
        ct_sort_coords = np.argsort(coords)
        x_ct = coords[ct_sort_coords]

        V0_y, y_dofs = u.function_space.sub(1).collapse()
        coords = V0_y.tabulate_dof_coordinates()[:, 0]
        y_sort_coords = np.argsort(coords)
        x_y = coords[y_sort_coords]

        coords_profile = V_profile.tabulate_dof_coordinates()[:, 0]
        num_profile_dofs = (
            V_profile.dofmap.index_map.size_local
        ) * V_profile.dofmap.index_map_bs

        profile_dofs = np.arange(num_profile_dofs)
        profile_sort_coords = np.argsort(coords_profile)
        # NOTE currently we don't use x_profile and use another x in the plotting script
        x_profile = coords_profile[profile_sort_coords]

        # scalar forms reused by SERIES exports
        ndot_T2_form = dolfinx.fem.form(
            u_g * P_g / (const.R * T) * y_T2_n * tank_area * ds(2)
        )
        n_T2_salt_form = dolfinx.fem.form(c_T2_n * tank_area * ufl.dx)

        # ---- EXPORTS registry: name -> (kind, units, expr_or_callable) ----
        exportable = {
            # time-varying spatial profiles (UFL expression)
            "c_T2": (ExportKind.PROFILE, "molT2/m^3", c_T2),
            "y_T2": (ExportKind.PROFILE, "dimensionless", y_T2),
            "P_T2": (ExportKind.PROFILE, "Pa", P_g * y_T2),
            "aJ_T2": (ExportKind.PROFILE, "molT2/m^3/s", aJ_T2),
            # time-invariant spatial profiles (UFL expression)
            "P_g": (ExportKind.STATIC, "Pa", P_g),
            "eps_g": (ExportKind.STATIC, "dimensionless", eps_g),
            "eps_l": (ExportKind.STATIC, "dimensionless", eps_l),
            "a": (ExportKind.STATIC, "1/m", a),
            "u_g": (ExportKind.STATIC, "m/s", u_g),
            # scalar time series (callable f(t) -> magnitude in `units`)
            "ndot_T2": (
                ExportKind.SERIES,
                "molT2/s",
                lambda t: dolfinx.fem.assemble_scalar(ndot_T2_form),
            ),
            "n_T2_salt": (
                ExportKind.SERIES,
                "molT2",
                lambda t: dolfinx.fem.assemble_scalar(n_T2_salt_form),
            ),
            "source_T2": (
                ExportKind.SERIES,
                "molT2/s",
                lambda t: Q_T2 * self.sim_input.signal_irr(t * ureg.s),
            ),
        }

        unknown = [name for name in self.exports if name not in exportable]
        if unknown:
            raise ValueError(
                f"Cannot export unknown quantities {unknown}. "
                f"Available exports: {sorted(exportable)}"
            )

        export_kinds = {n: exportable[n][0] for n in self.exports}
        export_units = {n: exportable[n][1] for n in self.exports}

        # interpolation machinery for PROFILE + STATIC
        export_funcs, export_exprs = {}, {}
        for name in self.exports:
            kind, _, thing = exportable[name]
            if kind in (ExportKind.PROFILE, ExportKind.STATIC):
                export_funcs[name] = dolfinx.fem.Function(V_profile)
                export_exprs[name] = dolfinx.fem.Expression(
                    thing, V_profile.element.interpolation_points
                )

        # callables for SERIES
        export_series_fn = {
            name: exportable[name][2]
            for name in self.exports
            if export_kinds[name] == ExportKind.SERIES
        }

        # storage: PROFILE and SERIES grow over time; STATIC filled once
        export_profiles = {
            n: [] for n in self.exports if export_kinds[n] == ExportKind.PROFILE
        }
        export_series = {
            n: [] for n in self.exports if export_kinds[n] == ExportKind.SERIES
        }
        export_static = {}

        # NOTE maybe we could take this function out and it would take a SimulationResults object as input + u + other things...
        def post_process(t):
            """
            Post-process the solution at time t.
            Extract solution profiles, compute fluxes and inventories, and store them in lists for later analysis.
            """
            c_T2_post, y_T2_post = u_n.split()

            c_T2_vals = u_n.x.array[ct_dofs][ct_sort_coords]
            y_T2_vals = u_n.x.array[y_dofs][y_sort_coords]
            aJ_T2_func.interpolate(aJ_T2_expr)
            aJ_T2_vals = aJ_T2_func.x.array[profile_dofs][ct_sort_coords]

            times.append(t)
            c_T2_profiles.append(c_T2_vals.copy())
            y_T2_profiles.append(y_T2_vals.copy())
            aJ_T2_profiles.append(aJ_T2_vals.copy())
            sources_T2_series.append(Q_T2 * self.sim_input.signal_irr(t * ureg.s))

            ndot_T2 = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(
                    u_g * P_g / (const.R * T) * y_T2_post * tank_area * ds(2)
                )
            )
            ndot_T2_series.append(ndot_T2)

            n_T2_salt = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(c_T2_post * ufl.dx)
            )
            n_T2_salt *= tank_area  # total amount of T2 in [mol]
            n_T2_salt_series.append(n_T2_salt)
            for name in self.exports:
                kind = export_kinds[name]
                if kind == ExportKind.PROFILE:
                    export_funcs[name].interpolate(export_exprs[name])
                    vals = export_funcs[name].x.array[profile_dofs][profile_sort_coords]
                    export_profiles[name].append(vals.copy())
                elif kind == ExportKind.SERIES:
                    export_series[name].append(export_series_fn[name](t))

        t = 0
        times = []
        c_T2_profiles = []
        y_T2_profiles = []
        aJ_T2_profiles = []
        sources_T2_series = []
        ndot_T2_series = []
        n_T2_salt_series = []
        # initialize (t=0)
        post_process(t)

        # SOLVE
        while t < t_final:
            # advance time
            t += dt
            # update time-dependent terms
            gen_T2_ave.value = (
                Q_T2 / tank_volume * self.sim_input.signal_irr(t * ureg.s)
            )
            h_l_const.value = h_l * self.sim_input.signal_sparging(t * ureg.s)

            problem.solve()  # solves for u (equivalent to u_n+1)

            # update previous solution
            u_n.x.array[:] = u.x.array[:]

            post_process(t)

        # STATIC profiles are time-invariant -> sample a single column
        for name in self.exports:
            if export_kinds[name] == ExportKind.STATIC:
                export_funcs[name].interpolate(export_exprs[name])
                export_static[name] = (
                    export_funcs[name].x.array[profile_dofs][profile_sort_coords].copy()
                )

        results = SimulationResults(
            times=np.array(times) * ureg("s"),
            c_T2_profiles=np.array(c_T2_profiles) * ureg("molT2/m^3"),
            y_T2_profiles=np.array(y_T2_profiles) * ureg("dimensionless"),
            aJ_T2_profiles=np.array(aJ_T2_profiles) * ureg("molT2/m^3/s"),
            x_ct=x_ct * ureg("m"),
            x_y=x_y * ureg("m"),
            n_T2_salt_series=np.array(n_T2_salt_series) * ureg("molT2"),
            sources_T2_series=np.array(sources_T2_series) * ureg("molT2/s"),
            ndot_T2_series=np.array(ndot_T2_series) * ureg("molT2/s"),
            sim_input=self.sim_input,
            dt=dt * ureg("s"),
            dx=dx * ureg("m"),
            exports={
                name: Export(
                    kind=export_kinds[name],
                    data=np.array(
                        export_profiles[name]
                        if export_kinds[name] == ExportKind.PROFILE
                        else export_series[name]
                        if export_kinds[name] == ExportKind.SERIES
                        else export_static[name]
                    )
                    * ureg(export_units[name]),
                )
                for name in self.exports
            },
            x_export=x_profile * ureg("m"),
        )
        return results
