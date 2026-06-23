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

logger = logging.getLogger(__name__)

hours_to_seconds = 3600
days_to_seconds = 24 * hours_to_seconds
T2_to_T = 2
T_to_T2 = 1 / T2_to_T

EPS = 1e-26
SEPARATOR_KEYWORD = "from"

# log.set_log_level(log.LogLevel.INFO)


@dataclass
class SimulationResults:
    times: np.ndarray[pint.Quantity]
    c_T2_solutions: np.ndarray[pint.Quantity]
    """ line : time step, column : spatial coordinate """
    y_T2_solutions: np.ndarray[pint.Quantity]
    aJ_T2_solutions: np.ndarray[pint.Quantity]
    x_ct: np.ndarray[pint.Quantity]
    x_y: np.ndarray[pint.Quantity]
    inventories_T2_salt: np.ndarray[pint.Quantity]
    sources_T2: np.ndarray[pint.Quantity]
    fluxes_T2: np.ndarray[pint.Quantity]
    dt: pint.Quantity = None
    dx: pint.Quantity = None
    sim_input: SimulationInput = None

    keys_to_ignore_results = [  # TODO do it the other way: keys_to_include_results
        # "c_T2_solutions",
        # "y_T2_solutions",
        # "J_T2_solutions",
        # "x_ct",
        # "x_y",
        # "inventories_T2_salt",
        # "times",
        # "sources_T2",
        # "fluxes_T2",
        "sim_input",
        "dt",
        "dx",
    ]

    def to_yaml(self, output_path: Path):
        sim_dict = self.sim_input.__dict__.copy()
        helpers.setup_yaml()

        # structure the output
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
        # remove c_T2_solutions and y_T2_solutions from results to avoid dumping large arrays in yaml, they can be saved separately if needed
        for key in self.keys_to_ignore_results:
            output["results"].pop(key, None)

        with open(output_path, "w") as f:
            yaml.dump(output, f, sort_keys=False)

    def serialize_output(self):
        sim_dict = self.sim_input.__dict__.copy()

        # structure the output
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

        # remove objects incompatible with serialization
        for key in self.keys_to_ignore_results:
            output["results"].pop(key, None)

        for key, value in output.items():
            if isinstance(value, np.ndarray):
                # convert numpy arrays to lists for JSON serialization
                output[key] = value.tolist()
                logger.verbose(
                    "found list in results, converting to list for JSON serialization"
                )
            if isinstance(value, pint.Quantity):
                # convert pint.Quantity to string for JSON serialization
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

    def to_json(self, output_path: Path):
        output = self.serialize_output()

        with open(output_path, "w") as f:
            json.dump(output, f, indent=3)

    def to_pickle(self, output_path: Path):
        import pickle

        output = self.serialize_output()

        with open(output_path, "wb") as f:
            pickle.dump(output, f)

    def profiles_to_csv(self, output_directory: Path):
        """Save c_T2 and y_T2 profiles at all time steps as CSV files."""
        times_s = np.array([t.to("seconds").magnitude for t in self.times])

        col_names = [f"t={t:.1f}s" for t in times_s]

        df_c_T2 = pd.DataFrame(
            np.column_stack([self.x_ct.magnitude, self.c_T2_solutions.magnitude.T]),
            columns=["x_metres", *col_names],
        )
        df_y_T2 = pd.DataFrame(
            np.column_stack([self.x_y.magnitude, self.y_T2_solutions.magnitude.T]),
            columns=["x_metres", *col_names],
        )

        df_aJ_T2 = pd.DataFrame(
            np.column_stack([self.x_y.magnitude, self.aJ_T2_solutions.magnitude.T]),
            columns=["x_metres", *col_names],
        )

        df_c_T2.to_csv(output_directory / "c_T2.csv", index=False, float_format="%.6e")
        df_y_T2.to_csv(output_directory / "y_T2.csv", index=False, float_format="%.6e")
        df_aJ_T2.to_csv(
            output_directory / "aJ_T2.csv", index=False, float_format="%.6e"
        )

    def profiles_to_cdf(self, output_directory: Path):
        """Export profiles to a self-describing NetCDF file, preserving units."""
        import xarray as xr

        # --- Helper: split a pint Quantity (scalar or array) into (magnitude, unit_str)
        def split(q, target_unit=None):
            if target_unit is not None:
                q = q.to(target_unit)
            return np.asarray(
                q.magnitude
            ), f"{q.units:~P}"  # "~" → short symbol, e.g. "mol/m³" -> "mol / m ** 3"

        # --- Coordinates ---
        t_mag, t_unit = split(self.times, "s")
        x_ct_mag, x_ct_unit = split(self.x_ct, "m")
        x_y_mag, x_y_unit = split(self.x_y, "m")

        # --- Data variables (note: c_T2_solutions is a 2D pint Quantity, shape (n_t, n_x)) ---
        aJ_T2_mag, aJ_T2_unit = split(self.aJ_T2_solutions, "molT2/m^3/s")
        c_mag, c_unit = split(self.c_T2_solutions, "molT2/m^3")
        y_mag, y_unit = split(self.y_T2_solutions)  # dimensionless → keep native

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

    @classmethod
    def deserialize_output(cls, data: dict) -> SimulationResults:
        # only read the "results" key
        # for each key in results, if the dict have "value" and "units" keys, convert it back to pint.Quantity
        results = data.get("results", {})
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
    profile_pressure_hydrostatic: bool = True
    dispersion_on: bool = True

    def hydrostatic_pressure(self, x: pint.Quantity) -> pint.Quantity:
        """returns the hydrostatic pressure at a given height x in the tank given P_bottom"""
        rho = self.sim_input.rho_l
        g = const_g
        return (self.sim_input.P_bottom + rho * g * x).to("Pa")

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
        a = self.sim_input.a.to("1/m").magnitude
        h_l = self.sim_input.h_l.to("m/s").magnitude
        K_s = self.sim_input.K_s.to("mol/m**3/Pa").magnitude  # convert to molT2 ?
        P_0 = self.sim_input.P_bottom.to("Pa").magnitude
        T = self.sim_input.temperature.to("K").magnitude
        eps_g = self.sim_input.eps_g.to("dimensionless").magnitude
        E_g = self.sim_input.E_g.to("m**2/s").magnitude
        E_l = self.sim_input.E_l.to("m**2/s").magnitude
        u_g0 = self.sim_input.u_g0.to("m/s").magnitude
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
        eps_l = 1 - eps_g

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

        c_T2, y_T2 = ufl.split(u)
        c_T2_n, y_T2_n = ufl.split(u_n)

        vel_x = u_g0  # TODO velocity should vary with hydrostatic pressure
        vel = dolfinx.fem.Constant(mesh, PETSc.ScalarType([vel_x]))

        h_l_const = dolfinx.fem.Constant(mesh, PETSc.ScalarType(h_l))

        gen_T2_ave = dolfinx.fem.Constant(
            mesh, Q_T2 / tank_volume * self.sim_input.signal_irr(0 * ureg.s)
        )  # magnitude of the generation term

        if (
            self.sim_input.profile_source_T is not None
        ):  # spatially varying profile is provided
            arbitrary_profile = dolfinx.fem.Function(V_profile)
            arbitrary_profile.interpolate(
                lambda x: x[0] * 0 + self.sim_input.profile_source_T(x[0] / tank_height)
            )
            profile_integral = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(
                    arbitrary_profile * 1 / tank_height * ufl.dx  # TODO
                )  # dimensionless integral
            )
            normalized_profile = (
                arbitrary_profile / profile_integral
            )  # normalize profile so that its integral over the dimensionless height is 1
        else:  # homogeneous generation
            normalized_profile = dolfinx.fem.Constant(mesh, PETSc.ScalarType(1.0))

        gen_T2 = gen_T2_ave * normalized_profile

        P_prof = dolfinx.fem.Function(V_profile)
        if self.profile_pressure_hydrostatic:
            P_prof.interpolate(
                lambda x: self.hydrostatic_pressure(x[0] * ureg.m).magnitude
            )
        else:
            P_prof.interpolate(lambda x: x[0] * 0 + P_0)

        P = P_prof

        # VARIATIONAL FORMULATION

        # mass transfer rate
        aJ_T2 = a * h_l_const * (c_T2 - K_s * (P * y_T2 + EPS))

        F = 0  # variational formulation

        # transient terms
        F += eps_l * ((c_T2 - c_T2_n) / dt) * v_c * ufl.dx
        F += eps_g * 1 / (const.R * T) * (P * (y_T2 - y_T2_n) / dt) * v_y * ufl.dx

        # dispersive terms
        if self.dispersion_on is True:
            F += eps_l * E_l * ufl.dot(ufl.grad(c_T2), ufl.grad(v_c)) * ufl.dx
            F += (
                eps_g
                * E_g
                * 1
                / (const.R * T)
                * ufl.dot(ufl.grad(P * y_T2), ufl.grad(v_y))
                * ufl.dx
            )

        # mass exchange (coupling term)
        F += aJ_T2 * v_c * ufl.dx - aJ_T2 * v_y * ufl.dx

        # Generation term in the breeder
        F += -gen_T2 * v_c * ufl.dx

        # advection of gas
        F += (
            1
            / (const.R * T)
            * ufl.inner(ufl.dot(ufl.grad(eps_g * P * y_T2), vel), v_y)
            * ufl.dx
        )

        # BOUNDARY CONDITIONS
        gas_inlet_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.isclose(x[0], 0.0)
        )
        gas_outlet_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, lambda x: np.isclose(x[0], tank_height)
        )
        bc1 = dolfinx.fem.dirichletbc(
            dolfinx.fem.Constant(mesh, 0.0),
            dolfinx.fem.locate_dofs_topological(V.sub(1), fdim, gas_inlet_facets),
            V.sub(1),
        )  # y_T2 = 0 at gas inlet

        # Custom measure
        all_facets = np.concatenate((gas_inlet_facets, gas_outlet_facets))
        all_tags = np.concatenate(
            (np.full_like(gas_inlet_facets, 1), np.full_like(gas_outlet_facets, 2))
        )
        facet_markers = dolfinx.mesh.meshtags(mesh, fdim, all_facets, all_tags)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_markers)

        # set up problem
        problem = NonlinearProblem(
            F,
            u,
            bcs=[bc1],  # Neumann BCs on c_T2 at inlet and outlet are naturally enforced
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

        # NOTE maybe we could take this function out and it would take a SimulationResults object as input + u + other things...
        def post_process(t):
            """
            Post-process the solution at time t.
            Extract solution profiles, compute fluxes and inventories, and store them in lists for later analysis.
            """
            c_T2_post, y_T2_post = u.split()

            c_T2_vals = u.x.array[ct_dofs][ct_sort_coords]
            y_T2_vals = u.x.array[y_dofs][y_sort_coords]
            aJ_T2_func.interpolate(aJ_T2_expr)
            aJ_T2_vals = aJ_T2_func.x.array[profile_dofs][ct_sort_coords]
            times.append(t)
            c_T2_solutions.append(c_T2_vals.copy())
            y_T2_solutions.append(y_T2_vals.copy())
            aJ_T2_solutions.append(aJ_T2_vals.copy())
            sources_T2.append(
                Q_T2 * self.sim_input.signal_irr(t * ureg.s)
            )  # total T generation rate in the tank [mol/s] TODO useless: signal_irr is already given

            n = ufl.FacetNormal(mesh)

            # flux_T2 = dolfinx.fem.assemble_scalar(
            #     dolfinx.fem.form(
            #         eps_g * vel_x * P / (const.R * T) * y_T2_post * tank_area * ds(2)
            #     )
            # )  # total T flux at the outlet [mol/s]
            flux_T2 = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(
                    eps_g * vel_x * P / (const.R * T) * y_T2_post * tank_area * ds(2)
                )
            )  # TODO replace with integral of J over volume
            flux_T2_2 = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(tank_area * aJ_T2_func * ufl.dx)
            )
            # flux_T_inlet = dolfinx.fem.assemble_scalar(
            #     dolfinx.fem.form(
            #         tank_area
            #         * E_g
            #         * P_0
            #         / (const.R * T)
            #         * y_T2_post.dx(0)
            #         * T2_to_T
            #         * ds(1)
            #     )
            # )  # total T dispersive flux at the inlet [mol/s]

            flux_T2_inlet = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(
                    -eps_g * E_g * ufl.inner(ufl.grad(P * y_T2_post), n) * ds(1)
                )
            )  # total T dispersive flux at the inlet [Pa T2 /s/m2]
            flux_T2_inlet *= 1 / (const.R * T)  # mol T2/s/m2
            flux_T2_inlet *= tank_area  # convert to molT2/s

            # fluxes_T2.append(flux_T2 + flux_T2_inlet)
            fluxes_T2.append(flux_T2)

            inventory_T2_salt = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(c_T2_post * ufl.dx)
            )
            inventory_T2_salt *= tank_area  # get total amount of T2 in [mol]
            inventories_T2_salt.append(inventory_T2_salt)

        # TODO Initialize results storage with zeros -> there's an inconsistency: times[0] = 0 but inventories[0] != 0 -> screws comparison with analytical solutions
        t = 0
        times = []
        c_T2_solutions = []
        y_T2_solutions = []
        aJ_T2_solutions = []
        sources_T2 = []
        fluxes_T2 = []
        inventories_T2_salt = []
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

            problem.solve()

            # update previous solution
            u_n.x.array[:] = u.x.array[:]

            post_process(t)

        # TODO reattach units using wrapping
        # https://pint.readthedocs.io/en/stable/advanced/performance.html#a-safer-method-wrapping
        results = SimulationResults(
            times=np.array(times) * ureg("s"),
            c_T2_solutions=np.array(c_T2_solutions) * ureg("molT2/m^3"),
            y_T2_solutions=np.array(y_T2_solutions) * ureg("dimensionless"),
            aJ_T2_solutions=np.array(aJ_T2_solutions) * ureg("molT2/m^3/s"),
            x_ct=x_ct * ureg("m"),
            x_y=x_y * ureg("m"),
            inventories_T2_salt=np.array(inventories_T2_salt) * ureg("molT2"),
            sources_T2=np.array(sources_T2) * ureg("molT2/s"),
            fluxes_T2=np.array(fluxes_T2) * ureg("molT2/s"),
            sim_input=self.sim_input,
            dt=dt * ureg("s"),
            dx=dx * ureg("m"),
        )
        return results
