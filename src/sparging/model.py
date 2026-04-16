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

# from dolfinx import log
import yaml
import sparging.helpers as helpers
import json
from pathlib import Path

from sparging.config import ureg, const_g

from sparging.inputs import SimulationInput

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pint

hours_to_seconds = 3600
days_to_seconds = 24 * hours_to_seconds
T2_to_T = 2
T_to_T2 = 1 / T2_to_T

EPS = 1e-26
SEPARATOR_KEYWORD = "from"

# log.set_log_level(log.LogLevel.INFO)


@dataclass
class SimulationResults:  # TODO implement pint in this class # TODO change list to np.array
    times: list
    c_T2_solutions: list
    y_T2_solutions: list
    J_T2_solutions: list
    x_ct: np.ndarray
    x_y: np.ndarray
    inventories_T2_salt: np.ndarray
    sources_T2: list
    fluxes_T2: list
    sim_input: SimulationInput
    dt: int | None = None
    dx: int | None = None

    keys_to_ignore_results = [  # TODO do it the other way: keys_to_include_results
        "c_T2_solutions",
        "y_T2_solutions",
        "J_T2_solutions",
        "x_ct",
        "x_y",
        "inventories_T2_salt",
        "times",
        "source_T2",
        "fluxes_T2",
        "sim_input",
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

    def to_json(self, output_path: Path):
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

        # remove c_T2_solutions and y_T2_solutions from results to avoid dumping large arrays in yaml, they can be saved separately if needed
        for key in self.keys_to_ignore_results:
            output["results"].pop(key, None)

        for key, value in output.items():
            if isinstance(value, np.ndarray):
                # convert numpy arrays to lists for JSON serialization
                output[key] = value.tolist()
                print(
                    "found list in results, converting to list for JSON serialization"
                )

        with open(output_path, "w") as f:
            json.dump(output, f, indent=3)

    def profiles_to_csv(self, output_path: Path):
        """save c_T2 and y_T2 profiles at all time steps to csv files, one for c_T2 and one for y_T2, with columns for each time step"""
        import pandas as pd

        df_c_T2 = pd.DataFrame({"x": self.x_ct})
        df_y_T2 = pd.DataFrame({"x": self.x_y})

        # add one column for each profile
        for i, (c_T2_profile, y_T2_profile) in enumerate(
            zip(self.c_T2_solutions, self.y_T2_solutions)
        ):
            df_c_T2[f"c_T2_t{i}"] = c_T2_profile
            df_y_T2[f"y_T2_t{i}"] = y_T2_profile

        df_c_T2.to_csv(output_path.joinpath("_c_T2.csv"), index=False)
        df_y_T2.to_csv(output_path.joinpath("_y_T2.csv"), index=False)


@dataclass
class Simulation:
    sim_input: SimulationInput
    t_final: pint.Quantity
    signal_irr: callable[pint.Quantity]
    signal_sparging: callable[pint.Quantity]
    profile_source_T: callable[pint.Quantity] | None = None
    """callable = f:[0,1] -> R+, it takes a dimensionless coordinate: (z / height)"""
    profile_pressure_hydrostatic: bool = False
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
        # unpack pint.Quantities
        t_final = self.t_final.to("seconds").magnitude
        tank_height = self.sim_input.height.to("m").magnitude
        tank_area = self.sim_input.area.to("m**2").magnitude
        tank_volume = self.sim_input.volume.to("m**3").magnitude
        a = self.sim_input.a.to("1/m").magnitude
        h_l = self.sim_input.h_l.to("m/s").magnitude
        K_s = self.sim_input.K_s.to("mol/m**3/Pa").magnitude
        P_0 = self.sim_input.P_bottom.to("Pa").magnitude
        T = self.sim_input.temperature.to("K").magnitude
        eps_g = self.sim_input.eps_g.to("dimensionless").magnitude
        E_g = self.sim_input.E_g.to("m**2/s").magnitude
        E_l = self.sim_input.E_l.to("m**2/s").magnitude
        D_l = self.sim_input.D_l.to("m**2/s").magnitude  # not needed (included in h_l)
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
            mesh, Q_T2 / tank_volume * self.signal_irr(0 * ureg.s)
        )  # magnitude of the generation term

        if self.profile_source_T is not None:  # spatially varying profile is provided
            arbitrary_profile = dolfinx.fem.Function(V_profile)
            arbitrary_profile.interpolate(
                lambda x: x[0] * 0 + self.profile_source_T(x[0] / tank_height)
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
        J_T2 = (
            a * h_l_const * (c_T2 - K_s * (P * y_T2 + EPS))
        )  # TODO pressure shouldn't be a constant (use hydrostatic pressure profile), how to deal with this ? -> use fem.Expression ?

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
        F += J_T2 * v_c * ufl.dx - J_T2 * v_y * ufl.dx

        # Generation term in the breeder
        F += -gen_T2 * v_c * ufl.dx

        # advection of gas
        F += (
            1
            / (const.R * T)
            * ufl.inner(ufl.dot(ufl.grad(P * y_T2), vel), v_y)
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
        # bc2 = dolfinx.fem.dirichletbc(
        #     dolfinx.fem.Constant(mesh, 0.0),
        #     dolfinx.fem.locate_dofs_topological(V.sub(0), fdim, gas_outlet_facets),
        #     V.sub(0),
        # )  # c_T2 = 0 at gas outlet

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
            # bcs=[bc1, bc2],
            bcs=[bc1],  # Neumann BCs on c_T2 at inlet and outlet are naturally enforced
            petsc_options_prefix="librasparge",
            # petsc_options={"snes_monitor": None},
        )

        # initialise post processing
        V0_ct, ct_dofs = u.function_space.sub(0).collapse()
        coords = V0_ct.tabulate_dof_coordinates()[:, 0]
        ct_sort_coords = np.argsort(coords)
        x_ct = coords[ct_sort_coords]

        V0_y, y_dofs = u.function_space.sub(1).collapse()
        coords = V0_y.tabulate_dof_coordinates()[:, 0]
        y_sort_coords = np.argsort(coords)
        x_y = coords[y_sort_coords]

        # TODO Initialize results storage with zeros -> there's an inconsistency: times[0] = 0 but inventories[0] != 0 -> screws comparison with analytical solutions
        times = []
        c_T2_solutions = []
        y_T2_solutions = []
        J_T2_solutions = []
        sources_T2 = []
        fluxes_T2 = []
        inventories_T2_salt = []

        # SOLVE
        t = 0
        while t < t_final:
            # update time-dependent terms
            gen_T2_ave.value = Q_T2 / tank_volume * self.signal_irr(t * ureg.s)
            h_l_const.value = h_l * self.signal_sparging(t * ureg.s)

            problem.solve()

            # update previous solution
            u_n.x.array[:] = u.x.array[:]

            # post process
            c_T2_post, y_T2_post = u.split()

            c_T2_vals = u.x.array[ct_dofs][ct_sort_coords]
            y_T2_vals = u.x.array[y_dofs][y_sort_coords]
            J_T2_vals = (
                a * h_l_const.value * (c_T2_vals - K_s * (P.x.array * y_T2_vals + EPS))
            )  # TODO there is a clever way of getting J_T2 for sure

            # store time and solution
            # TODO give units to the results
            times.append(t)
            c_T2_solutions.append(c_T2_vals.copy())
            y_T2_solutions.append(y_T2_vals.copy())
            J_T2_solutions.append(J_T2_vals.copy())
            sources_T2.append(
                Q_T2 * self.signal_irr(t * ureg.s)
            )  # total T generation rate in the tank [mol/s] TODO useless: signal_irr is already given

            n = ufl.FacetNormal(mesh)

            # flux_T2 = dolfinx.fem.assemble_scalar(
            #     dolfinx.fem.form(
            #         eps_g * vel_x * P / (const.R * T) * y_T2_post * tank_area * ds(2)
            #     )
            # )  # total T flux at the outlet [mol/s]
            flux_T2 = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(
                    vel_x * P / (const.R * T) * y_T2_post * tank_area * ds(2)
                )
            )  # TODO replace with integral of J over volume
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

            fluxes_T2.append(flux_T2 + flux_T2_inlet)
            # fluxes_T2.append(flux_T2)

            inventory_T2_salt = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(c_T2_post * ufl.dx)
            )
            inventory_T2_salt *= tank_area  # get total amount of T2 in [mol]
            inventories_T2_salt.append(inventory_T2_salt)

            # advance time
            t += dt

        inventories_T2_salt = np.array(inventories_T2_salt)

        # TODO reattach units using wrapping
        # https://pint.readthedocs.io/en/stable/advanced/performance.html#a-safer-method-wrapping
        results = SimulationResults(
            times=times,
            c_T2_solutions=c_T2_solutions,
            y_T2_solutions=y_T2_solutions,
            J_T2_solutions=J_T2_solutions,
            x_ct=x_ct,
            x_y=x_y,
            inventories_T2_salt=inventories_T2_salt,
            sources_T2=sources_T2,
            fluxes_T2=fluxes_T2,
            sim_input=self.sim_input,
            dt=dt,
            dx=dx,
        )
        return results
