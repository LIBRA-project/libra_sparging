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

from sparging.config import ureg

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
class SimulationResults:
    times: list
    c_T2_solutions: list
    y_T2_solutions: list
    x_ct: np.ndarray
    x_y: np.ndarray
    inventories_T2_salt: np.ndarray
    source_T2: list
    fluxes_T2: list
    sim_input: SimulationInput

    keys_to_ignore_results = [  # TODO do it the other way: keys_to_include_results
        "c_T2_solutions",
        "y_T2_solutions",
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


def solve(
    input: SimulationInput,
    t_final: pint.Quantity,
    t_irr: list[pint.Quantity],
    t_sparging: list[pint.Quantity],
):
    t_final = t_final.to("seconds").magnitude
    t_irr = t_irr.to("seconds").magnitude
    t_sparging = t_sparging.to("seconds").magnitude
    dt = 0.2 * ureg("hours").to("seconds").magnitude
    # unpack parameters
    tank_height = input.height.to("m").magnitude
    tank_area = input.area.to("m**2").magnitude
    tank_volume = input.volume.to("m**3").magnitude
    a = input.a.to("1/m").magnitude
    h_l = input.h_l.to("m/s").magnitude
    K_s = input.K_s.to("mol/m**3/Pa").magnitude
    P_0 = input.P_bottom.to("Pa").magnitude
    T = input.temperature.to("K").magnitude
    eps_g = input.eps_g.to("dimensionless").magnitude
    E_g = input.E_g.to("m**2/s").magnitude
    D_l = input.D_l.to("m**2/s").magnitude
    u_g0 = input.u_g0.to("m/s").magnitude
    # convert T generation rate to T2 generation rate for the gas phase [mol T2 /m3/s],
    # assuming bred T immediately combines to T2
    source_T2 = input.source_T.to("molT2/s/m**3").magnitude

    eps_l = 1 - eps_g

    # MESH AND FUNCTION SPACES
    mesh = dolfinx.mesh.create_interval(
        MPI.COMM_WORLD, 1000, points=[0, input.height.magnitude]
    )
    fdim = mesh.topology.dim - 1
    cg_el = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(2,))

    V = dolfinx.fem.functionspace(mesh, cg_el)

    u = dolfinx.fem.Function(V)
    u_n = dolfinx.fem.Function(V)
    v_c, v_y = ufl.TestFunctions(V)

    c_T2, y_T2 = ufl.split(u)
    c_T2_n, y_T2_n = ufl.split(u_n)

    vel_x = u_g0  # TODO velocity should vary with hydrostatic pressure
    vel = dolfinx.fem.Constant(mesh, PETSc.ScalarType([vel_x]))

    """ vel = fem_func(U)
        v,interpolate(lambda x: v0 + 2*x[0])"""

    h_l_const = dolfinx.fem.Constant(mesh, PETSc.ScalarType(h_l))

    gen_T2 = dolfinx.fem.Constant(
        mesh, PETSc.ScalarType(source_T2)
    )  # generation term [mol T2 /m3/s]

    # VARIATIONAL FORMULATION

    # mass transfer rate
    J_T2 = (
        a * h_l_const * (c_T2 - K_s * (P_0 * y_T2 + EPS))
    )  # TODO pressure shouldn't be a constant (use hydrostatic pressure profile), how to deal with this ? -> use fem.Expression ?

    F = 0  # variational formulation

    # transient terms
    F += eps_l * ((c_T2 - c_T2_n) / dt) * v_c * ufl.dx
    F += eps_g * 1 / (const.R * T) * (P_0 * (y_T2 - y_T2_n) / dt) * v_y * ufl.dx

    # diffusion/dispersion terms #TODO shouldn't use D_l, transport of T in liquid is dominated by dispersive effects due to gas sparging, find dispersion coeff for steady liquid in gas bubbles
    F += eps_l * D_l * ufl.dot(ufl.grad(c_T2), ufl.grad(v_c)) * ufl.dx

    # NOTE remove diffusive term for gas for now for mass balance
    # F += eps_g * E_g * ufl.dot(ufl.grad(P_0 * y_T2), ufl.grad(v_y)) * ufl.dx

    # mass exchange (coupling term)
    F += J_T2 * v_c * ufl.dx - J_T2 * v_y * ufl.dx

    # Generation term in the breeder
    F += -gen_T2 * v_c * ufl.dx

    # advection of gas
    F += 1 / (const.R * T) * ufl.inner(ufl.dot(ufl.grad(P_0 * y_T2), vel), v_y) * ufl.dx

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
    )
    bc2 = dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(mesh, 0.0),
        dolfinx.fem.locate_dofs_topological(V.sub(0), fdim, gas_outlet_facets),
        V.sub(0),
    )

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
        bcs=[bc1, bc2],
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

    times = []
    c_T2_solutions = []
    y_T2_solutions = []
    sources_T2 = []
    fluxes_T2 = []
    inventories_T2_salt = []

    # SOLVE
    t = 0
    while t < t_final:
        if isinstance(t_irr, (int, float)):
            if t >= t_irr:
                gen_T2.value = 0.0
        else:
            if t >= t_irr[0] and t < t_irr[1]:
                gen_T2.value = source_T2
            else:
                gen_T2.value = 0.0
        """ utiliser ufl.conditional TODO"""
        if t_sparging is not None:
            if t >= t_sparging[0] and t < t_sparging[1]:
                h_l_const.value = h_l
            else:
                h_l_const.value = 0.0

        problem.solve()

        # update previous solution
        u_n.x.array[:] = u.x.array[:]

        # post process
        c_T2_post, y_T2_post = u.split()

        c_T2_vals = u.x.array[ct_dofs][ct_sort_coords]
        y_T2_vals = u.x.array[y_dofs][y_sort_coords]

        # store time and solution
        # TODO give units to the results
        times.append(t)
        c_T2_solutions.append(c_T2_vals.copy())
        y_T2_solutions.append(y_T2_vals.copy())
        sources_T2.append(
            gen_T2.value.copy() * tank_volume
        )  # total T generation rate in the tank [mol/s]

        flux_T2 = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(
                tank_area * vel_x * P_0 / (const.R * T) * y_T2_post * ds(2)
            )
        )  # total T flux at the outlet [mol/s]

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

        n = ufl.FacetNormal(mesh)
        flux_T2_inlet = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(-E_g * ufl.inner(ufl.grad(P_0 * y_T2_post), n) * ds(1))
        )  # total T dispersive flux at the inlet [Pa T2 /s/m2]
        flux_T2_inlet *= 1 / (const.R * T) * T2_to_T  # convert to mol T/s/m2
        flux_T2_inlet *= tank_area  # convert to mol T/s

        # fluxes_T2.append(flux_T2 + flux_T2_inlet)
        fluxes_T2.append(flux_T2)

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
        x_ct=x_ct,
        x_y=x_y,
        inventories_T2_salt=inventories_T2_salt,
        source_T2=sources_T2,
        fluxes_T2=fluxes_T2,
        sim_input=input,
    )
    return results
