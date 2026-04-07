from mpi4py import MPI
import dolfinx
import basix
import ufl
import numpy as np
import scipy.constants as const
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc
from dataclasses import dataclass

import warnings
from datetime import datetime

# from dolfinx import log
import yaml
import sparging.helpers as helpers
import json
import pint
import inspect
import sparging.correlations as c

from sparging.config import ureg, const_R, const_g

from sparging.inputs import SimulationInput

hours_to_seconds = 3600
days_to_seconds = 24 * hours_to_seconds
T2_to_T = 2
T_to_T2 = 1 / T2_to_T

EPS = 1e-26
SEPARATOR_KEYWORD = "from"

# log.set_log_level(log.LogLevel.INFO)


class SimulationInputBak:
    P_0: pint.Quantity
    eps_l: pint.Quantity
    eps_g: pint.Quantity
    E_g: pint.Quantity
    D_l: pint.Quantity
    h_l: pint.Quantity
    K_s: pint.Quantity
    tank_height: pint.Quantity
    tank_diameter: pint.Quantity
    nozzle_diameter: pint.Quantity
    nb_nozzle: int
    T: pint.Quantity
    flow_g: pint.Quantity
    d_b: pint.Quantity
    Eo: pint.Quantity
    Mo: pint.Quantity
    Sc: pint.Quantity
    Re: pint.Quantity

    def __init__(self, input_dict: dict):
        self.input_dict = input_dict.copy()
        self.quantities_dict = {
            "inputed": {},
            "computed": {},
        }  # to keep track of inputed and calculated quantities and the correlation used, for output purposes
        # TODO useless, can use self.__dict__

        # -- System parameters --
        self.tank_height = self._get("tank_height").to("metre")
        self.tank_diameter = self._get("tank_diameter").to("metre")
        self.tank_area = np.pi * (self.tank_diameter / 2) ** 2
        self.tank_volume = self.tank_area * self.tank_height
        self.source_T = self._get("source_T").to(
            "molT/s/m**3"
        )  # tritium generation source term
        self.nozzle_diameter = self._get("nozzle_diameter").to("millimetre")
        self.nb_nozzle = self._get("nb_nozzle")
        self.P_top = self._get("P_top").to("bar")
        self.T = self._get("T").to("kelvin")  # temperature
        self.flow_g = self._get("flow_g").to("mol/s")  # inlet gas flow rate [mol/s]

        # -- FLiBe and tritium physical properties --
        self.rho_l = self._get("rho_l").to("kg/m**3")
        self.mu_l = self._get("mu_l").to("Pa*s")
        self.sigma_l = self._get("sigma_l").to("N/m")
        self.nu_l = self.mu_l / self.rho_l
        self.D_l = self._get("D_l").to("m**2/s")
        self.K_s = self._get("K_s").to("mol/m**3/Pa")
        self.P_0 = (self.P_top + self.rho_l * const_g * self.tank_height).to("bar")
        # gas inlet pressure [Pa] = hydrostatic pressure at the bottom of the tank (neglecting gas fraction)
        self.flow_g_vol = (self.flow_g.to("mol/s") * const_R * self.T / self.P_0).to(
            "m**3/s"
        )  # inlet gas volumetric flow rate

        # -- bubbles properties --
        self.d_b = self._get("d_b").to("millimetre")  # bubble diameter
        he_molar_mass = ureg("4.003e-3 kg/mol")
        self.rho_g = (self.P_0 * he_molar_mass / (const_R * self.T)).to(
            "kg/m**3"
        )  # bubbles density, using ideal gas law for He
        self.drho = self.rho_l - self.rho_g  # density difference between liquid and gas

        self.Eo = self._get("Eo")  # Eotvos number
        self.Mo = self._get("Mo")  # Morton number
        self.Sc = self._get("Sc")  # Schmidt number
        self.Re = self._get("Re")  # Reynolds number

        self.u_g0 = self._get("u_g0").to("m/s")  # mean bubble velocity

        self.Re = self._get("Re").to(
            "dimensionless"
        )  # update Reynolds number with the calculated bubble velocity

        self.eps_g = self._get("eps_g").to("dimensionless")  # gas void fraction
        self.eps_l = 1 - self.eps_g
        self.a = (6 * self.eps_g / self.d_b).to("1/m")  # specific interfacial area

        self.h_l = self._get("h_l", corr_name="h_l_briggs").to(
            "m/s"
        )  # mass transfer coefficient

        self.E_g = self._get("E_g").to(
            "m**2/s"
        )  # gas phase axial dispersion coefficient

        if input_dict:
            warnings.warn(f"Unused input parameters:{input_dict}")

        if VERBOSE:
            print(self)

    def _get(
        self,
        key: str,
        corr_name: str = None,
    ):
        """get a parameter value from input_dict, or compute it from correlation if not specified in input_dict
        - key: name of the parameter to get in the input_dict
        - corr_name: name of the correlation to use if not specified in input_dict, will be the key itself by default
        """
        # try keys (several names possible in input dictionary, for retrocompatibility)
        if key in self.input_dict.keys():
            print(key, f"found {key} in input_dict")
            quantity = get_quantity_or_correlation(self.input_dict, key)
            if callable(quantity):
                # if it's a correlation function, compute the quantity using the correlation
                return quantity(self)
            return quantity

        else:
            quantity = self.get_quantity_from_default_correlation(key, corr_name)

        if VERBOSE:
            print(f"{key} = {quantity} \t calculated using default correlation")

        return quantity

    # TODO simplify this
    def get_quantity_from_default_correlation(self, key, corr_name=None):
        # nothing found in input_dict -> use default correlation
        while True:
            # default corr_name is the name of the key itself
            corr_name = key if corr_name is None else corr_name
            # compute quantity using correlation, and if needed retrieve quantities it depends on
            if corr_name not in c.correlations_dict.keys():
                raise KeyError(
                    f"Correlation for '{key}' not found in correlations dictionary. Missing a required input or wrongcorrelation name. Available correlations: {list(c.correlations_dict.keys())}"
                )
            try:
                # compute the quantity using the default correlation function
                quantity = c.correlations_dict[corr_name](self)
                break
            except AttributeError as e:
                missing_attr = str(e).split("attribute '")[1].split("'")[0]
                print(f"AttributeError: missing attribute '{missing_attr}'")
                setattr(
                    self, missing_attr, self._get(missing_attr)
                )  # recursively get missing attributes

        return quantity

    def __str__(self):
        members = inspect.getmembers(self)
        return "\n".join(
            [
                f"{name}: {value}"
                for name, value in members
                if not name.startswith("_") and not inspect.isfunction(value)
            ]
        )


def find_correlation_from_library(corr_name) -> callable:
    if corr_name not in c.correlations_dict:
        raise KeyError(
            f"Correlation '{corr_name}' not found in correlations dictionary. Available correlations: {list(c.correlations_dict.keys())}"
        )

    func = c.correlations_dict[corr_name]
    assert callable(func), f"Correlation '{corr_name}' is not a function"
    return func


def get_correlation_from_string(s) -> callable | None:
    """
    If the input is a string that contains the keyword `SEPARATOR_KEYWORD`, we assume it's a correlation name
    and return the corresponding function from the correlations library, otherwise return None
    """
    # if the value is a string that contains keyword "from", we assume it's a correlation name
    if isinstance(s, str) and SEPARATOR_KEYWORD in s:
        return s.split(SEPARATOR_KEYWORD + " ")[1]
    return None


def get_quantity_or_correlation(input_dict, key) -> callable | pint.Quantity:
    value = input_dict[key]
    input_dict.pop(key)  # to keep track of unused input parameters, if any

    corr_name = get_correlation_from_string(value)
    if corr_name:
        corr_func = find_correlation_from_library(corr_name)
        return corr_func

    else:
        # The quantity is directly provided as a quantity
        # TODO implement logger
        if VERBOSE:
            print(f"{key} = {value} \t provided by input")
        quantity = ureg.parse_expression(str(value))
        return quantity


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

    # FIXME
    namespace = {
        "ramp": lambda s, e: helpers.string_to_ramp(times, s, e),
        "step": lambda s: helpers.string_to_step(times, s),
    }

    def to_yaml(self, output_path: str):
        sim_dict = self.sim_input.__dict__.copy()
        helpers.setup_yaml()

        # structure the output
        output = {
            "metadata": {
                "git_commit": helpers.get_git_hash(),
                "date": datetime.now().isoformat(),
            },
        }
        sim_dict.pop(
            "quantities_dict"
        )  # remove quantities_dict from input for cleaner output

        output["input"] = {}
        for key, value in sim_dict.items():
            output["input"][key] = str(value)

        output["results"] = self.__dict__.copy()
        # remove c_T2_solutions and y_T2_solutions from results to avoid dumping large arrays in yaml, they can be saved separately if needed
        for key in self.keys_to_ignore_results:
            output["results"].pop(key, None)

        with open(output_path, "w") as f:
            yaml.dump(output, f, sort_keys=False)

    def to_json(self, output_path: str):
        sim_dict = self.sim_input.quantities_dict.copy()

        # structure the output
        output = {
            "metadata": {
                "git_commit": helpers.get_git_hash(),
                "date": datetime.now().isoformat(),
            },
        }
        if sim_dict.get("inputed"):
            output["input parameters"] = sim_dict["inputed"]
        if sim_dict.get("computed"):
            output["calculated properties"] = sim_dict["computed"]
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

    def profiles_to_csv(self, output_path: str):
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

        df_c_T2.to_csv(output_path + "_c_T2.csv", index=False)
        df_y_T2.to_csv(output_path + "_y_T2.csv", index=False)


def solve(input: SimulationInput, t_final: float, t_irr, t_sparging):
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

    # tank_area = np.pi * (params["D"] / 2) ** 2
    # tank_volume = tank_area * tank_height

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
