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

import warnings
from datetime import datetime

# from dolfinx import log
import yaml
import helpers
import json

from pint import UnitRegistry
import inspect

ureg = UnitRegistry()
ureg.formatter.default_format = "3e~D"
ureg.define("triton = 1 * particle = T")
ureg.define(f"molT = {const.N_A} * triton")
ureg.define(f"molT2 = 2 * {const.N_A} * triton")
ureg.define("neutron = 1 * particle = n")
ureg.define("sccm = 7.44e-7 mol/s")

const_R = const.R * ureg("J/K/mol")  # ideal gas constant
const_g = const.g * ureg("m/s**2")  # gravitational acceleration


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

    keys_to_ignore_output = [
        "c_T2_solutions",
        "y_T2_solutions",
        "x_ct",
        "x_y",
        "inventories_T2_salt",
        "times",
        "source_T2",
        "fluxes_T2",
    ]

    def to_yaml(self, output_path: str, sim_dict: dict):

        helpers.setup_yaml_numpy()

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
        for key in self.keys_to_ignore_output:
            output["results"].pop(key, None)

        with open(output_path, "w") as f:
            yaml.dump(output, f, sort_keys=False)

    def to_json(self, output_path: str, sim_dict: dict):
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
        for key in self.keys_to_ignore_output:
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


hours_to_seconds = 3600
days_to_seconds = 24 * hours_to_seconds
T2_to_T = 2
T_to_T2 = 1 / T2_to_T

EPS = 1e-26
U_G0_DEFAULT = 0.25  # m/s, typical bubble velocity according to Chavez 2021
VERBOSE = False
SEPARATOR_KEYWORD = "from"

# log.set_log_level(log.LogLevel.INFO)


def get_d_b(flow_g_vol: float, nozzle_diameter: float, nb_nozzle: int) -> float:
    """
    mean bubble diameter [m], Kanai 2017 (reported by Evans 2026)
    """
    nozzle_flow = flow_g_vol / nb_nozzle  # volumetric flow per nozzle [m3/s]
    if nozzle_flow < ureg("3 cm**3/s") or nozzle_flow > ureg("10 cm**3/s"):
        warnings.warn(
            f"nozzle flow {nozzle_flow.to('cm**3/s')} is out of the validated range for the Kanai 2017 correlation (3-10 cm3/s)"
        )
    return ureg.Quantity(
        0.54
        * (
            nozzle_flow.to("cm**3/s").magnitude
            * np.sqrt(nozzle_diameter.to("cm").magnitude / 2)
        )
        ** 0.289,
        "cm",
    )


def get_Re(input: SimulationInput) -> float:
    try:
        u = input.u_g0
        Re_old = input.Re
    except AttributeError:
        if VERBOSE:
            print("in get_Re, use default bubble velocity")
        u = U_G0_DEFAULT * ureg("m/s")
        Re_old = 0

    Re = input.rho_l * u * input.d_b / input.mu_l
    if (
        Re_old != 0 and np.abs(np.log10(Re / Re_old)) > 1
    ):  # check if Reynolds number changed by more than an order of magnitude
        warnings.warn(
            f"Re number changed significantly from {Re_old:.2e} to {Re:.2e}, bubble velocity u = {u} might be off. Check assumed bubble velocity in initial Reynolds number calculation"
        )
    return Re.to("dimensionless")


def get_u_g0(input: SimulationInput) -> float:  # TODO move inside class ?
    """
    bubble initial velocity [m/s], correlation for terminal velocity from Clift 1978
    """
    H = (4 / 3 * input.Eo.magnitude * input.Mo.magnitude**-0.149) * (
        input.mu_l.magnitude / 0.0009
    ) ** -0.14
    if H > 59.3:
        J = 3.42 * H**0.441
    elif H > 2:
        J = 0.94 * H**0.757
    else:
        J = input.Re * input.Mo**0.149 + 0.857
        warnings.warn(
            f"Warning: low Reynolds number {input.Re:.2e}, bubble size d_b = {input.d_b} m might be too small."
            f"Clift correlation will use default value for bubble velocity u_g0 = {U_G0_DEFAULT} m/s"
        )
    u_g0 = input.mu_l / (input.rho_l * input.d_b) * input.Mo**-0.149 * (J - 0.857)
    if u_g0 > ureg("1 m/s") or u_g0 < ureg("0.1 m/s"):
        warnings.warn(f"Warning: bubble velocity {u_g0} is out of the typical range")

    return u_g0


def get_eps_g(input: SimulationInput) -> float:
    """computes gas void fraction from ideal gas law and Young-Laplace pressure in the bubbles (neglecting hydrostatic pressure variation)"""
    eps_g = (
        const_R
        * input.T
        / (input.P_0 + 4 * input.sigma_l / input.d_b)
        * input.flow_g
        / (np.pi * (input.tank_diameter / 2) ** 2 * input.u_g0)
    )
    if eps_g > 1 or eps_g < 0:
        warnings.warn(f"Warning: unphysical gas fraction: {eps_g}")
    elif eps_g > 0.1:
        warnings.warn(
            f"Warning: high gas fraction: {eps_g}, models assumptions may not hold"
        )
    return eps_g


def get_h_higbie(D_l: float, u_g: float, d_b: float) -> float:
    """mass transfer coefficient [m/s] for tritium in liquid FLiBe using Higbie penetration model"""
    h_l = (
        (D_l * u_g) / (ufl.pi * d_b)
    ) ** 0.5  # mass transport coefficient Higbie penetration model
    return h_l


def get_h_malara(D_l: float, d_b: float) -> float:
    """
    mass transfer coefficient [m/s] for tritium in liquid FLiBe using Malara 1995 correlation
    (used for inert gas stripping from breeder droplets, may not be valid here)
    """
    h_l = 2 * np.pi**2 * D_l / (3 * d_b)
    return h_l


def get_h_briggs(Re: float, Sc: float, D_l: float, d_b: float) -> float:
    """mass transfer coefficient [m/s] for tritium in liquid FLiBe using Briggs 1970 correlation"""
    Sh = 0.089 * Re**0.69 * Sc**0.33  # Sherwood number
    h_l = Sh * D_l / d_b
    return h_l


def get_E_g(diameter: float, u_g: float) -> float:
    """gas phase axial dispersion coefficient [m2/s], Malara 1995 correlation
    models dispersion of the gas velocity distribution around the mean bubble velocity"""
    E_g = 0.2 * ureg("1/m") * diameter**2 * u_g
    return E_g


correlations_dict = {
    "rho_l": lambda input: ureg.Quantity(
        2245 - 0.424 * input.T.to("celsius").magnitude, "kg/m**3"
    ),  # density of Li2BeF4, Vidrio 2022
    "mu_l": lambda input: ureg.Quantity(
        0.116e-3 * np.exp(3755 / input.T.to("kelvin").magnitude), "Pa*s"
    ),  # kinematic viscosity of Li2BeF4, Cantor 1968
    "sigma_l": lambda input: ureg.Quantity(
        260 - 0.12 * input.T.to("celsius").magnitude, "dyn/cm"
    ).to("N/m"),  # surface tension of Li2BeF4,Cantor 1968
    "D_l": lambda input: ureg.Quantity(
        9.3e-7 * np.exp(-42e3 / (const.R * input.T.to("kelvin").magnitude)), "m**2/s"
    ),  # diffusivity of T in FLiBe, Calderoni 2008
    "K_s": lambda input: ureg.Quantity(
        7.9e-2 * np.exp(-35e3 / (const.R * input.T.to("kelvin").magnitude)),
        "mol/m**3/Pa",
    ),  # solubility of T in FLiBe, Calderoni 2008
    "d_b": lambda input: get_d_b(
        flow_g_vol=input.flow_g_vol,
        nozzle_diameter=input.nozzle_diameter,
        nb_nozzle=input.nb_nozzle,
    ),  # mean bubble diameter, Kanai 2017
    "Eo": lambda input: (const_g * input.drho * input.d_b**2 / input.sigma_l).to(
        "dimensionless"
    ),  # Eotvos number
    "Mo": lambda input: (
        input.drho * const_g * input.mu_l**4 / (input.rho_l**2 * input.sigma_l**3)
    ).to("dimensionless"),  # Morton number
    "Sc": lambda input: (input.nu_l / input.D_l).to("dimensionless"),  # Schmidt number
    "Re": lambda input: get_Re(input),  # Reynolds number
    "u_g0": lambda input: get_u_g0(input),  # initial gas velocity
    "eps_g": lambda input: get_eps_g(input),  # gas void fraction
    "h_l_malara": lambda input: get_h_malara(
        input.D_l, input.d_b
    ),  # mass transfer coefficient with Malara correlation
    "h_l_briggs": lambda input: get_h_briggs(
        input.Re, input.Sc, input.D_l, input.d_b
    ),  # mass transfer coefficient with Briggs correlation
    "h_l_higbie": lambda input: get_h_higbie(
        input.D_l, input.u_g0, input.d_b
    ),  # mass transfer coefficient with Higbie correlation
    "E_g": lambda input: get_E_g(
        input.tank_diameter, input.u_g0
    ),  # gas phase axial dispersion coefficient
    "source_T": lambda input: (
        input.tbr * input.n_gen_rate / (input.tank_volume)
    ),  # tritium generation source term
}


class SimulationInput:
    def _get(self, input_dict: dict, key: str, corr_name: str = None):
        """get a parameter value from input_dict, or compute it from correlation if not specified in input_dict
        - input_dict: dictionary of input values and/or correlation names
        - key: name of the parameter to get in the input_dict
        - corr_name: name of the correlation to use if not specified in input_dict, will be the key itself by default
        """

        if key in input_dict:
            value = input_dict[key]
            input_dict.pop(key)  # to keep track of unused input parameters, if any

            if (
                isinstance(value, str) and value.__contains__(SEPARATOR_KEYWORD)
            ):  # if the value is a string that contains keyword "from", we assume it's a correlation name
                corr_name = value.split(SEPARATOR_KEYWORD + " ")[1]
                try:
                    func = correlations_dict[corr_name]
                    quantity = func(self)
                    self.quantities_dict["computed"][key] = {
                        "quantity": str(quantity),
                        "correlation": corr_name,
                    }
                    return quantity
                except KeyError:
                    raise KeyError(
                        f"Correlation '{corr_name}' not found in correlations dictionary. Available correlations: {list(correlations_dict.keys())}"
                    )
            else:  # we assume the quantity is directly provided
                if VERBOSE:
                    print(f"{key} = {value} \t provided by input")
                quantity = ureg.parse_expression(str(value))
                self.quantities_dict["inputed"][key] = str(quantity)
                return quantity
        else:  # input doesn't specify a value or a correlation name
            while True:
                # compute quantity using correlation, and if needed retrieve quantities it depends on
                try:
                    corr_name = key if corr_name is None else corr_name
                    quantity = correlations_dict[corr_name](self)
                    # compute the quantity using the default correlation function
                    self.quantities_dict["computed"][key] = {
                        "quantity": str(quantity),
                        "correlation": corr_name,
                    }
                    break
                except KeyError:
                    raise KeyError(
                        f"Correlation for '{corr_name}' not found in correlations dictionary. Available correlations: {list(correlations_dict.keys())}"
                    )
                except AttributeError as e:
                    missing_attr = str(e).split("attribute '")[1].split("'")[0]
                    print(f"AttributeError: missing attribute '{missing_attr}'")
                    setattr(
                        self, missing_attr, self._get(input_dict, missing_attr)
                    )  # recursively get missing attributes
            if VERBOSE:
                print(f"{key} = {quantity} \t calculated using default correlation")

            return quantity

    def __init__(self, input_dict: dict):
        input_dict = input_dict.copy()
        self.quantities_dict = {
            "inputed": {},
            "computed": {},
        }  # to keep track of inputed and calculated quantities and the correlation used, for output purposes
        # TODO useless, can use self.__dict__

        # -- System parameters --
        self.tank_height = self._get(input_dict, "tank_height")
        self.tank_diameter = self._get(input_dict, "D")
        self.tank_area = np.pi * (self.tank_diameter / 2) ** 2
        self.tank_volume = (self.tank_area * self.tank_height).to_base_units()
        self.source_T = self._get(input_dict, "source_T").to(
            "molT/s/m**3"
        )  # tritium generation source term
        self.nozzle_diameter = self._get(input_dict, "nozzle_diameter")
        self.nb_nozzle = self._get(input_dict, "nb_nozzle")
        self.P_top = self._get(input_dict, "P_top")
        self.T = self._get(input_dict, "T")  # temperature
        self.flow_g = self._get(input_dict, "flow_g").to(
            "mol/s"
        )  # inlet gas flow rate [mol/s]

        # -- FLiBe and tritium physical properties --
        self.rho_l = self._get(input_dict, "rho_l")
        self.mu_l = self._get(input_dict, "mu_l")
        self.sigma_l = self._get(input_dict, "sigma_l")
        self.nu_l = self.mu_l / self.rho_l
        self.D_l = self._get(input_dict, "D_l")
        self.K_s = self._get(input_dict, "K_s")

        self.P_0 = (self.P_top + self.rho_l * const_g * self.tank_height).to(
            "Pa"
        )  # gas inlet pressure [Pa] = hydrostatic pressure at the bottom of the tank (neglecting gas fraction)
        self.flow_g_vol = (self.flow_g.to("mol/s") * const_R * self.T / self.P_0).to(
            "m**3/s"
        )  # inlet gas volumetric flow rate

        # -- bubbles properties --
        self.d_b = self._get(input_dict, "d_b").to("metre")  # bubble diameter
        he_molar_mass = ureg("4.003e-3 kg/mol")
        self.rho_g = (
            self.P_0 * he_molar_mass / (const_R * self.T)
        ).to_base_units()  # bubbles density, using ideal gas law for He
        self.drho = self.rho_l - self.rho_g  # density difference between liquid and gas

        self.Eo = self._get(input_dict, "Eo")  # Eotvos number
        self.Mo = self._get(input_dict, "Mo")  # Morton number
        self.Sc = self._get(input_dict, "Sc")  # Schmidt number
        self.Re = self._get(input_dict, "Re")  # Reynolds number

        self.u_g0 = self._get(
            input_dict, "u_g0"
        ).to_base_units()  # mean bubble velocity

        self.Re = self._get(
            input_dict, "Re"
        )  # update Reynolds number with the calculated bubble velocity

        self.eps_g = self._get(input_dict, "eps_g").to(
            "dimensionless"
        )  # gas void fraction
        self.eps_l = 1 - self.eps_g
        self.a = (6 * self.eps_g / self.d_b).to("1/m")  # specific interfacial area

        self.h_l = self._get(input_dict, "h_l", corr_name="h_l_briggs").to(
            "m/s"
        )  # mass transfer coefficient

        self.E_g = self._get(input_dict, "E_g").to(
            "m**2/s"
        )  # gas phase axial dispersion coefficient

        if input_dict:
            warnings.warn(f"Unused input parameters:{input_dict}")

        if VERBOSE:
            print(self)

    def __str__(self):
        members = inspect.getmembers(self)
        return "\n".join(
            [
                f"{name}: {value}"
                for name, value in members
                if not name.startswith("_") and not inspect.isfunction(value)
            ]
        )


def solve(
    input: SimulationInput, t_final: float, t_irr: float | list, t_sparging: list = None
):
    dt = 0.2 * ureg("hours").to("seconds").magnitude
    # unpack parameters
    tank_height, a, h_l, K_s, P_0, T, eps_g, eps_l, E_g, D_l = (
        input.tank_height.magnitude,
        input.a.magnitude,
        input.h_l.magnitude,
        input.K_s.magnitude,
        input.P_0.magnitude,
        input.T.magnitude,
        input.eps_g.magnitude,
        input.eps_l.magnitude,
        input.E_g.magnitude,
        input.D_l.magnitude,
    )
    # tank_area = np.pi * (params["D"] / 2) ** 2
    # tank_volume = tank_area * tank_height

    # MESH AND FUNCTION SPACES
    mesh = dolfinx.mesh.create_interval(
        MPI.COMM_WORLD, 1000, points=[0, input.tank_height.magnitude]
    )
    fdim = mesh.topology.dim - 1
    cg_el = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(2,))

    V = dolfinx.fem.functionspace(mesh, cg_el)

    u = dolfinx.fem.Function(V)
    u_n = dolfinx.fem.Function(V)
    v_c, v_y = ufl.TestFunctions(V)

    c_T2, y_T2 = ufl.split(u)
    c_T2_n, y_T2_n = ufl.split(u_n)

    vel_x = input.u_g0.magnitude  # TODO velocity should vary with hydrostatic pressure
    vel = dolfinx.fem.Constant(mesh, PETSc.ScalarType([vel_x]))

    """ vel = fem_func(U)
        v,interpolate(lambda x: v0 + 2*x[0])"""

    h_l_const = dolfinx.fem.Constant(mesh, PETSc.ScalarType(input.h_l.magnitude))

    source_T2 = input.source_T.to(
        "molT2/s/m**3"
    )  # convert T generation rate to T2 generation rate for the gas phase [mol T2 /m3/s], assuming bred T immediately combines to T2
    gen_T2 = dolfinx.fem.Constant(
        mesh, PETSc.ScalarType(source_T2.magnitude)
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
            gen_T2.value.copy() * input.tank_volume.magnitude
        )  # total T generation rate in the tank [mol/s]

        flux_T2 = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(
                input.tank_area.magnitude
                * vel_x
                * P_0
                / (const.R * T)
                * y_T2_post
                * ds(2)
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
        flux_T2_inlet *= input.tank_area.magnitude  # convert to mol T/s

        # fluxes_T2.append(flux_T2 + flux_T2_inlet)
        fluxes_T2.append(flux_T2)

        inventory_T2_salt = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(c_T2_post * ufl.dx)
        )
        inventory_T2_salt *= (
            input.tank_area.magnitude
        )  # get total amount of T2 in [mol]
        inventories_T2_salt.append(inventory_T2_salt)

        # advance time
        t += dt

    inventories_T2_salt = np.array(inventories_T2_salt)

    results = SimulationResults(
        times=times,
        c_T2_solutions=c_T2_solutions,
        y_T2_solutions=y_T2_solutions,
        x_ct=x_ct,
        x_y=x_y,
        inventories_T2_salt=inventories_T2_salt,
        source_T2=sources_T2,
        fluxes_T2=fluxes_T2,
    )
    return results
