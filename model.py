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
from dolfinx import log
from main import ureg
import yaml
import helpers


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
    ]

    def to_yaml(self, output_path: str, inputs: dict, properties: dict):

        helpers.setup_yaml_numpy()

        # structure the output
        output = {
            "metadata": {
                "git_commit": helpers.get_git_hash(),
                "date": datetime.now().isoformat(),
            },
        }
        if inputs:
            output["input parameters"] = inputs
        if properties:
            output["calculated properties"] = properties
        output["results"] = self.__dict__.copy()
        # remove c_T2_solutions and y_T2_solutions from results to avoid dumping large arrays in yaml, they can be saved separately if needed
        for key in self.keys_to_ignore_output:
            output["results"].pop(key, None)

        with open(output_path, "w") as f:
            yaml.dump(output, f, sort_keys=False)

    def to_json(self, output_path: str, inputs: dict, properties: dict):
        import json

        # convert numpy arrays to lists for JSON serialization
        # structure the output
        output = {
            "metadata": {
                "git_commit": helpers.get_git_hash(),
                "date": datetime.now().isoformat(),
            },
        }
        if inputs:
            output["input parameters"] = inputs
        if properties:
            output["calculated properties"] = properties
        output["results"] = self.__dict__.copy()

        for key in self.keys_to_ignore_output:
            output["results"].pop(key, None)

        for key, value in output.items():
            if isinstance(value, np.ndarray):
                output[key] = value.tolist()
                print(
                    "found list in results, converting to list for JSON serialization"
                )

        with open(output_path, "w") as f:
            json.dump(output, f, indent=3)

    def profiles_to_csv(self, output_path: str):
        import pandas as pd

        # save c_T2 and y_T2 profiles at the final time step to csv
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


m3_to_cm3 = 1e6
cm3_to_m3 = 1e-6
m_to_cm = 1e2
cm_to_m = 1e-2
dynespercm_to_newtonpermeter = 1e-3
hours_to_seconds = 3600
days_to_seconds = 24 * hours_to_seconds
T2_to_T = 2
T_to_T2 = 1 / T2_to_T

EPS = 1e-26
U_G0_DEFAULT = 0.25  # m/s, typical bubble velocity according to Chavez 2021
VERBOSE = False
# log.set_log_level(log.LogLevel.INFO)


def get_rho_l(T: float) -> float:
    """density of FLiBe [kg/m3], Vidrio 2022"""
    return 2245 - 0.424 * (const.convert_temperature(T, "kelvin", "celsius"))


def get_mu_l(T: float) -> float:
    """dynamic viscosity of FLiBe [Pa.s], Cantor 1968"""
    return 0.116e-3 * np.exp(3755 / T)


def get_sigma_l(T: float) -> float:
    """surface tension of FLiBe [N/m], Cantor 1968"""
    return (
        260 - 0.12 * (const.convert_temperature(T, "kelvin", "celsius"))
    ) * dynespercm_to_newtonpermeter


def get_D_l(T: float) -> float:  # TODO
    """diffusivity of T in FLiBe [m2/s], Calderoni 2008"""
    return 9.3e-7 * np.exp(-42e3 / (const.R * T))


def get_K_s(T: float) -> float:
    """solubility of T in FLiBe [mol/m3/Pa], Calderoni 2008"""
    return 7.9e-2 * np.exp(-35e3 / (const.R * T))


def get_d_b(flow_g_vol: float, nozzle_diameter: float, nb_nozzle: int) -> float:
    """
    mean bubble diameter [m], Kanai 2017 (reported by Evans 2026)
    """
    nozzle_flow = flow_g_vol / nb_nozzle  # volumetric flow per nozzle [m3/s]
    if nozzle_flow < 3 * cm3_to_m3 or nozzle_flow > 10 * cm3_to_m3:
        warnings.warn(
            f"nozzle flow {nozzle_flow * m3_to_cm3:.2e} cm3/s is out of the validated range for the Kanai 2017 correlation (3-10 cm3/s)"
        )
    return (
        0.54
        * (nozzle_flow * m3_to_cm3 * np.sqrt(nozzle_diameter / 2 * m_to_cm)) ** 0.289
        * cm_to_m
    )


def get_Re(rho: float, mu: float, u: float, d: float, Re_old: float = 0) -> float:
    """
    Reynolds number
    """
    Re = rho * u * d / mu
    if (
        Re_old != 0 and np.abs(np.log10(Re / Re_old)) > 1
    ):  # check if Reynolds number changed by more than an order of magnitude
        warnings.warnngs.warn(
            f"Re number changed significantly from {Re_old:.2e} to {Re:.2e}, bubble velocity u = {u:.2f} m/s might be off. Check assumed bubble velocity in initial Reynolds number calculation"
        )
    return Re


def get_u_g0(
    Eo: float, Mo: float, mu_l: float, rho_l: float, d_b: float, Re: float
) -> float:
    """
    bubble initial velocity [m/s], correlation for terminal velocity from Clift 1978
    """
    H = 4 / 3 * Eo * Mo**-0.149 * (mu_l / 0.0009) ** -0.14
    if H > 59.3:
        J = 3.42 * H**0.441
    elif H > 2:
        J = 0.94 * H**0.757
    else:
        J = Re * Mo**0.149 + 0.857
        warnings.warn(
            f"Warning: low Reynolds number {Re:.2e}, bubble size d_b = {d_b} m might be too small."
            f"Clift correlation will use default value for bubble velocity u_g0 = {U_G0_DEFAULT} m/s"
        )
    u_g0 = mu_l / (rho_l * d_b) * Mo**-0.149 * (J - 0.857)
    if u_g0 > 1 or u_g0 < 0.1:
        warnings.warn(
            f"Warning: bubble velocity {u_g0:.2f} m/s is out of the typical range"
        )
    return u_g0


def get_eps_g(
    T: float,
    P_0: float,
    flow_g_mol: float,
    D: float,
    d_b: float,
    u_g: float,
    sigma_l: float,
) -> float:
    """computes gas void fraction from ideal gas law and Young-Laplace pressure in the bubbles (neglecting hydrostatic pressure variation)"""
    eps_g = (
        const.R
        * T
        / (P_0 + 4 * sigma_l / d_b)
        * flow_g_mol
        / (np.pi * (D / 2) ** 2 * u_g)
    )
    if eps_g > 1 or eps_g < 0:
        warnings.warn(f"Warning: unphysical gas fraction: {eps_g:.2f}")
    elif eps_g > 0.1:
        warnings.warn(
            f"Warning: high gas fraction: {eps_g:.2f}, models assumptions may not hold"
        )
    return eps_g


def get_h_l(
    Re: float, Sc: float, D_l: float, d_b: float, u_g: float, corr_name: str = "briggs"
) -> float:
    """mass transfer coefficient [m/s] for tritium in liquid FLiBe, computed using the specified correlation"""
    match corr_name:
        case "briggs":
            return get_h_briggs(Re, Sc, D_l, d_b)
        case "higbie":
            return get_h_higbie(D_l, u_g, d_b)
        case "malara":
            return get_h_malara(D_l, d_b)


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


def get_source_T(input: dict) -> float:
    """compute tritium generation source term [mol/m3/s] from TBR calculated by openMC for our geometry"""
    tank_volume = np.pi * (input["D"] / 2) ** 2 * input["tank_height"]
    return input["tbr"] * float(input["n_gen_rate"]) / (tank_volume * const.N_A)


def get_flow_g_mol(input: dict) -> float:
    """convert gas flow from sccm to mol/s"""
    T_standard = 0  # °C
    P_standard = 101325  # Pa
    min_to_sec = 60
    return (
        (input["flow_g_sccm"] * cm3_to_m3 / min_to_sec)
        * P_standard
        / (const.R * const.convert_temperature(T_standard, "celsius", "kelvin"))
    )


def get_E_g(diameter: float, u_g: float) -> float:
    """gas phase axial dispersion coefficient [m2/s], Malara 1995 correlation
    models dispersion of the gas velocity distribution around the mean bubble velocity"""
    E_g = 0.2 * diameter**2 * u_g
    return E_g


def _get(
    params: dict,
    key: str,
    func: callable,
    **kwargs,
):
    """
    Get a parameter value from params dictionnary.

    - If key is missing: compute it with default correlation func().
    - If key exists and is numeric: use that value.
    - If key exists and is a string: interpret it as a correlation name.

    Arguments:
    - params: dictionnary of parameters
    - key: name of the parameter to get (e.g. "d_b", "u_g0", "h_l", "eps_g", etc.)
    - func: default correlation to compute the parameter if key is missing (e.g. get_d_b, get_u_g0, get_h_briggs, get_eps_g, etc.)
    - correlations: dictionnary of available correlations to compute the parameter if key is a string
        (e.g. {"kanai": get_d_b, "higbie": get_h_higbie, "malara": get_h_malara, "briggs": get_h_briggs, etc.})
    - kwargs: additional arguments to pass to the correlation function if key is a string
    """
    if VERBOSE:
        print(f"in _get() for {key}, passed parameters: {kwargs}")
    if key in params:
        value = params[key]

        if isinstance(value, str):
            corr_name = value.lower()
            kwargs.update({"corr_name": corr_name})
            value = func(**kwargs)
            if VERBOSE:
                print(
                    f"{key} = {value:.2e} \t calculated using '{corr_name}' correlation as specified in input"
                )
            return value
        else:
            if VERBOSE:
                print(f"{key} = {value:.2e} \t provided by input")
            return value
    else:
        value = func(**kwargs)
        if VERBOSE:
            print(f"{key} = {value:.2e} \t calculated using default correlation")
        return value


def compute_properties(params):
    tank_height = params["tank_height"]
    D = params["D"]
    nozzle_diameter = params["nozzle_diameter"]
    nb_nozzle = params["nb_nozzle"]
    P_top = params["P_top"]
    T = params["T"]

    flow_g_mol = _get(
        params, "flow_g_mol", get_flow_g_mol, input=params
    )  # inlet gas flow rate [mol/s]

    # --- correlations for FLiBe properties ---
    rho_l = _get(params, "rho_l", get_rho_l, T=T)  # density [kg/m3] of Li2BeF4
    mu_l = _get(params, "mu_l", get_mu_l, T=T)  # dynamic viscosity [Pa.s] of Li2BeF4
    nu_l = mu_l / rho_l  # kinematic viscosity [m2/s] of Li2BeF4
    sigma_l = _get(
        params, "sigma_l", get_sigma_l, T=T
    )  # surface tension [N/m] of Li2BeF4

    # --- correlations for tritium in FLiBe ---
    D_l = _get(params, "D_l", get_D_l, T=T)  # diffusivity of T in FLiBe [m2/s]
    K_s = _get(params, "K_s", get_K_s, T=T)  # solubility of T in FLiBe [mol/m3/Pa]
    # - derived parameters -
    P_0 = (
        P_top + rho_l * const.g * tank_height
    )  # gas inlet pressure [Pa] = hydrostatic pressure at the bottom of the tank (neglecting gas fraction)
    flow_g_vol = flow_g_mol * const.R * T / P_0  # inlet gas volumetric flow rate [m3/s]

    # --- correlations for bubble properties ---
    d_b = _get(
        params,
        "d_b",
        get_d_b,
        flow_g_vol=flow_g_vol,
        nozzle_diameter=nozzle_diameter,
        nb_nozzle=nb_nozzle,
    )  # bubble diameter [m]

    he_molar_mass = 4.003e-3  # kg/mol
    rho_g = (
        P_0 * he_molar_mass / (const.R * T)
    )  # bubbles density [kg/m3], using ideal gas law for He
    drho = rho_l - rho_g  # density difference between liquid and gas [kg/m3]

    # --- dimensionless numbers used in correlations ---

    Eo = (drho * const.g * d_b**2) / sigma_l  # Eotvos (Bond) number
    Re = _get(
        params, "Re", get_Re, rho=rho_l, mu=mu_l, u=U_G0_DEFAULT, d=d_b
    )  # initial Reynolds number calculation, assuming typical gas velocity (~0.25 m/s according to Chavez 2021)
    Mo = (drho * const.g * mu_l**4) / (rho_l**2 * sigma_l**3)  # Morton number
    Sc = nu_l / D_l  # Schmidt number

    # --- bubble velocity ---
    u_g0 = _get(
        params, "u_g0", get_u_g0, Eo=Eo, Mo=Mo, mu_l=mu_l, rho_l=rho_l, d_b=d_b, Re=Re
    )  # bubble velocity [m/s], correlation from Clift 1978, reported by Chavez 2021
    Re = _get(
        params, "Re", get_Re, rho=rho_l, mu=mu_l, u=u_g0, d=d_b, Re_old=Re
    )  # update Reynolds number with the calculated bubble velocity

    # --- bubble volume fraction ---
    eps_g = _get(
        params,
        "eps_g",
        get_eps_g,
        T=T,
        P_0=P_0,
        flow_g_mol=flow_g_mol,
        D=D,
        d_b=d_b,
        u_g=u_g0,
        sigma_l=sigma_l,
    )  # gas fraction [-]
    eps_l = 1 - eps_g  # liquid fraction [-]
    a = 6 * eps_g / d_b  # specific interfacial area [m-1]

    # --- mass transfer coefficient ---
    h_l = _get(
        params,
        "h_l",
        get_h_l,
        Re=Re,
        Sc=Sc,
        D_l=D_l,
        d_b=d_b,
        u_g=u_g0,
    )  # T mass transfer coefficient [m/s], using Briggs 1970 correlation as default

    E_g = _get(
        params, "E_g", get_E_g, diameter=D, u_g=u_g0
    )  # gas phase axial dispersion coefficient [m2/s]

    source_T = _get(
        params,
        "source_T",
        get_source_T,
        input=params,
    )  # tritium generation source term [mol/m3/s]

    return {
        "rho_l": rho_l,
        "sigma_l": sigma_l,
        "mu_l": mu_l,
        "nu_l": nu_l,
        "K_s": K_s,
        "u_g0": u_g0,
        "d_b": d_b,
        "rho_g": rho_g,
        "P_0": P_0,
        "flow_g_vol": flow_g_vol,
        "flow_g_mol": flow_g_mol,
        "eps_g": eps_g,
        "eps_l": eps_l,
        "D_l": D_l,
        "E_g": E_g,
        "a": a,
        "h_l": h_l,
        "Re": Re,
        "Eo": Eo,
        "Mo": Mo,
        "Sc": Sc,
        "nozzle_flow": flow_g_vol / nb_nozzle,
        "source_T": source_T,
    }


def solve(params: dict, t_final: float, t_irr: float | list, t_sparging: list = None):
    dt = 0.2 * hours_to_seconds  # s
    # unpack parameters
    tank_height, u_g0, a, h_l, K_s, P_0, T, eps_g, eps_l, E_g, D_l, source_T = (
        params["tank_height"],
        params["u_g0"],
        params["a"],
        params["h_l"],
        params["K_s"],
        params["P_0"],
        params["T"],
        params["eps_g"],
        params["eps_l"],
        params["E_g"],
        params["D_l"],
        params["source_T"],
    )
    tank_area = np.pi * (params["D"] / 2) ** 2
    tank_volume = tank_area * tank_height

    # MESH AND FUNCTION SPACES
    mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, 1000, points=[0, tank_height])
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

    source_T2 = (
        source_T * T_to_T2
    )  # convert T generation rate to T2 generation rate for the gas phase [mol T2 /m3/s], assuming bred T immediately combines to T2
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
