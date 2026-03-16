from mpi4py import MPI
import dolfinx
import basix
import ufl
import numpy as np
import scipy.constants as const
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

import warnings

from dolfinx import log

m3_to_cm3 = 1e6

U_G0_DEFAULT = 0.25  # m/s, typical bubble velocity according to Chavez 2021
# log.set_log_level(log.LogLevel.INFO)

# -- Physical constants --
R = const.R  # J/mol/K
g = const.g  # m/s2


def get_d_b(flow_g_vol: float, nozzle_diameter: float, nb_nozzle: int) -> float:
    """
    mean bubble diameter [m], Kanai 2017 (reported by Evans 2026)
    """
    nozzle_flow = flow_g_vol / nb_nozzle  # volumetric flow per nozzle [m3/s]
    if nozzle_flow < 3e-6 or nozzle_flow > 10e-6:
        warnings.warn(
            f"nozzle flow {nozzle_flow * m3_to_cm3:.2e} cm3/s is out of the validated range for the Kanai 2017 correlation (3-10 cm3/s)"
        )
    return (
        0.54
        * (nozzle_flow * m3_to_cm3 * np.sqrt(nozzle_diameter / 2 * 1e2)) ** 0.289
        * 1e-2
    )


def _get(
    params: dict,
    key: str,
    func: callable,
    correlations: dict | None = None,
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
    if key in params:
        value = params[key]

        if isinstance(value, str):
            corr_name = value.lower()
            value = correlations[corr_name](**kwargs)
            print(
                f"{key} = {value:.2e} \t calculated using '{corr_name}' correlation as specified in input"
            )
            return value
        else:
            print(f"{key} = {value:.2e} \t provided by input")
            return value
    else:
        value = func(**kwargs)
        print(f"{key} = {value:.2e} \t calculated using default correlation")
        return value


def compute_properties(params):
    tank_height = params["tank_height"]
    D = params["D"]
    nozzle_diameter = params["nozzle_diameter"]
    nb_nozzle = params["nb_nozzle"]
    P_top = params["P_top"]
    T = params["T"]
    flow_g_mol = params["flow_g_mol"]

    # --- correlations for FLiBe properties ---
    rho_l = 2245 - 0.424 * (T - 272.15)  # density [kg/m3] of Li2BeF4, Vidrio 2022
    mu_l = 0.116e-3 * np.exp(
        3755 / T
    )  # dynamic viscosity [Pa.s] of Li2BeF4, Cantor 1968
    nu_l = mu_l / rho_l  # kinematic viscosity [m2/s] of Li2BeF4
    sigma_l = (
        260 - 0.12 * (T - 272.15)
    ) * 1e-3  # surface tension of FLiBe [N/m], Cantor 1968

    # --- correlations for tritium in FLiBe ---
    diffusivity = 9.3e-7 * np.exp(
        -42e3 / (R * T)
    )  # diffusivity of T in Li2BeF4 [m2/s], Calderoni 2008
    K_s = 7.9e-2 * np.exp(
        -35e3 / (R * T)
    )  # solubility of T in Li2BeF4 [mol/m3/Pa], Calderoni 2008

    # - derived parameters -
    P_0 = (
        P_top + rho_l * 9.81 * tank_height
    )  # gas inlet pressure, from hydrostatic pressure at the bottom of the tank (neglecting gas fraction)
    flow_g_vol = flow_g_mol * R * T / P_0  # inlet gas volumetric flow rate [m3/s]

    # --- correlations for bubble properties ---

    d_b = _get(
        params,
        "d_b",
        get_d_b,
        # flow_g_vol=flow_g_vol,
        # nozzle_diameter=nozzle_diameter,
        # nb_nozzle=nb_nozzle,
        kwargs={
            "flow_g_vol": flow_g_vol,
            "nozzle_diameter": nozzle_diameter,
            "nb_nozzle": nb_nozzle,
        },
    )  # bubble diameter [m]

    he_molar_mass = 4.003e-3  # kg/mol
    rho_g = (
        P_0 * he_molar_mass / (R * T)
    )  # bubbles density [kg/m3], using ideal gas law for He
    drho = rho_l - rho_g  # density difference between liquid and gas [kg/m3]

    # - dimensionless numbers used in correlations -
    def get_Re(u=U_G0_DEFAULT, Re_old=0):
        Re = rho_l * u * d_b / mu_l
        if (
            Re_old != 0 and np.abs(np.log10(Re / Re_old)) > 1
        ):  # check if Reynolds number changed by more than an order of magnitude
            print(
                f"Warning: Reynolds number changed significantly from {Re_old:.2e} to {Re:.2e}, resulting bubble velocity {u:.2f} m/s might be off. Check assumed bubble velocity in initial Reynolds number calculation"
            )
        return Re

    Eo = (drho * g * d_b**2) / sigma_l  # Eotvos (Bond) number
    Re = get_Re()  # Reynolds number, assuming velocity of gas = 0.25 m/s (typical according to Chavez 2021)
    Mo = (drho * g * mu_l**4) / (rho_l**2 * sigma_l**3)  # Morton number
    Sc = nu_l / diffusivity  # Schmidt number

    # - bubble velocity -
    def get_u_g0():  # Clift 1978 correlation for bubble terminal velocity
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

    u_g0 = _get(
        params, "u_g0", get_u_g0
    )  # bubble velocity [m/s], correlation from Clift 1978, reported by Chavez 2021
    Re = get_Re(u_g0, Re)  # update Reynolds number with the calculated bubble velocity

    # - bubble volume fraction -
    def get_eps_g():
        eps_g = (
            R
            * T
            / (P_0 + 4 * sigma_l / d_b)
            * (flow_g_mol / (np.pi * (D / 2) ** 2 * u_g0))
        )  # gas void fraction, from ideal gas law and Young-Laplace pressure (neglecting hydrostatic pressure variation)
        if eps_g > 1 or eps_g < 0:
            warnings.warn(f"Warning: unphysical gas fraction: {eps_g:.2f}")
        elif eps_g > 0.1:
            warnings.warn(
                f"Warning: high gas fraction: {eps_g:.2f}, models assumptions may not hold"
            )
        return eps_g

    eps_g = _get(params, "eps_g", get_eps_g)
    eps_l = 1 - eps_g
    a = 6 * eps_g / d_b  # specific interfacial area

    # - mass transfer coefficient -
    def get_h_higbie():
        h_l = (
            (diffusivity * u_g0) / (ufl.pi * d_b)
        ) ** 0.5  # mass transport coefficient Higbie penetration model
        return h_l

    def get_h_malara():  # mass transfer coefficient [m2/s], Malara 1995
        h_l = 2 * np.pi**2 * diffusivity / (3 * d_b)
        return h_l

    def get_h_briggs():  # mass transfer coefficient [m2/s] from Sherwood number, Briggs 1970
        Sh = 0.089 * Re**0.69 * Sc**0.33  # Sherwood number
        h_l = Sh * diffusivity / d_b
        return h_l

    h_l_correlations = {
        "higbie": get_h_higbie,
        "malara": get_h_malara,
        "briggs": get_h_briggs,
    }
    h_l = _get(
        params, "h_l", get_h_briggs, correlations=h_l_correlations
    )  # Briggs default

    E_g = 0.2 * D**2 * u_g0  # gas phase diffusivity (Malara 1995)
    E_l = diffusivity  # liquid phase diffusivity

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
        "eps_g": eps_g,
        "eps_l": eps_l,
        "E_l": E_l,
        "E_g": E_g,
        "a": a,
        "h_l": h_l,
        "Re": Re,
        "Eo": Eo,
        "Mo": Mo,
        "Sc": Sc,
    }


def solve(params):
    # unpack parameters
    tank_height, u_g0, a, h_l, K_s, P_0, T, eps_g, eps_l, E_g, E_l, source_term = (
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
        params["E_l"],
        params["source_term"],
    )

    # MESH AND FUNCTION SPACES
    mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, 1000, points=[0, tank_height])
    fdim = mesh.topology.dim - 1
    cg_el = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(2,))

    V = dolfinx.fem.functionspace(mesh, cg_el)

    u = dolfinx.fem.Function(V)
    u_n = dolfinx.fem.Function(V)
    v_c, v_y = ufl.TestFunctions(V)

    c_T, y_T2 = ufl.split(u)
    c_T_n, y_T2_n = ufl.split(u_n)

    dt = 0.2

    vel_x = u_g0  # TODO velocity should vary with hydrostatic pressure
    vel = dolfinx.fem.Constant(mesh, PETSc.ScalarType([vel_x]))

    EPS = 1e-16

    gen = dolfinx.fem.Constant(
        mesh, PETSc.ScalarType(source_term)
    )  # generation term (neutrons)

    # VARIATIONAL FORMULATION

    # mass transfer rate
    J = (
        a * h_l * (c_T - K_s * (P_0 * y_T2 + EPS))
    )  # TODO pressure shouldn't be a constant (use hydrostatic pressure profile), how to deal with this ? -> use fem.Expression ?

    F = 0  # variational formulation

    # transient terms
    F += eps_l * ((c_T - c_T_n) / dt) * v_c * ufl.dx
    F += eps_g * 1 / (R * T) * (P_0 * (y_T2 - y_T2_n) / dt) * v_y * ufl.dx

    # diffusion/dispersion terms
    F += eps_l * E_l * ufl.dot(ufl.grad(c_T), ufl.grad(v_c)) * ufl.dx
    F += eps_g * E_g * ufl.dot(ufl.grad(P_0 * y_T2), ufl.grad(v_y)) * ufl.dx

    # mass exchange (coupling term)
    F += J * v_c * ufl.dx - J * v_y * ufl.dx

    # Generation term in the breeder
    F += -gen * v_c * ufl.dx

    # advection of gas
    F += 1 / (R * T) * ufl.inner(ufl.dot(ufl.grad(P_0 * y_T2), vel), v_y) * ufl.dx

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
    c_T_solutions = []
    y_T2_solutions = []

    # SOLVE
    t = 0
    while t < 10:
        # t += dt
        # print("t:", t)
        problem.solve()

        # update previous solution
        u_n.x.array[:] = u.x.array[:]

        # post process
        c_T_vals = u.x.array[ct_dofs][ct_sort_coords]
        y_T2_vals = u.x.array[y_dofs][y_sort_coords]

        if t >= 1:
            gen.value = 0.0

        # store time and solution
        times.append(t)
        c_T_solutions.append(c_T_vals.copy())
        y_T2_solutions.append(y_T2_vals.copy())

        t += dt

    c_T_volume = np.zeros(len(times))
    for i in range(len(times)):
        c_T_volume[i] = np.trapezoid(
            c_T_solutions[i], x_ct
        )  # integrate concentration profile to get total amount of tritium in the tank at each time step
    c_T_volume *= np.pi * (params["D"] / 2) ** 2  # get total amount of T in [mol]
    return (times, c_T_solutions, y_T2_solutions, x_ct, x_y, c_T_volume)
