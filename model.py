from mpi4py import MPI
import dolfinx
import basix
import ufl
import numpy as np
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc
from animation import create_animation

from dolfinx import log

log.set_log_level(log.LogLevel.INFO)

# -- Physical constants -- 
avogadro_number = 6.022e23  # 1/mol
R = 8.314  # J/mol/K
g = 9.81  # m/s2, gravity acceleration

# -- input --

# - Geometry -
tank_height = 1  # m
D = 0.5  # m
nozzle_diameter = 0.001  # m
nb_nozzle = 1000 # number of nozzles at bottom of the tank 

# - Operating conditions -
source_term = 0.001  # mol/m3/s generation term
P_top = 151988 # Pa, gas pressure at the top of the tank (what we control in LIBRA)
T = 600 + 278  # K temperature
flow_g_mol = 0.19 # inlet gas flow [mol/s] 

def compute_properties(params):
    tank_height = params["tank_height"]
    D = params["D"]
    nozzle_diameter = params["nozzle_diameter"]
    nb_nozzle = params["nb_nozzle"]
    P_top = params["P_top"]
    T = params["T"]
    flow_g_mol = params["flow_g_mol"]

    # --- correlations for FLiBe properties ---
    rho_l = 2245 - 0.424 * (T - 272.15) # density [kg/m3] of Li2BeF4, Vidrio 2022
    mu_l = 0.116e-3 * np.exp(3755/T) # dynamic viscosity [Pa.s] of Li2BeF4, Cantor 1968
    nu_l = mu_l / rho_l  # kinematic viscosity [m2/s] of Li2BeF4
    sigma_l = (260 - 0.12 * (T - 272.15)) * 1e-3 # surface tension of FLiBe [N/m], Cantor 1968

    # --- correlations for tritium in FLiBe ---
    diffusivity = 9.3e-7 * np.exp(-42e3/(R*T))  # diffusivity of T in Li2BeF4 [m2/s], Calderoni 2008
    K_s = 7.9e-2 * np.exp(-35e3 / (R*T)) # solubility of T in Li2BeF4 [mol/m3/Pa], Calderoni 2008

    # - derived parameters -
    P_0 = P_top + rho_l * 9.81 * tank_height  # gas inlet pressure, from hydrostatic pressure at the bottom of the tank (neglecting gas fraction)
    flow_g_vol = flow_g_mol * R * T / P_0  # inlet gas volumetric flow rate [m3/s]
    
    # --- correlations for bubble properties ---
    def get_d_b ():
        nozzle_flow = flow_g_vol / nb_nozzle  # volumetric flow per nozzle [m3/s]
        if nozzle_flow < 3e-6 or nozzle_flow > 10e-6:
            print (f"Warning: nozzle flow {nozzle_flow*1e6:.2e} cm3/s is out of the validated range for the Kanai 2017 correlation (3-10 cm3/s)")
        return 0.54 * (nozzle_flow * 1e6 * np.sqrt(nozzle_diameter/2 * 1e2)) ** 0.289 * 1e-2 # mean bubble diameter [m], Kanai 2017 (reported by Evans 2026)

    d_b = get_d_b()

    rho_g = P_0 * 4.003e-3 / (R * T)  # bubbles density [kg/m3], using ideal gas law for He
    drho = rho_l - rho_g  # density difference between liquid and gas [kg/m3]
    
    # - dimensionless numbers used in correlations -
    def get_Re(u):
        return rho_l * u * d_b / mu_l
    Eo = (drho * g * d_b**2) / sigma_l  # Eotvos (Bond) number
    Re = get_Re(0.25)  # Reynolds number, assuming terminal velocity of gas = 0.25 m/s (typical according to Chavez 2021)
    Mo = (drho * g * mu_l**4) / (rho_l**2 * sigma_l**3)  # Morton number
    Sc = nu_l / diffusivity # Schmidt number 

    # - bubble velocity -
    def get_u_g0 (): # Clift 1978 correlation for bubble terminal velocity
        H = 4/3 * Eo * Mo**-0.149 * (mu_l / 0.0009)**-0.14
        if H > 59.3:
            J = 3.42 * H**0.441
        elif H > 2:
            J = 0.94 * H**0.757
        else:
            J = Re * Mo**0.149 + 0.857
        u_g0 = mu_l / (rho_l * d_b) * Mo**-0.149 * (J - 0.857)
        if u_g0 > 1 or u_g0 < 0.1:
            print (f"Warning: bubble velocity {u_g0:.2f} m/s is out of the typical range")
        return u_g0

    u_g0 = get_u_g0()  # bubble velocity [m/s], correlation from Clift 1978, reported by Chavez 2021
    Re = get_Re(u_g0) # update Reynolds number with the calculated bubble velocity

    # - bubble volume fraction -
    def get_eps():
        eps_g = R * T / (P_0 + 4 * sigma_l / d_b) * (flow_g_mol / (np.pi * (D/2)**2 * u_g0)) # gas void fraction, from ideal gas law and Young-Laplace pressure (neglecting hydrostatic pressure variation)
        if eps_g > 1 or eps_g < 0:
            print (f"Warning: unphysical gas fraction: {eps_g:.2f}")
        elif eps_g > 0.1:
            print (f"Warning: high gas fraction: {eps_g:.2f}, models assumptions may not hold")
        return (eps_g, 1 - eps_g)

    eps_g, eps_l = get_eps()

    a = 6 * eps_g / d_b  # specific interfacial area

    # - mass transfer coefficient -
    def get_h_higbie():
        h_l = (
            (diffusivity * u_g0) / (ufl.pi * d_b)
        ) ** 0.5  # mass transport coefficient Higbie penetration model
        return h_l

    def get_h_malara(): # mass transfer coefficient [m2/s], Malara 1995
        h_l = 2 * np.pi**2 * diffusivity / (3 * d_b)
        return h_l

    def get_h_briggs(): # mass transfer coefficient [m2/s] from Sherwood number, Briggs 1970
        Sh = 0.089 * Re**0.69 * Sc**0.33 # Sherwood number
        h_l = Sh * diffusivity / d_b
        return h_l

    h_l_higbie = get_h_higbie()
    h_l_malara = get_h_malara()
    h_l_briggs = get_h_briggs()

    h_l = h_l_briggs # choose mass transfer coefficient model

    E_g = 0.2 * D**2 * u_g0  # gas phase diffusivity (Malara 1995)
    E_l = diffusivity  # liquid phase diffusivity 

    return {
        "rho_l": rho_l, "sigma_l": sigma_l, "mu_l": mu_l, "nu_l": nu_l, "K_s": K_s,
        "u_g0": u_g0, "d_b": d_b, "rho_g": rho_g,
        "eps_g": eps_g, "eps_l": eps_l, "E_l": E_l, "E_g": E_g,
        "a": a, "h_l": h_l,
        "Re": Re, "Eo": Eo, "Mo": Mo, "Sc": Sc
    }

def solve():
    # MESH AND FUNCTION SPACES
    mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, 10000, points=[0, tank_height])
    fdim = mesh.topology.dim - 1
    cg_el = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(2,))


    V = dolfinx.fem.functionspace(mesh, cg_el)

    u = dolfinx.fem.Function(V)
    u_n = dolfinx.fem.Function(V)
    v_c, v_y = ufl.TestFunctions(V)

    c_T, y_T2 = ufl.split(u)
    c_T_n, y_T2_n = ufl.split(u_n)

    dt = 0.2

    vel_x = u_g0
    vel = dolfinx.fem.Constant(mesh, PETSc.ScalarType([vel_x]))


    EPS = 1e-16

    gen = dolfinx.fem.Constant(
        mesh, PETSc.ScalarType(source_term)
    )  # generation term (neutrons)


    # VARIATIONAL FORMULATION

    # mass transfer rate
    J = a * h_l * (c_T - K_s * (P_0 * y_T2 + EPS))    # TODO pressure shouldn't be a constant (use hydrostatic pressure profile), how to deal with this ?

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
        petsc_options={"snes_monitor": None},
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
        t += dt
        print("t:", t)
        problem.solve()

        # update previous solution
        u_n.x.array[:] = u.x.array[:]

        # post process
        c_T_vals = u.x.array[ct_dofs][ct_sort_coords]
        y_T2_vals = u.x.array[y_dofs][y_sort_coords]

        if t >= 5:
            gen.value = 0.0

        # store time and solution
        times.append(t)
        c_T_solutions.append(c_T_vals.copy())
        y_T2_solutions.append(y_T2_vals.copy())

    # Create interactive animation
    create_animation(times, c_T_solutions, y_T2_solutions, x_ct, x_y)
