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

# -- input --

# - Geometry -
tank_height = 1  # m
tank_diameter = 0.5  # m

# - Operating conditions -
source_term = 0.001  # mol/m3/s generation term
P = 151988  # total gas pressure  # TODO should be 7 PSIG - differential at the top
T = 900  # K temperature
n_g = 0.19 # inlet gas flow [mol/s] 

# -- derived parameters --
u_b = 0.3  # m/s bubble velocity
d_b = 0.002  # m bubble diameter

# - correlations for FLiBe properties -
rho_l = 2245 - 0.424 * (T - 272.15) # density [kg/m3] of Li2BeF4, Vidrio 2022

mu_l = 0.116e-3 * np.exp(3755/T) # dynamic viscosity [Pa.s] of Li2BeF4, Cantor 1968
nu_l = mu_l / rho_l  # kinematic viscosity [m2/s] of Li2BeF4

diffusivity = 9.3e-7 * np.exp(-42e3/(R*T))  # diffusivity of T in Li2BeF4 [m2/s], Calderoni 2008

K_S = 7.9e-2 * np.exp(-35e3 / (R*T)) # solubility of T in Li2BeF4 [mol/m3/Pa], Calderoni 2008


h_l = (
    (diffusivity * u_b) / (ufl.pi * d_b)
) ** 0.5  # mass transport coefficient Higbie penetration model

epsilon_g = 0.03  # gas void fraction  # TODO from correlations
epsilon_l = 1 - epsilon_g  # liquid void fraction
a = 6 * epsilon_g / d_b  # specific interfacial area

# FIXME is this homogeneous?
E_g = 0.2 * tank_diameter**2 * u_b  # gas phase diffusivity (dispersion coefficient)
E_l = diffusivity  # liquid phase diffusivity  # FIXME

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

vel_x = u_b
vel = dolfinx.fem.Constant(mesh, PETSc.ScalarType([vel_x]))


EPS = 1e-16

gen = dolfinx.fem.Constant(
    mesh, PETSc.ScalarType(source_term)
)  # generation term (neutrons)


# VARIATIONAL FORMULATION

# mass transfer rate
J = a * h_l * (c_T - K_S * (P * y_T2 + EPS))

F = 0  # variational formulation

# transient terms
F += epsilon_l * ((c_T - c_T_n) / dt) * v_c * ufl.dx
F += epsilon_g * 1 / (R * T) * (P * (y_T2 - y_T2_n) / dt) * v_y * ufl.dx

# diffusion/dispersion terms
F += epsilon_l * E_l * ufl.dot(ufl.grad(c_T), ufl.grad(v_c)) * ufl.dx
F += epsilon_g * E_g * ufl.dot(ufl.grad(P * y_T2), ufl.grad(v_y)) * ufl.dx


# mass exchange (coupling term)
F += J * v_c * ufl.dx - J * v_y * ufl.dx

# Generation term in the breeder
F += -gen * v_c * ufl.dx

# advection of gas
F += 1 / (R * T) * ufl.inner(ufl.dot(ufl.grad(P * y_T2), vel), v_y) * ufl.dx


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
