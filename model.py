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

mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, 1000, points=[0, 1])
fdim = mesh.topology.dim - 1
cg_el = basix.ufl.element("Lagrange", mesh.basix_cell(), degree=1, shape=(2,))


V = dolfinx.fem.functionspace(mesh, cg_el)

u = dolfinx.fem.Function(V)
u_n = dolfinx.fem.Function(V)
v_c, v_y = ufl.TestFunctions(V)

c_T, y_T2 = ufl.split(u)
c_T_n, y_T2_n = ufl.split(u_n)

dt = 0.2

vel_x = 1.0
vel = dolfinx.fem.Constant(mesh, PETSc.ScalarType([vel_x]))

K_S = 1  # solubility of breeder
D = 0.001
P = 1  # total gas pressure
h_l = 0.1  # mass transport coefficient
EPS = 1e-16

gen = dolfinx.fem.Constant(mesh, PETSc.ScalarType(1.0))  # generation term (neutrons)

# mass transfer rate
J = h_l * (c_T - K_S * (P * y_T2 + EPS) ** 0.5)

F = 0  # variational formulation

# transient terms
F += ((c_T - c_T_n) / dt) * v_c * ufl.dx + ((y_T2 - y_T2_n) / dt) * v_y * ufl.dx


# diffusion/dispersion terms
F += D * ufl.dot(ufl.grad(c_T), ufl.grad(v_c)) * ufl.dx
F += ufl.dot(ufl.grad(y_T2), ufl.grad(v_y)) * ufl.dx


# mass exchange (coupling term)
F += J * v_c * ufl.dx - 0.5 * J * v_y * ufl.dx

# Generation term in the breeder
F += -gen * v_c * ufl.dx

# advection of gas
F += ufl.inner(ufl.dot(ufl.grad(y_T2), vel), v_y) * ufl.dx


# BOUNDARY CONDITIONS
gas_inlet_facets = dolfinx.mesh.locate_entities_boundary(
    mesh, fdim, lambda x: np.isclose(x[0], 0.0)
)
gas_outlet_facets = dolfinx.mesh.locate_entities_boundary(
    mesh, fdim, lambda x: np.isclose(x[0], 1.0)
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
