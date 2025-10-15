# # A coupled 3D-1D manufactured solution
#
# This example is based on the manufactured solution from {cite}`masri2024coupled3d1d`.

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import ufl

from fenicsx_ii import Average, Circle, LinearProblem

# We create a 3D mesh (a cube)
M = 64
N = M
volume = dolfinx.mesh.create_box(
    MPI.COMM_WORLD,
    [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    [M, M, M],
    cell_type=dolfinx.mesh.CellType.tetrahedron,
)
volume.name = "volume"


# We create a 1D mesh (a line), which is not embedded in the 3D mesh
l_min = -0.5
l_max = 0.5
if MPI.COMM_WORLD.rank == 0:
    nodes = np.zeros((N, 3), dtype=np.float64)
    nodes[:, 0] = np.full(nodes.shape[0], 0)
    nodes[:, 1] = np.full(nodes.shape[0], 0)
    nodes[:, 2] = np.linspace(l_min, l_max, nodes.shape[0])[::-1]
    connectivity = np.repeat(np.arange(nodes.shape[0]), 2)[1:-1].reshape(
        nodes.shape[0] - 1, 2
    )
else:
    nodes = np.zeros((0, 3), dtype=np.float64)
    connectivity = np.zeros((0, 2), dtype=np.int64)
c_el = ufl.Mesh(
    basix.ufl.element("Lagrange", basix.CellType.interval, 1, shape=(nodes.shape[1],))
)
line_mesh = dolfinx.mesh.create_mesh(
    MPI.COMM_WORLD,
    x=nodes,
    cells=connectivity,
    e=c_el,
    partitioner=dolfinx.mesh.create_cell_partitioner(
        dolfinx.mesh.GhostMode.shared_facet
    ),
)


# We define the appropriate function spaces on each mesh

degree = 1
V = dolfinx.fem.functionspace(volume, ("Lagrange", degree))
Q = dolfinx.fem.functionspace(line_mesh, ("Lagrange", degree))

# We define a mixed function space, to automate block extraction of the system

W = ufl.MixedFunctionSpace(*[V, Q])
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

# We define the intermediate space used for the restriction operator, along with
# the operators themselves. Note that the test and trial function are restricted
# in different ways.

R = 0.05
q_degree = 20
restriction_trial = Circle(line_mesh, R, degree=q_degree)
restriction_test = Circle(line_mesh, R, degree=q_degree)

# Next, we define the integration measures on each mesh

dx_3D = ufl.Measure("dx", domain=volume)
dx_1D = ufl.Measure("dx", domain=line_mesh)

# We can define the intermediate space that we interpolate the
# 3D arguments into

q_el = basix.ufl.quadrature_element(
    line_mesh.basix_cell(), value_shape=(), degree=q_degree
)
Rs = dolfinx.fem.functionspace(line_mesh, q_el)


# Next, we define the averaging operator of each of the test and trial functions
# of the 3D spaces
avg_u = Average(u, restriction_trial, Rs)
avg_v = Average(v, restriction_test, Rs)

# We define the various constants used in the variational formulation

Alpha1 = dolfinx.fem.Constant(volume, 1.0)
A = ufl.pi * R**2

# and the variational form itself


def u_line(x):
    return ufl.sin(ufl.pi * x[2]) + 2


_xi_cache: dict[dolfinx.mesh.Mesh, dolfinx.fem.Constant] = {}


def xi(domain):
    if _xi_cache.get(domain, None) is None:
        _xi_cache[domain] = dolfinx.fem.Constant(domain, 1.0)
    return _xi_cache[domain]


def x(mesh: dolfinx.mesh.Mesh):
    return ufl.SpatialCoordinate(mesh)


P = 2 * ufl.pi * R

a = Alpha1 * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_3D
a += P * xi(line_mesh) * ufl.inner(avg_u - p, avg_v) * dx_1D
a += A * ufl.inner(ufl.grad(p), ufl.grad(q)) * dx_1D
a += P * xi(line_mesh) * ufl.inner(p - avg_u, q) * dx_1D

vol_rate = xi(volume) / (xi(volume) + 1)

u_inside = vol_rate * u_line(x(volume))
r = ufl.sqrt(x(volume)[0] ** 2 + x(volume)[1] ** 2)
u_outside = vol_rate * (1 - R * ufl.ln(r / R)) * u_line(x(volume))
u_ex = ufl.conditional(r < R, u_inside, u_outside)

p_ex = ufl.sin(ufl.pi * x(line_mesh)[2]) + 2

f_vol_in = vol_rate * ufl.pi**2 * ufl.sin(ufl.pi * x(volume)[2])
f_vol_out = (
    vol_rate * (1 - R * ufl.ln(r / R)) * ufl.pi**2 * ufl.sin(ufl.pi * x(volume)[2])
)
f_vol = ufl.conditional(r < R, f_vol_in, f_vol_out)
# - ufl.div(ufl.grad(u_ex))

line_rate = xi(line_mesh) / (xi(line_mesh) + 1)
# f_line = - A*ufl.div(ufl.grad(p_ex)) + P * line_rate * p_ex
f_line = A * ufl.sin(ufl.pi * x(line_mesh)[2]) * ufl.pi**2 + P * line_rate * p_ex

L = f_vol * v * dx_3D
L += f_line * q * dx_1D

# Exerior boundary conditions
volume.topology.create_connectivity(volume.topology.dim - 1, volume.topology.dim)
exterior_facets = dolfinx.mesh.exterior_facet_indices(volume.topology)
exterior_dofs = dolfinx.fem.locate_dofs_topological(
    V, volume.topology.dim - 1, exterior_facets
)
bc_expr = dolfinx.fem.Expression(u_ex, V.element.interpolation_points)
u_bc = dolfinx.fem.Function(V)
u_bc.interpolate(bc_expr)
bc = dolfinx.fem.dirichletbc(u_bc, exterior_dofs)
bcs = [bc]

# We assemble the arising linear system
uh = dolfinx.fem.Function(V, name="u_3D")
ph = dolfinx.fem.Function(Q, name="p_1D")
petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
}
problem = LinearProblem(
    a,
    L,
    u=[uh, ph],
    petsc_options_prefix="coupled_poisson",
    petsc_options=petsc_options,
    bcs=bcs,
)
uh, ph = problem.solve()

# We move the solution to a Function and save to file


with dolfinx.io.VTXWriter(volume.comm, "u_3D_solver.bp", [uh]) as vtx:
    vtx.write(0.0)
with dolfinx.io.VTXWriter(line_mesh.comm, "p_1D_solver.bp", [ph]) as vtx:
    vtx.write(0.0)

with dolfinx.io.VTXWriter(volume.comm, "u_bc.bp", [u_bc]) as vtx:
    vtx.write(0.0)


# Compute L2-errors for u and p


def L2(f: ufl.core.expr.Expr, dx: ufl.Measure):
    integral = ufl.inner(f, f) * dx
    comm = dx.ufl_domain().ufl_cargo().comm
    return np.sqrt(
        comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(integral)), op=MPI.SUM
        )
    )


def H1(f: ufl.core.expr.Expr, dx: ufl.Measure):
    integral = ufl.inner(f, f) * dx + ufl.inner(ufl.grad(f), ufl.grad(f)) * dx
    comm = dx.ufl_domain().ufl_cargo().comm
    return np.sqrt(
        comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(integral)), op=MPI.SUM
        )
    )


L2_error_u = L2(uh - u_ex, dx_3D)
L2_error_p = L2(ph - p_ex, dx_1D)
H1_error_u = H1(uh - u_ex, dx_3D)
H1_error_p = H1(ph - p_ex, dx_1D)

h_vol = volume.comm.allreduce(
    np.max(
        volume.h(
            volume.topology.dim,
            np.arange(
                volume.topology.index_map(volume.topology.dim).size_local,
                dtype=np.int32,
            ),
        )
    ),
    op=MPI.MAX,
)
h_line = line_mesh.comm.allreduce(
    np.max(
        line_mesh.h(
            line_mesh.topology.dim,
            np.arange(
                line_mesh.topology.index_map(line_mesh.topology.dim).size_local,
                dtype=np.int32,
            ),
        )
    ),
    op=MPI.MAX,
)
if MPI.COMM_WORLD.rank == 0:
    print(f"{degree=:d} {q_degree=:d}")
    print(f"{h_vol=:.5e}, {h_line=:.5e}")
    print(f"L2-error u: {L2_error_u:.6e}")
    print(f"L2-error p: {L2_error_p:.6e}")
    print(f"H1-error u: {H1_error_u:.6e}")
    print(f"H1-error p: {H1_error_p:.6e}")
