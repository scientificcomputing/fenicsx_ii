from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx
import numpy as np
import ufl

from fenicsx_ii import Average, Circle, PointwiseTrace, assemble_matrix, assemble_vector

# We create a 3D mesh (a cube)
M = 2**5
N = 2 * M
volume = dolfinx.mesh.create_unit_cube(
    MPI.COMM_WORLD, M, M, 2 * M, cell_type=dolfinx.mesh.CellType.tetrahedron
)
volume.name = "volume"


# We create a 1D mesh (a line), which is not embedded in the 3D mesh
l_min = 0
l_max = 1
if MPI.COMM_WORLD.rank == 0:
    nodes = np.zeros((N, 3), dtype=np.float64)
    nodes[:, 0] = np.full(nodes.shape[0], 0.5)
    nodes[:, 1] = np.full(nodes.shape[0], 0.5)
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

V = dolfinx.fem.functionspace(volume, ("Lagrange", 1))
Q = dolfinx.fem.functionspace(line_mesh, ("Lagrange", 1))

# We define a mixed function space, to automate block extraction of the system

W = ufl.MixedFunctionSpace(*[V, Q])
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

# We define the intermediate space used for the restriction operator, along with
# the operators themselves. Note that the test and trial function are restricted
# in different ways.

r = 0.2
restriction_trial = Circle(line_mesh, r, degree=5)
restriction_test = PointwiseTrace(line_mesh)

# Next, we define the integration measures on each mesh

dx_3D = ufl.Measure("dx", domain=volume)
dx_1D = ufl.Measure("dx", domain=line_mesh)

# We can define the intermediate space that we interpolate the
# 3D arguments into

q_degree = 10
q_el = basix.ufl.quadrature_element(
    line_mesh.basix_cell(), value_shape=(), degree=q_degree
)
R = dolfinx.fem.functionspace(line_mesh, q_el)


# Next, we define the averaging operator of each of the test and trial functions
# of the 3D spaces
avg_u = Average(u, restriction_trial, R)
avg_v = Average(v, restriction_test, R)

# We define the various constants used in the variational formulation

Alpha1 = dolfinx.fem.Constant(volume, 0.02)
Alpha0 = dolfinx.fem.Constant(volume, 0.01)
alpha1 = dolfinx.fem.Constant(line_mesh, 2.0)
alpha0 = dolfinx.fem.Constant(line_mesh, 0.01)
beta = dolfinx.fem.Constant(line_mesh, 10.0)  # Coupling strength

# and the variational form itself

a = Alpha1 * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_3D
a += Alpha0 * ufl.inner(u, v) * dx_3D
a -= beta * p * avg_v * dx_1D
a -= beta * avg_u * q * dx_1D
a += alpha1 * ufl.inner(ufl.grad(p), ufl.grad(q)) * dx_1D
a += (alpha0 + beta) * ufl.inner(p, q) * dx_1D
a += beta * ufl.inner(avg_u, avg_v) * dx_1D
x = ufl.SpatialCoordinate(line_mesh)
f = ufl.sin(x[2])
L = f * avg_v * dx_1D
L += f * q * dx_1D


A = assemble_matrix(a)
b = assemble_vector(L)

# We create the ksp solver and solve the system
ksp = PETSc.KSP().create(volume.comm)  # type: ignore
ksp.setOperators(A)
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")
ksp.setErrorIfNotConverged(True)
u_vec = b.duplicate()
ksp.solve(b, u_vec)
ksp.destroy()

# We move the solution to a Function and save to file

uh = dolfinx.fem.Function(V)
ph = dolfinx.fem.Function(Q)
dolfinx.fem.petsc.assign(u_vec, [uh, ph])
ph.x.scatter_forward()
uh.x.scatter_forward()
uh.name = "u_3D"
ph.name = "p_1D"

with dolfinx.io.VTXWriter(volume.comm, "u_3D_HL.bp", [uh]) as vtx:
    vtx.write(0.0)
with dolfinx.io.VTXWriter(line_mesh.comm, "p_1D_HL.bp", [ph]) as vtx:
    vtx.write(0.0)
