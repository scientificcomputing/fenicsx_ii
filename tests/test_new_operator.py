from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl

from fenicsx_ii import Circle, PointwiseTrace
from fenicsx_ii.assembly import assemble_matrix, assemble_vector
from fenicsx_ii.ufl_operations import Average

domain = dolfinx.mesh.create_box(
    MPI.COMM_WORLD, [[-1, -1, -1], [1, 1, 1]], [10, 10, 10]
)
N = 25
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


V = dolfinx.fem.functionspace(domain, ("Lagrange", 1, (2,)))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

Q = dolfinx.fem.functionspace(line_mesh, ("DG", 0, (2,)))  # Intermediate space


R = 0.1
restriction_trial = Circle(line_mesh, R, degree=5)
restriction_test = PointwiseTrace(line_mesh)

avg_u = Average(u, restriction_trial, Q)
avg_v = Average(v, restriction_test, Q)
W = dolfinx.fem.functionspace(line_mesh, ("Lagrange", 1, (2,)))

dGamma = ufl.Measure("dx", domain=line_mesh)

a_form = ufl.inner(avg_u, avg_v) * ufl.dx
a_form += ufl.inner(u, v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.ds

A = assemble_matrix(a_form)

z = ufl.TestFunction(W)

a01 = ufl.inner(avg_u, z) * dGamma
C = assemble_matrix(a01)


w = ufl.TrialFunction(W)
a10 = ufl.inner(w, avg_v) * dGamma
D = assemble_matrix(a10)

a11 = ufl.inner(z, w) * dGamma

E = assemble_matrix(a11)

PETSc.Sys.Print(A.getSizes(), C.getSizes(), D.getSizes(), E.getSizes())
norms = A.norm(0), C.norm(0), D.norm(0), E.norm(0)

PETSc.Sys.Print(norms)


f_vol = ufl.as_vector((2, 3))
f_line = dolfinx.fem.Constant(line_mesh, (1.0, 2.0))
L0 = ufl.inner(f_vol, avg_v) * dGamma
L1 = ufl.inner(f_line, z) * dGamma
b0 = assemble_vector(L0)
b1 = assemble_vector(L1)
b_norms = b0.norm(0), b1.norm(0)
PETSc.Sys.Print(b_norms)


T = ufl.MixedFunctionSpace(V, W)
u, w = ufl.TrialFunctions(T)
v, z = ufl.TestFunctions(T)
avg_u = Average(u, restriction_trial, Q)
avg_v = Average(v, restriction_test, Q)

ab_form = ufl.inner(avg_u, avg_v) * dGamma
ab_form += ufl.inner(u, v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.ds
ab_form += ufl.inner(avg_u, z) * dGamma
ab_form += ufl.inner(w, avg_v) * dGamma
ab_form += ufl.inner(w, z) * dGamma

B = assemble_matrix(ab_form)

PETSc.Sys.Print(B.getSizes())
for i in range(2):
    for j in range(2):
        PETSc.Sys.Print(f"B[{i}, {j}], {B.getNestSubMatrix(i, j).norm(0)}")


Lb_form = ufl.inner(f_vol, avg_v) * dGamma
Lb_form += ufl.inner(f_line, z) * dGamma

b_blocked = assemble_vector(Lb_form)

for i in range(2):
    n_i = b_blocked.getNestSubVecs()[i].norm(0)
    PETSc.Sys.Print(i, n_i)
