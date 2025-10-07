from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl

from fenicsx_ii import Circle, PointwiseTrace, create_interpolation_matrix


def assign_LG_map(
    C: PETSc.Mat,  # type: ignore
    row_map: dolfinx.common.IndexMap,
    col_map: dolfinx.common.IndexMap,
    row_bs: int,
    col_bs: int,
):
    global_row_map = row_map.local_to_global(
        np.arange(row_map.size_local + row_map.num_ghosts, dtype=np.int32)
    ).astype(PETSc.IntType)  # type: ignore
    global_col_map = col_map.local_to_global(
        np.arange(col_map.size_local + col_map.num_ghosts, dtype=np.int32)
    ).astype(PETSc.IntType)  # type: ignore
    row_map = PETSc.LGMap().create(global_row_map, bsize=row_bs, comm=row_map.comm)  # type: ignore
    col_map = PETSc.LGMap().create(global_col_map, bsize=col_bs, comm=col_map.comm)  # type: ignore
    C.setLGMap(row_map, col_map)  # type: ignore


M = 2**5
N = 2 * M
volume = dolfinx.mesh.create_unit_cube(
    MPI.COMM_WORLD, M, M, 2 * M, cell_type=dolfinx.mesh.CellType.tetrahedron
)
volume.name = "volume"

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
line_mesh.name = "line"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "meshes.xdmf", "w") as xdmf:
    xdmf.write_mesh(volume)
    xdmf.write_mesh(line_mesh)


V = dolfinx.fem.functionspace(volume, ("Lagrange", 1))
Q = dolfinx.fem.functionspace(line_mesh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
p = ufl.TrialFunction(Q)
q = ufl.TestFunction(Q)

q_degree = 10
q_el = basix.ufl.quadrature_element(
    line_mesh.basix_cell(), value_shape=(), degree=q_degree
)
W = dolfinx.fem.functionspace(line_mesh, q_el)
w = ufl.TestFunction(W)
z = ufl.TrialFunction(W)


dx_3D = ufl.Measure("dx", domain=volume)
dx_1D = ufl.Measure("dx", domain=line_mesh)

Alpha1 = dolfinx.fem.Constant(volume, 0.02)
Alpha0 = dolfinx.fem.Constant(volume, 0.01)
alpha1 = dolfinx.fem.Constant(line_mesh, 2.0)
alpha0 = dolfinx.fem.Constant(line_mesh, 0.01)
beta = dolfinx.fem.Constant(line_mesh, 10.0)  # Coupling strength


a00 = (
    Alpha1 * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_3D
    + Alpha0 * ufl.inner(u, v) * dx_3D
)

a01 = -beta * ufl.inner(p, w) * dx_1D
a10 = -beta * ufl.inner(z, q) * dx_1D

a11 = alpha1 * ufl.inner(ufl.grad(p), ufl.grad(q)) * dx_1D
a11 += (alpha0 + beta) * ufl.inner(p, q) * dx_1D

A00 = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a00))
A00.assemble()
A11 = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a11))
A11.assemble()
A10 = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a10))
A10.assemble()
A01 = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a01))
A01.assemble()

R = 0.2
# Create restriction operator for trial space
restriction_trial = Circle(line_mesh, R, degree=5)
T, imap_K, imap_V = create_interpolation_matrix(V, W, restriction_trial, use_petsc=True)
C = A10.matMult(T)
assign_LG_map(
    C, Q.dofmap.index_map, imap_V, Q.dofmap.index_map_bs, V.dofmap.index_map_bs
)

restriction_test = PointwiseTrace(line_mesh)
P, imap_K_n, imap_V_n = create_interpolation_matrix(
    V, W, restriction_test, use_petsc=True
)
Pt = P.copy()  # type: ignore
Pt.transpose()

x = ufl.SpatialCoordinate(line_mesh)
# f = dolfinx.fem.Constant(volume, 0.0)  # x[0]
f = ufl.sin(x[2])
# f = dolfinx.fem.Constant(line_mesh, 2.0)

L0 = f * w * dx_1D
b0_1D = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(L0))
b0_1D.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
b0_3D = dolfinx.fem.Function(V)
b0 = b0_3D.x.petsc_vec
Pt.mult(b0_1D, b0)

L1 = f * q * dx_1D
b1 = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(L1))
b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)  # type: ignore

# V_in = dolfinx.fem.Constant(line_mesh, 0.0)
# V_out = dolfinx.fem.Constant(line_mesh, 3.0)


# def bc_in(x):
#     return np.isclose(x[2], l_min)


# def bc_out(x):
#     return np.isclose(x[2], l_max)


# ltdim = line_mesh.topology.dim
# in_facets = dolfinx.mesh.locate_entities_boundary(
#     line_mesh, ltdim - 1, bc_in
# )
# in_dofs = dolfinx.fem.locate_dofs_topological(Q, ltdim - 1, in_facets)
# out_facets = dolfinx.mesh.locate_entities_boundary(
#     line_mesh, ltdim - 1, bc_out
# )
# out_dofs = dolfinx.fem.locate_dofs_topological(
#     Q, ltdim - 1, out_facets
# )


a00_W = dolfinx.fem.form(beta * ufl.inner(z, w) * dx_1D)
A00_W = dolfinx.fem.petsc.assemble_matrix(a00_W)
A00_W.assemble()
Z = Pt.matMult(A00_W).matMult(T)
assign_LG_map(
    Z,
    V.dofmap.index_map,
    V.dofmap.index_map,
    V.dofmap.index_map_bs,
    V.dofmap.index_map_bs,
)
A00.axpy(1, Z)

D = Pt.matMult(A01)

bcs: list[dolfinx.fem.DirichletBC] = [
    # dolfinx.fem.dirichletbc(V_in, in_dofs, Q),
    # dolfinx.fem.dirichletbc(V_out, out_dofs, Q),
]
for bc in bcs:
    dofs, lz = bc._cpp_object.dof_indices()
    C.zeroRowsLocal(dofs, diag=0)
    A11.zeroRowsLocal(dofs, diag=1)

A_block = PETSc.Mat().createNest([[A00, D], [C, A11]])  # type: ignore


dolfinx.fem.petsc.set_bc(b1, bcs)
b0.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
b1.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
b = PETSc.Vec().createNest([b0, b1])  # type: ignore


ksp = PETSc.KSP().create(volume.comm)  # type: ignore
ksp.setOperators(A_block)
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")
ksp.setErrorIfNotConverged(True)
u_vec = b.duplicate()
ksp.solve(b, u_vec)
ksp.destroy()


uh = dolfinx.fem.Function(V)
ph = dolfinx.fem.Function(Q)
dolfinx.fem.petsc.assign(u_vec, [uh, ph])
ph.x.scatter_forward()
uh.x.scatter_forward()
uh.name = "u_3D"
ph.name = "p_1D"

with dolfinx.io.VTXWriter(volume.comm, "u_3D.bp", [uh]) as vtx:
    vtx.write(0.0)
with dolfinx.io.VTXWriter(line_mesh.comm, "p_1D.bp", [ph]) as vtx:
    vtx.write(0.0)
