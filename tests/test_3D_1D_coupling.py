from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import dolfinx
import ufl
import basix.ufl
from fenicsx_ii import Circle, NaiveTrace, create_interpolation_matrix
import dolfinx.fem.petsc


def assign_LG_map(
    C: PETSc.Mat,
    row_map: dolfinx.common.IndexMap,
    col_map: dolfinx.common.IndexMap,
    row_bs: int,
    col_bs: int,
):
    global_row_map = row_map.local_to_global(
        np.arange(row_map.size_local + row_map.num_ghosts, dtype=np.int32)
    ).astype(PETSc.IntType)
    global_col_map = col_map.local_to_global(
        np.arange(col_map.size_local + col_map.num_ghosts, dtype=np.int32)
    ).astype(PETSc.IntType)
    row_map = PETSc.LGMap().create(global_row_map, bsize=row_bs, comm=row_map.comm)
    col_map = PETSc.LGMap().create(global_col_map, bsize=col_bs, comm=col_map.comm)
    C.setLGMap(row_map, col_map)


M = 33
N = 100
volume = dolfinx.mesh.create_unit_cube(
    MPI.COMM_WORLD, M, M, M, cell_type=dolfinx.mesh.CellType.tetrahedron
)
volume.name = "volume"

l_min = 0.25
l_max = 0.75
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


def dirichlet_boundary(x):
    return np.isclose(x[0], 0.0) & np.isclose(x[0], 1.0)


V = dolfinx.fem.functionspace(volume, ("Lagrange", 2))
Q = dolfinx.fem.functionspace(line_mesh, ("Lagrange", 2))
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

k = dolfinx.fem.Constant(volume, 2.0)
sigma = dolfinx.fem.Constant(line_mesh, 2.0)
gamma = 100  # Coupling strength
a00 = k * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_3D
a11 = sigma * ufl.inner(ufl.grad(p), ufl.grad(q)) * dx_1D
a10 = ufl.inner(z, q) * dx_1D
a01 = ufl.inner(p, w) * dx_1D

A00 = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a00))
A00.assemble()
A11 = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a11))
A11.assemble()
A10 = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a10))
A10.assemble()
A01 = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a01))
A01.assemble()

R = 0.05
# Create restriction operator for trial space
restriction_trial = Circle(line_mesh, R, degree=10)
T, imap_K, imap_V = create_interpolation_matrix(V, W, restriction_trial, use_petsc=True)
C = A10.matMult(T)
assign_LG_map(
    C, Q.dofmap.index_map, imap_V, Q.dofmap.index_map_bs, V.dofmap.index_map_bs
)


x = ufl.SpatialCoordinate(volume)
f = dolfinx.fem.Constant(volume, 0.0)  # x[0]
L0 = f * v * dx_3D
L1 = dolfinx.fem.Constant(line_mesh, 0.0) * q * dx_1D
b0 = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(L0))
b1 = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(L1))
b0.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

V_in = dolfinx.fem.Constant(line_mesh, 0.0)
V_out = dolfinx.fem.Constant(line_mesh, 3.0)


def bc_in(x):
    return np.isclose(x[2], l_min)


def bc_out(x):
    return np.isclose(x[2], l_max)


in_facets = dolfinx.mesh.locate_entities_boundary(
    line_mesh, line_mesh.topology.dim - 1, bc_in
)
in_dofs = dolfinx.fem.locate_dofs_topological(Q, line_mesh.topology.dim - 1, in_facets)
out_facets = dolfinx.mesh.locate_entities_boundary(
    line_mesh, line_mesh.topology.dim - 1, bc_out
)
out_dofs = dolfinx.fem.locate_dofs_topological(
    Q, line_mesh.topology.dim - 1, out_facets
)

restriction_test = NaiveTrace(line_mesh)
P, imap_K_n, imap_V_n = create_interpolation_matrix(
    V, W, restriction_test, use_petsc=True
)
Pt = P.copy()
Pt.transpose()

a00_W = dolfinx.fem.form(ufl.inner(z, w) * dx_1D)
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
A00.axpy(gamma, Z)

D = Pt.matMult(A01)
D.scale(-gamma)
C.scale(gamma)
A11.scale(gamma)

bc_in = dolfinx.fem.dirichletbc(V_in, in_dofs, Q)
bc_out = dolfinx.fem.dirichletbc(V_out, out_dofs, Q)
bcs = [bc_in, bc_out]
for bc in bcs:
    dofs, lz = bc._cpp_object.dof_indices()
    C.zeroRowsLocal(dofs, diag=0)
    A11.zeroRowsLocal(dofs, diag=1)

A_block = PETSc.Mat().createNest([[A00, D], [C, A11]])


dolfinx.fem.petsc.set_bc(b1, bcs)
b0.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
b1.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
b = PETSc.Vec().createNest([b0, b1])


ksp = PETSc.KSP().create(volume.comm)
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
