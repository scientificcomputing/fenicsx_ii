
from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl


from fenicsx_ii import Circle, PointwiseTrace, create_interpolation_matrix
from fenicsx_ii.ufl_operations import Average, get_replaced_argument_indices, apply_replacer

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


def assemble_matrix(a: ufl.Form) -> PETSc.Mat:
    """Assemble a matrix from a UFL form.

    Args:
        a: Bi-linear UFL form to assemble.
    """

    new_forms = apply_replacer(a)
    matrix = None
    for form in new_forms:
        a_c = dolfinx.fem.form(form)
        A = dolfinx.fem.petsc.assemble_matrix(a_c)
        A.assemble()

        replacement_indices = get_replaced_argument_indices(form)
        test_arg, trial_arg = form.arguments()
        match replacement_indices:
            case []:
                if matrix is None:
                    matrix = A
                else:
                    matrix.axpy(1.0, A)

            case [0]:
                # Replace rows, i.e. apply interpolation matrix on the left
                K, imap_test, _imap_trial = create_interpolation_matrix(
                    test_arg.parent_space,
                    test_arg.ufl_function_space(),
                    test_arg.restriction_operator,
                    use_petsc=True,
                )
                K.transpose()  # in-place transpose
                D = K.matMult(A)
                trial_space = trial_arg.ufl_function_space()
                test_space = test_arg.parent_space
                assign_LG_map(
                    D,
                    imap_test,
                    trial_space.dofmap.index_map,
                    test_space.dofmap.index_map_bs,
                    trial_space.dofmap.index_map_bs,
                )
                if matrix is None:
                    matrix = D
                else:
                    matrix.axpy(1.0, D)

            case [1]:
                # Replace columns, i.e. apply interpolation matrix on the right
                K, _imap_test, imap_trial = create_interpolation_matrix(
                    trial_arg.parent_space,
                    trial_arg.ufl_function_space(),
                    trial_arg.restriction_operator,
                    use_petsc=True,
                )
                D = A.matMult(K)
                trial_space = trial_arg.parent_space
                test_space = test_arg.ufl_function_space()
                assign_LG_map(
                    D,
                    test_space.dofmap.index_map,
                    imap_trial,
                    test_space.dofmap.index_map_bs,
                    trial_space.dofmap.index_map_bs,
                )
                if matrix is None:
                    matrix = D
                else:
                    matrix.axpy(1.0, D)

            case [0, 1]:
                # Replace both rows and columns, i.e. apply interpolation matrix on both sides
                # Start by replacing rows, multiplying by P.T from
                P, _, _ = create_interpolation_matrix(
                    test_arg.parent_space,
                    test_arg.ufl_function_space(),
                    test_arg.restriction_operator,
                    use_petsc=True,
                )
                Pt = P.copy()  # type: ignore
                Pt.transpose()
                Z = Pt.matMult(A)
                # Now replace columns, multiplying by K from the right
                K, _, _ = create_interpolation_matrix(
                    trial_arg.parent_space,
                    trial_arg.ufl_function_space(),
                    trial_arg.restriction_operator,
                    use_petsc=True,
                )
                D = Z.matMult(K)
                trial_space = trial_arg.parent_space
                test_space = test_arg.parent_space
                assign_LG_map(
                    D,
                    trial_space.dofmap.index_map,
                    test_space.dofmap.index_map,
                    trial_space.dofmap.index_map_bs,
                    test_space.dofmap.index_map_bs,
                )
                if matrix is None:
                    matrix = D
                else:
                    matrix.axpy(1.0, D)
    return matrix


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

Q = dolfinx.fem.functionspace(line_mesh, ("DG", 0, (2,))) # Intermediate space


R = 0.1
restriction_trial = Circle(line_mesh, R, degree=5)
restriction_test = PointwiseTrace(line_mesh)

avg_u = Average(u, restriction_trial, Q)
avg_v = Average(v, restriction_test, Q)


a_form = ufl.inner(avg_u, avg_v) * ufl.dx
a_form += ufl.inner(u, v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.ds

A = assemble_matrix(a_form)

W = dolfinx.fem.functionspace(line_mesh, ("Lagrange", 1, (2,)))
z = ufl.TestFunction(W)

dGamma = ufl.Measure("dx", domain=line_mesh)
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
