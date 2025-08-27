from mpi4py import MPI
import basix.ufl
import dolfinx
import numpy as np
import numpy.typing as npt
import ufl
import pytest
from fenicsx_ii import NaiveTrace, create_interpolation_matrix

def create_line(ghost_mode: dolfinx.mesh.GhostMode) -> dolfinx.mesh.Mesh:
    if MPI.COMM_WORLD.rank == 0:
        M = 125
        nodes = np.zeros((M, 3), dtype=np.float64)

        nodes[:, 0] = np.linspace(0.23, 0.7, nodes.shape[0])
        nodes[:, 1] = np.linspace(0.32, 0.6, nodes.shape[0])[::-1]
        nodes[:, 2] = np.linspace(0.15, 0.7, nodes.shape[0])[::-1]

        # Test for curved line
        theta = np.linspace(0, 3 * 2 * np.pi, nodes.shape[0])
        nodes[:, 0] = 0.5 + 0.2 * np.cos(theta)
        nodes[:, 1] = 0.5 + 0.2 * np.sin(theta)
        nodes[:, 2] = np.linspace(0.2, 0.8, nodes.shape[0])[::-1]

        connectivity = np.repeat(np.arange(nodes.shape[0]), 2)[1:-1].reshape(
            nodes.shape[0] - 1, 2
        )
    else:
        nodes = np.zeros((0, 3), dtype=np.float64)
        connectivity = np.zeros((0, 2), dtype=np.int32)

    c_el = ufl.Mesh(
        basix.ufl.element("Lagrange", basix.CellType.interval, 1, shape=(nodes.shape[1],))
    )
    line_mesh = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD,
        x=nodes,
        cells=connectivity,
        e=c_el,
        partitioner=dolfinx.mesh.create_cell_partitioner(ghost_mode),
    )
    line_mesh.name = "line"
    return line_mesh

@pytest.mark.parametrize("use_petsc", [True, False])
@pytest.mark.parametrize("family", ["DG", "Quadrature"])
@pytest.mark.parametrize("degree", [1, 2, 4])
@pytest.mark.parametrize("ghost_mode", [dolfinx.mesh.GhostMode.none, dolfinx.mesh.GhostMode.shared_facet])
def test_naive_trace(use_petsc, family, degree, ghost_mode):
   
    N = 17
    cube = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD, N, N, N, ghost_mode=ghost_mode
    )
    cube.name = "Cube"
    line = create_line(ghost_mode) 
    assert line.comm.size == cube.comm.size
    assert line.comm.rank == cube.comm.rank

    V = dolfinx.fem.functionspace(cube, ("Lagrange", 1))

    if family == "DG":
        el = ("DG", degree)
    elif family == "Quadrature":
        el = basix.ufl.quadrature_element(line.basix_cell(), value_shape=(), degree=degree)
    else:
        raise NotImplementedError(f"Test for {family=} not implemented")
    K_hat = dolfinx.fem.functionspace(
        line, el)



    def f(x):
        return x[0] - x[1] + 2*x[2]

    # Interpolate reference solution onto `u`
    uh = dolfinx.fem.Function(V)
    uh.interpolate(f)

    restriction = NaiveTrace(line)

    bh = dolfinx.fem.Function(K_hat)
    use_petsc = False
    A = create_interpolation_matrix(V, K_hat, restriction, use_petsc=use_petsc)
   
    if use_petsc:
        A.mult(uh.x.petsc_vec, bh.x.petsc_vec)
    else:
        # NOTE: Implicit assumption in DOLFINx that the dofs map of input vector
        # is the same as the once used within the matrix

        # Transfer from standard function to compatible vector
        num_owned_dofs = A.index_map(1).size_local * A.block_size[1]
        u_vec = dolfinx.la.vector(A.index_map(1), A.block_size[1])
        u_vec.array[:num_owned_dofs] = uh.x.array[:num_owned_dofs]
        u_vec.scatter_forward()
        b_vec = dolfinx.la.vector(A.index_map(0), A.block_size[0])
        A.mult(u_vec, b_vec)
        b_vec.scatter_forward()
        # Transfer back to DOLFINx vector
        num_owned_dofs_b = A.index_map(0).size_local * A.block_size[0]
        bh.x.array[:num_owned_dofs_b] = b_vec.array[:num_owned_dofs_b]
    bh.x.scatter_forward()

    bh_ref = dolfinx.fem.Function(K_hat)
    bh_ref.interpolate(f)
    np.testing.assert_allclose(bh.x.array, bh_ref.x.array)

