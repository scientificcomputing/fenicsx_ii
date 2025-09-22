import itertools

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import pytest
import ufl

from fenicsx_ii import Circle, PointwiseTrace, create_interpolation_matrix

# Only run ghost mode parametrization in parallel
if MPI.COMM_WORLD.size == 1:
    ghost_modes = [
        dolfinx.mesh.GhostMode.none,
    ]
else:
    ghost_modes = [
        dolfinx.mesh.GhostMode.none,
        dolfinx.mesh.GhostMode.shared_facet,
    ]

three_dimensional_cell_types: list[dolfinx.mesh.CellType] = [
    dolfinx.mesh.CellType.tetrahedron,
    dolfinx.mesh.CellType.hexahedron,
]

one_dim_combinations = list(itertools.product(ghost_modes))
three_dim_combinations = list(
    itertools.product(three_dimensional_cell_types, ghost_modes)
)


def create_line(
    ghost_mode: dolfinx.mesh.GhostMode, curved: bool = True
) -> dolfinx.mesh.Mesh:
    if MPI.COMM_WORLD.rank == 0:
        M = 63
        nodes = np.zeros((M, 3), dtype=np.float64)

        if curved:
            theta = np.linspace(0, 3 * 2 * np.pi, nodes.shape[0])
            nodes[:, 0] = 0.5 + 0.2 * np.cos(theta)
            nodes[:, 1] = 0.5 + 0.2 * np.sin(theta)
            nodes[:, 2] = np.linspace(0.2, 0.8, nodes.shape[0])[::-1]
        else:
            nodes[:, 0] = np.zeros(nodes.shape[0])
            nodes[:, 1] = np.zeros(nodes.shape[0])
            nodes[:, 2] = np.linspace(0.15, 0.7, nodes.shape[0])[::-1]

        connectivity = np.repeat(np.arange(nodes.shape[0]), 2)[1:-1].reshape(
            nodes.shape[0] - 1, 2
        )
    else:
        nodes = np.zeros((0, 3), dtype=np.float64)
        connectivity = np.zeros((0, 2), dtype=np.int32)

    c_el = ufl.Mesh(
        basix.ufl.element(
            "Lagrange", basix.CellType.interval, 1, shape=(nodes.shape[1],)
        )
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


@pytest.fixture(params=one_dim_combinations, scope="module")
def line(request):
    (ghost_mode,) = request.param
    mesh = create_line(ghost_mode, curved=False)
    return mesh


@pytest.fixture(params=one_dim_combinations, scope="module")
def curved_line(request):
    (ghost_mode,) = request.param
    mesh = create_line(ghost_mode, curved=True)
    return mesh


@pytest.fixture(params=three_dim_combinations, scope="module")
def unit_cube(request):
    cell_type, ghost_mode = request.param
    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD, 5, 7, 3, cell_type=cell_type, ghost_mode=ghost_mode
    )
    return mesh


@pytest.fixture(params=three_dim_combinations, scope="module")
def box(request):
    cell_type, ghost_mode = request.param
    N = 15
    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [[-1, -1, -1], [1, 1, 1]],
        [N, N, N],
        cell_type=cell_type,
        ghost_mode=ghost_mode,
    )
    mesh.name = "Box"
    return mesh


@pytest.mark.parametrize("use_petsc", [True, False])
@pytest.mark.parametrize("family", ["DG", "Quadrature", "P"])
@pytest.mark.parametrize("degree", [1, 2, 4])
def test_naive_trace(use_petsc, family, degree, curved_line, unit_cube):
    V = dolfinx.fem.functionspace(unit_cube, ("Lagrange", 1))

    if family == "Quadrature":
        el = basix.ufl.quadrature_element(
            curved_line.basix_cell(), value_shape=(), degree=degree
        )
    else:
        el = (family, degree)
    K_hat = dolfinx.fem.functionspace(curved_line, el)

    def f(x):
        return x[0] - x[1] + 2 * x[2]

    # Interpolate reference solution onto `u`
    uh = dolfinx.fem.Function(V)
    uh.interpolate(f)

    restriction = PointwiseTrace(curved_line)
    bh = dolfinx.fem.Function(K_hat)
    A, _, _ = create_interpolation_matrix(V, K_hat, restriction, use_petsc=use_petsc)

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


@pytest.mark.parametrize("use_petsc", [True, False])
@pytest.mark.parametrize("family", ["P", "DG", "Quadrature"])
@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("case", [1, 2, 3, 4])
@pytest.mark.parametrize("radius", [0.53, lambda x: x[2]])
def test_circle_trace(use_petsc, family, degree, line, box, radius, case):
    V = dolfinx.fem.functionspace(box, ("Lagrange", 3))

    if family == "Quadrature":
        el = basix.ufl.quadrature_element(
            line.basix_cell(), value_shape=(), degree=degree
        )
    else:
        el = (family, degree)
    K_hat = dolfinx.fem.functionspace(line, el)

    def f(x):
        if case == 1:
            return x[2]
        elif case == 2:
            return x[1] * x[1] + x[0] * x[0]
        elif case == 3:
            return x[2] * (x[0] * x[0] + x[1] * x[1])
        elif case == 4:
            return np.sin(0.5 * np.pi * x[2])
        else:
            raise ValueError(f"{case=} is not supported")

    def Pi_f(x):
        if callable(radius):
            radius_val = radius(x)
        else:
            radius_val = np.full_like(x[0], radius)

        if case == 1:
            return x[2]
        elif case == 2:
            return radius_val**2
        elif case == 3:
            return x[2] * radius_val**2
        elif case == 4:
            return np.sin(0.5 * np.pi * x[2])
        else:
            raise ValueError(f"{case=} is not supported")

    # Interpolate reference solution onto `u`
    uh = dolfinx.fem.Function(V)
    uh.interpolate(f)
    restriction = Circle(line, radius, degree=6)

    bh = dolfinx.fem.Function(K_hat)
    use_petsc = False
    A, _, _ = create_interpolation_matrix(V, K_hat, restriction, use_petsc=use_petsc)

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
    bh_ref.interpolate(Pi_f)

    np.testing.assert_allclose(bh.x.array, bh_ref.x.array, atol=1e-6)


@pytest.mark.parametrize("use_petsc", [True, False])
@pytest.mark.parametrize("family", ["DG", "P"])
@pytest.mark.parametrize("degree", [2])
def test_naive_trace_vector(use_petsc, family, degree, curved_line, unit_cube):
    shape = (unit_cube.geometry.dim,)
    V = dolfinx.fem.functionspace(unit_cube, ("Lagrange", 1, shape))

    if family == "Quadrature":
        el = basix.ufl.quadrature_element(
            curved_line.basix_cell(), value_shape=shape, degree=degree
        )
    else:
        el = (family, degree, shape)
    K_hat = dolfinx.fem.functionspace(curved_line, el)

    def f(x):
        return x[0] - x[1] + 2 * x[2], x[2] + 3 * x[1], x[0] + x[1]

    # Interpolate reference solution onto `u`
    uh = dolfinx.fem.Function(V)
    uh.interpolate(f)

    restriction = PointwiseTrace(curved_line)
    bh = dolfinx.fem.Function(K_hat)
    A, _, _ = create_interpolation_matrix(V, K_hat, restriction, use_petsc=use_petsc)

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
