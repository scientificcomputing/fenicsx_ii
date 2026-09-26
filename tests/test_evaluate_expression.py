from functools import partial

from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import numpy.typing as npt
import pytest
import ufl

from fenicsx_ii import PointExchange, evaluate_expression
from fenicsx_ii.utils import get_physical_points


def deform(x: np.ndarray) -> np.ndarray:
    """A smooth non-affine map of the unit square."""
    y = x.copy()
    y[:, 0] += 0.1 * np.sin(np.pi * x[:, 1])
    y[:, 1] += 0.1 * np.sin(np.pi * x[:, 0])
    return y


def create_curved_mesh(
    N: int, degree: int, dtype: npt.DTypeLike = np.float64
) -> dolfinx.mesh.Mesh:
    """Unit square of triangles with a Lagrange geometry of `degree`, deformed so
    that the cells are curved."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, dtype=dtype)
    cmap = dolfinx.fem.coordinate_element(
        dolfinx.mesh.CellType.triangle,
        degree,
        int(basix.LagrangeVariant.gll_warped),
        dtype=dtype,
    )
    mesh = dolfinx.fem.interpolate_geometry(mesh, cmap)
    mesh.geometry.x[:, :2] = deform(mesh.geometry.x[:, :2])
    return mesh


def create_bilinear_quad_mesh(
    N: int, dtype: npt.DTypeLike = np.float64
) -> dolfinx.mesh.Mesh:
    """Unit square of quadrilaterals, deformed so that no cell is a parallelogram."""
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, N, N, dolfinx.mesh.CellType.quadrilateral, dtype=dtype
    )
    mesh.geometry.x[:, :2] = deform(mesh.geometry.x[:, :2])
    return mesh


dolfinx_version = tuple(int(v) for v in dolfinx.__version__.split(".")[:2])
mesh_types = [
    *(
        pytest.param(
            partial(create_curved_mesh, degree=degree),
            id=f"curved-P{degree}",
            marks=pytest.mark.skipif(
                dolfinx_version < (0, 11),
                reason="dolfinx.fem.interpolate_geometry requires DOLFINx >= 0.11",
            ),
        )
        for degree in (2, 3, 4)
    ),
    pytest.param(create_bilinear_quad_mesh, id="bilinear-quad"),
]


def u_exact(x):
    return np.sin(2 * x[0]) * np.cos(3 * x[1]) + x[0] ** 2 * x[1]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("create_mesh", mesh_types)
def test_derivatives(create_mesh, dtype):
    mesh = create_mesh(4, dtype=dtype)
    eps = np.finfo(mesh.geometry.x.dtype).eps
    # Central differences: h balances truncation O(h^2) against roundoff O(eps / h),
    # leaving an error of O(eps^(2/3))
    h = eps ** (1 / 3)
    fd_tol = 1e3 * eps ** (2 / 3)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 3))
    u = dolfinx.fem.Function(V, dtype=dtype)
    u.interpolate(u_exact)

    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cells = np.arange(min(num_cells, 8), dtype=np.int32)
    reference_point = np.array([[0.2, 0.3]])
    x = np.zeros((len(cells), 3), dtype=dtype)
    x[:, :2] = get_physical_points(mesh, cells, reference_point)

    # Values against the pull-back in Function.eval
    values = evaluate_expression(u, mesh, x, cells)
    assert values.shape == (len(cells),)
    np.testing.assert_allclose(
        values, u.eval(x, cells).reshape(-1), rtol=1e3 * eps, atol=1e3 * eps
    )

    # Gradients against central differences of Function.eval, and Hessians against
    # central differences of the gradients. Shifted points stay in their cells.
    shifts = [np.zeros(3)] + [s * h * np.eye(3)[k] for k in range(2) for s in (1, -1)]
    x_shifted = np.vstack([x + shift for shift in shifts])
    cells_shifted = np.tile(cells, len(shifts))
    grads = evaluate_expression(ufl.grad(u), mesh, x_shifted, cells_shifted)
    grads = grads.reshape(len(shifts), len(cells), 2)
    u_shifted = u.eval(x_shifted, cells_shifted).reshape(len(shifts), len(cells))
    for k in range(2):
        fd_grad = (u_shifted[1 + 2 * k] - u_shifted[2 + 2 * k]) / (2 * h)
        np.testing.assert_allclose(grads[0, :, k], fd_grad, rtol=fd_tol, atol=fd_tol)

    hessians = evaluate_expression(ufl.grad(ufl.grad(u)), mesh, x, cells)
    assert hessians.shape == (len(cells), 2, 2)
    for k in range(2):
        fd_hessian = (grads[1 + 2 * k] - grads[2 + 2 * k]) / (2 * h)
        np.testing.assert_allclose(
            hessians[:, :, k], fd_hessian, rtol=fd_tol, atol=fd_tol
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_forward_reverse_adjoint(dtype):
    """<w, A c> = <A^T w, c> for A c = grad(u_c) at points, with A^T built from the
    basis values of an Argument expression."""
    comm = MPI.COMM_WORLD
    mesh = create_bilinear_quad_mesh(5, dtype=dtype)
    eps = np.finfo(mesh.geometry.x.dtype).eps
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
    rng = np.random.default_rng(17 + comm.rank)
    points = np.zeros((25, 3), dtype=dtype)
    # Some points fall outside the deformed square
    points[:, :2] = rng.uniform(-0.1, 1.1, size=(25, 2))
    ownership = dolfinx.geometry.determine_point_ownership(mesh, points, 1e-10)
    exchange = PointExchange(comm, ownership)

    # Exchange alone
    a = rng.normal(size=(len(exchange.evaluated_cells), 2)).astype(dtype)
    b = rng.normal(size=(exchange.num_points, 2)).astype(dtype)
    lhs = comm.allreduce(np.sum(exchange.forward(a) * b), op=MPI.SUM)
    rhs = comm.allreduce(np.sum(a * exchange.reverse(b)), op=MPI.SUM)
    assert np.isclose(lhs, rhs, rtol=1e2 * eps, atol=1e2 * eps)

    # Forward: coefficients to gradients at the points
    c = dolfinx.fem.Function(V, dtype=dtype)
    c.x.array[:] = rng.normal(size=c.x.array.size)
    c.x.scatter_forward()
    grad_at_points = exchange.forward(
        evaluate_expression(
            ufl.grad(c),
            mesh,
            exchange.evaluated_points,
            exchange.evaluated_cells,
        )
    )
    assert grad_at_points.shape == (exchange.num_points, 2)
    w = rng.normal(size=(exchange.num_points, 2)).astype(dtype)
    lhs = comm.allreduce(np.sum(w * grad_at_points), op=MPI.SUM)

    # Transpose: weights at the points to the dofs of V
    basis = evaluate_expression(
        ufl.grad(ufl.TestFunction(V)),
        mesh,
        exchange.evaluated_points,
        exchange.evaluated_cells,
        dtype=dtype,
    )
    local = np.einsum("pv,pvd->pd", exchange.reverse(w), basis)
    At_w = dolfinx.fem.Function(V, dtype=dtype)
    np.add.at(At_w.x.array, V.dofmap.list[exchange.evaluated_cells], local)
    At_w.x.scatter_reverse(dolfinx.la.InsertMode.add)
    size_local = V.dofmap.index_map.size_local
    rhs = comm.allreduce(
        np.sum(At_w.x.array[:size_local] * c.x.array[:size_local]), op=MPI.SUM
    )
    assert np.isclose(lhs, rhs, rtol=1e3 * eps, atol=1e3 * eps)


def test_points_outside_mesh():
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, 4, 4)
    eps = np.finfo(mesh.geometry.x.dtype).eps
    points = np.array([[0.3, 0.2 + 0.1 * comm.rank / comm.size, 0.0], [5.0, 5.0, 0.0]])
    ownership = dolfinx.geometry.determine_point_ownership(mesh, points, 1e-10)
    exchange = PointExchange(comm, ownership)
    x = ufl.SpatialCoordinate(mesh)
    values = exchange.forward(
        evaluate_expression(
            ufl.as_vector((x[0] + 2, x[1])),
            mesh,
            exchange.evaluated_points,
            exchange.evaluated_cells,
        )
    )
    np.testing.assert_allclose(values[0], points[0, :2] + [2, 0], rtol=1e2 * eps)
    np.testing.assert_array_equal(values[1], 0)
    assert exchange.ownership.src_owner[1] == -1
    # The reverse direction drops the unfound point
    back = exchange.reverse(np.array([[1.0], [7.0]]))
    assert comm.allreduce(np.sum(back), op=MPI.SUM) == comm.size
