"""Utilities for FEniCSx ii"""

import dolfinx
import numpy as np
import numpy.typing as npt
import ufl


def get_physical_points(
    mesh: dolfinx.mesh.Mesh,
    cells: npt.NDArray[np.int32],
    reference_points: npt.NDArray[np.floating],
) -> npt.NDArray:
    """Given a mesh, a set of cells and some points on the reference cells,
    get the physical coordinates of the points."""
    x_line = ufl.SpatialCoordinate(mesh)
    x_expr = dolfinx.fem.Expression(x_line, reference_points)
    return x_expr.eval(mesh, cells).reshape(-1, mesh.geometry.dim)


def get_cell_normals(
    mesh: dolfinx.mesh.Mesh, cells: npt.NDArray[np.int32]
) -> npt.NDArray[np.floating]:
    """Given a mesh and a set of cells, compute the normals at the cell faces."""
    assert mesh.topology.dim == 1, "Can only compute cell normals for 1D meshes."
    # Compute cell-normal at each interpolation point.
    expr = dolfinx.fem.Expression(ufl.geometry.CellVertices(mesh), np.array([0.0]))
    cell_vertices = expr.eval(mesh, cells)
    normals = cell_vertices[:, 0][:, 0] - cell_vertices[:, 0][:, 1]
    cell_normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    return cell_normals


def unroll_dofmap(dofs: npt.NDArray[np.int32], bs: int) -> npt.NDArray[np.int32]:
    """
    Given a two-dimensional dofmap of size `(num_cells, num_dofs_per_cell)`
    Expand the dofmap by its block size such that the resulting array
    is of size `(num_cells, bs*num_dofs_per_cell)`
    """
    num_cells, num_dofs_per_cell = dofs.shape
    unrolled_dofmap = (
        np.repeat(dofs, bs).reshape(num_cells, num_dofs_per_cell * bs) * bs
    )
    unrolled_dofmap += np.tile(np.arange(bs), num_dofs_per_cell)
    return unrolled_dofmap
