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
