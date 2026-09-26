"""Layer for small backward compatibility wrappers for DOLFINx"""

import dolfinx


def get_cell_permutation_info(mesh: dolfinx.mesh.Mesh):
    """Compute (if needed) and return the packed cell permutation info."""
    if hasattr(mesh.topology, "create_cell_permutations"):
        mesh.topology.create_cell_permutations()
    else:
        mesh.topology.create_entity_permutations()  # type: ignore[call-arg]
    return mesh.topology.get_cell_permutation_info()


def get_cmap(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.CoordinateElement:
    """Get the basix Cmap for the mesh."""
    if hasattr(mesh.geometry, "cmaps"):
        if len(mesh.geometry.cmaps) > 1:
            raise RuntimeError(
                "Mesh has more than one cmap, cannot determine which to use."
            )
        else:
            return mesh.geometry.cmaps[0]
    if callable(mesh.geometry.cmap):
        return mesh.geometry.cmap()
    else:
        return mesh.geometry.cmap


def get_geom_dofmap(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.DofMap:
    """Get the geometry dofmap for the mesh."""
    if hasattr(mesh.geometry, "dofmaps"):
        if len(mesh.geometry.dofmaps) > 1:
            raise RuntimeError(
                "Mesh has more than one geometry dofmap, cannot determine which to use."
            )
        else:
            return mesh.geometry.dofmaps[0]
    else:
        return mesh.geometry.dofmap


def expression_uses_wrong_cell_info() -> bool:
    """Check if {py:meth}`dolfinx.fem.Expression.eval` applies the dof transformations
    of the wrong cells.

    In DOLFINx < 0.11, `Expression.eval(mesh, cells)` looks up the cell
    permutation info with the position of each cell in `cells` rather than the
    cell index, so the result is only correct if `cells[i] == i`.
    """
    major, minor = (int(v) for v in dolfinx.__version__.split(".")[:2])
    return (major, minor) < (0, 11)
