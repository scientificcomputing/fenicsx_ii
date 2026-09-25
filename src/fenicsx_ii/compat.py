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
