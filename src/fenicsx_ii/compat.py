"""Layer for small backward compatibility wrappers for DOLFINx"""

import dolfinx


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
