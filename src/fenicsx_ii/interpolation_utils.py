from mpi4py import MPI as _MPI

import dolfinx
import numpy as np
import numpy.typing as npt
import ufl


def evaluate_basis_function(
    V: dolfinx.fem.FunctionSpace,
    points: npt.NDArray[np.inexact],
    cells: npt.NDArray[np.int32],
    batch_size: int = 200,
):
    """Evaluate basis functions in `V` at a set of points.

    Args:
        V: The function space
        points: The points to evaluate at (in physical space)
        cells: The cells. The ith cell corresponds to the ith point.
        batch_size: The number of points/cells to evaluate at the time, to find a balance between
            flooding the expression cache and the memory usage.

    Returns:
        The evaluated basis functions at the given points.
    """
    # Pull owning points back to reference cell
    mesh = V.mesh
    mesh_nodes = mesh.geometry.x
    cmap = mesh.geometry.cmap

    ref_x = np.zeros((len(cells), mesh.topology.dim), dtype=mesh.geometry.x.dtype)
    for i, (point, cell) in enumerate(zip(points, cells)):
        geom_dofs = mesh.geometry.dofmap[cell]
        ref_x[i] = cmap.pull_back(point.reshape(-1, 3), mesh_nodes[geom_dofs])

    # Create expression evaluating a trial function (i.e. just the basis function)
    u = ufl.TestFunction(V)
    bs = V.dofmap.bs
    num_dofs = V.dofmap.dof_layout.num_dofs * bs
    value_size = V.value_size

    basis_values = np.zeros((len(cells), num_dofs), dtype=dolfinx.default_scalar_type)
    if len(cells) > 0:
        # NOTE: Expression lives on only this communicator rank
        # Expression is evaluated for every point in every cell, which means that we
        # need to discard values that are not on the diagonal.
        assert ref_x.shape[0] == len(cells)
        num_batches = len(cells) // batch_size + 1
        for b in range(num_batches):
            x_batch = ref_x[b * batch_size : (b + 1) * batch_size]
            cell_batch = cells[b * batch_size : (b + 1) * batch_size]
            expr = dolfinx.fem.Expression(u, x_batch, comm=_MPI.COMM_SELF)
            all_values = expr.eval(mesh, cell_batch)

            # Diagonalize values (num_cells, num_points, num_dofs, bs) -> (num_cells, num_dofs)
            # or Diagonalize values (num_cells, num_points, value_size, num_dofs) -> (num_cells, num_dofs)
            if bs == 1:
                if value_size > 1:
                    # Values have shape (num_cells, num_points,value_size, num_dofs)
                    for i in range(len(cell_batch)):
                        basis_values[b * batch_size + i] = sum(
                            all_values[i, i, j, :] for j in range(value_size)
                        )
                else:
                    # Values have shape (num_cells, num_points, num_dofs)
                    for i in range(len(cell_batch)):
                        basis_values[b * batch_size + i] = all_values[i, i, :]
            else:
                # Values have shape (num_cells, num_points, bs, num_dofs*bs)
                for i in range(len(cell_batch)):
                    basis_values[b * batch_size + i] = sum(
                        all_values[i, i, j, :] for j in range(bs)
                    )
    else:
        basis_values = np.zeros((0, num_dofs), dtype=dolfinx.default_scalar_type)
    return basis_values


def create_extended_indexmap(
    comm: _MPI.Intracomm,
    imap: dolfinx.common.IndexMap,
    potential_new_dofs: npt.NDArray[np.int64],
    owners: npt.NDArray[np.int32],
    tag: int,
) -> dolfinx.common.IndexMap:
    """
    Create an extended index map that includes new dofs and their owners.
    """
    local_indices = imap.global_to_local(potential_new_dofs)
    new_ghosts = local_indices < 0
    new_owners = owners[new_ghosts]
    extended_ghosts = np.concatenate([imap.ghosts, potential_new_dofs[new_ghosts]])
    extended_owners = np.concatenate([imap.owners, new_owners])

    return dolfinx.common.IndexMap(
        comm,
        imap.size_local,
        extended_ghosts,
        extended_owners,
        tag,
    )
