"""Utilities for FEniCSx ii"""

from mpi4py import MPI as _MPI

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


def send_dofs_to_other_process(
    K: dolfinx.fem.FunctionSpace,
    dest_processes: npt.NDArray[np.int32],
    recv_processes: npt.NDArray[np.int32],
    cells: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    """Send slices of a dofmap, i.e. `dofmap[cells]` to the `recv_processes`
        and corresponding owner indices for each global dof.

    Args:
        K: Function space that holds the dofmap
        dest_processes: List of which processes the current rank gets data from
        recv_processes: List of processes which each entry in `cells` is sent to
        cells: List of cell indices (local to process) to extract dofmap from

    Returns:
        The dofmap (in global indices) that was sent to the process, and the
        corresponding dof owner for each entry in the map.
    """
    sort_data_per_proc = np.argsort(
        recv_processes, stable=True
    )  # Sort data to send per process that has taken ownership
    num_dofs_per_cell = K.dofmap.list.shape[1]
    # Pack global DOFs of K to send to V for insertion on extended index map.
    K_dofs_to_send = np.empty((len(cells), num_dofs_per_cell), dtype=np.int32)
    for i, cell in enumerate(cells):
        K_dofs_to_send[i] = K.dofmap.list[cell]
    K_dofs_to_send = K_dofs_to_send[sort_data_per_proc]
    K_global_dofs = K.dofmap.index_map.local_to_global(K_dofs_to_send.flatten())

    # Send global DOF numbering from K to V for sparsity pattern insertion
    # We also send who owns the global dofs
    line_sends_to, send_counts_K = np.unique(recv_processes, return_counts=True)
    volume_recv_from, recv_counts_K = np.unique(dest_processes, return_counts=True)
    line_to_volume_comm = K.mesh.comm.Create_dist_graph_adjacent(
        volume_recv_from.tolist(), line_sends_to.tolist(), reorder=False
    )

    incoming_K_dofs = np.full(
        (sum(recv_counts_K), num_dofs_per_cell), -1, dtype=np.int64
    )
    incoming_offsets_K = np.zeros(len(recv_counts_K) + 1, dtype=np.intc)
    incoming_offsets_K[1:] = np.cumsum(recv_counts_K) * num_dofs_per_cell
    send_counts_K *= num_dofs_per_cell
    recv_counts_K *= num_dofs_per_cell
    outgoing_offsets_K = np.zeros(len(send_counts_K) + 1, dtype=np.intc)
    outgoing_offsets_K[1:] = np.cumsum(send_counts_K) * num_dofs_per_cell
    send_message = [K_global_dofs, send_counts_K, _MPI.INT64_T]
    recv_message = [incoming_K_dofs, recv_counts_K, _MPI.INT64_T]
    line_to_volume_comm.Neighbor_alltoallv(send_message, recv_message)

    # Send ownership info
    num_K_dofs_local = K.dofmap.index_map.size_local
    is_K_ghost = K_dofs_to_send.flatten() < num_K_dofs_local
    send_dof_owners_K = np.empty(sum(send_counts_K), dtype=np.int32)
    send_dof_owners_K[is_K_ghost] = K.mesh.comm.rank
    local_ghost_index = K_dofs_to_send.flatten() - num_K_dofs_local
    send_dof_owners_K[~is_K_ghost] = K.dofmap.index_map.owners[
        local_ghost_index[~is_K_ghost]
    ]
    send_message = [send_dof_owners_K, send_counts_K, _MPI.INT32_T]
    incoming_K_owners = np.empty(sum(recv_counts_K), dtype=np.int32)
    recv_message = [incoming_K_owners, recv_counts_K, _MPI.INT32_T]
    line_to_volume_comm.Neighbor_alltoallv(send_message, recv_message)

    if len(incoming_K_dofs) == 0:
        assert np.all(incoming_K_dofs > 0)
    line_to_volume_comm.Free()
    return incoming_K_dofs, incoming_K_owners
