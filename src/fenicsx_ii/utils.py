"""Utilities for FEniCSx ii"""

from functools import cached_property

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
    dtype = mesh.geometry.x.dtype
    x_expr = dolfinx.fem.Expression(
        x_line, np.asarray(reference_points, dtype=dtype), dtype=dtype
    )
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
    comm = K.mesh.comm
    assert isinstance(comm, _MPI.Intracomm)
    line_to_volume_comm = comm.Create_dist_graph_adjacent(
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


class PointExchange:
    """Move per-point data between the ranks owning points and the ranks
    evaluating them.

    The owners are the ranks that passed the points to
    {py:func}`dolfinx.geometry.determine_point_ownership`; the evaluating rank of a
    point is the rank holding the cell that contains it. Data is communicated with
    neighbourhood collectives, so every rank of `comm` has to call the same methods
    in the same order, with rows of the same trailing shape and dtype.

    Args:
        comm: The communicator the ownership was determined on.
        ownership: Output of {py:func}`dolfinx.geometry.determine_point_ownership`.
    """

    def __init__(
        self, comm: _MPI.Intracomm, ownership: dolfinx.geometry.PointOwnershipData
    ):
        self._comm = comm
        self._ownership = ownership

    @cached_property
    def _received(self) -> npt.NDArray[np.int64]:
        """Local point index of each forward-received row. Rows arrive grouped by
        evaluating rank, each group in local point order."""
        src_owner = np.asarray(self._ownership.src_owner)
        found = np.flatnonzero(src_owner >= 0)
        return found[np.argsort(src_owner[found], stable=True)]

    @cached_property
    def _evaluators(self) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]:
        """Ranks evaluating this rank's points, and the number of points each."""
        src_owner = np.asarray(self._ownership.src_owner)
        return np.unique(src_owner[src_owner >= 0], return_counts=True)

    @cached_property
    def _owners(self) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]:
        """Ranks owning the evaluated points, and the number of points each."""
        dest_owner = np.asarray(self._ownership.dest_owner)
        assert np.all(np.diff(dest_owner) >= 0), (
            "Evaluated points are expected grouped by owning rank"
        )
        return np.unique(dest_owner, return_counts=True)

    @property
    def ownership(self) -> dolfinx.geometry.PointOwnershipData:
        """The point ownership the exchange was built from."""
        return self._ownership

    @property
    def num_points(self) -> int:
        """Number of points owned by this rank."""
        return len(self._ownership.src_owner)

    @property
    def evaluated_points(self) -> npt.NDArray[np.floating]:
        """Points this rank evaluates, shape `(num_evaluated, 3)`."""
        return np.asarray(self._ownership.dest_points).reshape(-1, 3)

    @property
    def evaluated_cells(self) -> npt.NDArray[np.int32]:
        """Local cell holding each evaluated point."""
        return np.asarray(self._ownership.dest_cells, dtype=np.int32)

    def _exchange(
        self,
        data: npt.NDArray,
        sources: tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]],
        destinations: tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]],
    ) -> npt.NDArray:
        """Send rows of `data`, grouped by destination rank, and receive rows
        grouped by source rank. `sources` and `destinations` are (ranks, counts)."""
        source_ranks, recv_counts = sources
        destination_ranks, send_counts = destinations
        data = np.ascontiguousarray(data)
        row_shape = data.shape[1:]
        width = int(np.prod(row_shape, dtype=int))
        received = np.zeros((int(np.sum(recv_counts)), *row_shape), dtype=data.dtype)
        graph_comm = self._comm.Create_dist_graph_adjacent(
            source_ranks.tolist(), destination_ranks.tolist(), reorder=False
        )
        try:
            graph_comm.Neighbor_alltoallv(
                # mpi4py infers the (predefined) MPI datatype from the buffers
                [data.reshape(-1), send_counts * width],
                [received.reshape(-1), recv_counts * width],
            )
        finally:
            graph_comm.Free()
        return received

    def forward(self, rows_at_evaluated: npt.NDArray) -> npt.NDArray:
        """Send rows at the evaluated points to the owners of the points.

        Args:
            rows_at_evaluated: One row per evaluated point, shape
                `(len(evaluated_cells), ...)`.

        Returns:
            One row per owned point, shape `(num_points, ...)`. Rows of points not
            found in the mesh are zero.
        """
        if len(rows_at_evaluated) != len(self.evaluated_cells):
            raise ValueError(
                f"Expected {len(self.evaluated_cells)} rows, got "
                f"{len(rows_at_evaluated)}."
            )
        received = self._exchange(rows_at_evaluated, self._evaluators, self._owners)
        values = np.zeros(
            (self.num_points, *received.shape[1:]), dtype=rows_at_evaluated.dtype
        )
        values[self._received] = received
        return values

    def reverse(self, rows_at_points: npt.NDArray) -> npt.NDArray:
        """Send rows at the owned points to the ranks evaluating them.

        Args:
            rows_at_points: One row per owned point, shape `(num_points, ...)`.
                Rows of points not found in the mesh are ignored.

        Returns:
            One row per evaluated point, shape `(len(evaluated_cells), ...)`.
        """
        if len(rows_at_points) != self.num_points:
            raise ValueError(
                f"Expected {self.num_points} rows, got {len(rows_at_points)}."
            )
        return self._exchange(
            np.asarray(rows_at_points)[self._received], self._owners, self._evaluators
        )
