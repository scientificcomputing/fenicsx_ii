from __future__ import annotations

from mpi4py import MPI as _MPI

import dolfinx
import numpy as np

from .interpolation_utils import create_extended_indexmap, evaluate_basis_function
from .restriction_operators import ReductionOperator


def create_interpolation_matrix(
    V: dolfinx.fem.FunctionSpace,
    K: dolfinx.fem.FunctionSpace,
    red_op: ReductionOperator,
    tol: float = 1.0e-8,
    use_petsc: bool = False,
) -> "PETSc.Mat" | dolfinx.la.MatrixCSR:  # type: ignore[name-defined] # noqa: F821
    """
    Create an interpolation matrix from `V` to `K` with a specific reduction operator
    applied to the interpolation points of `K`.

    Args:
        V: The function space to interpolate from.
        K: The function space to interpolate to.
        red_op: Reduction operator for each interpolation point on `K`.
        tol: Tolerance for determining point ownership across processes.
        use_petsc: Flag to indicate whether to use PETSc for the matrix.

    Returns:
        The interpolation matrix, as a DOLFINx built in matrix or a PETSc matrix.
        The accumulation of contributions from all processes has been accounted before
        returning the matrix.
    """
    mesh_to = K.mesh
    mesh_from = V.mesh
    num_line_cells = mesh_to.topology.index_map(mesh_to.topology.dim).size_local
    reference_interpolation_points = K.element.interpolation_points

    num_ip_per_cell = reference_interpolation_points.shape[0]
    line_cells = np.arange(num_line_cells, dtype=np.int32)

    quadrature_rule = red_op.compute_quadrature(
        line_cells, reference_interpolation_points
    )
    quad_points = quadrature_rule.points

    interpolation_coordinates = quad_points.reshape(-1, mesh_to.geometry.dim)
    num_average_qp = red_op.num_points

    point_ownership = dolfinx.cpp.geometry.determine_point_ownership(
        mesh_from._cpp_object, interpolation_coordinates, tol
    )
    cells_on_proc = (
        point_ownership.dest_cells
    )  # Cells in 3D domain that has interpolation points
    points_on_proc = (
        point_ownership.dest_points
    )  # Interpolation points relating to `cells_on_proc`
    ip_sender = (
        point_ownership.src_owner
    )  # For IP in 1D grid, what process has taken ownership
    assert (ip_sender >= 0).all()
    ip_owner = point_ownership.dest_owners  # For received data, who sent it

    num_dofs_per_cell_K = K.dofmap.list.shape[1]
    K_dofs_to_send = np.empty(
        (num_line_cells * num_ip_per_cell * num_average_qp, num_dofs_per_cell_K),
        dtype=np.int32,
    )
    sort_data_per_proc = np.argsort(
        ip_sender, stable=True
    )  # Sort data to send per process that has taken ownership

    # Pack global DOFs of K to send to V for insertion on extended index map.
    for i in range(interpolation_coordinates.shape[0]):
        local_cell_index = i // (num_ip_per_cell * num_average_qp)
        K_dofs_to_send[i] = K.dofmap.list[local_cell_index]

    K_dofs_to_send = K_dofs_to_send[sort_data_per_proc]
    K_global_dofs = K.dofmap.index_map.local_to_global(K_dofs_to_send.flatten())
    # Send global DOF numbering from K to V for sparsity pattern insertion
    # We also send who owns the global dofs
    line_sends_to, send_counts_K = np.unique(ip_sender, return_counts=True)
    volume_recv_from, recv_counts_K = np.unique(ip_owner, return_counts=True)
    line_to_volume_comm = mesh_to.comm.Create_dist_graph_adjacent(
        volume_recv_from.tolist(), line_sends_to.tolist(), reorder=False
    )

    incoming_K_dofs = np.full(
        (sum(recv_counts_K), num_dofs_per_cell_K), -1, dtype=np.int64
    )
    incoming_offsets_K = np.zeros(len(recv_counts_K) + 1, dtype=np.intc)
    incoming_offsets_K[1:] = np.cumsum(recv_counts_K) * num_dofs_per_cell_K
    send_counts_K *= num_dofs_per_cell_K
    recv_counts_K *= num_dofs_per_cell_K
    outgoing_offsets_K = np.zeros(len(send_counts_K) + 1, dtype=np.intc)
    outgoing_offsets_K[1:] = np.cumsum(send_counts_K) * num_dofs_per_cell_K
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

    # Create extended index map
    new_imap_K = create_extended_indexmap(
        K.mesh.comm,
        K.dofmap.index_map,
        incoming_K_dofs.flatten(),
        incoming_K_owners,
        123,
    )
    assert (new_imap_K.global_to_local(incoming_K_dofs.flatten()) >= 0).all()

    # Send global DOF numbering from V to K for sparsity pattern insertion
    line_to_volume_comm = mesh_to.comm.Create_dist_graph_adjacent(
        volume_recv_from.tolist(), line_sends_to.tolist(), reorder=False
    )
    num_dofs_per_cell_V = V.dofmap.list.shape[1]

    send_counts_V = (
        recv_counts_K / (num_dofs_per_cell_K) * num_dofs_per_cell_V
    ).astype(np.intc)
    recv_counts_V = (
        send_counts_K / (num_dofs_per_cell_K) * num_dofs_per_cell_V
    ).astype(np.intc)
    volume_to_line_comm = mesh_from.comm.Create_dist_graph_adjacent(
        line_sends_to.tolist(), volume_recv_from.tolist(), reorder=False
    )

    V_dofs_to_send = np.empty(
        (points_on_proc.shape[0], num_dofs_per_cell_V), dtype=np.int32
    )

    # Pack V_dofs to send
    for i, cell_V in enumerate(cells_on_proc):
        V_dofs_to_send[i] = V.dofmap.list[cell_V]
    V_global_dofs = V.dofmap.index_map.local_to_global(V_dofs_to_send.flatten())
    # Check that point ownership has sorted input already
    sorted_V = np.argsort(ip_owner, stable=True)
    assert np.allclose(sorted_V, np.arange(len(sorted_V)))

    incoming_V_dofs = np.full(sum(recv_counts_V), -1, dtype=np.int64)
    incoming_offsets_V = np.zeros(len(recv_counts_V) + 1, dtype=np.intc)
    incoming_offsets_V[1:] = np.cumsum(recv_counts_V) * num_dofs_per_cell_V
    outgoing_offsets_V = np.zeros(len(send_counts_V) + 1, dtype=np.intc)
    outgoing_offsets_V[1:] = np.cumsum(send_counts_V) * num_dofs_per_cell_V
    send_message = [V_global_dofs, send_counts_V, _MPI.INT64_T]
    recv_message = [incoming_V_dofs, recv_counts_V, _MPI.INT64_T]
    volume_to_line_comm.Neighbor_alltoallv(send_message, recv_message)

    # Send ownership info
    num_V_dofs_local = V.dofmap.index_map.size_local
    is_V_ghost = V_dofs_to_send.flatten() < num_V_dofs_local

    send_dof_owners_V = np.empty(sum(send_counts_V), dtype=np.int32)
    send_dof_owners_V[is_V_ghost] = V.mesh.comm.rank
    local_ghost_index = V_dofs_to_send.flatten() - num_V_dofs_local
    send_dof_owners_V[~is_V_ghost] = V.dofmap.index_map.owners[
        local_ghost_index[~is_V_ghost]
    ]

    send_message = [send_dof_owners_V, send_counts_V, _MPI.INT32_T]
    incoming_V_owners = np.empty(sum(recv_counts_V), dtype=np.int32)
    recv_message = [incoming_V_owners, recv_counts_V, _MPI.INT32_T]
    volume_to_line_comm.Neighbor_alltoallv(send_message, recv_message)

    # Create extended index map
    new_imap_V = create_extended_indexmap(
        V.mesh.comm,
        V.dofmap.index_map,
        incoming_V_dofs.flatten(),
        incoming_V_owners,
        321,
    )
    new_local_V_dofs = new_imap_V.global_to_local(incoming_V_dofs.flatten())
    assert (new_local_V_dofs >= 0).all()
    new_local_V_dofs = new_local_V_dofs.reshape(-1, num_dofs_per_cell_V)

    # Evaluate basis functions in 3D space
    basis_values_on_V = evaluate_basis_function(V, points_on_proc, cells_on_proc)
    recv_basis_functions = np.empty((len(ip_sender), num_dofs_per_cell_V * V.dofmap.bs))
    basis_send_counts = send_counts_V * V.dofmap.bs
    basis_recv_counts = recv_counts_V * V.dofmap.bs
    send_message = [basis_values_on_V, basis_send_counts, _MPI.DOUBLE]
    recv_messhage = [recv_basis_functions, basis_recv_counts, _MPI.DOUBLE]
    volume_to_line_comm.Neighbor_alltoallv(send_message, recv_messhage)

    # Free communicators post communication
    line_to_volume_comm.Free()
    volume_to_line_comm.Free()

    # Create sparsity pattern for the interpolation matrix
    sp = dolfinx.cpp.la.SparsityPattern(
        K.mesh.comm,
        [new_imap_K, new_imap_V],
        [K.dofmap.index_map_bs, V.dofmap.index_map_bs],
    )
    insert_position = np.argsort(ip_sender, stable=True)
    V_in_Q_order = np.argsort(insert_position, stable=True)

    assert K.element.interpolation_ident
    assert not K.element.needs_dof_transformations
    for i in range(num_line_cells):
        local_k_dofs = K.dofmap.list[i]
        local_v_dofs = new_local_V_dofs[
            V_in_Q_order[
                num_dofs_per_cell_K * num_average_qp * i : num_dofs_per_cell_K
                * num_average_qp
                * (i + 1)
            ]
        ]
        for j in range(num_dofs_per_cell_K):
            for k in range(num_average_qp):
                ldofs = local_v_dofs[j * num_average_qp + k]
                sp.insert(local_k_dofs[j : j + 1], ldofs)
    sp.finalize()

    # Create distributed petsc matrix and insert basis function values
    weights = quadrature_rule.weights
    scales = quadrature_rule.scales

    if use_petsc:
        assert dolfinx.has_petsc, (
            "DOLFINx has to be installed with PETSc support to use PETSc matrices"
        )
        from petsc4py import PETSc

        A = dolfinx.cpp.la.petsc.create_matrix(K.mesh.comm, sp)

        def insert_function(A, rows, columns, values):
            A.setValuesLocal(rows, columns, values, addv=PETSc.InsertMode.ADD)

        def finalize(A):
            A.assemble()
    else:
        A = dolfinx.la.matrix_csr(
            sp,
            block_mode=dolfinx.la.BlockMode.compact,
            dtype=dolfinx.default_scalar_type,
        )

        def insert_function(A, rows, columns, values):
            A.add(values, rows, columns)

        def finalize(A):
            A.scatter_reverse()

    # Keep track of dofs that are local to process, to ensure that we only insert for
    # - Once per degree of freedom
    # - Only on the process that owns the degree of freedom
    dofs_visited = np.full(
        (K.dofmap.index_map.size_local + K.dofmap.index_map.num_ghosts)
        * K.dofmap.index_map_bs,
        False,
        dtype=np.bool_,
    )
    dofs_visited[K.dofmap.index_map.size_local * K.dofmap.index_map_bs :] = True
    local_visit = np.full(num_average_qp, False, dtype=np.bool_)
    for i in range(num_line_cells):
        local_k_dofs = K.dofmap.list[i]
        V_slice = V_in_Q_order[
            num_average_qp * num_dofs_per_cell_K * i : num_average_qp
            * num_dofs_per_cell_K
            * (i + 1)
        ]
        local_v_dofs = new_local_V_dofs[V_slice]
        local_v_values = recv_basis_functions[V_slice]
        for j in range(num_dofs_per_cell_K):
            local_dofs = local_v_dofs[j * num_average_qp : (j + 1) * num_average_qp]
            local_values = local_v_values[j * num_average_qp : (j + 1) * num_average_qp]
            average_weights = weights[i * num_dofs_per_cell_K + j]
            lv = (
                local_values
                * average_weights[:, None]
                / scales[i * num_dofs_per_cell_K + j]
            )
            # Get visited dofs from previous run
            local_visit[:] = dofs_visited[local_k_dofs[j]]
            for k in range(num_average_qp):
                # We insert for all average nodes, thus local visit
                # is only updated next time we pass through the `j` loop
                lv[k][:] = 0 if local_visit[k] else lv[k]
                insert_function(A, local_k_dofs[j : j + 1], local_dofs[k], lv[k])
                dofs_visited[local_k_dofs[j]] = True
    finalize(A)
    return A
