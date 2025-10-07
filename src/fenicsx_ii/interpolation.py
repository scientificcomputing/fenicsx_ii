from __future__ import annotations

from mpi4py import MPI as _MPI

import dolfinx
import numpy as np
from dolfinx.common import IndexMap as _im

from .interpolation_utils import create_extended_indexmap, evaluate_basis_function
from .restriction_operators import ReductionOperator
from .utils import send_dofs_to_other_process, unroll_dofmap


def create_interpolation_matrix(
    V: dolfinx.fem.FunctionSpace,
    K: dolfinx.fem.FunctionSpace,
    red_op: ReductionOperator,
    tol: float = 1.0e-8,
    use_petsc: bool = False,
) -> tuple["PETSc.Mat" | dolfinx.la.MatrixCSR, _im, _im]:  # type: ignore[name-defined] # noqa: F821
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
    assert mesh_to.geometry.dim == 3

    num_average_qp = red_op.num_points

    point_ownership = dolfinx.geometry.determine_point_ownership(
        mesh_from, interpolation_coordinates, padding=tol
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
    ip_owner = point_ownership.dest_owner  # For received data, who sent it

    num_dofs_per_cell_K = K.dofmap.list.shape[1]
    incoming_K_dofs, incoming_K_owners = send_dofs_to_other_process(
        K,
        ip_owner,
        ip_sender,
        np.repeat(np.arange(num_line_cells), num_ip_per_cell * num_average_qp),
    )
    # Create extended index map
    new_imap_K = create_extended_indexmap(
        K.mesh.comm,
        K.dofmap.index_map,
        incoming_K_dofs.flatten(),
        incoming_K_owners,
        123,
    )
    assert (new_imap_K.global_to_local(incoming_K_dofs.flatten()) >= 0).all()

    incoming_V_dofs, incoming_V_owners = send_dofs_to_other_process(
        V, ip_sender, ip_owner, cells_on_proc
    )

    # Create extended index map
    new_imap_V = create_extended_indexmap(
        V.mesh.comm,
        V.dofmap.index_map,
        incoming_V_dofs.flatten(),
        incoming_V_owners,
        321,
    )
    num_dofs_per_cell_V = V.dofmap.list.shape[1]
    new_local_V_dofs = new_imap_V.global_to_local(incoming_V_dofs.flatten())
    assert (new_local_V_dofs >= 0).all()
    new_local_V_dofs = new_local_V_dofs.reshape(-1, num_dofs_per_cell_V)

    # Evaluate basis functions in 3D space
    basis_values_on_V = evaluate_basis_function(V, points_on_proc, cells_on_proc)
    second_dimension = max(V.dofmap.bs, np.prod(V.element.basix_element.value_shape))
    recv_basis_functions = np.empty(
        (len(ip_sender), basis_values_on_V.shape[1], basis_values_on_V.shape[2]),
        dtype=basis_values_on_V.dtype,
    )
    volume_send_to, send_counts_V = np.unique(ip_owner, return_counts=True)
    line_recv_from, recv_counts_V = np.unique(ip_sender, return_counts=True)
    basis_send_counts = (
        send_counts_V * num_dofs_per_cell_V * V.dofmap.bs * second_dimension
    )
    basis_recv_counts = (
        recv_counts_V * num_dofs_per_cell_V * V.dofmap.bs * second_dimension
    )
    send_message = [basis_values_on_V.flatten(), basis_send_counts, _MPI.DOUBLE]
    recv_message = [recv_basis_functions, basis_recv_counts, _MPI.DOUBLE]
    volume_to_line_comm = mesh_from.comm.Create_dist_graph_adjacent(
        line_recv_from.tolist(), volume_send_to.tolist(), reorder=False
    )
    volume_to_line_comm.Neighbor_alltoallv(send_message, recv_message)
    # Free communicators post communication
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
    K_bs = K.dofmap.bs
    dofs_visited[K.dofmap.index_map.size_local * K.dofmap.index_map_bs :] = True
    padded_K_dm = unroll_dofmap(K.dofmap.list, K_bs)
    local_visit = np.full(num_average_qp * K_bs, False, dtype=np.bool_)
    for i in range(num_line_cells):
        local_k_dofs = padded_K_dm[i]
        V_slice = V_in_Q_order[
            num_average_qp * num_dofs_per_cell_K * i : num_average_qp
            * num_dofs_per_cell_K
            * (i + 1)
        ]
        local_v_dofs = unroll_dofmap(new_local_V_dofs[V_slice], V.dofmap.index_map_bs)
        local_v_values = recv_basis_functions[V_slice]
        for j in range(num_dofs_per_cell_K):
            local_dofs = local_v_dofs[j * num_average_qp : (j + 1) * num_average_qp]
            local_values = local_v_values[j * num_average_qp : (j + 1) * num_average_qp]
            average_weights = weights[i * num_dofs_per_cell_K + j]
            # Get visited dofs from previous run
            for b in range(K_bs):
                local_visit[:] = dofs_visited[local_k_dofs[j * K_bs + b]]
                lv = (
                    local_values[:, :, b]
                    * average_weights[:, None]
                    / scales[i * num_dofs_per_cell_K + j]
                )
                for k in range(num_average_qp):
                    # We insert for all average nodes, thus local visit
                    # is only updated next time we pass through the `j` loop
                    lv[k][:] = 0 if local_visit[b * num_average_qp + k] else lv[k]
                    insert_function(
                        A,
                        local_k_dofs[j * K_bs + b : j * K_bs + b + 1],
                        local_dofs[k],
                        lv[k],
                    )
                    dofs_visited[local_k_dofs[j * K_bs + b]] = True
    finalize(A)
    return A, new_imap_K, new_imap_V
