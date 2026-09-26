from __future__ import annotations

from mpi4py import MPI as _MPI

import basix
import dolfinx
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx.common import IndexMap as _im

from .compat import get_cell_permutation_info
from .interpolation_utils import create_extended_indexmap, evaluate_basis_function
from .restriction_operators import ReductionOperator
from .utils import PointExchange, send_dofs_to_other_process, unroll_dofmap


def create_interpolation_matrix(
    V: dolfinx.fem.FunctionSpace,
    K: dolfinx.fem.FunctionSpace,
    red_op: ReductionOperator,
    tol: float = 1.0e-8,
    use_petsc: bool = False,
    cells_K: npt.NDArray[np.int32] | None = None,
    complex_dtype: bool = False,
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
        complex_dtype: Create a matrix with complex dtype

    Returns:
        The interpolation matrix, as a DOLFINx built in matrix or a PETSc matrix.
        The accumulation of contributions from all processes has been accounted before
        returning the matrix.
    """
    mesh_to = K.mesh
    mesh_from = V.mesh
    reference_interpolation_points = K.element.interpolation_points

    num_ip_per_cell = reference_interpolation_points.shape[0]

    if cells_K is None:
        num_cells_K = mesh_to.topology.index_map(mesh_to.topology.dim).size_local
        cells_K = np.arange(num_cells_K, dtype=np.int32)
    else:
        num_cells_K = len(cells_K)

    quadrature_rule = red_op.compute_quadrature(cells_K, reference_interpolation_points)
    quad_points = quadrature_rule.points

    # Pad interpolation coordinates for 3D
    _interpolation_coordinates = quad_points.reshape(-1, mesh_to.geometry.dim)
    interpolation_coordinates = np.zeros(
        (_interpolation_coordinates.shape[0], 3), dtype=_interpolation_coordinates.dtype
    )
    interpolation_coordinates[:, : mesh_to.geometry.dim] = _interpolation_coordinates
    del _interpolation_coordinates

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
        np.repeat(cells_K, num_ip_per_cell * num_average_qp),
    )
    # Create extended index map
    comm = mesh_to.comm
    assert isinstance(comm, _MPI.Intracomm)
    new_imap_K = create_extended_indexmap(
        comm,
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
    comm_from = mesh_from.comm
    assert isinstance(comm_from, _MPI.Intracomm)
    new_imap_V = create_extended_indexmap(
        comm_from,
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
    # Basis values at each interpolation point of K, in point order
    exchange = PointExchange(comm_from, point_ownership)
    basis_values = exchange.forward(basis_values_on_V)

    # Create sparsity pattern for the interpolation matrix
    if hasattr(dolfinx.la, "sparsity_pattern"):
        sp = dolfinx.la.sparsity_pattern(
            K.mesh.comm,
            [new_imap_K, new_imap_V],
            [K.dofmap.index_map_bs, V.dofmap.index_map_bs],
        )
    else:
        sp = dolfinx.cpp.la.SparsityPattern(  # type: ignore
            K.mesh.comm,
            [new_imap_K, new_imap_V],  # type: ignore
            [K.dofmap.index_map_bs, V.dofmap.index_map_bs],
        )
    insert_position = np.argsort(ip_sender, stable=True)
    V_in_Q_order = np.argsort(insert_position, stable=True)

    # Point-evaluation targets: interpolation point `j` is dof `j` (per block).
    # Otherwise (e.g. Piola-mapped or moment-based elements), each dof of a cell
    # depends on the source values at all interpolation points of the cell.
    point_evaluation = (
        K.element.interpolation_ident and not K.element.needs_dof_transformations
    )
    num_points_per_cell = num_ip_per_cell * num_average_qp
    for i, cell_K in enumerate(cells_K):
        local_k_dofs = K.dofmap.list[cell_K]
        local_v_dofs = new_local_V_dofs[
            V_in_Q_order[num_points_per_cell * i : num_points_per_cell * (i + 1)]
        ]
        if point_evaluation:
            for j in range(num_dofs_per_cell_K):
                for k in range(num_average_qp):
                    ldofs = local_v_dofs[j * num_average_qp + k]
                    sp.insert(local_k_dofs[j : j + 1], ldofs)
        else:
            sp.insert(local_k_dofs, np.unique(local_v_dofs))
    sp.finalize()

    # Create distributed petsc matrix and insert basis function values
    weights = quadrature_rule.weights
    scales = quadrature_rule.scales
    A: "PETSc.Mat" | dolfinx.la.MatrixCSR  # type: ignore[name-defined]
    if use_petsc:
        assert dolfinx.has_petsc, (
            "DOLFINx has to be installed with PETSc support to use PETSc matrices"
        )
        from petsc4py import PETSc

        if (
            np.issubdtype(PETSc.ScalarType, np.complexfloating) != complex_dtype  # type: ignore
        ):
            raise RuntimeError(
                "PETSc has been compiled with dtype {PETSc.ScalarType}, ",
                "requested complex={complex_dtype}.",
            )

        if hasattr(sp, "_cpp_object"):
            sp_cpp = sp._cpp_object
        else:
            sp_cpp = sp  # type: ignore
        A = dolfinx.cpp.la.petsc.create_matrix(K.mesh.comm, sp_cpp, None)

        def insert_function(A, rows, columns, values):
            A.setValuesLocal(rows, columns, values, addv=PETSc.InsertMode.ADD)

        def finalize(A):
            A.assemble()
    else:
        dtype = np.dtype(V.element.dtype)
        if complex_dtype:
            dtype = np.result_type(dtype, 1j)
        A = dolfinx.la.matrix_csr(
            sp,
            block_mode=dolfinx.la.BlockMode.compact,
            dtype=dtype,
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
    padded_K_dm = unroll_dofmap(K.dofmap.list[cells_K], K_bs)
    if not point_evaluation:
        _insert_interpolation_operator_rows(
            A,
            insert_function,
            K,
            cells_K,
            padded_K_dm,
            local_V_dofs=new_local_V_dofs[V_in_Q_order],
            V_bs=V.dofmap.index_map_bs,
            V_basis_values=basis_values,
            weights=weights,
            scales=scales,
            dofs_visited=dofs_visited,
        )
        finalize(A)
        return A, new_imap_K, new_imap_V

    local_visit = np.full(num_average_qp * K_bs, False, dtype=np.bool_)
    for i in range(num_cells_K):
        local_k_dofs = padded_K_dm[i]
        point_slice = slice(
            num_average_qp * num_dofs_per_cell_K * i,
            num_average_qp * num_dofs_per_cell_K * (i + 1),
        )
        V_slice = V_in_Q_order[point_slice]
        local_v_dofs = unroll_dofmap(new_local_V_dofs[V_slice], V.dofmap.index_map_bs)
        local_v_values = basis_values[point_slice]
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


def _insert_interpolation_operator_rows(
    A,
    insert_function,
    K: dolfinx.fem.FunctionSpace,
    cells_K: npt.NDArray[np.int32],
    padded_K_dm: npt.NDArray[np.int32],
    local_V_dofs: npt.NDArray[np.int32],
    V_bs: int,
    V_basis_values: npt.NDArray[np.inexact],
    weights: npt.NDArray[np.floating],
    scales: npt.NDArray[np.floating],
    dofs_visited: npt.NDArray[np.bool_],
):
    """Insert the rows of the interpolation matrix for a target space `K` whose
    dofs are not point evaluations.

    Per target cell, the dofs are computed as in DOLFINx's interpolation,
    `T M P^-1(v)`, where `v` are the (averaged) source basis values at the
    interpolation points of `K`, `P^-1` the pull-back of `K`, `M` the basix
    interpolation matrix and `T` the dof transformation of the cell. The resulting
    dense `(num_dofs_K, num_source_dofs)` block is inserted in one call.

    Args:
        A: Matrix to insert into.
        insert_function: Adds `values` to `A` at `(rows, cols)`, as
            `insert_function(A, rows, cols, values)`.
        K: The function space to interpolate to.
        cells_K: Local target cells.
        padded_K_dm: Unrolled dofs of `K` for each cell in `cells_K`.
        local_V_dofs: Source dofs (blocked, local to the extended index map)
            of the source cell containing each point, ordered by target cell,
            interpolation point and averaging point.
        V_bs: Block size of `local_V_dofs`.
        V_basis_values: Source basis values at each point, shape
            `(num_points, num_dofs_per_cell_V * V_bs, value_size)`.
        weights: Averaging weights, shape `(num_cells * num_ip, num_average_qp)`.
        scales: Averaging scales, shape `(num_cells * num_ip,)`.
        dofs_visited: Marks unrolled dofs of `K` that must not be inserted
            (ghosts, or rows already inserted from a neighbouring cell). Updated in
            place.
    """
    element = K.element.basix_element
    K_bs = K.dofmap.bs
    X = K.element.interpolation_points
    num_ip = X.shape[0]
    num_qp = weights.shape[1]
    value_size = V_basis_values.shape[2]
    if value_size != int(np.prod(K.element.value_shape, dtype=int)):
        raise ValueError(
            f"Source value size {value_size} does not match target value shape "
            f"{K.element.value_shape}."
        )
    needs_transformation = K.element.needs_dof_transformations
    needs_pull_back = element.map_type != basix.MapType.identity
    if K_bs > 1 and (needs_pull_back or needs_transformation):
        raise NotImplementedError(
            "Blocked target elements must use an identity map without dof "
            "transformations."
        )
    M = element.interpolation_matrix
    real_type = K.element.dtype
    # Basis functions are real valued, also in complex mode
    V_basis_values = np.real(V_basis_values).astype(real_type, copy=False)

    mesh = K.mesh
    if needs_pull_back:
        J_all, detJ_all, K_all = (
            dolfinx.fem.Expression(op(mesh), X)
            .eval(mesh, cells_K)
            .reshape(len(cells_K), num_ip, *op(mesh).ufl_shape)
            .astype(real_type)
            for op in (ufl.Jacobian, ufl.JacobianDeterminant, ufl.JacobianInverse)
        )
    if needs_transformation:
        cell_info = get_cell_permutation_info(mesh)

    # Notation, per target cell: interpolation points x_p (p < num_ip), each with
    # averaging points x_pk, weights w_pk and scale s_p (for a pointwise trace:
    # x_p0 = x_p, w_p0 = s_p = 1). phi_c is the source basis function of
    # (unrolled) source dof c.
    points = np.arange(num_ip)[:, None, None]
    for i, cell in enumerate(cells_K):
        point_slice = slice(i * num_ip * num_qp, (i + 1) * num_ip * num_qp)
        # Columns: the dofs of every source cell containing some x_pk.
        # `inverse` maps (p, k, local source dof) to its column in `cols`.
        dofs = unroll_dofmap(local_V_dofs[point_slice], V_bs)
        cols, inverse = np.unique(dofs, return_inverse=True)

        # Averaged physical values, shape (num_ip, len(cols), value_size):
        #   F[p, c] = sum_k (w_pk / s_p) phi_c(x_pk),
        # where phi_c(x_pk) = 0 if c is not a dof of the source cell containing
        # x_pk. A column repeats over k, so accumulate unbuffered with np.add.at.
        w = (
            weights[i * num_ip : (i + 1) * num_ip]
            / scales[i * num_ip : (i + 1) * num_ip, None]
        )
        values = V_basis_values[point_slice].reshape(num_ip, num_qp, -1, value_size)
        F = np.zeros((num_ip, len(cols), value_size), dtype=real_type)
        np.add.at(
            F,
            (points, inverse.reshape(num_ip, num_qp, -1)),
            values * w[..., None, None],
        )

        # Pull back to the reference cell of K with the Jacobian J_p at x_p:
        #   F[p, c] <- P_p^-1(F[p, c]),
        # e.g. det(J_p) J_p^-1 F for contravariant Piola, J_p^T F for covariant.
        if needs_pull_back:
            F = element.pull_back(F, J_all[i], detJ_all[i], K_all[i])

        # Reference dofs, with M the basix interpolation matrix:
        #   B[d, c] = sum_{m, p} M[d, m * num_ip + p] F[p, c, m]
        if K_bs == 1:
            block = M @ np.transpose(F, (2, 0, 1)).reshape(-1, len(cols))
        else:
            # Scalar sub-element: block b interpolates value component b,
            #   B[d * K_bs + b, c] = sum_p M[d, p] F[p, c, b]
            block = np.einsum("dp,pcb->dbc", M, F).reshape(-1, len(cols))
        block = np.ascontiguousarray(block, dtype=real_type)
        # Map reference dofs to the cell's dofs, B <- T^-T B, with T the dof
        # transformation of the cell (entity orientations)
        if needs_transformation:
            flat_block = block.reshape(-1)
            K.element.Tt_inv_apply(flat_block, cell_info[cell : cell + 1], len(cols))

        # A[r, c] = B[r, c]; a dof shared with a neighbouring cell gets the same
        # row from both, so it is inserted once
        rows = padded_K_dm[i]
        block[dofs_visited[rows]] = 0
        insert_function(A, rows, cols, block.reshape(-1))
        dofs_visited[rows] = True
