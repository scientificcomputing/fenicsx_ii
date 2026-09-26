from mpi4py import MPI as _MPI

import dolfinx
import numpy as np
import numpy.typing as npt
import ufl

from .compat import expression_uses_wrong_cell_info, get_cmap, get_geom_dofmap


def _check_dof_transformations(expr: ufl.core.expr.Expr):
    """Check if any terminal requires dof transformations.

    Raise if `expr` has an {py:class}`ufl.Argument`
    or {py:class}`ufl.Coefficient` whose space needs dof
    transformations, which {py:meth}`dolfinx.fem.Expression.eval`
    applies for the wrong cells in DOLFINx < 0.11."""
    if not expression_uses_wrong_cell_info():
        return
    terminals = [
        *ufl.algorithms.extract_arguments(expr),
        *ufl.algorithms.extract_coefficients(expr),
    ]
    for terminal in terminals:
        element = terminal.ufl_function_space().element
        if element.needs_dof_transformations:
            raise RuntimeError(
                "Cannot evaluate expressions in "
                f"{element.basix_element.family.name} (which requires dof "
                f"transformations) with DOLFINx {dolfinx.__version__}. "
                "In DOLFINx < 0.11, `dolfinx.fem.Expression.eval(mesh, cells)` "
                "applies the dof transformations of cell `i` to the `i`th entry of "
                "`cells` rather than of `cells[i]`, which gives wrong values. Please "
                "upgrade to DOLFINx >= 0.11."
            )


def pull_back(
    mesh: dolfinx.mesh.Mesh,
    points: npt.NDArray[np.inexact],
    cells: npt.NDArray[np.int32],
) -> npt.NDArray[np.floating]:
    """Pull physical points back to the reference cell of the cell holding them.

    Args:
        mesh: The mesh.
        points: Physical points, shape `(num_points, 3)`. Trimmed to
            `(num_points, gdim)` if the geometric dimension `gdim < 3`.
        cells: The cell of each point.

    Returns:
        Reference coordinates, shape `(num_points, tdim)`.
    """
    gdim = mesh.geometry.dim
    cmap = get_cmap(mesh)
    geom_dm = get_geom_dofmap(mesh)
    # The pull-back works in gdim; points and `geometry.x` are padded to 3
    mesh_nodes = mesh.geometry.x[:, :gdim]
    points = np.asarray(points, dtype=mesh_nodes.dtype).reshape(len(cells), 3)[:, :gdim]
    kwargs = {}
    if hasattr(cmap, "pull_back_working_size"):
        kwargs["working_array"] = np.zeros(
            cmap.pull_back_working_size(points.shape[1]), dtype=mesh_nodes.dtype
        )
    ref_x = np.zeros((len(cells), mesh.topology.dim), dtype=mesh_nodes.dtype)
    # One pull-back per cell, over all points in it
    order = np.argsort(cells, kind="stable")
    unique_cells, starts = np.unique(cells[order], return_index=True)
    ends = np.append(starts[1:], len(cells))
    for cell, start, end in zip(unique_cells, starts, ends):
        rows = order[start:end]
        ref_x[rows] = cmap.pull_back(points[rows], mesh_nodes[geom_dm[cell]], **kwargs)  # type: ignore[arg-type]
    return ref_x


def evaluate_expression(
    expr: ufl.core.expr.Expr,
    mesh: dolfinx.mesh.Mesh,
    points: npt.NDArray[np.inexact],
    cells: npt.NDArray[np.int32],
    batch_size: int = 200,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray:
    """Evaluate a UFL expression at physical points in given cells.

    Each point is pulled back to the reference cell of its cell, and `expr` is
    evaluated there with a {py:class}`dolfinx.fem.Expression` on `MPI.COMM_SELF`.

    Args:
        expr: The expression, with no or one {py:class}`ufl.Argument`.
        mesh: The mesh `expr` lives on.
        points: Physical points, shape `(num_points, 3)`. Trimmed to
            `(num_points, gdim)` if the geometric dimension `gdim < 3`.
        cells: The cell holding each point.
        batch_size: Points per compiled Expression. Every point in a batch is
            evaluated in every cell of the batch, so this trades JIT compilations
            against evaluation cost and memory.
        dtype: Scalar type of the Expression. Defaults to the dtype of the
            coefficients in `expr`, else the geometry dtype of `mesh`.

    Returns:
        `expr` at `points[i]` in `cells[i]`, shape `(num_points, *expr.ufl_shape)`,
        with a trailing axis over the dofs of the Argument's cell if `expr` has one.
    """
    arguments = ufl.algorithms.extract_arguments(expr)
    if len(arguments) > 1:
        raise ValueError(
            f"Expression has {len(arguments)} arguments, at most one is supported."
        )
    _check_dof_transformations(expr)
    if dtype is None:
        coefficients = ufl.algorithms.extract_coefficients(expr)
        dtype = np.result_type(
            mesh.geometry.x.dtype, *(c.x.array.dtype for c in coefficients)
        )

    cells = np.asarray(cells, dtype=np.int32)
    shape = tuple(expr.ufl_shape)
    if len(arguments) == 1:
        shape += (arguments[0].ufl_function_space().element.space_dimension,)
    values = np.zeros((len(cells), *shape), dtype=dtype)
    if len(cells) == 0:
        return values

    ref_x = pull_back(mesh, points, cells)
    for start in range(0, len(cells), batch_size):
        batch = slice(start, start + batch_size)
        num_batch = len(cells[batch])
        compiled = dolfinx.fem.Expression(
            expr, ref_x[batch], comm=_MPI.COMM_SELF, dtype=dtype
        )
        all_values = compiled.eval(mesh, cells[batch])
        # Every point was evaluated in every cell of the batch; keep the diagonal
        diagonal = np.arange(num_batch)
        values[batch] = all_values[diagonal, diagonal].reshape(num_batch, *shape)
    return values


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
        batch_size: The number of points/cells to evaluate at the time,
            to find a balance between flooding the expression cache and
            the memory usage.

    Returns:
        The evaluated basis functions at the given points, shape
        `(num_points, num_dofs_per_cell * bs, max(bs, value_size))`.
    """
    bs = int(V.dofmap.bs)
    value_size = int(np.prod(V.element.basix_element.value_shape))
    if bs > 1 and value_size > 1:
        raise ValueError(
            f"A function space cant have both {value_size=} and {bs=} bigger than 1."
        )
    mesh = V.mesh
    values = evaluate_expression(
        ufl.TestFunction(V),
        mesh,
        points,
        cells,
        batch_size=batch_size,
        dtype=mesh.geometry.x.dtype,
    )
    # (num_points, *value_shape, num_dofs * bs) -> (num_points, num_dofs * bs, value)
    values = values.reshape(
        len(cells), int(np.prod(values.shape[1:-1])), values.shape[-1]
    )
    return np.swapaxes(values, 1, 2).astype(dolfinx.default_scalar_type)


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
    # backward compatibility for index map:
    if hasattr(dolfinx.common, "index_map"):
        return dolfinx.common.index_map(
            comm,
            imap.size_local,
            ghosts=(extended_ghosts, extended_owners),
            tag=tag,
        )
    else:
        return dolfinx.common.IndexMap(
            comm,  # type: ignore
            imap.size_local,
            extended_ghosts,
            extended_owners,
            tag,
        )
