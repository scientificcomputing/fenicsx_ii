import typing

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import numpy as np
import ufl

from .interpolation import create_interpolation_matrix
from .ufl_operations import (
    AveragedCoefficient,
    apply_replacer,
)

__all__ = ["assemble_scalar", "assign_LG_map", "average_coefficients"]


def assign_LG_map(
    C: PETSc.Mat,  # type: ignore
    row_map: dolfinx.common.IndexMap,
    col_map: dolfinx.common.IndexMap,
    row_bs: int,
    col_bs: int,
):
    """
    Assign a Local to Global map (LG-map) to a PETSc matrix.

    Args:
        C: The PETSc matrix to which the LG-map is assigned.
        row_map: The row index map.
        col_map: The column index map.
        row_bs: The row block size.
        col_bs: The column block size.
    """
    global_row_map = row_map.local_to_global(
        np.arange(row_map.size_local + row_map.num_ghosts, dtype=np.int32)
    ).astype(PETSc.IntType)  # type: ignore
    global_col_map = col_map.local_to_global(
        np.arange(col_map.size_local + col_map.num_ghosts, dtype=np.int32)
    ).astype(PETSc.IntType)  # type: ignore
    row_map = PETSc.LGMap().create(global_row_map, bsize=row_bs, comm=row_map.comm)  # type: ignore
    col_map = PETSc.LGMap().create(global_col_map, bsize=col_bs, comm=col_map.comm)  # type: ignore
    C.setLGMap(row_map, col_map)  # type: ignore


def average_coefficients(form: ufl.Form):
    """Perform the average operation on all coefficients of a given form.

    Note:
        This updates the underlying vector of each coefficient.

    Args:
        form: The form whose coefficients are averaged.
    """
    for coeff in form.coefficients():
        if isinstance(coeff, AveragedCoefficient):
            u = coeff.parent_coefficient
            res_op = coeff.restriction_operator
            # Replace rows, i.e. apply interpolation matrix on the left
            K, _, _ = create_interpolation_matrix(
                u.function_space,
                coeff.function_space,
                res_op,
                use_petsc=True,
            )
            K.mult(u.x.petsc_vec, coeff.x.petsc_vec)
        coeff.x.scatter_forward()


def assemble_scalar(
    form: ufl.Form,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
    entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
    op: MPI.Op = MPI.SUM,
) -> np.inexact | float | complex:
    """
    Assemble a rank-0 form into a scalar value. Perform necessary MPI communication.

    Args:
        form: The rank-0 form to assemble.
        form_compiler_options: Options for the form compiler.
        jit_options: Options for just-in-time compilation.
        entity_maps: Entity maps for the mesh.
        op: The MPI operation to use for communication (default is
            :py:obj:`mpi4py.MPI.SUM`).
    Returns
        The assembled scalar value.
    """
    new_forms = apply_replacer(form)
    loc_val: float | complex = 0.0
    for avg_form in new_forms:
        L_c = dolfinx.fem.form(
            avg_form,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        average_coefficients(avg_form)
        loc_val += dolfinx.fem.assemble_scalar(L_c)
    val = L_c.mesh.comm.allreduce(loc_val, op=op)
    return val
