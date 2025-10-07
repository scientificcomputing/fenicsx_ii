from petsc4py import PETSc

import dolfinx.fem.petsc
import numpy as np
import ufl

from .interpolation import create_interpolation_matrix
from .ufl_operations import apply_replacer, get_replaced_argument_indices

__all__ = ["assemble_matrix"]


def assign_LG_map(
    C: PETSc.Mat,  # type: ignore
    row_map: dolfinx.common.IndexMap,
    col_map: dolfinx.common.IndexMap,
    row_bs: int,
    col_bs: int,
):
    """
    Assign a Local to Global map (LG-map) to a PETSc matrix
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


def assemble_matrix(
    a: ufl.Form,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
) -> PETSc.Mat:
    """Assemble a matrix from a UFL form.

    Args:
        a: Bi-linear UFL form to assemble.
    """

    new_forms = apply_replacer(a)
    matrix = None
    for form in new_forms:
        a_c = dolfinx.fem.form(form)
        A = dolfinx.fem.petsc.assemble_matrix(a_c)
        A.assemble()

        replacement_indices = get_replaced_argument_indices(form)
        test_arg, trial_arg = form.arguments()
        match replacement_indices:
            case []:
                if matrix is None:
                    matrix = A
                else:
                    matrix.axpy(1.0, A)

            case [0]:
                # Replace rows, i.e. apply interpolation matrix on the left
                K, imap_test, _imap_trial = create_interpolation_matrix(
                    test_arg.parent_space,
                    test_arg.ufl_function_space(),
                    test_arg.restriction_operator,
                    use_petsc=True,
                )
                K.transpose()  # in-place transpose
                D = K.matMult(A)
                trial_space = trial_arg.ufl_function_space()
                test_space = test_arg.parent_space
                assign_LG_map(
                    D,
                    imap_test,
                    trial_space.dofmap.index_map,
                    test_space.dofmap.index_map_bs,
                    trial_space.dofmap.index_map_bs,
                )
                if matrix is None:
                    matrix = D
                else:
                    matrix.axpy(1.0, D)

            case [1]:
                # Replace columns, i.e. apply interpolation matrix on the right
                K, _imap_test, imap_trial = create_interpolation_matrix(
                    trial_arg.parent_space,
                    trial_arg.ufl_function_space(),
                    trial_arg.restriction_operator,
                    use_petsc=True,
                )
                D = A.matMult(K)
                trial_space = trial_arg.parent_space
                test_space = test_arg.ufl_function_space()
                assign_LG_map(
                    D,
                    test_space.dofmap.index_map,
                    imap_trial,
                    test_space.dofmap.index_map_bs,
                    trial_space.dofmap.index_map_bs,
                )
                if matrix is None:
                    matrix = D
                else:
                    matrix.axpy(1.0, D)

            case [0, 1]:
                # Replace both rows and columns, i.e. apply interpolation matrix on both sides
                # Start by replacing rows, multiplying by P.T from
                P, _, _ = create_interpolation_matrix(
                    test_arg.parent_space,
                    test_arg.ufl_function_space(),
                    test_arg.restriction_operator,
                    use_petsc=True,
                )
                Pt = P.copy()  # type: ignore
                Pt.transpose()
                Z = Pt.matMult(A)
                # Now replace columns, multiplying by K from the right
                K, _, _ = create_interpolation_matrix(
                    trial_arg.parent_space,
                    trial_arg.ufl_function_space(),
                    trial_arg.restriction_operator,
                    use_petsc=True,
                )
                D = Z.matMult(K)
                trial_space = trial_arg.parent_space
                test_space = test_arg.parent_space
                assign_LG_map(
                    D,
                    trial_space.dofmap.index_map,
                    test_space.dofmap.index_map,
                    trial_space.dofmap.index_map_bs,
                    test_space.dofmap.index_map_bs,
                )
                if matrix is None:
                    matrix = D
                else:
                    matrix.axpy(1.0, D)
    return matrix
