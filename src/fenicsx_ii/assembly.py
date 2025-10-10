from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem.petsc
import numpy as np
import ufl

from .interpolation import create_interpolation_matrix
from .ufl_operations import (
    AveragedCoefficient,
    apply_replacer,
    get_replaced_argument_indices,
)

__all__ = ["assemble_matrix", "assemble_vector", "assemble_scalar"]


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
    bcs: list[dolfinx.fem.DirichletBC] | None = None,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
) -> PETSc.Mat:
    """Assemble a matrix from a UFL form.

    Args:
        a: Bi-linear UFL form to assemble.
    """
    bcs = [] if bcs is None else bcs
    num_arguments = len(a.arguments())
    if num_arguments == 2:
        A = assemble_matrix_and_apply_restriction(
            None, a, form_compiler_options, jit_options
        )
        # Unsymmetric application of Dirichlet BCs
        spaces = [arg.ufl_function_space()._cpp_object for arg in a.arguments()]
        on_diagonal = float(spaces[0] == spaces[1])
        for bc in bcs:
            if bc.function_space == spaces[0]:
                dofs, lz = bc._cpp_object.dof_indices()
                A.zeroRowsLocal(dofs[:lz], diag=on_diagonal)
        return A
    else:
        bilinear_form = ufl.extract_blocks(a)
        num_spaces = len(bilinear_form)
        A = [[None for _ in range(num_spaces)] for _ in range(num_spaces)]
        for i in range(num_spaces):
            for j in range(num_spaces):
                if bilinear_form[i][j] is not None:
                    A[i][j] = assemble_matrix_and_apply_restriction(
                        A[i][j],
                        bilinear_form[i][j],
                        form_compiler_options,
                        jit_options,
                    )
                    # Unsymmetric application of Dirichlet BCs
                    spaces = [
                        arg.ufl_function_space()._cpp_object
                        for arg in bilinear_form[i][j].arguments()
                    ]
                    on_diagonal = float(spaces[0] == spaces[1])
                    for bc in bcs:
                        if bc.function_space == spaces[0]:
                            dofs, lz = bc._cpp_object.dof_indices()
                            A[i][j].zeroRowsLocal(dofs[:lz], diag=on_diagonal)
        return PETSc.Mat().createNest(A)  # type: ignore


def assemble_matrix_and_apply_restriction(
    matrix: None | PETSc.Mat,
    form: ufl.Form,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
) -> PETSc.Mat:
    new_forms = apply_replacer(form)
    for avg_form in new_forms:
        a_c = dolfinx.fem.form(
            avg_form,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
        )
        average_coefficients(avg_form)
        A = dolfinx.fem.petsc.assemble_matrix(a_c)
        A.assemble()

        replacement_indices = get_replaced_argument_indices(avg_form)
        test_arg, trial_arg = avg_form.arguments()
        match replacement_indices:
            case []:
                if matrix is None:
                    matrix = A
                else:
                    matrix.axpy(1.0, A)

            case [0]:
                # Replace rows, i.e. apply interpolation matrix on the left
                K, _imap_test, _imap_trial = create_interpolation_matrix(
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
                    test_space.dofmap.index_map,
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
                    test_space.dofmap.index_map,
                    trial_space.dofmap.index_map,
                    test_space.dofmap.index_map_bs,
                    trial_space.dofmap.index_map_bs,
                )
                if matrix is None:
                    matrix = D
                else:
                    matrix.axpy(1.0, D)
            case _:
                raise ValueError(
                    f"Unexpected replacement indices {replacement_indices}"
                )
    return matrix


def assemble_vector(
    L: ufl.Form,
    bcs: list[dolfinx.fem.DirichletBC] | None = None,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
) -> PETSc.Vec:
    """Assemble a matrix from a UFL form.

    Args:
        b: Linear UFL form to assemble.
    """
    bcs = [] if bcs is None else bcs
    num_arguments = len(L.arguments())
    if num_arguments == 1:
        b = assemble_vector_and_apply_restriction(
            None, L, form_compiler_options, jit_options
        )
        space = L.arguments()[0].ufl_function_space()._cpp_object
        for bc in bcs:
            if bc.function_space == space:
                dolfinx.fem.petsc.set_bc(b, [bc])
        b.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )  # type: ignore
        return b
    else:
        linear_form = ufl.extract_blocks(L)
        num_spaces = len(linear_form)
        b = [None for _ in range(num_spaces)]
        for i in range(num_spaces):
            if linear_form[i] is not None:
                b[i] = assemble_vector_and_apply_restriction(
                    b[i],
                    linear_form[i],
                    form_compiler_options,
                    jit_options,
                )
                space = linear_form[i].arguments()[0].ufl_function_space()._cpp_object
                for bc in bcs:
                    if bc.function_space == space:
                        dolfinx.fem.petsc.set_bc(b[i], [bc])
                b[i].ghostUpdate(
                    addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
                )  # type: ignore

        return PETSc.Vec().createNest(b)  # type: ignore


def assemble_vector_and_apply_restriction(
    vec: None | PETSc.Vec,
    form: ufl.Form,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
) -> PETSc.Vec:
    new_forms = apply_replacer(form)
    for avg_form in new_forms:
        L_c = dolfinx.fem.form(
            avg_form,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
        )
        average_coefficients(avg_form)
        b = dolfinx.fem.petsc.assemble_vector(L_c)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)  # type: ignore

        replacement_indices = get_replaced_argument_indices(avg_form)
        test_arg = avg_form.arguments()[0]
        match replacement_indices:
            case []:
                if vec is None:
                    vec = b
                else:
                    vec.axpy(1.0, b)

            case [0]:
                # Replace rows, i.e. apply interpolation matrix on the left
                K, _, _ = create_interpolation_matrix(
                    test_arg.parent_space,
                    test_arg.ufl_function_space(),
                    test_arg.restriction_operator,
                    use_petsc=True,
                )
                K.transpose()  # in-place transpose

                z = dolfinx.fem.petsc.create_vector(
                    form.arguments()[0].ufl_function_space()
                )
                with z.localForm() as z_loc:
                    z_loc.set(0)
                K.mult(b, z)
                if vec is None:
                    vec = z
                else:
                    vec.axpy(1.0, z)
            case _:
                raise ValueError(
                    f"Unexpected replacement indices {replacement_indices}"
                )
    return vec


def assemble_vector(
    L: ufl.Form,
    bcs: list[dolfinx.fem.DirichletBC] | None = None,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
) -> PETSc.Vec:
    """Assemble a matrix from a UFL form.

    Args:
        b: Linear UFL form to assemble.
    """
    bcs = [] if bcs is None else bcs
    num_arguments = len(L.arguments())
    if num_arguments == 1:
        b = assemble_vector_and_apply_restriction(
            None, L, form_compiler_options, jit_options
        )
        space = L.arguments()[0].ufl_function_space()._cpp_object
        for bc in bcs:
            if bc.function_space == space:
                dolfinx.fem.petsc.set_bc(b, [bc])
        b.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )  # type: ignore
        return b
    else:
        linear_form = ufl.extract_blocks(L)
        num_spaces = len(linear_form)
        b = [None for _ in range(num_spaces)]
        for i in range(num_spaces):
            if linear_form[i] is not None:
                b[i] = assemble_vector_and_apply_restriction(
                    b[i],
                    linear_form[i],
                    form_compiler_options,
                    jit_options,
                )
                space = linear_form[i].arguments()[0].ufl_function_space()._cpp_object
                for bc in bcs:
                    if bc.function_space == space:
                        dolfinx.fem.petsc.set_bc(b[i], [bc])
                b[i].ghostUpdate(
                    addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
                )  # type: ignore

        return PETSc.Vec().createNest(b)  # type: ignore


def average_coefficients(form: ufl.Form):
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
    op: MPI.Op = MPI.SUM,
) -> np.inexact:
    new_forms = apply_replacer(form)
    val = 0.0
    for avg_form in new_forms:
        L_c = dolfinx.fem.form(
            avg_form,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
        )
        average_coefficients(avg_form)
        loc_val = dolfinx.fem.assemble_scalar(L_c)
        val += L_c.mesh.comm.allreduce(loc_val, op=op)
    return val
