import typing

from petsc4py import PETSc

import dolfinx.fem.petsc
import ufl

from .assembly import assign_LG_map, average_coefficients
from .interpolation import create_interpolation_matrix
from .ufl_operations import apply_replacer, get_replaced_argument_indices

__all__ = ["assemble_matrix"]


def assemble_matrix(
    a: ufl.Form,
    bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
    entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
    A: PETSc.Mat | None = None,  # type: ignore[name-defined]
) -> PETSc.Mat:  # type: ignore[name-defined]
    """Assemble a bi-linear :py:class:`ufl.Form` into a :py:class:`petsc4py.PETSc.Mat`.

    The form might be a block form, in which case a nested matrix is returned.

    Args:
        a: Bi-linear UFL form to assemble.
        bcs: List of Dirichlet boundary conditions to apply.
        form_compiler_options: Options to pass to the form compiler.
        jit_options: Options to pass to the JIT compiler.
        A: An optional PETSc matrix to which the resulting matrix is added. If `None`,
            a new matrix is created.
    """
    bcs = [] if bcs is None else bcs
    num_arguments = len(a.arguments())
    if A is None:
        A = create_matrix(
            a,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )

    if num_arguments == 2:
        assemble_matrix_and_apply_restriction(
            A,
            a,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
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
        for i in range(num_spaces):
            for j in range(num_spaces):
                Aij = A.getNestSubMatrix(i, j)  # type: ignore
                if bilinear_form[i][j] is not None:
                    assemble_matrix_and_apply_restriction(
                        Aij,
                        bilinear_form[i][j],
                        form_compiler_options=form_compiler_options,
                        jit_options=jit_options,
                        entity_maps=entity_maps,
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
                            Aij.zeroRowsLocal(dofs[:lz], diag=on_diagonal)
        return A  # type: ignore


def assemble_matrix_and_apply_restriction(
    matrix: None | PETSc.Mat,  # type: ignore[name-defined]
    form: ufl.Form,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
    entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
) -> PETSc.Mat:  # type: ignore[name-defined]
    """Convenince function to assemble a matrix from a UFL form, applying any
    restriction operators on the test and trial functions.

    Args:
        matrix: An optional PETSc matrix to which the resulting matrix is added.
            If `None`, a new matrix is created.
        form: Bi-linear UFL form to assemble.
        form_compiler_options: Options to pass to the form
        jit_options: Options to pass to the JIT compiler.
        entity_maps: Entity maps to use for assembly.
    """

    def assemble_restricted_matrix(form):
        a_c = dolfinx.fem.form(
            form,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        average_coefficients(form)
        A = dolfinx.fem.petsc.assemble_matrix(a_c)
        return A

    def post_assembly(A):
        A.assemble()

    A = apply_matrix_replacer(
        form,
        assemble_restricted_matrix,
        post_assembly,
        matrix,
    )
    return A


def apply_matrix_replacer(
    a: ufl.Form,
    get_matrix: typing.Callable[[ufl.Form], PETSc.Mat],  # type: ignore[name-defined]
    post_operation: typing.Callable[[PETSc.Mat], None] = lambda *args, **kwargs: None,  # type: ignore[name-defined]
    matrix: PETSc.Mat | None = None,  # type: ignore[name-defined]
) -> PETSc.Mat:  # type: ignore[name-defined]
    """
    Given a bi-linear form, replace all averaged coefficients and arguments
    by the corresponding :py:class:`fenicsx_ii.ufl_operations.AveragedCoefficient`
    and :py:class:`fenicsx_ii.ufl_operations.ReplacedArgument` and apply the
    `get_matrix` function to the resulting form(s).
    The `post_operation` function is then applied to each resulting matrix,
    before combining them into a single matrix, which is returned.

    Args:
        a: The bi-linear form to process.
        get_matrix: A function that takes a UFL form and returns a PETSc matrix.
        post_operation: A function that takes a PETSc matrix and performs some
            operation on it.
        form_compiler_options: Options to pass to the form compiler.
        jit_options: Options to pass to the JIT compiler.
        matrix: An optional PETSc matrix to which the resulting matrix is added.
            If None, a new matrix is created.

    Returns:
        The resulting PETSc matrix.

    """
    if len(a.arguments()) != 2:
        raise ValueError(
            "The form has more than two arguments, cannot assemble a matrix."
        )
    new_forms = apply_replacer(a)
    for avg_form in new_forms:
        A = get_matrix(avg_form)
        post_operation(A)
        test_arg, trial_arg = avg_form.arguments()
        replacement_indices = get_replaced_argument_indices(avg_form)
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
                assert isinstance(K, PETSc.Mat)  # type: ignore[attr-defined]
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
                # Replace both rows and columns, i.e. apply interpolation
                # matrix on both sides.
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


def create_submatrix(
    form: ufl.Form,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
    entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
) -> PETSc.Mat:  # type: ignore[name-defined]
    """Convenince function to create a matrix from a UFL with the Average operators"""

    def create_restricted_matrix(form):
        a_c = dolfinx.fem.form(
            form,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        A = dolfinx.fem.petsc.create_matrix(a_c)
        return A

    def post_assembly(A):
        A.assemble()

    A = apply_matrix_replacer(
        form,
        create_restricted_matrix,
        post_assembly,
        matrix=None,
    )
    return A


def create_matrix(
    a: ufl.Form,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
    entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
) -> PETSc.Mat:  # type: ignore[name-defined]
    """Create a :py:class:`petsc4py.PETSc.Mat` from a :py:class:`ufl.Form`.

    The form might contain :py:class:`fenicsx_ii.Average` operators, in which case
    the appropriate restriction operators are applied to the test and trial functions
    before creating the matrix. The form might also be a block form, in which case
    a nested matrix is created.

    Args:
        a: Bi-linear UFL form to assemble.
        form_compiler_options: Options to pass to the form compiler.
        jit_options: Options to pass to the JIT compiler.
    """
    num_arguments = len(a.arguments())
    if num_arguments == 2:
        return create_submatrix(
            a, form_compiler_options, jit_options, entity_maps=entity_maps
        )
    else:
        bilinear_form = ufl.extract_blocks(a)
        num_spaces = len(bilinear_form)
        A = [[None for _ in range(num_spaces)] for _ in range(num_spaces)]
        for i in range(num_spaces):
            for j in range(num_spaces):
                if bilinear_form[i][j] is not None:
                    A[i][j] = create_submatrix(
                        bilinear_form[i][j],
                        form_compiler_options,
                        jit_options,
                        entity_maps=entity_maps,
                    )
        return PETSc.Mat().createNest(A)  # type: ignore
