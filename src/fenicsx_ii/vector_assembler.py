import typing

from petsc4py import PETSc

import dolfinx.fem.petsc
import ufl

from .assembly import average_coefficients
from .interpolation import create_interpolation_matrix
from .ufl_operations import apply_replacer, get_replaced_argument_indices

__all__ = ["assemble_vector"]


def assemble_vector(
    L: ufl.Form,
    bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
    entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
    b: PETSc.Vec | None = None,  # type: ignore[name-defined]
) -> PETSc.Vec:  # type: ignore[name-defined]
    """Assemble a :py:class:`ufl.Form` into a :py:class:`petsc4py.PETSc.Vec`.

    The form might contain :py:class:`fenicsx_ii.Average` operators, in which case
    the appropriate restriction operators are applied to the test and trial functions
    before creating the vector. The form might also be a block form, in which case
    a nested vector is created.

    Args:
        L: Linear UFL form to assemble.
        bcs: List of Dirichlet boundary conditions to apply.
        form_compiler_options: Options to pass to the form compiler.
        jit_options: Options to pass to the JIT compiler.
        entity_maps: Entity maps to use for assembly.
        b: An optional PETSc vector to which the assembled vector is added.
            If None, a new vector is created.
    """
    bcs = [] if bcs is None else bcs
    num_arguments = len(L.arguments())
    if b is None:
        b = create_vector(
            L,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
    if num_arguments == 1:
        assemble_vector_and_apply_restriction(
            b,
            L,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        space = L.arguments()[0].ufl_function_space()._cpp_object
        for bc in bcs:
            if bc.function_space == space:
                dolfinx.fem.petsc.set_bc(b, [bc])
        b.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES,  # type: ignore[attr-defined]
            mode=PETSc.ScatterMode.FORWARD,  # type: ignore[attr-defined]
        )
        return b
    else:
        linear_form = ufl.extract_blocks(L)
        num_spaces = len(linear_form)
        _vecs = b.getNestSubVecs()  # type: ignore

        for i in range(num_spaces):
            if linear_form[i] is not None:
                assemble_vector_and_apply_restriction(
                    _vecs[i],
                    linear_form[i],
                    form_compiler_options=form_compiler_options,
                    jit_options=jit_options,
                    entity_maps=entity_maps,
                )
                space = linear_form[i].arguments()[0].ufl_function_space()._cpp_object
                for bc in bcs:
                    if bc.function_space == space:
                        dolfinx.fem.petsc.set_bc(_vecs[i], [bc])
                _vecs[i].ghostUpdate(
                    addv=PETSc.InsertMode.INSERT_VALUES,  # type: ignore[attr-defined]
                    mode=PETSc.ScatterMode.FORWARD,  # type: ignore[attr-defined]
                )
        [vec.destroy() for vec in _vecs]
        return b


def assemble_vector_and_apply_restriction(
    vec: None | PETSc.Vec,  # type: ignore[name-defined]
    form: ufl.Form,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
    entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
) -> PETSc.Vec:  # type: ignore[name-defined]
    def assemble_restricted_vector(form: ufl.Form) -> PETSc.Vec:  # type: ignore[name-defined]
        L_c = dolfinx.fem.form(
            form,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        average_coefficients(form)
        b = dolfinx.fem.petsc.assemble_vector(L_c)
        return b

    def post_assembly(b):
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)  # type: ignore

    b = apply_vector_replacer(
        form,
        assemble_restricted_vector,
        post_assembly,
        vec,
    )
    return b


def apply_vector_replacer(
    form: ufl.Form,  # type: ignore[name-defined]
    get_vector: typing.Callable[[ufl.Form], PETSc.Vec],  # type: ignore[name-defined]
    post_assembly: typing.Callable[[PETSc.Vec], None],  # type: ignore[name-defined]
    vec: PETSc.Vec | None = None,  # type: ignore[name-defined]
) -> PETSc.Vec:  # type: ignore[name-defined]
    """Given a linear form, replace all averaged coefficients and arguments
    by the corresponding :py:class:`fenicsx_ii.AveragedCoefficient` and
    :py:class:`fenicsx_ii.AveragedArgument` and assemble the resulting
    form into a vector.

    Args:
        form: The linear UFL form to assemble.
        get_vector: A callable that takes a UFL form and returns the
            assembled PETSc vector
        post_assembly: A callable that takes a PETSc vector and performs any
            post-assembly operations.
        form_compiler_options: Options to pass to the form compiler.
        jit_options: Options to pass to the JIT compiler.
        vec: An optional PETSc vector to which the assembled vector is added. If None,
            a new vector is created.
    Returns:
        The assembled PETSc vector.
    """
    if len(form.arguments()) != 1:
        raise ValueError("The form must have exactly one argument.")
    new_forms = apply_replacer(form)
    for avg_form in new_forms:
        b = get_vector(avg_form)
        post_assembly(b)

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
                assert isinstance(K, PETSc.Mat)  # type: ignore[attr-defined]
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


def create_subvector(
    form,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
    entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
) -> PETSc.Vec:  # type: ignore[name-defined]
    """Convenince function to create a matrix from a UFL with the Average operators"""

    def create_restricted_matrix(form):
        b_c = dolfinx.fem.form(
            form,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        b = dolfinx.fem.petsc.create_vector(b_c.function_spaces[0])
        return b

    def post_assembly(b):
        pass

    b = apply_vector_replacer(
        form,
        create_restricted_matrix,
        post_assembly,
        vec=None,
    )
    return b


def create_vector(
    L: ufl.Form,
    form_compiler_options: dict | None = None,
    jit_options: dict | None = None,
    entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
) -> PETSc.Vec:  # type: ignore[name-defined]
    """Create a :py:class:`petsc4py.PETSc.Vec` from a :py:class:`ufl.Form`.

    The form might contain :py:class:`fenicsx_ii.Average` operators, in which case
    the appropriate restriction operators are applied to the test functions before
    creating the vector. The form might also be a block form, in which case
    a nested vector is created.

    Args:
        b: Linear UFL form to assemble.
        form_compiler_options: Options to pass to the form compiler.
        jit_options: Options to pass to the JIT compiler.
    """
    num_arguments = len(L.arguments())
    if num_arguments == 1:
        return create_subvector(L, form_compiler_options, jit_options, entity_maps)
    else:
        linear_form = ufl.extract_blocks(L)
        num_spaces = len(linear_form)
        b = [None for _ in range(num_spaces)]
        for i in range(num_spaces):
            if linear_form[i] is not None:
                b[i] = create_subvector(
                    linear_form[i], form_compiler_options, jit_options, entity_maps
                )
        return PETSc.Vec().createNest(b)  # type: ignore
