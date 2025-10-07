from functools import singledispatchmethod

from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from ufl.algorithms.map_integrands import map_integrands
from ufl.core.operator import Operator
from ufl.core.ufl_type import ufl_type
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.domain import extract_unique_domain

from fenicsx_ii import Circle, PointwiseTrace, create_interpolation_matrix


class CustomArgument(ufl.Argument):
    """Custom Argument with extra data."""

    __slots__ = ("_part", "_number", "_parent_space", "_restriction_operator")

    def __init__(
        self,
        function_space: ufl.FunctionSpace,
        part: int | None = None,
        number: int = 0,
        restriction_operator=None,
        parent_space: ufl.FunctionSpace | None = None,
    ):
        """Create a custom argument."""
        super().__init__(function_space, part=part, number=number)
        assert parent_space is not None, "Parent space must be provided"
        assert restriction_operator is not None, "Restriction operator must be provided"
        self._restriction_operator = restriction_operator
        self._parent_space = parent_space

    @property
    def parent_space(self) -> ufl.FunctionSpace:
        """Return the parent space associated with this argument."""
        return self._parent_space

    def __repr__(self) -> str:
        return f"CustomArgument({self.ufl_function_space()}, part={self._part}, number={self._number}, restriction_operator={self._restriction_operator} parent_space={self._parent_space})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def restriction_operator(self):
        return self._restriction_operator


@ufl_type(num_ops=1, inherit_shape_from_operand=0, inherit_indices_from_operand=0)
class Average(Operator):
    """Custom operator."""

    __slots__ = ("_u", "_restriction_operator", "_restriction_space")

    def __init__(self, u: ufl.Argument, restriction_operator, restriction_space):
        """Create a custom operator."""
        self._u = u
        self._restriction_operator = restriction_operator
        self._restriction_space = restriction_space

    @property
    def ufl_operands(self):
        return (self._u,)

    @property
    def _hash(self) -> int:
        return hash(("Average", self._u, self._restriction_operator))

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"Average({self._u}, {self._restriction_operator}, {self._restriction_space})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def restriction_operator(self):
        return self._restriction_operator

    @property
    def restriction_space(self):
        return self._restriction_space


class AverageReplacer(DAGTraverser):
    """DAGTraverser to split mixed coefficients."""

    def __init__(
        self,
        compress: bool | None = True,
        visited_cache: dict[tuple, ufl.core.expr.Expr] | None = None,
        result_cache: dict[ufl.core.expr.Expr, ufl.core.expr.Expr] | None = None,
    ) -> None:
        """Initialise.

        Args:
            compress: If True, ``result_cache`` will be used.
            visited_cache: cache of intermediate results; expr -> r = self.process(expr, ...).
            result_cache: cache of result objects for memory reuse, r -> r.

        """
        super().__init__(
            compress=compress, visited_cache=visited_cache, result_cache=result_cache
        )

    @singledispatchmethod
    def process(
        self,
        o: ufl.core.expr.Expr,
        reference_value: bool | None = False,
        reference_grad: int | None = 0,
        restricted: str | None = None,
    ) -> ufl.core.expr.Expr:
        """Replace averaged arguments with intermediate space.

        Args:
            o: `ufl.core.expr.Expr` to be processed.
            reference_value: Whether `ReferenceValue` has been applied or not.
            reference_grad: Number of `ReferenceGrad`s that have been applied.
            restricted: '+', '-', or None.
        """
        return super().process(o)

    @process.register(Average)
    def _(
        self,
        o: ufl.core.expr.Expr,
        reference_value: bool | None = False,
        reference_grad: int | None = 0,
        restricted: str | None = None,
    ) -> ufl.core.expr.Expr:
        """Handle Expr."""
        ops = o.ufl_operands
        assert len(ops) == 1, "Expected single operator in averaging"
        u = ops[0]
        assert isinstance(u, ufl.Argument), "Expected Argument in averaging"
        res_op = o.restriction_operator
        new_u = CustomArgument(
            o.restriction_space,
            part=u.part(),
            number=u.number(),
            parent_space=u.ufl_function_space(),
            restriction_operator=res_op,
        )
        return new_u

    @process.register(ufl.core.expr.Expr)
    def _(
        self,
        o: ufl.Argument,
        reference_value: bool | None = False,
        reference_grad: int | None = 0,
        restricted: str | None = None,
    ) -> ufl.core.expr.Expr:
        """Handle Argument."""
        return self.reuse_if_untouched(
            o,
            reference_value=reference_value,
            reference_grad=reference_grad,
            restricted=restricted,
        )


def apply_replacer(form: ufl.Form) -> ufl.Form:
    """Replace averaged arguments with intermediate space.

    Args:
        expr: UFL expression.

    """
    rule = AverageReplacer()
    mapped_integrals: list[ufl.Integral] = []
    for itg in form.integrals():
        new_itg = map_integrands(rule, itg)
        new_domain = extract_unique_domain(new_itg.integrand())
        old_domain = itg.ufl_domain()
        if new_domain != old_domain:
            new_itg = new_itg.reconstruct(domain=new_domain)
        mapped_integrals.append(new_itg)
    return [ufl.Form([mapped_integral]) for mapped_integral in mapped_integrals]


def get_replaced_argument_indices(form: ufl.Form) -> list[int, ...]:
    """Check if a form has a replaced argument."""
    indices = []
    for arg in form.arguments():
        if isinstance(arg, CustomArgument):
            indices.append(arg.number())
    return indices


def assign_LG_map(
    C: PETSc.Mat,  # type: ignore
    row_map: dolfinx.common.IndexMap,
    col_map: dolfinx.common.IndexMap,
    row_bs: int,
    col_bs: int,
):
    global_row_map = row_map.local_to_global(
        np.arange(row_map.size_local + row_map.num_ghosts, dtype=np.int32)
    ).astype(PETSc.IntType)  # type: ignore
    global_col_map = col_map.local_to_global(
        np.arange(col_map.size_local + col_map.num_ghosts, dtype=np.int32)
    ).astype(PETSc.IntType)  # type: ignore
    row_map = PETSc.LGMap().create(global_row_map, bsize=row_bs, comm=row_map.comm)  # type: ignore
    col_map = PETSc.LGMap().create(global_col_map, bsize=col_bs, comm=col_map.comm)  # type: ignore
    C.setLGMap(row_map, col_map)  # type: ignore


def assemble_matrix(a: ufl.Form) -> PETSc.Mat:
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


domain = dolfinx.mesh.create_box(
    MPI.COMM_WORLD, [[-1, -1, -1], [1, 1, 1]], [10, 10, 10]
)
N = 25
l_min = 0
l_max = 1
if MPI.COMM_WORLD.rank == 0:
    nodes = np.zeros((N, 3), dtype=np.float64)
    nodes[:, 0] = np.full(nodes.shape[0], 0.5)
    nodes[:, 1] = np.full(nodes.shape[0], 0.5)
    nodes[:, 2] = np.linspace(l_min, l_max, nodes.shape[0])[::-1]
    connectivity = np.repeat(np.arange(nodes.shape[0]), 2)[1:-1].reshape(
        nodes.shape[0] - 1, 2
    )
else:
    nodes = np.zeros((0, 3), dtype=np.float64)
    connectivity = np.zeros((0, 2), dtype=np.int64)

c_el = ufl.Mesh(
    basix.ufl.element("Lagrange", basix.CellType.interval, 1, shape=(nodes.shape[1],))
)
line_mesh = dolfinx.mesh.create_mesh(
    MPI.COMM_WORLD,
    x=nodes,
    cells=connectivity,
    e=c_el,
    partitioner=dolfinx.mesh.create_cell_partitioner(
        dolfinx.mesh.GhostMode.shared_facet
    ),
)


V = dolfinx.fem.functionspace(domain, ("Lagrange", 1, (2,)))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

Q = dolfinx.fem.functionspace(line_mesh, ("DG", 0, (2,))) # Intermediate space


R = 0.1
restriction_trial = Circle(line_mesh, R, degree=5)
restriction_test = PointwiseTrace(line_mesh)

avg_u = Average(u, restriction_trial, Q)
avg_v = Average(v, restriction_test, Q)


a_form = ufl.inner(avg_u, avg_v) * ufl.dx
a_form += ufl.inner(u, v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.ds

A = assemble_matrix(a_form)

W = dolfinx.fem.functionspace(line_mesh, ("Lagrange", 1, (2,)))
z = ufl.TestFunction(W)

dGamma = ufl.Measure("dx", domain=line_mesh)
a01 = ufl.inner(avg_u, z) * dGamma
C = assemble_matrix(a01)


w = ufl.TrialFunction(W)
a10 = ufl.inner(w, avg_v) * dGamma
D = assemble_matrix(a10)

a11 = ufl.inner(z, w) * dGamma

E = assemble_matrix(a11)

PETSc.Sys.Print(A.getSizes(), C.getSizes(), D.getSizes(), E.getSizes())
norms = A.norm(0), C.norm(0), D.norm(0), E.norm(0)

PETSc.Sys.Print(norms)
