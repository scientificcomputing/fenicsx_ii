"""Custom UFL operations for handling average operations on arguments.

TODO: Average operations on coefficients.

"""

from functools import singledispatchmethod

import ufl
from ufl.algorithms.map_integrands import map_integrands
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.domain import extract_unique_domain

__all__ = ["Average", "get_replaced_argument_indices", "apply_replacer"]


class AveragedArgument(ufl.Argument):
    """Argument that is restricted to a different mesh through a restriction operator.

    Args:
        function_space: Function space on the new mesh.
        part: Part of the function space.
        number: Argument number.
        restriction_operator: Operator that restricts from the parent space to this space.
        parent_space: Function space on the original mesh.
    """

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
        return (
            f"AveragedArgument({self.ufl_function_space()}, part={self._part}, number={self._number},"
            + f" restriction_operator={self._restriction_operator}, parent_space={self._parent_space})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def restriction_operator(self):
        return self._restriction_operator


@ufl.core.ufl_type.ufl_type(
    num_ops=1, inherit_shape_from_operand=0, inherit_indices_from_operand=0
)
class Average(ufl.core.operator.Operator):
    """Averaging operator.

    Args:
        u: Argument to be averaged.
        restriction_operator: Operator that restricts from the parent space to the averaged space.
        restriction_space: Function space on the new mesh.
    """

    __slots__ = ("_u", "_restriction_operator", "_restriction_space")

    def __init__(self, u: ufl.Argument, restriction_operator, restriction_space):
        """Create a custom operator."""
        assert isinstance(u, ufl.Argument), "Can only average arguments"
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

    def _ufl_expr_reconstruct_(self, *operands):
        if isinstance(operands[0], ufl.constantvalue.Zero) or isinstance(
            operands[0], ufl.operators.Zero
        ):
            return ufl.operators.Zero(operands[0].ufl_shape)
        else:
            return Average(
                operands[0], self._restriction_operator, self._restriction_space
            )


class AverageReplacer(DAGTraverser):
    """DAGTraverser to replaced averaged arguments with an argument in an
    intermediate space."""

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
        """Handle AverageOperator."""
        ops = o.ufl_operands
        assert len(ops) == 1, "Expected single operator in averaging"
        u = ops[0]
        if isinstance(u, ufl.Argument):
            res_op = o.restriction_operator
            new_u = AveragedArgument(
                o.restriction_space,
                part=u.part(),
                number=u.number(),
                parent_space=u.ufl_function_space(),
                restriction_operator=res_op,
            )
            return new_u
        elif isinstance(u, ufl.constantvalue.Zero):
            return ufl.operators.Zero(o.ufl_shape)
        elif isinstance(u, ufl.operators.Zero):
            return ufl.operators.Zero(o.ufl_shape)
        else:
            raise NotImplementedError(f"Can only average arguments, got {type(u)}")

    @process.register(ufl.core.expr.Expr)
    def _(
        self,
        o: ufl.Argument,
        reference_value: bool | None = False,
        reference_grad: int | None = 0,
        restricted: str | None = None,
    ) -> ufl.core.expr.Expr:
        """Handle anything else in UFL."""
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
        if not isinstance(itg.integrand(), ufl.operators.Zero):
            mapped_integrals.append(new_itg)
    return [ufl.Form([mapped_integral]) for mapped_integral in mapped_integrals]


def get_replaced_argument_indices(form: ufl.Form) -> list[int, ...]:
    """Check if a form has a replaced argument."""
    indices = []
    for arg in form.arguments():
        if isinstance(arg, AveragedArgument):
            indices.append(arg.number())
    return indices
