# # Long-range coupling using FEniCSx_ii
# Author: Jørgen S. Dokken
#
# SPDX License identifier: MIT

# The following example shows how to couple two non-overlapping meshes using FEniCSx_ii.

# This example will consider two domains, $\Omega_0=[0,1]\times[0,1]$ and
# $\Omega_1=[4, 3]\times[5,4]$.
# We will compute the following integral
#
# $$
# \begin{align}
# J &= \int_{\Omega_1} \mathbf{x_1}[0] \cdot u_0(T(\mathbf{x_1}))~\mathrm{d}\mathbf{x_1}
# \end{align}
# $$
#
# where $\mathbf{x_1}\in \Omega_1$, and $T: \Omega_1\mapsto \Omega_0$
# through a translation.
#
# First we start by importing the necessary modules:

# +
from typing import Callable

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import numpy.typing as npt
import ufl

from fenicsx_ii import Average, assemble_scalar
from fenicsx_ii.quadrature import Quadrature

# -

# Next we create the two domains

# +
mesh0 = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, 10, 10, cell_type=dolfinx.mesh.CellType.triangle
)

translation_vector = np.array([4, 3])

mesh1 = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    [translation_vector, translation_vector + np.ones(2)],
    [20, 20],
    cell_type=dolfinx.mesh.CellType.quadrilateral,
)
# -

# Furthermore, we implement the translation-restriction from $\Omega_1$ to $\Omega_0$
# as a general mapping restriction, using the {py:class}`fenicsx_ii.ReductionOperator`
# protocol, where we have to implement a mapping that takes a set of reference points
# in $\mathbb{R}^2$ and a set of cells in $\Omega_1$ and computes the mapping
# $T(F(x_{ref}))$ where $F$ is the push forward operation from $K_{ref}$
# (reference element) to $K$ (element in physical space) and $T$ the mapping operator.


class MappedRestriction:
    quadrature_degree: int

    def __init__(
        self,
        mesh,
        operator: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
    ):
        self._mesh = mesh
        self._operator = operator

    def compute_quadrature(
        self, cells: npt.NDArray[np.int32], reference_points: npt.NDArray[np.floating]
    ) -> Quadrature:
        phys_points = np.zeros(
            (len(cells) * reference_points.shape[0], self._mesh.geometry.dim)
        )
        x_geom = self._mesh.geometry.x[
            self._mesh.geometry.dofmap[cells], : self._mesh.geometry.dim
        ]
        for i, x_i in enumerate(x_geom):
            phys_points[
                i * reference_points.shape[0] : (i + 1) * reference_points.shape[0], :
            ] = self._mesh.geometry.cmap.push_forward(reference_points, x_i)
        translated_points = self._operator(phys_points.T).T
        return Quadrature(
            name="Translation",
            points=translated_points,
            weights=np.ones((phys_points.shape[0], 1)),
            scales=np.ones(phys_points.shape[0]),
        )

    @property
    def num_points(self) -> int:
        return 1


# Next we specify the actual translation operator from $\Omega_1$ to $\Omega_0$ and
# create the restriction operator.


# +
def translate_1_to_0(x):
    x_out = x.copy()
    for i, ti in enumerate(translation_vector):
        x_out[i] -= ti
    return x_out


restriction = MappedRestriction(mesh1, translate_1_to_0)
# -

# Next, we are ready to test the implementation.
# We start by defining $u_0$ as a known function.


# +
def f(x):
    return x[0] + 2 * x[1] * x[1]


V0 = dolfinx.fem.functionspace(mesh0, ("Lagrange", 2))
u_0 = dolfinx.fem.Function(V0, name="u0")
u_0.interpolate(f)
# -

# Next we define $u_0(T(\Omega_1))$ as

q_degree = 2
quadrature = basix.ufl.quadrature_element(
    mesh1.basix_cell(), value_shape=(), degree=q_degree
)
Q1 = dolfinx.fem.functionspace(mesh1, quadrature)
u0_on_omega1 = Average(u_0, restriction, Q1)

# where we have used an intermediate space $Q1$ to accurately capture $u_0$.
# Finally, we can assemble the integral and compare it with the exact solution

# +
dx1 = ufl.Measure("dx", domain=mesh1)
x1 = ufl.SpatialCoordinate(mesh1)
J = x1[0] * u0_on_omega1 * dx1

print(f"J: {assemble_scalar(J):.5e}")
# -

J_exact = x1[0] * f(x1 - ufl.as_vector(translation_vector.tolist())) * dx1
print(f"J_exact: {assemble_scalar(J_exact):.5e}")

# +
diff = x1[0] * u0_on_omega1 - x1[0] * f(x1 - ufl.as_vector(translation_vector.tolist()))
L2_error = ufl.inner(diff, diff) * dx1
error = np.sqrt(assemble_scalar(L2_error))
print(f"L2 error {error:.5e}")

tol = 200 * np.finfo(mesh1.geometry.x.dtype).eps
assert np.isclose(error, 0.0, atol=tol, rtol=tol)
# -
