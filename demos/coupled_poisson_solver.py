# # A coupled 3D-1D manufactured solution
#
# This example is based on the strong formulation and manufactured solution
# from {cite}`3D1Dmasri-masri2024coupled3d1d`.
# We consider a 3D domain $\Omega$ with a 1D inclusion $\Lambda$.
# We assume that $\Lambda$ can be parameterised as a curve $\lambda(s)$, $s\in(0,L)$.
# Next, we define a generalized cylinder $B_{\Lambda,R(s)}$ with centerline $\Lambda$
# and radius $R$.
# A cross section of the cylinder at $s$ is defined as the circle $\Theta_{R}(s)$
# with radius $R: [0, L] \to \mathbb{R}^+$, area $A(s)$ and perimeter $P(s)$.
# The boundary of $B_\Lambda$ is denoted $\Gamma$.
# For a function $u\in V(\Omega)$, we define the restriction operator
# $\Pi_R(s): V(\Omega) \to L^2(\Lambda)$ as
#
# $$
# \Pi_R(u)(s) = \frac{1}{|P(s)|} \int_{\partial \Theta_R(s)} u~\mathrm{d}\gamma.
# $$
#
# ## Mesh generation
# We start by generating the two domains that we will consider in this demo.
# ```{admonition} Strict inclusion
# :class: dropdown
# Note that $\Gamma$ should be strictly included in $\Omega$.
# Additionally, we assume that $B_\Lambda \subset \Omega$.
# ```
# We start by defining a cube that spans $[-0.5,-0.5,-0.5]\times[0.5,0.5,0.5]$

# +
import inspect

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import ufl

from fenicsx_ii import Average, Circle, LinearProblem, assemble_scalar

M = 32  # Number of elements in each spatial direction in the box
N = M  # Number of elements in the line
comm = MPI.COMM_WORLD
omega = dolfinx.mesh.create_box(
    comm,
    [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    [M, M, M],
    cell_type=dolfinx.mesh.CellType.tetrahedron,
)
# -

# Next we create a line that spans $[0,0,-0.5]\times[0,0,0.5]$.
# Note that the discretized lines does not have to conform with the
# 3D domain.
# ```{admonition} Line embedded in 3D
# Note that the geometrical dimension of the line should be 3. Therefore,
# the built in mesh generators {py:func}`dolfinx.mesh.create_interval`
# and {py:func}`dolfinx.mesh.create_unit_interval` does not work as
# they use geometrical dimension 1.
# ```

# +
l_min = -0.5
l_max = 0.5
if comm.rank == 0:
    nodes = np.zeros((N, 3), dtype=np.float64)
    nodes[:, 0] = np.full(nodes.shape[0], 0)
    nodes[:, 1] = np.full(nodes.shape[0], 0)
    nodes[:, 2] = np.linspace(l_min, l_max, nodes.shape[0])
    connectivity = np.repeat(np.arange(nodes.shape[0]), 2)[1:-1].reshape(
        nodes.shape[0] - 1, 2
    )
else:
    nodes = np.zeros((0, 3), dtype=np.float64)
    connectivity = np.zeros((0, 2), dtype=np.int64)
c_el = ufl.Mesh(
    basix.ufl.element("Lagrange", basix.CellType.interval, 1, shape=(nodes.shape[1],))
)

sig = inspect.signature(dolfinx.mesh.create_cell_partitioner)
kwargs = {}
if "max_facet_to_cell_links" in list(sig.parameters.keys()):
    kwargs["max_facet_to_cell_links"] = 2
lmbda = dolfinx.mesh.create_mesh(
    comm,
    x=nodes,
    cells=connectivity,
    e=c_el,
    partitioner=dolfinx.mesh.create_cell_partitioner(
        dolfinx.mesh.GhostMode.shared_facet, **kwargs
    ),
    **kwargs,
)
# -

# ## Variational formulation
#
# We consider the following coupled 3D-1D problem, which is used in
# {cite}`3D1Dmasri-masri2024coupled3d1d` and stems from
# {cite}`3D1Dmasri-laurino20193d1d`:
# Find $u\in V(\Omega)$, $p\in Q(\Lambda)$ such that
#
# $$
# \begin{align}
#   - \nabla \cdot (\alpha_1 \nabla u) + \xi (\Pi_R(u) - p))\delta_\Gamma &= f
#   && \text{in } \Omega, \\
#   - d_s(A d_s p) + P \xi (p - \Pi(u)) &= A \hat f  &&  \text{in } \Lambda, \\
#   u&=g &&\text{on } \partial\Omega,\\
#   A d_s p &=0 && \text{at } s\in\{0, 1\}.
# \end{align}
# $$
#
# where $\xi > 0$ is the permeability constant, with the variational formulation
#
# $$
# \begin{align*}
#   \int_\Omega \alpha_1 \nabla u \cdot \nabla v~\mathrm{d}x
#   + \int_\Gamma P\xi (\Pi_R(u) - p)\Pi_R(v)~\mathrm{d}s
#   &= \int_\Omega f\cdot v~\mathrm{d}x\\
#   \int_\Gamma A d_s p \cdot d_s q~\mathrm{d}s
#   + \int_\Gamma P\xi (p - \Pi_R(u))q~\mathrm{d}s
#   &= \int_\Gamma A\hat f\cdot q~\mathrm{d}s
# \end{align*}
# $$
#
# We define the appropriate function spaces on each mesh

degree = 1
V = dolfinx.fem.functionspace(omega, ("Lagrange", degree))
Q = dolfinx.fem.functionspace(lmbda, ("Lagrange", degree))

# We define a {py:class}`mixed function space<ufl.MixedFunctionSpace>`,
# to automate {py:func}`block extraction<ufl.extract_blocks>` of the system

W = ufl.MixedFunctionSpace(*[V, Q])
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

# We start by defining the restriction operator, which we use to represent $\Pi_R(u)$ as
# {py:class}`Circle<fenicsx_ii.Circle>` is used to define the restriction operator
# defined above, where we specify the radius of the vessel
# (can be a spatiallydependent function).
# In addition, we need to specify the quadrature degree used for numerical
# integration over $\partial\Theta_R(s)$.
# In this example, the restriction for the trial and test functions are the same,
# but this can vary based on the discretization, see for instance
# {cite}`3D1Dmasri-dangelo20083d1d` for an example where `restriction_test`
# is {py:class}`PointwiseTrace<fenicsx_ii.PointwiseTrace>` rather than
# a {py:class}`Circle<fenicsx_ii.Circle>`.

R = 0.05
q_degree = 20
restriction_trial = Circle(lmbda, R, degree=q_degree)
restriction_test = Circle(lmbda, R, degree=q_degree)

# Next, we define the integration measures on each mesh

dx_3D = ufl.Measure("dx", domain=omega)
dx_1D = ufl.Measure("dx", domain=lmbda)

# To represent the variational formulations in {py:mod}`Unified Form Language<ufl>`,
# we will use an intermediate space to represent the averages as a
# {py:func}`test function<ufl.TestFunction>`,
# {py:func}`trial function<ufl.TrialFunction>`
# or {py:class}`function<dolfinx.fem.Function>` on $\Lambda$.

# In this demo, we choose to use a
# {py:func}`quadrature element<basix.ufl.quadrature_element>`,
# as we can the easily control the accuracy of the representation on $\Gamma$.

q_el = basix.ufl.quadrature_element(lmbda.basix_cell(), value_shape=(), degree=q_degree)
Rs = dolfinx.fem.functionspace(lmbda, q_el)

# We are now ready to define the averaging operator $\Pi_R(\Gamma)$ for the test
# and trial functions.

avg_u = Average(u, restriction_trial, Rs)
avg_v = Average(v, restriction_test, Rs)

# We define the physical parameters, the area and perimeter of $\Gamma$

alpha1 = dolfinx.fem.Constant(omega, 1.0)
A = ufl.pi * R**2
P = 2 * ufl.pi * R

# ```{admonition} Spatially dependent quantities and constants
# We will define `xi` and the {py:class}`spatial coordinates<ufl.SpatialCoordinate>`
# with respect to the 3D mesh.
# However, we will use them in forms which uses `dx_1D` as an
# {py:class}`integration measure<ufl.Measure>`.
# This is usually not supported in UFL. However, internally in FEniCSx_ii, we implement
# a {py:class}`DAGTraverser<fenicsx_ii.ufl_operations.DomainReplacer>` that replaces
# these domains with the appropriate one, defined in the initialization of the
# {py:class}`integration measure<ufl.Measure>`.
# ````

xi = dolfinx.fem.Constant(omega, 1.0)
x = ufl.SpatialCoordinate(omega)


# ### Defining the bilinear form
# We can define the bilinear form `a` with standard {py:class}`UFL<ufl>` operations.

a = alpha1 * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_3D
a += P * xi * ufl.inner(avg_u - p, avg_v) * dx_1D
a += A * ufl.inner(ufl.grad(p), ufl.grad(q)) * dx_1D
a += P * xi * ufl.inner(p - avg_u, q) * dx_1D

# ### Defining the linear form
# We will use a manufactured solution to define the right-hand side.
# We recall the solution from {cite}`3D1Dmasri-masri2024coupled3d1d`:
#
# $$
# \begin{align*}
#   u_{ex} &=
#   \begin{cases}
#     \frac{\xi}{\xi + 1} \left(1 - R\log\left(\frac{r}{R}\right)\right)p_{ex}
#     , & r > R, \\
#     \frac{\xi}{\xi + 1} p_{ex}, & r \leq R,
#   \end{cases} \\
#  p_{ex} &= \sin(\pi z) + 2 \\
# \end{align*}
# $$
#
# We use {py:mod}`UFL<ufl>` operations to define the manufactured solution
# and the right-hand side.
# Note that since $p_ex$ doesn't vary in the radial direction, the average is:
#
# $$
# \Pi_R(u_{ex}) = \frac{\xi}{\xi+1}p_{ex}
# $$
#
# and therefore
#
# $$
# (p_{ex}- \Pi_R(u_{ex}) = \left(1-\frac{\xi}{\xi+1}\right)p_{ex}
# = \frac{1}{\xi+1}p_{ex}
# $$
#
# and we can compute
#
# $$
# f=-\nabla \cdot (\alpha_1 \nabla u_{ex})
# $$
#
# and
#
# $$
# \begin{align*}
#   A\hat{f} &= -d_s(A d_s p_{ex}) + P\xi(p_{ex} - \Pi_R(u_{ex}))\\
#   &=-d_s (A d_s p_{ex}) + \frac{P\xi}{\xi+1}p_{ex}.
# \end{align*}
# $$

# +
p_ex = ufl.sin(ufl.pi * x[2]) + 2
xi_rate = xi / (xi + 1)

u_inside = xi_rate * p_ex
r = ufl.sqrt(x[0] ** 2 + x[1] ** 2)
u_outside = xi_rate * (1 - R * ufl.ln(r / R)) * p_ex
u_ex = ufl.conditional(r < R, u_inside, u_outside)

f_vol = -ufl.div(alpha1 * ufl.grad(u_ex))
A_fhat = -ufl.div(A * ufl.grad(p_ex)) + P * xi_rate * p_ex

L = f_vol * v * dx_3D
L += A_fhat * q * dx_1D
# -

# Next, we set up the strong
# {py:class}`Dirichlet boundary conditions<dolfinx.fem.DirichletBC>` using
# the manufactured solution. We interpolate it into the appropriate function
# space by wrapping it as a {py:class}`Expression<dolfinx.fem.Expression>`.

omega.topology.create_connectivity(omega.topology.dim - 1, omega.topology.dim)
exterior_facets = dolfinx.mesh.exterior_facet_indices(omega.topology)
exterior_dofs = dolfinx.fem.locate_dofs_topological(
    V, omega.topology.dim - 1, exterior_facets
)
bc_expr = dolfinx.fem.Expression(u_ex, V.element.interpolation_points)
u_bc = dolfinx.fem.Function(V)
u_bc.interpolate(bc_expr)
bc = dolfinx.fem.dirichletbc(u_bc, exterior_dofs)
bcs = [bc]

# ## Solving the linear system

# Next, we solve the arising linear system with the
# {py:class}`LinearProblem<fenicsx_ii.LinearProblem>` class.
# Note that you have to use the {mod}`fenicsx_ii-class<fenicsx_ii.LinearProblem>`
# rather than the standard {py:class}`LinearProblem<dolfinx.fem.petsc.LinearProblem>`
# from {py:mod}`dolfinx.fem.petsc`

petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
}
problem = LinearProblem(
    a,
    L,
    petsc_options_prefix="coupled_poisson",
    petsc_options=petsc_options,
    bcs=bcs,
)
uh, ph = problem.solve()
uh.name = "uh"
ph.name = "ph"

# We store the solutions to file using the {py:class}`VTXWriter<dolfinx.io.VTXWriter>`.

with dolfinx.io.VTXWriter(omega.comm, "u_3D_solver.bp", [uh]) as vtx:
    vtx.write(0.0)
with dolfinx.io.VTXWriter(lmbda.comm, "p_1D_solver.bp", [ph]) as vtx:
    vtx.write(0.0)

# ## Error-computations
# We compute the error in the $L^2$ and $H^1$ norms for both variables.
# Note that we employ a custom implementation of {py:func}`dolfinx.fem.assemble_scalar`,
# namely {py:func}`fenicsx_ii.assemble_scalar`, which supports restricted coefficients,
# {py:class}`dolfinx.fem.Constant<dolfinx.fem.Constant>` defined on different domains.
# It additionally takes care of the necessary communication for parallel computations.


# +
def L2(f: ufl.core.expr.Expr, dx: ufl.Measure) -> float:
    integral = ufl.inner(f, f) * dx
    return np.sqrt(assemble_scalar(integral, op=MPI.SUM))


def H1(f: ufl.core.expr.Expr, dx: ufl.Measure) -> float:
    integral = ufl.inner(f, f) * dx + ufl.inner(ufl.grad(f), ufl.grad(f)) * dx
    return np.sqrt(assemble_scalar(integral, op=MPI.SUM))


# -

# +
L2_error_u = L2(uh - u_ex, dx_3D)
L2_error_p = L2(ph - p_ex, dx_1D)
H1_error_u = H1(uh - u_ex, dx_3D)
H1_error_p = H1(ph - p_ex, dx_1D)

if MPI.COMM_WORLD.rank == 0:
    print(f"{degree=:d} {q_degree=:d}")
    print(f"L2-error u: {L2_error_u:.6e}")
    print(f"L2-error p: {L2_error_p:.6e}")
    print(f"H1-error u: {H1_error_u:.6e}")
    print(f"H1-error p: {H1_error_p:.6e}")
# -

# ## References

# ```{bibliography}
# :filter: cited
# :labelprefix:
# :keyprefix: 3D1Dmasri-
# ```
