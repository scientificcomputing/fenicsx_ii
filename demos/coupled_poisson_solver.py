# # A coupled 3D-1D manufactured solution
#
# This example is based on the strong formulation and manufactured solution
# from {cite}`3D1Dmasri-masri2024coupled3d1d`.
# We consider a 3D domain $\Omega$ with a 1D inclusion $\Lambda$.
# We assume that $\Lambda$ can be parameterised as a curve $\lambda(s)$, $s\in(0,L)$.
# Next, we define a generalized cylinder $B_{\Lambda,R(s)}$ with centerline $\Lambda$ and radius $R$.
# A cross section of the cylinder at $s$ is defined as the circle $\Theta_{R}(s)$ with radius
# $R: [0, L] \to \mathbb{R}^+$, area $A(s)$ and perimeter $P(s)$.
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
from mpi4py import MPI
import dolfinx
import numpy as np
import basix.ufl
import ufl

M = 32 # Number of elements in each spatial direction in the box
N = M # Number of elements in the line
comm = MPI.COMM_WORLD
omega = dolfinx.mesh.create_box(
    comm,
    [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    [M, M, M],
    cell_type=dolfinx.mesh.CellType.tetrahedron,
)
omega.name = "volume"
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
lmbda = dolfinx.mesh.create_mesh(
    comm,
    x=nodes,
    cells=connectivity,
    e=c_el,
    partitioner=dolfinx.mesh.create_cell_partitioner(
        dolfinx.mesh.GhostMode.shared_facet
    ),
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
#   - \nabla \cdot (\nabla u) + \xi (\Pi_R(u) - p))\delta_\Gamma &= f && \text{in } \Omega, \\
#   - d_s(A d_s p) + P \xi (p - \Pi(u)) &= A \hat f  &&  \text{in } \Lambda, \\
#   u&=g &&\text{on } \partial\Omega,\\
#   A d_s p &=0 && \text{at } s\in\{0, 1\}.
# \end{align}
# $$
#
# with the variational formulation
#
# $$
# \begin{align*}
#   \int_\Omega \nabla u \cdot \nabla v~\mathrm{d}x
#   + \int_\Gamma P\xi (\Pi_R(u) - p)\Pi_R(v)~\mathrm{d}s
#   &= \int_\Omega f\cdot v~\mathrm{d}x\\
#   \int_\Gamma d_s p \cdot d_s q~\mathrm{d}s
#   + \int_\Gamma P\xi (p - \Pi_R(u))q~\mathrm{d}s
#   &= \int_\Gamma A\hat f\cdot q~\mathrm{d}s
# \end{align*}
# $$

from fenicsx_ii import Average, Circle, LinearProblem



# We define the appropriate function spaces on each mesh

degree = 1
V = dolfinx.fem.functionspace(omega, ("Lagrange", degree))
Q = dolfinx.fem.functionspace(lmbda, ("Lagrange", degree))

# We define a mixed function space, to automate block extraction of the system

W = ufl.MixedFunctionSpace(*[V, Q])
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

# We define the intermediate space used for the restriction operator, along with
# the operators themselves. Note that the test and trial function are restricted
# in different ways.

R = 0.05
q_degree = 20
restriction_trial = Circle(lmbda, R, degree=q_degree)
restriction_test = Circle(lmbda, R, degree=q_degree)

# Next, we define the integration measures on each mesh

dx_3D = ufl.Measure("dx", domain=omega)
dx_1D = ufl.Measure("dx", domain=lmbda)

# We can define the intermediate space that we interpolate the
# 3D arguments into

q_el = basix.ufl.quadrature_element(
    lmbda.basix_cell(), value_shape=(), degree=q_degree
)
Rs = dolfinx.fem.functionspace(lmbda, q_el)


# Next, we define the averaging operator of each of the test and trial functions
# of the 3D spaces
avg_u = Average(u, restriction_trial, Rs)
avg_v = Average(v, restriction_test, Rs)

# We define the various constants used in the variational formulation

Alpha1 = dolfinx.fem.Constant(omega, 1.0)
A = ufl.pi * R**2

# and the variational form itself


def u_line(x):
    return ufl.sin(ufl.pi * x[2]) + 2


_xi_cache: dict[dolfinx.mesh.Mesh, dolfinx.fem.Constant] = {}


def xi(domain):
    if _xi_cache.get(domain, None) is None:
        _xi_cache[domain] = dolfinx.fem.Constant(domain, 1.0)
    return _xi_cache[domain]


def x(mesh: dolfinx.mesh.Mesh):
    return ufl.SpatialCoordinate(mesh)


P = 2 * ufl.pi * R

a = Alpha1 * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_3D
a += P * xi(lmbda) * ufl.inner(avg_u - p, avg_v) * dx_1D
a += A * ufl.inner(ufl.grad(p), ufl.grad(q)) * dx_1D
a += P * xi(lmbda) * ufl.inner(p - avg_u, q) * dx_1D

vol_rate = xi(omega) / (xi(omega) + 1)

u_inside = vol_rate * u_line(x(omega))
r = ufl.sqrt(x(omega)[0] ** 2 + x(omega)[1] ** 2)
u_outside = vol_rate * (1 - R * ufl.ln(r / R)) * u_line(x(omega))
u_ex = ufl.conditional(r < R, u_inside, u_outside)

p_ex = ufl.sin(ufl.pi * x(lmbda)[2]) + 2

f_vol_in = vol_rate * ufl.pi**2 * ufl.sin(ufl.pi * x(omega)[2])
f_vol_out = (
    vol_rate * (1 - R * ufl.ln(r / R)) * ufl.pi**2 * ufl.sin(ufl.pi * x(omega)[2])
)
f_vol = ufl.conditional(r < R, f_vol_in, f_vol_out)
# - ufl.div(ufl.grad(u_ex))

line_rate = xi(lmbda) / (xi(lmbda) + 1)
# f_line = - A*ufl.div(ufl.grad(p_ex)) + P * line_rate * p_ex
f_line = A * ufl.sin(ufl.pi * x(lmbda)[2]) * ufl.pi**2 + P * line_rate * p_ex

L = f_vol * v * dx_3D
L += f_line * q * dx_1D

# Exerior boundary conditions
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

# We assemble the arising linear system
uh = dolfinx.fem.Function(V, name="u_3D")
ph = dolfinx.fem.Function(Q, name="p_1D")
petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
}
problem = LinearProblem(
    a,
    L,
    u=[uh, ph],
    petsc_options_prefix="coupled_poisson",
    petsc_options=petsc_options,
    bcs=bcs,
)
uh, ph = problem.solve()

# We move the solution to a Function and save to file


with dolfinx.io.VTXWriter(omega.comm, "u_3D_solver.bp", [uh]) as vtx:
    vtx.write(0.0)
with dolfinx.io.VTXWriter(lmbda.comm, "p_1D_solver.bp", [ph]) as vtx:
    vtx.write(0.0)

with dolfinx.io.VTXWriter(omega.comm, "u_bc.bp", [u_bc]) as vtx:
    vtx.write(0.0)


# Compute L2-errors for u and p


def L2(f: ufl.core.expr.Expr, dx: ufl.Measure):
    integral = ufl.inner(f, f) * dx
    comm = dx.ufl_domain().ufl_cargo().comm
    return np.sqrt(
        comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(integral)), op=MPI.SUM
        )
    )


def H1(f: ufl.core.expr.Expr, dx: ufl.Measure):
    integral = ufl.inner(f, f) * dx + ufl.inner(ufl.grad(f), ufl.grad(f)) * dx
    comm = dx.ufl_domain().ufl_cargo().comm
    return np.sqrt(
        comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(integral)), op=MPI.SUM
        )
    )


L2_error_u = L2(uh - u_ex, dx_3D)
L2_error_p = L2(ph - p_ex, dx_1D)
H1_error_u = H1(uh - u_ex, dx_3D)
H1_error_p = H1(ph - p_ex, dx_1D)

h_vol = omega.comm.allreduce(
    np.max(
        omega.h(
            omega.topology.dim,
            np.arange(
                omega.topology.index_map(omega.topology.dim).size_local,
                dtype=np.int32,
            ),
        )
    ),
    op=MPI.MAX,
)
h_line = lmbda.comm.allreduce(
    np.max(
        lmbda.h(
            lmbda.topology.dim,
            np.arange(
                lmbda.topology.index_map(lmbda.topology.dim).size_local,
                dtype=np.int32,
            ),
        )
    ),
    op=MPI.MAX,
)
if MPI.COMM_WORLD.rank == 0:
    print(f"{degree=:d} {q_degree=:d}")
    print(f"{h_vol=:.5e}, {h_line=:.5e}")
    print(f"L2-error u: {L2_error_u:.6e}")
    print(f"L2-error p: {L2_error_p:.6e}")
    print(f"H1-error u: {H1_error_u:.6e}")
    print(f"H1-error p: {H1_error_p:.6e}")


## References

# ```{bibliography}
# :filter: cited
# :labelprefix:
# :keyprefix: 3D1Dmasri-
# ```