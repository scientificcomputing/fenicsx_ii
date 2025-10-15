from mpi4py import MPI

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import sympy as sp
import ufl

from fenicsx_ii import Disk, PointwiseTrace
from fenicsx_ii.assembly import assemble_scalar
from fenicsx_ii.ufl_operations import Average


def test_trace_average_coefficient():
    M = 12
    c_min = -1
    c_max = 1
    assert M % 2 == 0, "M must be even"
    domain = dolfinx.mesh.create_box(
        MPI.COMM_WORLD, [[c_min, c_min, c_min], [c_max, c_max, c_max]], [M, M, M]
    )
    N = M

    if MPI.COMM_WORLD.rank == 0:
        nodes = np.zeros((N, 3), dtype=np.float64)
        nodes[:, 0] = np.full(nodes.shape[0], c_min + (c_max - c_min) / 2)
        nodes[:, 1] = np.full(nodes.shape[0], c_min + (c_max - c_min) / 2)
        nodes[:, 2] = np.linspace(c_min, c_max, nodes.shape[0])[::-1]
        connectivity = np.repeat(np.arange(nodes.shape[0]), 2)[1:-1].reshape(
            nodes.shape[0] - 1, 2
        )
    else:
        nodes = np.zeros((0, 3), dtype=np.float64)
        connectivity = np.zeros((0, 2), dtype=np.int64)

    c_el = ufl.Mesh(
        basix.ufl.element(
            "Lagrange", basix.CellType.interval, 1, shape=(nodes.shape[1],)
        )
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

    V = dolfinx.fem.functionspace(domain, ("Lagrange", 2, ()))
    Q = dolfinx.fem.functionspace(line_mesh, ("Lagrange", 2, ()))  # Intermediate space

    restriction = PointwiseTrace(line_mesh)

    uh = dolfinx.fem.Function(V)
    uh.interpolate(lambda x: 2 + x[1] ** 2 + x[2] + x[2] ** 2)
    avg_u = Average(uh, restriction, Q)
    dGamma = ufl.Measure("dx", domain=line_mesh)

    J = avg_u * dGamma

    J_gamma = assemble_scalar(J)

    domain.topology.create_connectivity(domain.topology.dim - 2, domain.topology.dim)
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 2)
    interior_ridges = dolfinx.mesh.locate_entities(
        domain,
        domain.topology.dim - 2,
        lambda x: np.isclose(x[0], c_min + (c_max - c_min) / 2)
        & np.isclose(x[1], c_min + (c_max - c_min) / 2),
    )

    ridge_integral_entities = []
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 2)
    c_to_r = domain.topology.connectivity(domain.topology.dim, domain.topology.dim - 2)
    r_to_c = domain.topology.connectivity(domain.topology.dim - 2, domain.topology.dim)
    num_ridges_local = domain.topology.index_map(domain.topology.dim - 2).size_local
    for ridge in interior_ridges:
        if ridge >= num_ridges_local:
            continue
        cell = r_to_c.links(ridge)[0]
        ridges = c_to_r.links(cell)
        local_idx = np.flatnonzero(ridges == ridge)[0]
        ridge_integral_entities.extend((cell, local_idx))
    ridge_integral_entities = np.array(ridge_integral_entities, dtype=np.int32)

    tag = 3
    dR = ufl.Measure(
        "dr",
        domain=domain,
        subdomain_data=[(tag, ridge_integral_entities)],
        subdomain_id=tag,
    )

    J_ridge = dolfinx.fem.form(uh * dR(tag))
    J_ridge_loc = dolfinx.fem.assemble_scalar(J_ridge)
    J_ridge_tot = domain.comm.allreduce(J_ridge_loc, op=MPI.SUM)

    np.testing.assert_allclose(J_ridge_tot, J_gamma)


def test_circle_average_coefficient():
    M = 12
    c_min = -1
    c_max = 1
    assert M % 4 == 0, "M must be a multiple of 4"
    domain = dolfinx.mesh.create_box(
        MPI.COMM_WORLD, [[c_min, c_min, c_min], [c_max, c_max, c_max]], [M, M, M]
    )
    N = M
    x_c = c_min + (c_max - c_min) / 4
    y_c = c_min + (c_max - c_min) / 4
    if MPI.COMM_WORLD.rank == 0:
        nodes = np.zeros((N, 3), dtype=np.float64)
        nodes[:, 0] = np.full(nodes.shape[0], x_c)
        nodes[:, 1] = np.full(nodes.shape[0], y_c)
        nodes[:, 2] = np.linspace(c_min, c_max, nodes.shape[0])[::-1]
        connectivity = np.repeat(np.arange(nodes.shape[0]), 2)[1:-1].reshape(
            nodes.shape[0] - 1, 2
        )
    else:
        nodes = np.zeros((0, 3), dtype=np.float64)
        connectivity = np.zeros((0, 2), dtype=np.int64)

    c_el = ufl.Mesh(
        basix.ufl.element(
            "Lagrange", basix.CellType.interval, 1, shape=(nodes.shape[1],)
        )
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

    V = dolfinx.fem.functionspace(domain, ("Lagrange", 2, ()))
    Q = dolfinx.fem.functionspace(line_mesh, ("Lagrange", 2, ()))  # Intermediate space

    R = 0.1
    restriction = Disk(line_mesh, R, degree=5)

    def f(x):
        return 2 + x[1] ** 2 + x[2] + x[2] ** 2

    uh = dolfinx.fem.Function(V)
    uh.interpolate(f)
    avg_u = Average(uh, restriction, Q)
    dGamma = ufl.Measure("dx", domain=line_mesh)

    J = avg_u * dGamma

    J_gamma = assemble_scalar(J)

    # Symbolic integration of cylinder
    x, y, z, R_s, Z_lower, Z_upper = sp.symbols("x y z R Z_lower Z_upper")
    integral_result = sp.integrate(
        1 / (sp.pi * R_s**2) * f([x, y, z]),
        (z, Z_lower, Z_upper),
        (
            y,
            y_c - sp.sqrt(R_s**2 - (x - x_c) ** 2),
            y_c + sp.sqrt(R_s**2 - (x - x_c) ** 2),
        ),
        (x, x_c - R_s, x_c + R_s),
        manual=True,
    )
    float_result = complex(
        integral_result.subs({R_s: R, Z_lower: c_min, Z_upper: c_max})
    ).real

    np.testing.assert_allclose(float_result, J_gamma)
