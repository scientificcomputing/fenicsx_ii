from collections.abc import Sized
from dataclasses import dataclass

import basix
import numpy as np
import numpy.typing as npt

__all__ = ["Quadrature", "compute_disk_quadrature"]


@dataclass
class Quadrature:
    name: str
    """Name of the quadrature rule"""
    points: npt.NDArray[np.floating]
    """Points in the rule (physical space)"""
    weights: npt.NDArray[np.floating]
    """Weights of the quadrature points"""
    scales: npt.NDArray[np.floating]
    """Scaling factors for the quadrature points"""


def _A(k, n):
    return np.pi / (n + 1) * np.sin(k * np.pi / (n + 1))


def _eta(k, n):
    return np.cos(k * np.pi / (n + 1))


def _bounds(k, n):
    return -np.sqrt(1 - _eta(k, n) ** 2), np.sqrt(1 - _eta(k, n) ** 2)


def rotation_matrix(
    n: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute the rotation matrix for circle with center(s) at x0, normal(s) n,
    radius(es) R)"""
    # Normalize normal vector
    if n.ndim == 1:
        n = n.reshape(-1, 3)
    n = (n.T / np.linalg.norm(n, axis=1)).T

    # Find a plane that is normal to n
    ez = np.array([0, 0, 1])
    axis = np.cross(ez, n)

    axis_norm = np.linalg.norm(axis, axis=1)
    axis[np.isclose(axis_norm, 0)] = ez
    axis = axis / np.linalg.norm(axis, axis=1).reshape(-1, 1)

    cos_theta = np.dot(n, ez)
    sin_theta = np.sqrt(1 - cos_theta**2)
    # Rotation matrices
    Rot = np.broadcast_to(np.eye(3), (n.shape[0], 3, 3))
    Rot = np.einsum("ijk,i->ijk", Rot, cos_theta)
    smat = np.zeros((n.shape[0], 3, 3))
    smat[:, 0, 1] = -axis[:, 2]
    smat[:, 0, 2] = axis[:, 1]
    smat[:, 1, 0] = axis[:, 2]
    smat[:, 1, 2] = -axis[:, 0]
    smat[:, 2, 0] = -axis[:, 1]
    smat[:, 2, 1] = axis[:, 0]
    Rot += np.einsum("ijk,i->ijk", smat, sin_theta)
    outer_prod = np.einsum("ij,ik->ijk", axis, axis)
    Rot += np.einsum("ijk,i->ijk", outer_prod, 1 - cos_theta)

    return Rot


def compute_disk_quadrature(
    n: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute an `n`-point quadrature rule on a unit disk.

    The quadrature has accuracy `2*n-1` and is based on integrating over Gaussian
    quadrature lines, see: Bojanov and Petrova (1998), DOI: 10.1007/s002110050358
    for details.
    """

    # Line quadrature
    qp, qw = basix.make_quadrature(basix.CellType.interval, 2 * n - 1)

    def _I(k, n):
        L_b, U_b = _bounds(k, n)
        q_w_scaled = (U_b - L_b) * qw
        q_p_scaled = qp * L_b + (1 - qp) * U_b
        q_p_2D = np.zeros((len(q_p_scaled), 2))
        q_p_2D[:, 0] = _eta(k, n)
        q_p_2D[:, 1] = q_p_scaled[:, 0]
        return q_w_scaled, q_p_2D

    assert isinstance(qp, Sized)
    points = np.zeros((n * len(qp), 2))
    weights = np.zeros(n * len(qp))
    for i in range(1, n + 1):
        I_i_w, I_i_p = _I(i, n)
        I_i_w *= _A(i, n)
        points[(i - 1) * len(qp) : i * len(qp), :] = I_i_p
        weights[(i - 1) * len(qp) : i * len(qp)] = I_i_w

    return points, weights
