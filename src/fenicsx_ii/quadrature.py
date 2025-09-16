from dataclasses import dataclass

import basix
import numpy as np
import numpy.typing as npt

__all__ = ["Quadrature", "compute_disk_quadrature"]


@dataclass
class Quadrature:
    name: str
    points: npt.NDArray[np.floating]
    weights: npt.NDArray[np.floating]
    scales: npt.NDArray[np.floating]


def _A(k, n):
    return np.pi / (n + 1) * np.sin(k * np.pi / (n + 1))


def _eta(k, n):
    return np.cos(k * np.pi / (n + 1))


def _bounds(k, n):
    return -np.sqrt(1 - _eta(k, n) ** 2), np.sqrt(1 - _eta(k, n) ** 2)


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

    points = np.zeros((n * len(qp), 2))
    weights = np.zeros(n * len(qp))
    for i in range(1, n + 1):
        I_i_w, I_i_p = _I(i, n)
        I_i_w *= _A(i, n)
        points[(i - 1) * len(qp) : i * len(qp), :] = I_i_p
        weights[(i - 1) * len(qp) : i * len(qp)] = I_i_w

    return points, weights
