import basix
import numpy as np
import numpy.typing as npt
import pytest
import sympy as sp

from fenicsx_ii.quadrature import _eta, compute_disk_quadrature


def reference_coefficients(n: int) -> npt.NDArray[np.floating]:
    """Reference coefficients from the Bojanov and Petrova paper.

    Args:
        n: Number of points
    """
    match n:
        case 1:
            return np.array([1.57079632679415], dtype=np.float64)
        case 2:
            return np.array([0.90689968211791, 0.90689968211791], dtype=np.float64)
        case 3:
            return np.array(
                [0.55536036726971, 0.78539816339708, 0.55536036726971],
                dtype=np.float64,
            )
        case 4:
            return np.array(
                [
                    0.36931636609870,
                    0.59756643294895,
                    0.59756643294895,
                    0.36931636609870,
                ],
                dtype=np.float64,
            )
        case 5:
            return np.array(
                [
                    0.26179938779933,
                    0.45344984105895,
                    0.52359877559866,
                    0.45344984105895,
                    0.26179938779933,
                ],
                dtype=np.float64,
            )
        case 6:
            return np.array(
                [
                    0.19472656676044,
                    0.35088514880954,
                    0.43754662381298,
                    0.43754662381298,
                    0.35088514880954,
                    0.19472656676044,
                ],
                dtype=np.float64,
            )
        case 7:
            return np.array(
                [
                    0.15027943247105,
                    0.27768018363486,
                    0.36280664401738,
                    0.39269908169854,
                    0.36280664401738,
                    0.27768018363486,
                    0.15027943247105,
                ],
                dtype=np.float64,
            )
        case 8:
            return np.array(
                [
                    0.11938755218353,
                    0.22437520360108,
                    0.302299894039155,
                    0.343762755784618,
                    0.34376275578461,
                    0.30229989403870,
                    0.22437520360108,
                    0.11938755218353,
                ],
                dtype=np.float64,
            )
        case 9:
            return np.array(
                [
                    0.09708055193641,
                    0.18465818304935,
                    0.25416018461601,
                    0.29878321647448,
                    0.31415926535919,
                    0.29878321647402,
                    0.25416018461601,
                    0.18465818304935,
                    0.09708055193641,
                ],
                dtype=np.float64,
            )
        case 10:
            return np.array(
                [
                    0.08046263007725,
                    0.15440665639539,
                    0.21584157370421,
                    0.25979029037080,
                    0.28269234274376,
                    0.28269234274376,
                    0.25979029037080,
                    0.21584157370421,
                    0.15440665639539,
                    0.08046263007725,
                ],
                dtype=np.float64,
            )
        case _:
            raise NotImplementedError(
                f"Reference coefficients not implemented for {n=}"
            )


def reference_x_coordinates(n: int) -> npt.NDArray[np.floating]:
    """Reference x-coordinates from the Bojanov and Petrova paper.

    Args:
        n: Number of points
    """
    match n:
        case 1:
            return np.zeros(1, dtype=np.float64)
        case 2:
            return np.array([0.5, -0.5], dtype=np.float64)
        case 3:
            return np.array([0.70710678118655, 0, -0.70710678118655], dtype=np.float64)
        case 4:
            return np.array(
                [
                    0.80901699437495,
                    0.30901699437495,
                    -0.30901699437495,
                    -0.80901699437495,
                ],
                dtype=np.float64,
            )
        case 5:
            return np.array(
                [
                    0.86602540378444,
                    0.5,
                    0,
                    -0.5,
                    -0.86602540378444,
                ],
                dtype=np.float64,
            )
        case 6:
            return np.array(
                [
                    0.90096886790242,
                    0.62348980185873,
                    0.22252093395631,
                    -0.22252093395631,
                    -0.62348980185873,
                    -0.90096886790242,
                ],
                dtype=np.float64,
            )
        case 7:
            return np.array(
                [
                    0.92387953251129,
                    0.70710678118655,
                    0.38268343236509,
                    0,
                    -0.38268343236509,
                    -0.70710678118655,
                    -0.92387953251129,
                ],
                dtype=np.float64,
            )
        case 8:
            return np.array(
                [
                    0.93969262078591,
                    0.76604444311898,
                    0.5,
                    0.17364817766693,
                    -0.17364817766693,
                    -0.5,
                    -0.76604444311898,
                    -0.93969262078591,
                ],
                dtype=np.float64,
            )
        case 9:
            return np.array(
                [
                    0.95105651629515,
                    0.80901699437495,
                    0.58778525229247,
                    0.30901699437495,
                    0,
                    -0.30901699437495,
                    -0.58778525229247,
                    -0.80901699437495,
                    -0.95105651629515,
                ],
                dtype=np.float64,
            )
        case 10:
            return np.array(
                [
                    0.95949297361450,
                    0.84125353283118,
                    0.65486073394529,
                    0.41541501300189,
                    0.14231483827329,
                    -0.14231483827329,
                    -0.41541501300189,
                    -0.65486073394529,
                    -0.84125353283118,
                    -0.95949297361450,
                ],
                dtype=np.float64,
            )
        case _:
            raise NotImplementedError(f"Reference coordinates not implemented for {n=}")


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_circle_quadrature(n):
    points, weights = compute_disk_quadrature(n)

    ref_x_coords = reference_x_coordinates(n)
    qp, qw = basix.make_quadrature(basix.CellType.interval, 2 * n - 1)
    num_coords_per_line = len(qw)
    np.testing.assert_allclose(
        points[:, 0], np.repeat(ref_x_coords, num_coords_per_line), atol=1e-16
    )

    reference_weights = reference_coefficients(n)
    for i in range(ref_x_coords.shape[0] // num_coords_per_line):
        scale = 2 * np.sqrt(1 - _eta(i + 1, n) ** 2)

        np.testing.assert_allclose(
            weights[i * num_coords_per_line : (i + 1) * num_coords_per_line],
            scale * qw * reference_weights[i],
            atol=1e-16,
        )


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7])
def test_quadrature_accuracy(n):
    points, weights = compute_disk_quadrature(n)

    for m in range(n):

        def f(x):
            return 2 + (x[0] ** (n - m) - 0.1) * (x[1] ** m - 0.2)

        integral = np.sum(weights * f(points.T))
        x, y = sp.symbols("x y")
        exact_integral = sp.integrate(
            f([x, y]), (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)
        )
        np.testing.assert_allclose(
            integral, float(exact_integral), rtol=1e-12, atol=1e-12
        )
