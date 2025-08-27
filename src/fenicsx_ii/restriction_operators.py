"""Module for restriction operators from 3D to 1D"""

import typing

import basix
import dolfinx
import numpy as np
import numpy.typing as npt

from .quadrature import Quadrature
from .utils import get_cell_normals, get_physical_points

__all__ = ["Circle", "NaiveTrace", "ReductionOperator"]


class ReductionOperator(typing.Protocol):
    """Protocol for reduction operators from 3D to 1D."""

    _mesh: dolfinx.mesh.Mesh

    def compute_quadrature(
        self, cells: npt.NDArray[np.int32], reference_points: npt.NDArray[np.floating]
    ) -> Quadrature: ...

    @property
    def num_points(self) -> int: ...


class NaiveTrace:
    def __init__(self, mesh: dolfinx.mesh.Mesh):
        """Naive trace on a 1D line segment.

        Just evaluates input points in physical space.
        """
        self._mesh = mesh

    def compute_quadrature(
        self, cells: npt.NDArray[np.int32], reference_points: npt.NDArray[np.floating]
    ) -> Quadrature:
        points = get_physical_points(self._mesh, cells, reference_points)
        weights = np.ones((points.shape[0], 1), dtype=points.dtype)
        scales = np.ones(points.shape[0], dtype=points.dtype)
        return Quadrature(
            name="CircleAvg", points=points, weights=weights, scales=scales
        )

    @property
    def num_points(self) -> int:
        return 1


class Circle:
    def __init__(
        self,
        mesh: dolfinx.mesh.Mesh,
        radius: float
        | typing.Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        degree: int = 15,
    ):
        if degree < 3:
            raise ValueError(
                "Degree must be bigger than 3 for a sensible approximation"
            )
        if mesh.topology.dim != 1:
            raise ValueError("Circle quadrature can only be used on 1D meshes")
        if mesh.geometry.cmap.degree > 1:
            raise NotImplementedError("Not implemented for curved meshes")

        self._mesh = mesh
        xp, w = basix.make_quadrature(
            basix.CellType.interval, degree, basix.QuadratureType.default
        )
        self._w = np.asarray(w)
        xp = np.asarray(xp)
        if not callable(radius):
            assert radius >= 0, "Radius cannot be zero or negative"
            self._radius = lambda x: np.full(x.shape[1], radius)
        else:
            self._radius = radius

        self._xp = np.vstack(
            [
                np.cos(2 * np.pi * xp.T[0]),
                np.sin(2 * np.pi * xp.T[0]),
                np.zeros_like(xp.T[0]),
            ]
        ).T

    @property
    def num_points(self):
        assert self._xp.shape[0] == len(self._w)
        return len(self._w)

    @property
    def interpolation_matrix(self):
        return np.eye(self.num_points)

    @staticmethod
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

    def compute_quadrature(
        self, cells: npt.NDArray[np.int32], reference_points: npt.NDArray[np.floating]
    ) -> Quadrature:
        if self._mesh.geometry.cmap.degree > 1:
            raise NotImplementedError("Non-affine lines not supported yet")
        x_coords = get_physical_points(self._mesh, cells, reference_points)
        cell_normals = get_cell_normals(self._mesh, cells)
        normals_at_quadrature = np.repeat(
            cell_normals, reference_points.shape[0], axis=0
        )
        assert normals_at_quadrature.shape[0] == x_coords.shape[0]
        assert normals_at_quadrature.shape[1] == x_coords.shape[1]
        points, weights, scales = self.quadrature(x_coords, normals_at_quadrature)
        return Quadrature(
            name="CircleAvg", points=points, weights=weights, scales=scales
        )

    def quadrature(
        self, x0: npt.NDArray[np.floating], normals: npt.NDArray[np.floating]
    ) -> tuple[
        npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]
    ]:
        """
        Compute quadrature points in physical space for circles
        with centers `x0` and `normals`.

        Args:
            x0: List of center points of the circles, shape (num_points, 3)
            normals: List of normal vectors to the circles, shape (num_points, 3)
        Returns:
            Quadrature points and weights and scales for each weight in 3D.
        """
        assert x0.shape[0] == normals.shape[0], (
            "Number of centers and normals must match"
        )
        assert x0.shape[1] == 3, "Center points must be 3D"
        assert normals.shape[1] == 3, "Normal vectors must be 3D"

        if x0.shape[0] == 0:
            return (
                np.zeros((0, x0.shape[1])),
                np.zeros((0, self._w.shape[0])),
                np.zeros((0,)),
            )

        R = self._radius(x0.T)
        Rot = self.rotation_matrix(normals)
        xp = np.tile(x0, self._xp.shape[0]).reshape(x0.shape[0], self._xp.shape[0], 3)

        applied_rot = np.einsum("ijk,lk->ilj", Rot, self._xp)
        if x0.shape[0] == 0:
            assert False
        R_tiles = np.repeat(R, self._xp.shape[0]).reshape(x0.shape[0], -1)

        xp += np.einsum("ijk,ij->ijk", applied_rot, R_tiles)

        # Circumradius of the circle 2*pi*R
        # Quadrature rule is made on interval [0,1]
        scale = 2 * R * np.pi
        tiled_scale = np.repeat(scale, self._xp.shape[0])
        weights = np.tile(self._w, x0.shape[0])
        weights *= tiled_scale
        return xp, weights.reshape(x0.shape[0], -1), scale
