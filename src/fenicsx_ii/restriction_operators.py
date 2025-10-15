"""Module for restriction operators from 3D to 1D"""

import typing

import basix
import dolfinx
import numpy as np
import numpy.typing as npt

from .quadrature import Quadrature, compute_disk_quadrature, rotation_matrix
from .utils import get_cell_normals, get_physical_points

__all__ = ["Circle", "PointwiseTrace", "ReductionOperator"]


class ReductionOperator(typing.Protocol):
    """Protocol for reduction operators from 3D to 1D."""

    _mesh: dolfinx.mesh.Mesh

    def compute_quadrature(
        self, cells: npt.NDArray[np.int32], reference_points: npt.NDArray[np.floating]
    ) -> Quadrature:
        """
        Function that should compute the :py:class:`Quadrature<fenicsx_ii.Quadrature>`
        on the given cells and reference points.

        Args:
            cells: List of cell indices on which to compute
                the quadrature (local to process).
            reference_points: List of reference points in the cells,
                shape ``(num_points, tdim)``.

        Returns:
            The quadrature rule on the given cells and reference points.
        """
        ...

    @property
    def num_points(self) -> int:
        """Number of quadrature points per cell."""
        ...

    def __str__(self) -> str:
        """String representation of the operator."""
        ...


class PointwiseTrace:
    """Trace of the 3D function on the line segment.

    Args:
        mesh: The 1D mesh on which the trace is taken.
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh):
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

    def __str__(self) -> str:
        return f"PointwiseTrace({self._mesh})"


class Circle:
    """
    The average on the perimiter of a circle with a given `radius`.

    Args:
        mesh: The 1D mesh on which the circles are centered.
        radius: The radius of the circles. Either a float or a function that
            takes in a list of points (shape (3, num_points)) and returns a list
            of radii (shape (num_points,)).
        degree: The degree of the quadrature rule on the circle.
            Must be at least 3 for a sensible approximation.
    """

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

    def __str__(self) -> str:
        return (
            f"Circle({self._mesh}, radius={self._radius}"
            + f", num_points={self._w.shape[0]})"
        )

    @property
    def num_points(self):
        assert self._xp.shape[0] == len(self._w)
        return len(self._w)

    @property
    def interpolation_matrix(self):
        return np.eye(self.num_points)

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
        Rot = rotation_matrix(normals)
        xp = np.tile(x0, self._xp.shape[0]).reshape(x0.shape[0], self._xp.shape[0], 3)

        applied_rot = np.einsum("ijk,lk->ilj", Rot, self._xp)

        R_tiles = np.repeat(R, self._xp.shape[0]).reshape(x0.shape[0], -1)

        xp += np.einsum("ijk,ij->ijk", applied_rot, R_tiles)

        # Circumradius of the circle 2*pi*R
        # Quadrature rule is made on interval [0,1]
        scale = 2 * R * np.pi
        tiled_scale = np.repeat(scale, self._xp.shape[0])
        weights = np.tile(self._w, x0.shape[0])
        weights *= tiled_scale
        return xp, weights.reshape(x0.shape[0], -1), scale


class Disk:
    """
    The average on a disk with a given `radius`.
    The quadrature rule is based on Bojanov and Petrova (1998),
    https://doi.org/10.1007/s002110050358.

    Args:
        mesh: The 1D mesh on which the disks are centered.
        radius: The radius of the disks. Either a float or a function that
            takes in a list of points (shape (3, num_points)) and returns a list
            of radii (shape (num_points,)).
        degree: The degree of the quadrature rule on the disk.
    """

    def __init__(
        self,
        mesh: dolfinx.mesh.Mesh,
        radius: float
        | typing.Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        degree: int = 7,
    ):
        if mesh.topology.dim != 1:
            raise ValueError("Circle quadrature can only be used on 1D meshes")
        if mesh.geometry.cmap.degree > 1:
            raise NotImplementedError("Not implemented for curved meshes")

        self._mesh = mesh

        # Compute quadrature points and weights on unit disk and pad for 3D
        xp, self._w = compute_disk_quadrature(degree)
        self._xp = np.zeros((xp.shape[0], 3), dtype=xp.dtype)
        self._xp[:, :2] = xp
        if not callable(radius):
            assert radius >= 0, "Radius cannot be zero or negative"
            self._radius = lambda x: np.full(x.shape[1], radius)
        else:
            self._radius = radius

    @property
    def num_points(self):
        assert self._xp.shape[0] == len(self._w)
        return len(self._w)

    @property
    def interpolation_matrix(self):
        return np.eye(self.num_points)

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
        return Quadrature(name="DiskAvg", points=points, weights=weights, scales=scales)

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
        Rot = rotation_matrix(normals)
        xp = np.tile(x0, self._xp.shape[0]).reshape(x0.shape[0], self._xp.shape[0], 3)
        applied_rot = np.einsum("ijk,lk->ilj", Rot, self._xp)
        R_tiles = np.repeat(R, self._xp.shape[0]).reshape(x0.shape[0], -1)

        xp += np.einsum("ijk,ij->ijk", applied_rot, R_tiles)

        # area of unit disk is pi R^2
        # Quadrature rule is made on a unit disk
        scale = np.pi * R**2
        tiled_scale = np.repeat(R**2, self._xp.shape[0])
        weights = np.tile(self._w, x0.shape[0])
        weights *= tiled_scale
        return xp, weights.reshape(x0.shape[0], -1), scale
