from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from typing import Final

from autoware_mtr.datatype import PolylineLabel
import numpy as np
from numpy.typing import NDArray

__all__ = ("Polyline",)


# TODO(ktro2828): Type definition


@dataclass
class Polyline:
    """
    A dataclass of Polyline.

    Attributes
    ----------
        polyline_type (PolylineType): `PolylineType` instance.
        waypoints (NDArray): Waypoints of polyline.

    """

    polyline_type: PolylineLabel
    waypoints: NDArray

    # NOTE: For the 1DArray indices must be a list.
    XYZ_IDX: ClassVar[list[int]] = [0, 1, 2]
    XY_IDX: ClassVar[list[int]] = [0, 1]
    FULL_DIM3D: ClassVar[int] = 7
    FULL_DIM2D: ClassVar[int] = 5

    def __post_init__(self) -> None:
        if not isinstance(self.waypoints, np.ndarray):
            self.waypoints = np.array(self.waypoints, dtype=np.float32)
        assert isinstance(self.waypoints, np.ndarray)
        min_ndim: Final[int] = 1
        point_dim: Final[int] = 3
        assert self.waypoints.ndim > min_ndim and self.waypoints.shape[1] == point_dim
        assert isinstance(self.polyline_type, PolylineLabel)

    @property
    def xyz(self) -> NDArray:
        """
        Return 3D positions.

        Returns
        -------
            NDArray: (x, y, z) positions.

        """
        return self.waypoints[..., self.XYZ_IDX]

    @xyz.setter
    def xyz(self, xyz: NDArray) -> None:
        self.waypoints[..., self.XYZ_IDX] = xyz

    @property
    def xy(self) -> NDArray:
        """
        Return 2D positions.

        Returns
        -------
            NDArray: (x, y) positions.

        """
        return self.waypoints[..., self.XY_IDX]

    @xy.setter
    def xy(self, xy: NDArray) -> None:
        self.waypoints[..., self.XY_IDX] = xy

    @property
    def dxyz(self) -> NDArray:
        """
        Return 3D normalized directions. The first element always becomes (0, 0, 0).

        Returns
        -------
            NDArray: (dx, dy, dz) positions.

        """
        if self.is_empty():
            return np.empty((0, 3), dtype=np.float32)
        diff = np.diff(self.xyz, axis=0, prepend=self.xyz[0].reshape(-1, 3))
        norm = np.linalg.norm(diff, axis=-1, keepdims=True)
        zero: Final[float] = 0.0
        return np.divide(diff, norm, where=(diff != zero) & (norm != zero))

    @property
    def dxy(self) -> NDArray:
        """
        Return 2D normalized directions. The first element always becomes (0, 0).

        Returns
        -------
            NDArray: (dx, dy) positions.

        """
        if self.is_empty():
            return np.empty((0, 2), dtype=np.float32)
        diff = np.diff(self.xy, axis=0, prepend=self.xy[0].reshape(-1, 2))
        norm = np.linalg.norm(diff, axis=-1, keepdims=True)
        zero: Final[float] = 0.0
        return np.divide(diff, norm, where=(diff != zero) & (norm != zero))

    @property
    def type_id(self) -> int:
        """
        Return the type ID in `int`.

        Returns
        -------
            int: Type ID.

        """
        return self.polyline_type.value

    @property
    def type_str(self) -> str:
        """
        Return the type in `str`.

        Returns
        -------
            str: Type in `str`.

        """
        return self.polyline_type.as_str()

    def __len__(self) -> int:
        return len(self.waypoints)

    def is_empty(self) -> bool:
        """
        Indicate whether waypoints is empty array.

        Returns
        -------
            bool: Return `True` if the number of points is 0.

        """
        return len(self.waypoints) == 0

    def as_array(self, *, full: bool = False, as_3d: bool = True) -> NDArray:
        """
        Return the polyline as `NDArray`.

        Args:
        ----
            full (bool, optional): Indicates whether to return
                `(x, y, z, dx, dy, dz, type_id)`. If `False`, returns `(x, y, z)`.
                Defaults to False.
            as_3d (bool, optional): If `True` returns array containing 3D coordinates.
                Otherwise, 2D coordinates. Defaults to True.

        Returns:
        -------
            NDArray: Polyline array.

        """
        if full:
            if self.is_empty():
                return (
                    np.empty((0, self.FULL_DIM3D), dtype=np.float32)
                    if as_3d
                    else np.empty((0, self.FULL_DIM2D), dtype=np.float32)
                )

            shape = self.waypoints.shape[:-1]
            type_id = np.full((*shape, 1), self.type_id)
            return (
                np.concatenate([self.xyz, self.dxyz, type_id], axis=1, dtype=np.float32)
                if as_3d
                else np.concatenate([self.xy, self.dxy, type_id], axis=1, dtype=np.float32)
            )
        else:
            return self.xyz if as_3d else self.xy
