from __future__ import annotations

from dataclasses import asdict, dataclass
from attr import define, field

from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import Self

from .polyline import Polyline

if TYPE_CHECKING:
    from awml_pred.typing import NDArrayF32

__all__ = ("AWMLStaticMap", "LaneSegment", "BoundarySegment", "CrosswalkSegment")


@dataclass(frozen=True)
class AWMLStaticMap:
    """Represents a static map information.

    Attributes
    ----------
        id (str): Unique ID associated with this map.
        lane_segments (dict[int, LaneSegment]): Container of lanes stored by its id.
        crosswalk_segments (dict[int, CrosswalkSegment]): Container of crosswalks stored by its id.
        boundary_segments (dict[int, BoundarySegment]): Container of boundaries stored by its id
            except of contained in lanes.

    """

    id: str
    lane_segments: dict[int, LaneSegment]
    crosswalk_segments: dict[int, CrosswalkSegment]
    boundary_segments: dict[int, BoundarySegment] = field(factory=dict)

    def __post_init__(self) -> None:
        assert all(
            isinstance(item, LaneSegment) for _, item in self.lane_segments.items()
        ), "Expected all items are LaneSegments."
        assert all(
            isinstance(item, CrosswalkSegment) for _, item in self.crosswalk_segments.items()
        ), "Expected all items are CrosswalkSegments."
        assert all(
            isinstance(item, BoundarySegment) for _, item in self.boundary_segments.items()
        ), "Expected all items are BoundarySegments."

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Construct instance from dict data.

        Args:
        ----
            data (dict): Dict data for `AWMLStaticMap`.

        Returns:
        -------
            AWMLStaticMap: Constructed instance.

        """
        map_id: int = data["id"]

        lane_segments: dict[int, LaneSegment] = {
            seg_id: LaneSegment.from_dict(seg) for seg_id, seg in data["lane_segments"].items()
        }

        crosswalk_segments: dict[int, CrosswalkSegment] = {
            seg_id: CrosswalkSegment.from_dict(seg) for seg_id, seg in data["crosswalk_segments"].items()
        }

        boundary_segments: dict[int, BoundarySegment] = {
            seg_id: BoundarySegment.from_dict(seg) for seg_id, seg in data.get("boundary_segments", {}).items()
        }
        return cls(map_id, lane_segments, crosswalk_segments, boundary_segments)

    def get_lane_segments(self) -> list[LaneSegment]:
        """Return all lane segments as a list.

        Returns
        -------
            list[LaneSegment]: List of `LaneSegment`.

        """
        return [seg for _, seg in self.lane_segments.items()]

    def get_crosswalk_segments(self) -> list[CrosswalkSegment]:
        """Return all crosswalk segments as a list.

        Returns
        -------
            list[CrosswalkSegment]: List of `CrosswalkSegment`.

        """
        return [seg for _, seg in self.crosswalk_segments.items()]

    def get_boundary_segments(self) -> list[BoundarySegment]:
        """Return all the other boundary segments except of contained lanes as a list.

        Returns
        -------
            list[BoundarySegments]: List of `BoundarySegment`.

        """
        return [seg for _, seg in self.boundary_segments.items()]

    def get_all_polyline(
        self,
        *,
        as_array: bool = False,
        full: bool = False,
        as_3d: bool = True,
    ) -> list[Polyline] | NDArrayF32:
        """Return all segments polyline.

        Args:
        ----
            as_array (bool, optional): Indicates whether to return polyline as `NDArray`. Defaults to False.
            full (bool, optional): This is used only if `as_array=True`.
                Indicates whether to return `(x, y, z, dx, dy, dz, type_id)`.
                If `False`, returns `(x, y, z)`. Defaults to False.
            as_3d (bool, optional): This is used only if `as_array=True`.
                If `True` returns array containing 3D coordinates.
                Otherwise, 2D coordinates. Defaults to True.

        Returns:
        -------
            list[Polyline] | NDArrayF32: List of `Polyline` instances or `NDArray`.

        """
        all_polyline: list[Polyline | NDArrayF32] = []

        duplicate_boundary_ids = []

        def _append_boundaries(boundaries: list[BoundarySegment]) -> None:
            for bound in boundaries:
                if bound.id in duplicate_boundary_ids:
                    continue
                duplicate_boundary_ids.append(bound.id)
                all_polyline.append(bound.polyline)

        for _, lane in self.lane_segments.items():
            if as_array:
                all_polyline.append(lane.as_array(full=full, as_3d=as_3d))
            else:
                all_polyline.append(lane.polyline)
                _append_boundaries(lane.left_boundaries)
                _append_boundaries(lane.right_boundaries)
        for _, crosswalk in self.crosswalk_segments.items():
            if as_array:
                all_polyline.append(crosswalk.as_array(full=full, as_3d=as_3d))
            else:
                all_polyline.append(crosswalk.polygon)
        for _, boundary in self.boundary_segments.items():
            if as_array:
                all_polyline.append(boundary.as_array(full=full, as_3d=as_3d))
            else:
                all_polyline.append(boundary.polyline)
        return np.concatenate(all_polyline, axis=0, dtype=np.float32) if as_array else all_polyline


def _to_boundary_segment(x: list[dict | BoundarySegment]) -> list[BoundarySegment]:
    return [BoundarySegment.from_dict(v) if isinstance(v, dict) else v for v in x]


@define
class LaneSegment:
    """Represents a lane segment.

    Attributes
    ----------
        id (int): Unique ID associated with this lane.
        polyline (Polyline): `Polyline` instance.
        is_intersection (bool): Flag indicating if this lane is intersection.
        left_boundaries (list[BoundarySegment]): List of `BoundarySegment` instances.
        right_boundaries (list[BoundarySegment]): List of `BoundarySegment` instances.
        left_neighbor_ids (list[int]): List of left neighbor ids on left side.
        right_neighbor_ids (list[int]): List of neighbor ids on right side.
        speed_limit_mph (float | None, optional): Lane speed limit in [miles/h].

    """

    id: int
    polyline: Polyline = field(converter=lambda x: Polyline.from_dict(x)
                               if isinstance(x, dict) else x)
    is_intersection: bool
    left_boundaries: list[BoundarySegment] = field(converter=_to_boundary_segment, factory=list)
    right_boundaries: list[BoundarySegment] = field(converter=_to_boundary_segment, factory=list)
    left_neighbor_ids: list[int] = field(factory=list)
    right_neighbor_ids: list[int] = field(factory=list)
    speed_limit_mph: float | None = field(default=None)

    @property
    def lane_type(self) -> MapType:
        """Return the type of the lane.

        Returns
        -------
            MapType: Lane type.

        """
        return self.polyline.polyline_type

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Construct a instance from a dict data.

        Args:
        ----
            data (dict[str, Any]): Dict data of `LaneSegment`.

        Returns:
        -------
            LaneSegment: Constructed instance.

        """
        return cls(**data)

    def is_drivable(self) -> bool:
        """Whether the lane is allowed to drive by car like vehicle.

        Returns
        -------
            bool: Return True if the lane is allowed to drive.

        """
        return self.lane_type.is_drivable()

    def as_array(self, *, full: bool = False, as_3d: bool = True) -> NDArrayF32:
        """Return polyline containing all points on the road segment.

        Args:
        ----
            full (bool, optional): Indicates whether to return `(x, y, z, dx, dy, dz, type_id)`.
                If `False`, returns `(x, y, z)`. Defaults to False.
            as_3d (bool, optional): If `True` returns array containing 3D coordinates.
                Otherwise, 2D coordinates. Defaults to True.

        Returns:
        -------
            NDArrayF32: Polyline of the road segment in shape (N, D).

        """
        all_polyline: list[NDArrayF32] = [self.polyline.as_array(full=full, as_3d=as_3d)]
        duplicate_boundary_ids: list[int] = []

        def _append_boundaries(boundaries: list[BoundarySegment]) -> None:
            for bound in boundaries:
                if bound.id in duplicate_boundary_ids:
                    continue
                duplicate_boundary_ids.append(bound.id)
                all_polyline.append(bound.as_array(full=full, as_3d=as_3d))

        _append_boundaries(self.left_boundaries)
        _append_boundaries(self.right_boundaries)

        return np.concatenate(all_polyline, axis=0, dtype=np.float32)

    def is_left_crossable(self) -> bool:
        """Whether all left boundaries of lane are allowed to cross.

        Returns
        -------
            bool: Return True if all boundaries are allowed to cross.

        """
        if len(self.left_boundaries) == 0:
            return False
        return all(bound.is_crossable() for bound in self.left_boundaries)

    def is_right_crossable(self) -> bool:
        """Whether all right boundaries of this lane are allowed to cross.

        Returns
        -------
            bool: Return True if all boundaries are allowed to cross.

        """
        if len(self.right_boundaries) == 0:
            return False
        return all(bound.is_crossable() for bound in self.right_boundaries)

    def is_left_virtual(self) -> bool:
        """Whether all left boundaries of this lane are virtual (=unknown).

        Returns
        -------
            bool: Return True if all boundaries are virtual.

        """
        if len(self.left_boundaries) == 0:
            return False
        return all(bound.is_virtual() for bound in self.left_boundaries)

    def is_right_virtual(self) -> bool:
        """Whether all right boundaries of this lane are virtual (=unknown).

        Returns
        -------
            bool: Return True if all boundaries are virtual.

        """
        if len(self.right_boundaries) == 0:
            return False
        return all(bound.is_virtual() for bound in self.right_boundaries)

    def has_left_neighbor(self) -> bool:
        """Whether the lane segment has the neighbor lane on its left side.

        Returns
        -------
            bool: Return True if it has at least one `left_neighbor_ids`.

        """
        return len(self.left_neighbor_ids) > 0

    def has_right_neighbor(self) -> bool:
        """Whether the lane segment has the neighbor lane on its right side.

        Returns
        -------
            bool: Return True if it has at least one `right_neighbor_ids`.

        """
        return len(self.right_neighbor_ids) > 0


@dataclass
class BoundarySegment:
    """Represents a boundary segment which is RoadLine or RoadEdge.

    Attributes
    ----------
        id (int): Unique ID associated with this boundary.
        boundary_type (BoundaryType): `BoundaryType` instance.
        polyline (Polyline): `Polyline` instance.

    """

    id: int
    polyline: Polyline

    def __post_init__(self) -> None:
        assert isinstance(self.polyline, Polyline), "Expected Polyline."

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundarySegment:
        """Construct a instance from a dict data.

        Args:
        ----
            data (dict[str, Any]): Dict data of `BoundarySegment`.

        Returns:
        -------
            BoundarySegment: Constructed instance.

        """
        return cls(**data)

    def as_dict(self) -> dict:
        """Convert the instance to a dict.

        Returns
        -------
            dict: Converted data.

        """
        return asdict(self)

    def is_crossable(self) -> bool:
        """Indicate whether the boundary is allowed to cross or not.

        Return value depends on the `BoundaryType` definition.

        Returns
        -------
            bool: Return True if the boundary is allowed to cross.

        """
        return self.boundary_type.is_crossable()

    def is_virtual(self) -> bool:
        """Indicate whether the boundary is virtual(or Unknown) or not.

        Returns
        -------
            bool: Return True if the boundary is virtual.

        """
        return self.boundary_type.is_virtual()

    def as_array(self, *, full: bool = False, as_3d: bool = True) -> NDArrayF32:
        """Return the polyline as `NDArray`.

        Args:
        ----
            full (bool, optional): Indicates whether to return `(x, y, z, dx, dy, dz, type_id)`.
                If `False`, returns `(x, y, z)`. Defaults to False.
            as_3d (bool, optional): If `True` returns array containing 3D coordinates.
                Otherwise, 2D coordinates. Defaults to True.

        Returns:
        -------
            NDArrayF32: Polyline array.

        """
        return self.polyline.as_array(full=full, as_3d=as_3d)


@dataclass
class CrosswalkSegment:
    """Represents a crosswalk segment.

    Attributes
    ----------
        id (int): Unique ID associated with this crosswalk.
        polygon (Polyline): `Polyline` instance represents crosswalk polygon.

    """

    id: int
    polygon: Polyline

    def __post_init__(self) -> None:
        assert isinstance(self.polygon, Polyline), "Expected Polyline."

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrosswalkSegment:
        """Construct a instance from a dict data.

        Args:
        ----
            data (dict[str, Any]): Dict data of `CrosswalkSegment`.

        Returns:
        -------
            CrosswalkSegment: Constructed instance.

        """
        crosswalk_id = data["id"]
        polygon = Polyline.from_dict(data["polygon"])
        return cls(crosswalk_id, polygon)

    def as_dict(self) -> dict:
        """Convert the instance to a dict.

        Returns
        -------
            dict: Converted data.

        """
        return asdict(self)

    def as_array(self, *, full: bool = False, as_3d: bool = True) -> NDArrayF32:
        """Return the polyline as `NDArray`.

        Args:
        ----
            full (bool, optional): Indicates whether to return `(x, y, z, dx, dy, dz, type_id)`.
                If `False`, returns `(x, y, z)`. Defaults to False.
            as_3d (bool, optional): If `True` returns array containing 3D coordinates.
                Otherwise, 2D coordinates. Defaults to True.

        Returns:
        -------
            NDArrayF32: Polyline array.

        """
        return self.polygon.as_array(full=full, as_3d=as_3d)
