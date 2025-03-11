from __future__ import annotations

from enum import unique

from .base import LabelBaseType
from .context import ContextType

__all__ = ["MapType"]


@unique
class MapType(LabelBaseType):
    # Lane
    ROADWAY = 0
    BUS_LANE = 1
    BIKE_LANE = 2

    # RoadLine
    DASH_SOLID = 3
    DASHED = 4
    DOUBLE_DASH = 5

    # RoadEdge
    SOLID = 6
    DOUBLE_SOLID = 7
    SOLID_DASH = 8

    # Crosswalk
    CROSSWALK = 9

    # Catch-all other/unknown map features
    UNKNOWN = 10

    def is_drivable(self) -> bool:
        """Indicate whether the lane is drivable area by vehicle.

        Returns
        -------
            bool: True if drivable are by car-like vehicle.

        """
        return self in (MapType.ROADWAY, MapType.BUS_LANE)

    def is_crossable(self) -> bool:
        """Whether the boundary is allowed to cross or not.

        Reference:
            https://github.com/HKUST-Aerial-Robotics/SIMPL/blob/2a33314c92f42feee2b1b1b863f394672cd23eca/data_av2/av2_preprocess.py#L374-L390

        Returns
        -------
            bool: Return `True` if the boundary is allowed to cross.

        """
        return self in (MapType.DASH_SOLID, MapType.DASHED, MapType.DOUBLE_DASH)

    def is_virtual(self) -> bool:
        """Whether the boundary is virtual or not.

        Returns
        -------
            bool: Return `True` if the boundary is unknown.

        """
        return self == MapType.UNKNOWN

    def to_context(self, *, as_str: bool = False) -> ContextType | str:
        """Convert the enum member to `ContextType`.

        Args:
        ----
            as_str (bool, optional): Whether to return as str. Defaults to False.

        Returns:
        -------
            ContextType | str: Return always `ContextType.POLYLINE` or its value as str.

        """
        if self in (MapType.ROADWAY, MapType.BUS_LANE, MapType.BIKE_LANE):
            ctx = ContextType.LANE
        elif self in (MapType.DASH_SOLID, MapType.DASHED, MapType.DOUBLE_DASH):
            ctx = ContextType.ROADLINE
        elif self in (MapType.SOLID, MapType.DOUBLE_SOLID, MapType.SOLID_DASH):
            ctx = ContextType.ROADEDGE
        elif self == MapType.CROSSWALK:
            ctx = ContextType.CROSSWALK
        else:
            ctx = ContextType.UNKNOWN

        return ctx.value if as_str else ctx
