from awml_pred.datatype.context import ContextType

from .base import BoundaryType, LaneType, PolylineType

__all__ = ("T4Polyline", "T4Lane", "T4RoadLine", "T4RoadEdge")


class T4Polyline(PolylineType):
    """Polyline types in T4."""

    # for lane
    ROAD = 0
    HIGHWAY = 1
    ROAD_SHOULDER = 2
    BICYCLE_LANE = 3
    PEDESTRIAN_LANE = 4
    WALKWAY = 5

    # for road line
    DASHED = 6
    SOLID = 7
    DASHED_DASHED = 8
    VIRTUAL = 9

    # for road edge
    ROAD_BORDER = 10

    # for crosswalk
    CROSSWALK = 11

    # for stop sign
    TRAFFIC_SIGN = 12

    # for speed bump
    SPEED_BUMP = 13

    # catch otherwise
    UNKNOWN = -1


class T4Lane(LaneType):
    """Lane types in T4."""

    ROAD = 0
    HIGHWAY = 1
    ROAD_SHOULDER = 2
    BICYCLE_LANE = 3
    PEDESTRIAN_LANE = 4
    WALKWAY = 5

    def is_drivable(self) -> bool:
        """Indicate whether the lane is drivable.

        Returns
        -------
            bool: True if drivable.

        """
        return self in (T4Lane.ROAD, T4Lane.HIGHWAY, T4Lane.ROAD_SHOULDER)


class T4RoadLine(BoundaryType):
    """Road line types in T4."""

    DASHED = 0
    SOLID = 1
    DASHED_DASHED = 2
    VIRTUAL = 3

    def is_crossable(self) -> bool:
        """Whether the boundary is allowed to cross or not.

        Returns
        -------
            bool: Return `True` if the boundary is allowed to cross.

        """
        return self in (T4RoadLine.DASHED, T4RoadLine.DASHED_DASHED)

    def is_virtual(self) -> bool:
        """Whether the boundary is virtual or not.

        Returns
        -------
            bool: Return `True` if the boundary is virtual.

        """
        return self == T4RoadLine.VIRTUAL

    @staticmethod
    def to_context(*, as_str: bool = False) -> ContextType | str:
        """Convert the enum member to `ContextType`.

        Args:
        ----
            as_str (bool, optional): Whether to return as str. Defaults to False.

        Returns:
        -------
            ContextType | str: Return always `ContextType.ROADLINE`, or its value as str.

        """
        ctx = ContextType.ROADLINE
        return ctx.value if as_str else ctx


class T4RoadEdge(BoundaryType):
    """Road edge types in T4."""

    ROAD_BORDER = 0

    def is_crossable(self) -> bool:
        """Whether the boundary is allowed to cross or not.

        Returns
        -------
            bool: Return always `False`.

        """
        return False

    def is_virtual(self) -> bool:
        """Whether the boundary is virtual or not.

        Returns
        -------
            bool: Return always `False`.

        """
        return False

    @staticmethod
    def to_context(*, as_str: bool = False) -> ContextType | str:
        """Convert the enum member to `ContextType`.

        Args:
        ----
            as_str (bool, optional): Whether to return as str. Defaults to False.

        Returns:
        -------
            ContextType | str: Return always `ContextType.ROADEDGE`, or its value as str.

        """
        ctx = ContextType.ROADEDGE
        return ctx.value if as_str else ctx
