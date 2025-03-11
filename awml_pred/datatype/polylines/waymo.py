from awml_pred.datatype.context import ContextType

from .base import BoundaryType, LaneType, PolylineType, SignalType

__all__ = ("WaymoPolyline", "WaymoLane", "WaymoRoadLine",
           "WaymoRoadEdge", "WaymoRoadEdge", "WaymoSignal")


class WaymoPolyline(PolylineType):
    """Polyline types in waymo."""

    # for lane
    TYPE_UNDEFINED = -1
    TYPE_FREEWAY = 1
    TYPE_SURFACE_STREET = 2
    TYPE_BIKE_LANE = 3

    # for road line
    TYPE_UNKNOWN = -2  # -1
    TYPE_BROKEN_SINGLE_WHITE = 6
    TYPE_SOLID_SINGLE_WHITE = 7
    TYPE_SOLID_DOUBLE_WHITE = 8
    TYPE_BROKEN_SINGLE_YELLOW = 9
    TYPE_BROKEN_DOUBLE_YELLOW = 10
    TYPE_SOLID_SINGLE_YELLOW = 11
    TYPE_SOLID_DOUBLE_YELLOW = 12
    TYPE_PASSING_DOUBLE_YELLOW = 13

    # for road edge
    TYPE_ROAD_EDGE_BOUNDARY = 15
    TYPE_ROAD_EDGE_MEDIAN = 16

    # for stop sign
    TYPE_STOP_SIGN = 17

    # for crosswalk
    TYPE_CROSSWALK = 18

    # for speed bump
    TYPE_SPEED_BUMP = 19


class WaymoLane(LaneType):
    """Lane types in Waymo."""

    TYPE_UNDEFINED = 0
    TYPE_FREEWAY = 1
    TYPE_SURFACE_STREET = 2
    TYPE_BIKE_LANE = 3

    def is_drivable(self) -> bool:
        """Indicate whether the lane is drivable.

        Returns
        -------
            bool: Return `True` if drivable.

        """
        return self in (WaymoLane.TYPE_FREEWAY, WaymoLane.TYPE_SURFACE_STREET)


class WaymoRoadLine(BoundaryType):
    """Road line types in Waymo."""

    TYPE_UNKNOWN = 0
    TYPE_BROKEN_SINGLE_WHITE = 1
    TYPE_SOLID_SINGLE_WHITE = 2
    TYPE_SOLID_DOUBLE_WHITE = 3
    TYPE_BROKEN_SINGLE_YELLOW = 4
    TYPE_BROKEN_DOUBLE_YELLOW = 5
    TYPE_SOLID_SINGLE_YELLOW = 6
    TYPE_SOLID_DOUBLE_YELLOW = 7
    TYPE_PASSING_DOUBLE_YELLOW = 8

    def is_crossable(self) -> bool:
        """Whether the boundary is allowed to cross or not.

        Returns
        -------
            bool: Return `True` if the boundary is allowed to cross.

        """
        return self in (
            WaymoRoadLine.TYPE_BROKEN_SINGLE_WHITE,
            WaymoRoadLine.TYPE_BROKEN_SINGLE_YELLOW,
            WaymoRoadLine.TYPE_BROKEN_DOUBLE_YELLOW,
            WaymoRoadLine.TYPE_PASSING_DOUBLE_YELLOW,
        )

    def is_virtual(self) -> bool:
        """Whether the boundary is virtual or not.

        Returns
        -------
            bool: Return `True` if the boundary is virtual.

        """
        return self == WaymoRoadLine.TYPE_UNKNOWN

    @staticmethod
    def to_context(*, as_str: bool = False) -> ContextType | str:
        """Convert the enum member to `ContextType`.

        Args:
        ----
            as_str (bool, optional): Whether to return as str. Defaults to False.

        Returns:
        -------
            ContextType | str: Return always `ContextType.LANE`, or its value as str.

        """
        ctx = ContextType.ROADLINE
        return ctx.value if as_str else ctx


class WaymoRoadEdge(BoundaryType):
    """RoadEdge types in Waymo."""

    TYPE_UNKNOWN = 0
    TYPE_ROAD_EDGE_BOUNDARY = 1
    TYPE_ROAD_EDGE_MEDIAN = 2

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
            bool: Return `True` if the boundary is virtual.

        """
        return self == WaymoRoadEdge.TYPE_UNKNOWN

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
        ctx = ContextType.ROADEDGE
        return ctx.value if as_str else ctx


class WaymoSignal(SignalType):
    """Signal states in Waymo."""

    LANE_STATE_UNKNOWN = 0

    # States for traffic signals with arrows.
    LANE_STATE_ARROW_STOP = 1
    LANE_STATE_ARROW_CAUTION = 2
    LANE_STATE_ARROW_GO = 3

    # Standard round traffic signals.
    LANE_STATE_STOP = 4
    LANE_STATE_CAUTION = 5
    LANE_STATE_GO = 6

    # Flashing light signals.
    LANE_STATE_FLASHING_STOP = 7
    LANE_STATE_FLASHING_CAUTION = 8
