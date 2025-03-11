from awml_pred.datatype.context import ContextType

from .base import BoundaryType, LaneType, PolylineType

__all__ = ("ArgoversePolyline", "ArgoverseLane", "ArgoverseRoadLine", "ArgoverseRoadEdge")


class ArgoversePolyline(PolylineType):
    """Polyline types in Argoverse."""

    # for lane
    VEHICLE = 0
    BIKE = 1
    BUS = 2

    # for road line
    DASH_SOLID_YELLOW = 3
    DASH_SOLID_WHITE = 4
    DASHED_WHITE = 5
    DASHED_YELLOW = 6
    DOUBLE_SOLID_YELLOW = 7
    DOUBLE_SOLID_WHITE = 8
    DOUBLE_DASH_YELLOW = 9
    DOUBLE_DASH_WHITE = 10
    SOLID_YELLOW = 11
    SOLID_WHITE = 12
    SOLID_DASH_WHITE = 13
    SOLID_DASH_YELLOW = 14
    SOLID_BLUE = 15
    UNKNOWN = -1
    NONE = -2

    # for crosswalk
    CROSSWALK = 16


class ArgoverseLane(LaneType):
    """Lane types in Argoverse."""

    VEHICLE = 0
    BIKE = 1
    BUS = 2

    def is_drivable(self) -> bool:
        """Indicate whether the lane is drivable.

        Returns
        -------
            bool: True if drivable.

        """
        return self in (ArgoverseLane.VEHICLE, ArgoverseLane.BUS)


class ArgoverseRoadLine(BoundaryType):
    """Road line types in Argoverse."""

    DASH_SOLID_YELLOW = 0
    DASH_SOLID_WHITE = 1
    DASHED_WHITE = 2
    DASHED_YELLOW = 3
    DOUBLE_SOLID_YELLOW = 4
    DOUBLE_SOLID_WHITE = 5
    DOUBLE_DASH_YELLOW = 6
    DOUBLE_DASH_WHITE = 7
    SOLID_YELLOW = 8
    SOLID_WHITE = 9
    SOLID_DASH_WHITE = 10
    SOLID_DASH_YELLOW = 11
    SOLID_BLUE = 12
    UNKNOWN = -1
    NONE = -2

    def is_crossable(self) -> bool:
        """Whether the boundary is allowed to cross or not.

        Returns
        -------
            bool: Return `True` if the boundary is allowed to cross.

        """
        return self in (
            ArgoverseRoadLine.DASH_SOLID_YELLOW,
            ArgoverseRoadLine.DASH_SOLID_WHITE,
            ArgoverseRoadLine.DASHED_WHITE,
            ArgoverseRoadLine.DASHED_YELLOW,
            ArgoverseRoadLine.DOUBLE_DASH_YELLOW,
            ArgoverseRoadLine.DOUBLE_DASH_WHITE,
            ArgoverseRoadLine.SOLID_DASH_WHITE,
            ArgoverseRoadLine.SOLID_DASH_YELLOW,
        )

    def is_virtual(self) -> bool:
        """Whether the boundary is virtual or not.

        Returns
        -------
            bool: Return `True` if the boundary is virtual.

        """
        return self in (ArgoverseRoadLine.UNKNOWN, ArgoverseRoadLine.NONE)

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


class ArgoverseRoadEdge(BoundaryType):
    """Road edge types in Argoverse."""

    DASH_SOLID_YELLOW = 0
    DASH_SOLID_WHITE = 1
    DASHED_WHITE = 2
    DASHED_YELLOW = 3
    DOUBLE_SOLID_YELLOW = 4
    DOUBLE_SOLID_WHITE = 5
    DOUBLE_DASH_YELLOW = 6
    DOUBLE_DASH_WHITE = 7
    SOLID_YELLOW = 8
    SOLID_WHITE = 9
    SOLID_DASH_WHITE = 10
    SOLID_DASH_YELLOW = 11
    SOLID_BLUE = 12
    UNKNOWN = -1
    NONE = -2

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
        return self in (ArgoverseRoadEdge.UNKNOWN, ArgoverseRoadEdge.NONE)

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
