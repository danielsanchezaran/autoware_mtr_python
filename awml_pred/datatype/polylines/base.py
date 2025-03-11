from abc import abstractmethod

from awml_pred.datatype.base import LabelBaseType
from awml_pred.datatype.context import ContextType

__all__ = ("PolylineType", "LaneType", "BoundaryType", "SignalType")


class PolylineType(LabelBaseType):
    """A base enum of Polyline."""

    def is_dynamic(self) -> bool:
        """Whether the item is dynamic.

        Returns
        -------
            bool: Return always False.

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
            ContextType | str: Return always `ContextType.POLYLINE` or its value as str.

        """
        ctx = ContextType.POLYLINE
        return ctx.value if as_str else ctx


class LaneType(LabelBaseType):
    """A base enum of Lane."""

    def is_dynamic(self) -> bool:
        """Whether the item is dynamic.

        Returns
        -------
            bool: Return always False.

        """
        return False

    @abstractmethod
    def is_drivable(self) -> bool:
        """Indicate whether the lane is drivable.

        Returns
        -------
            bool: Return `True` if drivable.

        """

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
        ctx = ContextType.LANE
        return ctx.value if as_str else ctx


class BoundaryType(LabelBaseType):
    """A base enum of RoadLine and RoadEdge."""

    def is_dynamic(self) -> bool:
        """Indicate whether the lane is drivable.

        Returns
        -------
            bool: Return always False.

        """
        return False

    @abstractmethod
    def is_virtual(self) -> bool:
        """Whether the boundary is virtual or not.

        Returns
        -------
            bool: Return `True` if the boundary is virtual.

        """

    @abstractmethod
    def is_crossable(self) -> bool:
        """Whether the boundary is allowed to cross or not.

        Returns
        -------
            bool: Return `True` if the boundary is allowed to cross.

        """


class SignalType(LabelBaseType):
    """A base enum of Signal."""

    def is_dynamic(self) -> bool:
        """Indicate whether the lane is drivable.

        Returns
        -------
            bool: Return always True.

        """
        return True

    @staticmethod
    def to_context(*, as_str: bool = False) -> ContextType | str:
        """Convert the enum member to `ContextType`.

        Args:
        ----
            as_str (bool, optional): Whether to return as str. Defaults to False.

        Returns:
        -------
            ContextType | str: Return always `ContextType.SIGNAL`, or its value as str.

        """
        ctx = ContextType.SIGNAL
        return ctx.value if as_str else ctx
