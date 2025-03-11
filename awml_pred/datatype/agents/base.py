from awml_pred.datatype.base import LabelBaseType
from awml_pred.datatype.context import ContextType


class AgentType(LabelBaseType):
    """A base enum of Agent."""
    # Dynamic movers
    VEHICLE = 0
    PEDESTRIAN = 1
    MOTORCYCLIST = 2
    CYCLIST = 3
    LARGE_VEHICLE = 4

    # Catch-all dynamic agents
    UNKNOWN = 5

    # Static objects
    STATIC = 6

    @staticmethod
    def to_context(*, as_str: bool = False) -> ContextType | str:
        """Convert the enum member to `ContextType`.

        Args:
        ----
            as_str (bool, optional): Whether to return as str. Defaults to False.

        Returns:
        -------
            ContextType | str: Converted object.

        """
        ctx = ContextType.AGENT
        return ctx.value if as_str else ctx
