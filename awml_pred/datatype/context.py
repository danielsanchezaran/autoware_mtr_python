from __future__ import annotations

from enum import Enum, unique

__all__ = ["ContextType"]


@unique
class ContextType(str, Enum):
    """Context types, this is only used in visualization."""

    # Agent context
    EGO = "EGO"
    TARGET_AGENT = "TARGET_AGENT"
    OTHER_AGENT = "OTHER_AGENT"

    LANE = "LANE"  # = CENTERLINE
    ROADLINE = "ROADLINE"
    ROADEDGE = "ROADEDGE"
    CROSSWALK = "CROSSWALK"
    SIGNAL = "SIGNAL"

    # Catch all contexts
    UNKNOWN = "UNKNOWN"
