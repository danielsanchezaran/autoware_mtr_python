from __future__ import annotations

from enum import Enum

__all__ = ["DatasetName"]


class DatasetName(str, Enum):
    """Represents the name of dataset."""

    ARGOVERSE = "ARGOVERSE"
    T4 = "T4"

    @classmethod
    def from_str(cls, name: str) -> DatasetName:
        """Construct instance from string.

        Args:
        ----
            name (str): Name of dataset. Both Upper or lower cases are allowed.

        Returns:
        -------
            TargetTrackMode: Enum member.

        """
        name = name.upper()
        assert name in cls.__members__, f"{name} is not in enum members."
        return cls.__members__[name]
