from __future__ import annotations

from abc import abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING, TypeVar

from typing_extensions import Self

if TYPE_CHECKING:
    from .context import ContextType

__all__ = ["LabelTypeLike"]


class LabelBaseType(IntEnum):
    """Base of label types.

    All types must have the following enum format.
    * TYPE_NAME = TYPE_ID <int>
    """

    def as_str(self) -> str:
        """Return the type name.

        Returns
        -------
            str: Name in str.

        """
        return self.name

    @classmethod
    def from_str(cls, name: str) -> Self:
        """Construct from the name of member.

        Args:
        ----
            name (str): Name of an enum member.

        Returns:
        -------
            Self: Constructed member.

        """
        name = name.upper()
        assert name in cls.__members__, f"{name} is not in enum members of {cls.__name__}."
        return cls.__members__[name]

    @classmethod
    def from_id(cls, type_id: int) -> Self:
        """Construct from the value of member.

        Args:
        ----
            type_id (int): Value of enum member.

        Returns:
        -------
            Self: Constructed member.

        """
        for _, item in cls.__members__.items():
            if item.value == type_id:
                return item
        msg = f"{type_id} is not in enum ids."
        raise ValueError(msg)

    @classmethod
    def contains(cls, name: str) -> bool:
        """Check whether the input name is contained in members.

        Args:
        ----
            name (str): Name of enum member.

        Returns:
        -------
            bool: Whether it is contained.

        """
        return name.upper() in cls.__members__

    @classmethod
    def encode(cls, object_type: LabelTypeLike) -> list[int]:
        """Return One-hot encoding of the specified type ID.

        Args:
        ----
            object_type (LabelTypeLike): Object type.

        Returns:
        -------
            list[int]: One-hot encoding.

        """
        onehot = [0] * len(cls)
        onehot[object_type.value] = 1
        return onehot

    @abstractmethod
    def to_context(self, *, as_str: bool = False) -> ContextType | str:
        """Convert the enum member to `ContextType`.

        Args:
        ----
            as_str (bool, optional): Whether to return as str.

        Returns:
        -------
            ContextType | str: Converted object.

        """


LabelTypeLike = TypeVar("LabelTypeLike", bound=LabelBaseType)
