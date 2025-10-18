from __future__ import annotations
from typing import Generic, TypeVar, TYPE_CHECKING
import weakref

if TYPE_CHECKING:
    import argparse

T = TypeVar("T", bound="argparse.ArgumentParser | None")


class SubcommandMixin(Generic[T]):
    """Adds a parent reference to subcommand parsers to allow access to parent parser attributes."""

    # do not annotate directly as used as Tap subclass
    __parent = None  # type: T | weakref.ReferenceType[T] | None

    @property
    def parent(self) -> T | None:
        """Parent or reference to parent ArgumentParser, note that this attribute must be set by the parent."""
        if self.__parent is None:
            return None
        if isinstance(self.__parent, weakref.ReferenceType):
            return self.__parent()
        # Assure that we have a weakref
        self.__parent = weakref.ref(self.__parent)
        return self.__parent

    @parent.setter
    def parent(self, value: T) -> None:
        """Sets the parent ArgumentParser."""
        self.__parent = weakref.ref(value)
