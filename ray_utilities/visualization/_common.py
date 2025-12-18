from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from typing_extensions import Final, Literal


class _PlaceholderType(str):
    __slots__ = ()

    def __new__(cls, *args, **kwargs) -> "_PlaceholderType":
        return str.__new__(cls, "<Placeholder>")

    def __lt__(self, value: object) -> bool:
        return False

    def __gt__(self, value: object) -> bool:
        return True

    def __hash__(self) -> int:
        return hash("<Placeholder>")


Placeholder: Final = _PlaceholderType()

assert Placeholder == "<Placeholder>"


@dataclass
class PlotOption:
    main_only: bool = False
    plot_reduced: bool = True
    main_vs_second_best: bool = False
    main_vs_rest: bool = False

    exclude_metric: Sequence[str] | None = ()
    colorbar: bool = False
    exclude_groupby: Sequence[str] | Literal["_auto_"] | None = "_auto_"
    """If auto and groupby is only pbt_group_key will exclude main_* options"""

    def __post_init__(self):
        # at most one of these can be true
        if self.main_only + self.plot_reduced + self.main_vs_second_best + self.main_vs_rest > 1:
            raise ValueError("At most one of main_only, plot_reduced, main_vs_second_best, main_vs_rest can be True.")

    def exclude(self, metric: str | Sequence[str], groupby: str | Sequence[str] | None) -> bool:
        if self.exclude_metric is not None and metric in self.exclude_metric:
            return True
        if self.exclude_groupby is not None and groupby is not None:
            if self.exclude_groupby == "_auto_":
                if len(groupby) != 1 or groupby[0] != "pbt_group_key":
                    return False
                if self.main_only or self.main_vs_second_best or self.main_vs_rest:
                    return True
                return False
            if groupby in self.exclude_groupby:
                return True
        return False
