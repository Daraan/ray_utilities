from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from typing_extensions import Final


class _PlaceholderType:
    def __eq__(self, value: object) -> bool:
        if isinstance(value, _PlaceholderType):
            return True
        return False

    def __lt__(self, value: object) -> bool:
        return False

    def __gt__(self, value: object) -> bool:
        return True

    def __repr__(self) -> str:
        return "<Placeholder>"

    def __hash__(self) -> int:
        return hash("ru.vis.Placeholder")


Placeholder: Final = _PlaceholderType()


@dataclass
class PlotOption:
    main_only: bool = False
    plot_reduced: bool = True
    main_vs_second_best: bool = False
    main_vs_rest: bool = False

    exclude_metric: Sequence[str] | None = ()
    colorbar: bool = False

    def __post_init__(self):
        # at most one of these can be true
        if self.main_only + self.plot_reduced + self.main_vs_second_best + self.main_vs_rest > 1:
            raise ValueError("At most one of main_only, plot_reduced, main_vs_second_best, main_vs_rest can be True.")
