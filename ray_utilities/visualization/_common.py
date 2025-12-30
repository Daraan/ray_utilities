from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypedDict

import seaborn as sns
from typing_extensions import Final, Literal

_logger = logging.getLogger(__name__)

# Set seaborn and matplotlib style for publication-quality plots
# sns.set_theme()
sns.set_theme(
    style="dark",
    context="talk",
    rc={
        "axes.grid": False,  # Disable all grid lines
        "axes.spines.top": False,
        "axes.spines.right": True,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "font.size": 20.0,
        "axes.labelsize": 20.0,
        "axes.titlesize": 20.0,
        "legend.fontsize": 18,
        "legend.title_fontsize": 20.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
    },
)

DEFAULT_SNS_CONTEXT = sns.plotting_context()


class SubmissionRun(TypedDict):
    group_name: str
    run_key: str
    """For example "(Cartpole-v5)\""""

    run_id: str
    status: str
    submission_name: str | None
    """Derived from the submission file name"""

    file_path: str | None
    submission_id: str | None


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

    title: bool = True
    """Whether to show titles on the plots."""

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


def _relative_path_to_any_base(path: Path, bases: Sequence[Path]) -> Path | None:
    """
    Return `path` relative to the first matching base in `bases`.

    Args:
        path: The file path to relativize.
        bases: Candidate base directories. If `path` lies under any of these
            bases, the relative subpath is returned.

    Returns:
        The relative path if a base matches; otherwise `None`.
    """
    rp = path.resolve()
    for base in bases:
        try:
            rel = rp.relative_to(Path(base).resolve())
        except Exception:
            continue
        else:
            return rel
    return None


# DIR_FLAGS = {"use_kl_loss", "vf_share_layers", "large", "no_kl_loss", "no_shared_layers"}


def make_zip_arcname(
    file_path,
    base_candidates: list[Path],
    run_infos: dict[str, SubmissionRun] | None = None,
    *,
    use_dir_flags: bool = True,
) -> str:
    """
    Parts:
     -1 filename
     -2 metric
     -3 "plots"
     -4 experiment_dir
     -5 tune group folder
     -6 project dir
    """
    EXP_DIR_IDX = -4
    GROUP_DIR_IDX = -5
    rel = _relative_path_to_any_base(file_path, base_candidates)
    parts = list(rel.parts) if rel else list(file_path.parts)
    if run_infos:
        run_id = "RunIDNotFound In path"
        # get run_id from file_path
        if len(parts) >= abs(EXP_DIR_IDX):
            run_id = parts[EXP_DIR_IDX].rsplit("-")[-1]
            run_info = run_infos.get(run_id)
        else:
            run_info = None
        if not run_info:
            _logger.warning("Could not get a runinfo from path %s <- %s", run_id, rel)
        else:
            parts[EXP_DIR_IDX] += run_info["group_name"].replace("pbt_", "")
            if run_info.get("submission_name"):
                parts[EXP_DIR_IDX] += run_info["submission_name"]  # pyright: ignore[reportOperatorIssue]
            if use_dir_flags:
                insert = []
                submission_name = run_info.get("submission_name", "")
                if submission_name:
                    insert.append(submission_name)
                submission_id = run_info.get("submission_id", "")
                if submission_id and ("large" in submission_id or "8192" in submission_id):
                    insert.append("large")
                if insert:
                    parts[GROUP_DIR_IDX + 1] = "_".join(insert)
        rel = Path(*parts)
    if use_dir_flags and "large" in str(file_path.name):
        # rename parent
        # path should have structure of **/experiment/plots/*large
        # arcname to **/experiment(large)/plots/*large
        if len(parts) >= abs(EXP_DIR_IDX) and not any("large" in p for p in parts[EXP_DIR_IDX]):
            if "-mlp-" not in parts[EXP_DIR_IDX]:
                _logger.warning(
                    "Part '%s' does not contain the expected '-mlp-' substring. Parts: %s", parts[EXP_DIR_IDX], parts
                )
            rel = Path(*parts)
    arcname = rel if rel is not None else Path(file_path.name)
    arcname_str = str(arcname).replace(":", "_")
    return arcname_str
