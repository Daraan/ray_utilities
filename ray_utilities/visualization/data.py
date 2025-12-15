from __future__ import annotations

import concurrent.futures
import logging
import traceback
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Sequence, TypeAlias, TypeVar, cast
from zipfile import ZIP_DEFLATED, ZipFile

import base62
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm, colormaps, patheffects
from tqdm import tqdm
from typing_extensions import Final, Sentinel, TypeVarTuple, Unpack

# from ray_utilities.constants import EPISODE_RETURN_MEAN_EMA
# from ray_utilities.testing_utils import remote_breakpoint

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

PATHS = [Path("outputs/experiments/shared"), Path("outputs/experiments/shared_backup")]

DROP_COLUMNS = [
    "hostname",
    "done",
    ("config", "cli_args", "evaluate_every_n_steps_before_step"),
    ("config", "cli_args", "offline_loggers"),
    ("config", "cli_args", "fcnet_hiddens"),
    ("config", "cli_args", "perturbation_factors"),
    ("config", "cli_args", "comment"),
    ("config", "cli_args", "head_fcnet_hiddens"),
    ("config", "cli_args", "tune"),
    ("config", "_config_files"),
    ("config", "env_seed"),  # can be list in older versions
    # Maybe want to keep but not hashable
    ("config", "fork_from", "parent_time"),
]

LOG_SETTINGS = {"lr", "batch_size"}

nan: Final = np.nan

Placeholder = Sentinel("Placeholder")
Ts = TypeVarTuple("Ts")

# Experiment output directory structure is - if sorted in groups else one level higher

# Project/group/Name*run_id/

logger = logging.getLogger(__name__)

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
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
    },
)

# mpl.rcParams["savefig.facecolor"] = "#f7f7f7"


def ifill(
    *cols: Unpack[Ts], n: int
) -> tuple[Unpack[Ts]] | tuple[Unpack[Ts], Sentinel] | tuple[Unpack[Ts], Sentinel, Sentinel] | tuple[Sentinel, ...]:
    return (*cols, *(Placeholder,) * (n - len(cols)))


def load_run_data(offline_run: str | Path, experiment_dir=None):
    if isinstance(offline_run, str):
        if "/" in offline_run:
            # Assume path
            offline_run = Path(offline_run)
        else:
            # find it
            base_paths = [Path(experiment_dir)] if experiment_dir else PATHS
            for base_path in base_paths:
                # search in project/group/*run_id and project/*run_id then go to next base path
                search_with_group = list(base_path.glob(f"*/*/*{offline_run}"))
                if search_with_group:
                    run_path = search_with_group
                    break
                search_without_group = list(base_path.glob(f"*/*{offline_run}"))
                if search_without_group:
                    run_path = search_without_group
                    break
            else:
                raise FileNotFoundError(f"Run ID {offline_run} not found in any of the base paths.")
            if len(run_path) > 1:
                raise ValueError(f"Multiple runs found for ID {offline_run}: {run_path}")
            offline_run = run_path[0]
    if not offline_run.exists():
        raise FileNotFoundError(f"Run path {offline_run} does not exist.")
    # Load data
    result_files = offline_run.glob("*/result*.json")
    run_data = {}
    num_errors = 0
    for result_file in tqdm(result_files):
        if result_file.name.endswith(".parent.json"):
            continue
        try:
            df = pd.read_json(result_file, lines=True)
            # Drop unwanted columns if they exist
            # Flatten nested dict columns and convert to MultiIndex
            # ptimes = df[("config", "fork_from", "parent_time")]
            # mask = ~ptimes.isna()

            # df.loc[mask.values, ("config", "fork_from", "parent_time")] = ptimes[mask.values].map(str).values
            df = pd.json_normalize(df.to_dict(orient="records"), sep="/")
            # Convert columns to MultiIndex, replacing NaN with Placeholder
            tuples = [
                tuple(Placeholder if (isinstance(part, float) and np.isnan(part)) else part for part in col.split("/"))
                for col in df.columns
            ]
            df.columns = pd.MultiIndex.from_tuples(tuples)
            df = df.sort_index(axis=1, level=range(len(df.columns[0]))).drop(columns=DROP_COLUMNS, errors="ignore")
            # df.columns = pd.MultiIndex.from_tuples(cast("list[tuple[str, ...]]", df.columns))
            try:
                experiment_key = df.config.experiment_key.iloc[-1].item()
                assert result_file.name == "result.json" or experiment_key in result_file.name, (
                    f"Experiment key {experiment_key} does does not match id in file name {result_file}"
                )
            except AttributeError as ae:
                if "experiment_key" not in str(ae):
                    raise
                # Older versions without this field, take from filename
                if result_file.name == "result.json":
                    assert df.trial_id.nunique().item() == 1
                    experiment_key = df.iloc[0].trial_id.item()
                else:
                    experiment_key = result_file.stem.split("-")[-1]
                    if "fork_from" in df.config:
                        # can be nan if it is a continued fork and we did not include this info
                        last_non_nan: int | None = df.config.fork_from["fork_id_this_trial"].last_valid_index()
                        key_info_from_config = "Not fork_id_this_trial found"
                        if last_non_nan is not None:
                            key_info_from_config = df.iloc[last_non_nan].config.fork_from["fork_id_this_trial"].item()
                        assert last_non_nan is None or experiment_key == key_info_from_config, (
                            f"Experiment key {experiment_key} does not match fork_from key {key_info_from_config}"
                        )
                # experiment_key = df.config.experiment_id.values.item()+"_"+df.config.trial_id.values.item()
                if experiment_key in run_data:
                    raise ValueError(f"Duplicate experiment_key {experiment_key} found in {offline_run}")  # noqa: B904
        except Exception as e:  # noqa: PERF203
            num_errors += 1
            logger.error(f"Failed to load run data for {result_file}: {e!r}", exc_info=num_errors <= 2)  # noqa: G004
            if num_errors >= 6:
                logger.error("Too many errors encountered while loading run data; aborting further attempts.")
                raise
        else:
            run_data[experiment_key] = df

    logger.info(f"Loaded data for run {offline_run} with {len(run_data.keys())} experiments")  # noqa: G004
    return run_data


__logged_base62_value_error = False


def __base62_sort_key(s: str) -> tuple[int, int]:
    if not isinstance(s, str):
        return s
    if s.endswith("Z") or "S" not in s:
        secondary = 0
        try:
            if "_" in s:
                # trial_00012
                secondary = int(s.rsplit("_", 1)[-1])
            elif "C" in s:
                _, part2 = s.split("C", 1)
                secondary = int(part2[:2])
        except ValueError:
            global __logged_base62_value_error  # noqa: PLW0603
            if not __logged_base62_value_error:
                logger.exception(f"Failed to parse secondary sort key for {s!r}")
                __logged_base62_value_error = True
        return 0, secondary
    # We should not need a secondary here as they should be already ordered.
    return base62.decode(s.split("S", 1)[-1]), 0


def _base62_sort_key(s: pd.Index):
    return s.map(__base62_sort_key)


def combine_df(dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple DataFrames into a single DataFrame by concatenation.

    Args:
        dataframes: Dictionary of DataFrames to combine.

    Returns:
        A single DataFrame resulting from concatenation of the input DataFrames.
    """

    # Remove 'run_id' columns at any level if they exist to avoid conflicts
    def drop_run_id_columns(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            mask = [not any(lvl == "run_id" for lvl in col) for col in df.columns]
            return df.loc[:, mask]
        return df.drop(columns=["run_id"], errors="ignore")

    dfs: list[pd.DataFrame] = []
    for df in dataframes.values():
        df_clean = drop_run_id_columns(df)
        # Find the correct column name for 'training_iteration' in MultiIndex or single index
        if isinstance(df_clean.columns, pd.MultiIndex):
            idx_name = next((col for col in df_clean.columns if col[0] == "training_iteration"), None)
            if idx_name is None:
                raise KeyError("No 'training_iteration' column found in DataFrame columns.")
        else:
            idx_name = "training_iteration"
        df_clean = df_clean.set_index(idx_name, drop=True)
        dfs.append(df_clean)
    if len(dfs) == 0:
        raise ValueError("No DataFrames to combine.")
    try:
        combined_df = pd.concat(dfs, keys=dataframes.keys(), names=["run_id", "training_iteration"]).sort_index(
            key=_base62_sort_key
        )
        # Replace NaN in MultiIndex with the Placeholder sentinel
        if isinstance(combined_df.columns, pd.MultiIndex):
            new_tuples = [
                tuple(Placeholder if (isinstance(v, float) and np.isnan(v)) else v for v in idx)
                for idx in combined_df.columns
            ]
            combined_df.columns = pd.MultiIndex.from_tuples(new_tuples, names=combined_df.columns.names)
    except Exception as e:  # noqa: PERF203
        logger.error(f"Failed to concatenate DataFrames: {e!r}", exc_info=True)  # noqa: G004
        breakpoint()
        raise
    shape_before = combined_df.shape
    try:
        combined_df = combined_df.drop_duplicates(keep="first")
    except TypeError:
        configs = combined_df.pop("config")  # not hashable
        duplicates = combined_df.index.duplicated(keep="first")
        combined_df = combined_df[~duplicates]
        configs = configs[~duplicates]
        # Merge again
        # Add a MultiIndex level "config" to all columns in configs
        if not isinstance(configs.columns, pd.MultiIndex):
            configs.columns = pd.MultiIndex.from_product([["config"], configs.columns])
        else:
            configs.columns = pd.MultiIndex.from_tuples(
                [("config", *col) if isinstance(col, tuple) else ("config", col) for col in configs.columns]
            )
        combined_df = pd.concat([combined_df, configs], axis=1)
    logger.debug("Removing duplicates changed the shape from %s to %s", shape_before, combined_df.shape)

    # Reset only the outermost 'run_id' index, not all levels
    # combined_df = combined_df.reset_index(level=0)
    # Ensure 'run_id' is the first index level, and avoid duplicate index names
    # index_names = [n for n in combined_df.index.names if n != "run_id"]
    # combined_df = combined_df.set_index(["run_id", "training_iteration"], drop=True)
    # combined_df = combined_df.reorder_levels(["run_id", *index_names])
    depth = len(combined_df.columns[0])

    def ifill(*cols):
        return (*cols, *(Placeholder,) * (depth - len(cols)))

    continued_runs = which_continued(combined_df)
    if continued_runs.empty:
        # likely because failed before first perturb
        combined_df.loc[:, ifill("config", "__pbt_main_branch__")] = False
        return combined_df
    main_branch_mask = np.zeros_like(
        combined_df.index, dtype=bool
    )  # combined_df.index.get_level_values("run_id").isin(continued_runs.index)
    # BUT when the run a fork is no longer continued we may NOT set it to True in their last pbt_epoch
    last_pbt_epoch = combined_df[("config", "pbt_epoch")].groupby(level="run_id").max()
    for run_id in continued_runs.index:
        if run_id in last_pbt_epoch.index:
            # Set to true were we have a continued run but not where it has its max pbt_epoch
            run_mask = (combined_df.index.get_level_values("run_id") == run_id) & (
                combined_df.config.pbt_epoch != last_pbt_epoch.loc[run_id]
            ).to_numpy().flatten()
            main_branch_mask[run_mask] = True
        else:
            assert False

    try:
        combined_df.loc[:, ("config", "__pbt_main_branch__")] = main_branch_mask
    except KeyError:
        combined_df.loc[:, ifill("config", "__pbt_main_branch__")] = main_branch_mask

    return combined_df.sort_index(axis=1)


def make_cmap(
    values,
    name: str = "viridis",
    *,
    log: bool = False,
):
    # Create a continuous colormap for the group_stat (logarithmic scale)
    if log:
        norm = mcolors.LogNorm(
            vmin=values.replace(0, nan).min(),
            vmax=values.max(),
        )
    else:
        norm = mcolors.Normalize(
            vmin=values.min(),
            vmax=values.max(),
        )
    cmap = colormaps.get_cmap(name)

    # Map group_stat values to colors
    unique_stats = values.unique()
    color_map = {val: mcolors.to_hex(cmap(norm(val))) for val in unique_stats if val > 0}
    # For zero or negative values, fallback to a default color
    for val in unique_stats:
        if log and val <= 0:
            color_map[val] = "#cccccc"
    return color_map, cmap, norm


def _which_continued_legacy(df: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.str_]]:
    # When we have no pbt_epoch we need other methods.
    try:
        parent_ids = df.config.fork_from["parent_fork_id"].dropna().unique()
    except AttributeError as ae:
        if "unique" not in str(ae):
            raise
        # need to drop levels
        depth = df.config.fork_from.columns.nlevels
        parent_ids = df.config.fork_from.droplevel(axis=1, level=list(range(1, depth))).parent_fork_id.dropna().unique()
    return parent_ids


def which_continued(df: pd.DataFrame) -> pd.DataFrame:
    # there is no continued key we need to check which run_id (in the index) spans over multiple pbt_epochs
    resort = False
    try:
        pbt_epochs = df[("config", "pbt_epoch")]
    except KeyError as ke:
        if "pbt_epoch" not in str(ke):
            raise
        continued_ids = _which_continued_legacy(df)
        # bring into expected format
        continued = pd.DataFrame(index=continued_ids, columns=["continued"])
        # How many epochs where they continued?
        # parent_env_steps = df.config.fork_from.parent_env_steps

        # This might be off when there was a reload or some other error:
        perturbation_intervals = (
            df.config.fork_from.droplevel(axis=1, level=list(range(1, df.config.fork_from.columns.nlevels)))
            .parent_env_steps.fillna(0)
            .astype(int)
            .unique()
        )
        if len(perturbation_intervals) <= 1:
            # was never perturbed, return empty
            return pd.DataFrame(columns=["continued"])
        perturbation_interval = perturbation_intervals[1]
        # pbt_epochs_estimated = (parent_env_steps // perturbation_interval).groupby(level="run_id").max().reindex(continued.index, fill_value=0)
        indexer = ifill("config", "pbt_epoch", n=df.columns.nlevels)
        # As the perturbation_interval is the last step of an epoch we need to subtract 1
        df.loc[:, indexer] = (df.current_step - 1) // perturbation_interval
        pbt_epochs = df[("config", "pbt_epoch")]
        resort = True
    if "pbt_group_key" not in df.config:
        from ray_utilities.tune.scheduler.grouped_top_pbt_scheduler import GroupedTopPBTTrialScheduler  # noqa: PLC0415

        def map_group_key(row: pd.Series) -> str:
            groups = row.config.cli_args.get("tune")
            if not groups:
                groups = [row.config.experiment_group.values.item().split(":")[-1]]  # noqa: PD011
            mutations = dict.fromkeys(groups, None)

            ns = SimpleNamespace(_hyperparam_mutations=mutations)
            return GroupedTopPBTTrialScheduler._build_group_key_from_config(
                ns,
                row.droplevel(level=list(range(1, row.index.nlevels))),  # pyright: ignore[reportArgumentType]
            )

        group_keys = df.apply(map_group_key, axis=1)
        df.loc[:, ifill("config", "pbt_group_key", n=df.columns.nlevels)] = group_keys
        resort = True

    if resort:
        df.sort_index(axis=1, inplace=True)
    # Ignore "training_iteration" in the index by resetting it if present
    idx_names = list(df.index.names)
    # if "training_iteration" in idx_names:
    #    pbt_epochs = pbt_epochs.reset_index("training_iteration", drop=True)
    epoch_counts = pbt_epochs.groupby(level="run_id").nunique()

    # Method 1: TODO: Maybe not correct for older runs that do not set it on perturb
    continued = epoch_counts[(epoch_counts > 1).values]  # type: ignore  # noqa: PD011
    continued.columns = ["continued"]
    return continued
    # Method 2:
    mask = df[("config", "__pbt_main_branch__")] == True  # noqa: E712
    return df[mask.values].index.get_level_values("run_id").unique()


def _connect_groups(last: pd.DataFrame, now: pd.DataFrame, stat_value: str | None) -> pd.DataFrame:
    last_entries = last.xs(
        last.index.get_level_values("training_iteration").max(),
        level=1,
        drop_level=False,
        axis=0,
    )
    # Add immediate connection to start value
    if stat_value is None:
        return pd.concat(
            [
                last_entries,
                now,
            ],
            axis=0,
        )
    new_value = now.iloc[0][stat_value]
    last_entries2 = last_entries.copy()
    last_entries2["current_step"] += 128
    last_entries2[stat_value] = new_value
    return pd.concat(
        [
            last_entries,
            last_entries2,
            now,
        ],
        axis=0,
    )


TEST = 0


def shade_background(ax: Axes, color: str, left: float, right: float, **kwargs):
    kwargs.setdefault("alpha", 0.2)
    ax.axvspan(
        left,
        right,
        color=color,
        zorder=0,
        **kwargs,
    )


def plot_run_data(
    df: pd.DataFrame,
    metrics: Sequence[str | tuple[str, ...]],
    experiment_keys: list[str] | None = None,
    figsize: tuple[int, int] = (12, 8),
    group_stat: str | None = None,
    group_by=("pbt_epoch", "pbt_group_key"),
    *,
    pbt_metric: str | tuple[str, ...] | None = None,
    log: bool | None = None,
    main_only: bool = False,
    plot_reduced: bool = True,
    show: bool = True,
    pbt_plot_interval: int = 4,
) -> Figure:
    """Plot specified metrics from the run data.

    Args:
        df: Combined DataFrame with MultiIndex columns and index.
        metrics: List of metric names to plot.
        pbt_metric: The selection metric that was used during PBT. This metric
            will be used to determine the best runs for highlighting and ordering.
        experiment_keys: Optional list of experiment keys to include in the plot.
            If None, all experiments in df are used.
        smoothing: Smoothing window size for rolling mean. Default is 1 (no smoothing).
        figsize: Size of the figure to create.
        group_stat: Refers to the column that relates to the grouping statistic, e.g. pbt_group_key.
            By default the last key of group_by is used.
        group_by: Tuple of column names (as strings, not tuples) to group by under 'config'.
        log: Whether to use logarithmic scale for the group_stat color mapping. If None, checks LOG_SETTINGS.
        show: Whether to display the plot immediately.
        main_only: Whether to plot only the main branch runs.
        plot_reduced: When True, to reduce clutter will plot curves of the best, second best, and worst of each group_stat,
            the old value +/- 1 level and the new value +/- 1 level and the second best.
    """
    depth = len(df.columns[0])

    def ifill(*cols):
        return (*cols, *(Placeholder,) * (depth - len(cols)))

    if experiment_keys is None:
        experiment_keys = df.index.get_level_values(0).unique().to_list()
        if TEST:
            experiment_keys = experiment_keys[:5]
    if group_stat is None:
        # assumes a name=... format
        group_stat = df.iloc[0].config[group_by[-1]].str.split("=").iloc[0][0]
        assert group_stat is not None
    if group_stat == "batch_size" and "batch_size" not in df.config:
        df.loc[:, ifill("config", "batch_size")] = df.config.train_batch_size_per_learner
    final_metric_was_none = pbt_metric is None
    log = log if log is not None else group_stat in LOG_SETTINGS

    # Secondary x-axis mappers
    first_change: tuple[str, int] = df[ifill("config", "pbt_epoch")].diff().iloc[1:].ne(0).idxmax()  # pyright: ignore[reportAssignmentType]
    perturbation_interval = df.current_step.loc[(first_change[0], first_change[1] - 1)].item()
    secondard_to_main = lambda xs: (xs + 0.5) * perturbation_interval  # noqa: E731
    main_to_secondary = lambda xs: (xs - perturbation_interval / 2) / perturbation_interval  # noqa: E731
    num_pbt_epochs = df[ifill("config", "pbt_epoch")].max().item() + 1

    # Select the group-by columns from MultiIndex columns
    group_cols = [("config", k) for k in group_by]
    # Extract the group-by columns as a DataFrame (each column is 1D)
    try:
        group_values: list[pd.Series] = [
            df.current_step.droplevel(0, axis=1),
            *(df[col] for col in group_cols),
            df.config.seed,
            df[group_stat].droplevel(0, axis=1),
        ]
    except AttributeError as ae:
        if "seed" not in str(ae):
            raise
        df.loc[:, ifill("config", "seed")] = df.config.cli_args.seed.to_numpy()
        df.sort_index(axis=1, inplace=True)
        group_values: list[pd.Series] = [
            df.current_step.droplevel(0, axis=1),
            *(df[col] for col in group_cols),
            df.config.seed,
            df[group_stat].droplevel(0, axis=1),
        ]

    # Combine into a DataFrame for groupby
    group_df = pd.concat(group_values, axis=1, keys=["current_step", *group_by, "seed", group_stat], copy=False)
    # Problem we get duplicated values as the dfs contain their parents data - need to drop these when we aggregate
    group_df = group_df[~group_df.duplicated(keep="first")]
    group_df.columns = ["current_step", *group_by, "seed", group_stat]  # flatten for groupby
    group_df = group_df.drop("seed", axis=1)

    # Example usage: grouped = df.groupby([group_df[k] for k in group_keys])
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=figsize)

    if num_metrics == 1:
        axes = [axes]

    # continued_runs = which_continued(df)
    # cont_mask = df.index.get_level_values("run_id").isin(continued_runs)
    # df.loc[cont_mask, ("config", "__pbt_main_branch__")] = True

    for i, metric in enumerate(metrics):
        if final_metric_was_none:
            pbt_metric = metric
        if (metric == "episode_reward_mean" or pbt_metric == "episode_reward_mean") and "episode_reward_mean" not in df:
            from ray_utilities.constants import EPISODE_RETURN_MEAN, EPISODE_RETURN_MEAN_EMA  # noqa: PLC0415

            # introduced it later
            if EPISODE_RETURN_MEAN_EMA in df.evaluation:
                backport_metric = ("evaluation", EPISODE_RETURN_MEAN_EMA)
            else:
                backport_metric = ("evaluation", EPISODE_RETURN_MEAN)
            logger.info(
                "Metric 'episode_reward_mean' was not found in the data, changing metric to: %s", backport_metric
            )
            if pbt_metric == "episode_reward_mean":
                pbt_metric = backport_metric
            if metric == "episode_reward_mean":
                metric = backport_metric  # noqa: PLW2901
        assert pbt_metric is not None
        ax: Axes = axes[i]
        ax2 = ax.twinx()
        ax2.set_ylabel(group_stat)
        # we did not save perturbation interval, check where pbt_epoch first changes
        # Use transform to add it relative to the data
        if pbt_plot_interval:
            secax = ax.secondary_xaxis("top", functions=(main_to_secondary, secondard_to_main), transform=None)
            secax.set_xlabel("PBT Epoch")
            secax.set_xticks([e for e in range(num_pbt_epochs) if e % pbt_plot_interval == 1])
            # Show tick labels for secondary xaxis, inside and closer to the plot
            secax.xaxis.set_tick_params(which="both", bottom=False, top=False, labelbottom=True, labeltop=False, pad=-1)
            secax.xaxis.set_label_position("top")
            # Also move the label closer to the plot
            secax.set_xlabel("PBT Epoch", labelpad=3)
        # Move main x-axis tick labels closer to the plot
        ax.xaxis.set_tick_params(pad=2)
        ax.set_xlabel("Environment Steps", labelpad=3)
        if log:
            ax2.set_yscale("log")

        metric_key = metric if isinstance(metric, str) else metric[-1]

        plot_df: pd.DataFrame = pd.concat(
            [
                (
                    df[["current_step", metric]]
                    if isinstance(metric, str)
                    else df[[ifill("current_step"), ifill(*metric)]]
                ),
                df[[ifill("config", group_by[0]), ifill("config", "__pbt_main_branch__")]],
                df[("config", group_stat)].round(10),
                *(
                    ()
                    if metric == pbt_metric
                    else [df[pbt_metric if isinstance(pbt_metric, str) else ifill(*pbt_metric)]]
                ),
            ],
            axis=1,
        )
        pbt_metric_key = pbt_metric if isinstance(pbt_metric, str) else pbt_metric[-1]
        if metric != pbt_metric and pbt_metric_key == metric_key:
            pbt_metric_key = f"pbt_{pbt_metric_key}"
        plot_df.columns = ["current_step", metric_key, group_by[0], "__pbt_main_branch__", group_stat] + (
            [pbt_metric_key] if metric != pbt_metric else []
        )
        plot_df["__pbt_main_branch__"] = plot_df["__pbt_main_branch__"].infer_objects(copy=False).fillna(False)  # noqa: FBT003
        # plot_df.sort_values(["training_iteration", "__pbt_main_branch__", metric], inplace=True)
        assert group_stat in plot_df

        color_map, cmap, norm = make_cmap(plot_df[group_stat], log=log)
        # Sort each subgroup by their std from highest to lowest
        # Sort plot_df within each (group_by[0], group_stat) group by the std of the metric (descending)
        if sort_by_std := True:
            std_by_group = (
                plot_df.groupby([group_by[0], group_stat])[metric_key]  # no multiindex here
                .std()
                .sort_values(ascending=False)
            )
            # Reorder plot_df so that rows belonging to groups with higher std appear first
            plot_df["__group_sort__"] = list(zip(plot_df[group_by[0]], plot_df[group_stat], strict=True))
            plot_df["__group_sort__"] = pd.Categorical(
                plot_df["__group_sort__"], categories=list(std_by_group.index), ordered=True
            )
        else:
            plot_df["__group_sort__"] = 0

        # Assign ranks for each epoch based on their last iteration metric value
        # TODO: do this by final_metric_key?
        max_group_steps = plot_df.groupby([group_by[0], group_stat, "current_step"])[pbt_metric_key].mean()
        max_group_steps_frame = max_group_steps.to_frame().reset_index()
        # Select only the value from highest current step
        max_group_current_step = max_group_steps_frame.groupby([group_by[0], group_stat])["current_step"].max()
        max_step_indexer = (
            max_group_current_step.to_frame().reset_index().set_index([group_by[0], group_stat, "current_step"])
        )
        group_ranks = max_group_steps.loc[max_step_indexer.index].groupby([group_by[0]]).rank(method="max")
        max_rank = group_ranks.max()
        second_max_rank = group_ranks[group_ranks != max_rank].max()
        min_rank = group_ranks.min()
        ranks_to_keep = {max_rank, second_max_rank, min_rank}

        # For each (group_by[0], group_stat) assign their rank based on last iteration metric value
        # Assign group_ranks to the respective rows with matching (group_by[0], group_stat)
        try:
            plot_df["__group_rank__"] = plot_df[[group_by[0], group_stat]].apply(
                lambda x: group_ranks.loc[(x[group_by[0]], x[group_stat])].item(), axis=1
            )
        except KeyError:  # noqa: TRY203
            # remote_breakpoint()
            raise

        # ATTENTION # XXX - for older runs the pbt_epoch might go into the next epoch if a run was continued!

        # As there is not last perturbation check which run is at the end highest and sort it to the top

        # TODO: Does likely not work for uneven lengths of runs, i.e. when we tune batch_size
        # then need to check for current_step max
        if plot_reduced and not main_only:
            max_stat = plot_df[group_stat].max()
            min_stat = plot_df[group_stat].min()
            stats_to_keep = {max_stat, min_stat}
            continued_runs = which_continued(df)
            # Need to know which the main groups are
            # Remove those that do not have the highest rank, second highest rank, lowest rank,
            # And highest lowest group_stat, however keep last group_stat always

            def filter_groups(group: pd.DataFrame) -> pd.DataFrame:
                main_group = group.index.get_level_values("run_id").isin(continued_runs.index).any().item()  # noqa: B023
                example = group.iloc[0]
                logger.debug(
                    "Filtering group: epoch %s stat %s value - Rank %s Main group: %s Keep: %s",
                    example[group_by[0]],
                    example[group_stat],
                    example["__group_rank__"],
                    main_group,
                    main_group or example["__group_rank__"] in ranks_to_keep or example[group_stat] in stats_to_keep,  # noqa: B023
                )

                def keep_row(row: pd.Series) -> bool:
                    r = row["__group_rank__"]
                    s = row[group_stat]

                    return main_group or r in ranks_to_keep or s in stats_to_keep  # noqa: B023

                return group[group.apply(keep_row, axis=1)]

            logger.debug("Applying group filtering to reduce clutter.... Shape before %s", plot_df.shape)
            plot_df = plot_df.groupby([group_by[0], group_stat], group_keys=False)[plot_df.columns].apply(filter_groups)
            logger.debug("Shape after %s", plot_df.shape)
        plot_df = plot_df.sort_values(
            ["training_iteration", "__pbt_main_branch__", "__group_sort__", "__group_rank__", metric_key]
        )
        # Iterators:
        max_epoch = plot_df[group_by[0]].max()
        last_data = plot_df[plot_df[group_by[0]] == max_epoch]
        # TODO: What about those runs that were trained too long
        last_data = last_data.sort_values(["training_iteration", "__group_sort__", "__group_rank__", metric_key])
        last_group_iter = iter(last_data.groupby([group_by[0], group_stat], sort=False))
        grouper = plot_df.groupby([group_by[0], group_stat], sort=False)

        # Plot Non continues
        # Plot each group separately to control color
        last_main_group = None
        last_epoch = -1
        # Track background shading for group changes
        background_colors = ["#f0f0f0", "#909090"]
        last_bg_epoch = None
        bg_color_idx = 0
        prev_group: pd.DataFrame = None  # type: ignore[assignment]
        seen_labels = set()
        for i, ((pbt_epoch, stat_val), group) in enumerate(grouper):
            # Add background shade when group changes
            if pbt_epoch == max_epoch:
                (pbt_epoch, stat_val), group = next(last_group_iter)  # noqa: PLW2901

            if pbt_epoch != last_epoch:
                last_epoch = pbt_epoch
            if TEST and pbt_epoch > 1:
                break
            is_main = False
            if group["__pbt_main_branch__"].any() or (
                pbt_epoch == max_epoch and group["__group_rank__"].max() == max_rank
            ):
                is_main = True
                if last_main_group is not None:
                    # Add connection to last points
                    group = _connect_groups(last_main_group, group, group_stat)  # noqa: PLW2901
                logger.debug("Main group: %s %s", pbt_epoch, stat_val)
                last_main_group = group
                sns.lineplot(
                    data=group[["current_step", group_stat]].drop_duplicates(),
                    x="current_step",
                    y=group_stat,
                    ax=ax2,
                    color=color_map[stat_val],
                    linestyle="-",
                    linewidth=2,
                    path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
                )

                # On a secondary axis we want to plot the group_stat value over time
                # Plot group_stat value over time on a secondary y-axis
            elif main_only:
                logger.debug("Skipping non-main group: %s %s", pbt_epoch, stat_val)
                continue
            elif last_main_group is not None:
                logger.debug("Plotting non-main group: %s %s", pbt_epoch, stat_val)
                group = _connect_groups(last_main_group, group, group_stat)  # noqa: PLW2901

            # Shading:
            if last_bg_epoch is None or pbt_epoch != last_bg_epoch:
                logger.debug("plotting shade %s", pbt_epoch)
                # Shade the region for this epoch
                if last_bg_epoch is not None:
                    # Shade from previous group's min step to its max step (the region of the previous group)
                    prev_min: float = prev_group["current_step"].min()
                    prev_max: float = prev_group["current_step"].max()
                    shade_background(ax, background_colors[bg_color_idx % 2], prev_min, prev_max)
                    bg_color_idx += 1
                else:
                    # Shade from left edge to first group's min step
                    curr_min = group["current_step"].min()
                    shade_background(ax, background_colors[bg_color_idx % 2], ax.get_xlim()[0], curr_min)
                    bg_color_idx += 1
            if pbt_epoch % pbt_plot_interval == 0:
                # print
                ...

            last_bg_epoch = pbt_epoch
            prev_group = group

            # Plot
            sns.lineplot(
                data=group,
                x="current_step",
                y=metric_key,
                ax=ax,
                # linestyle="--",
                color=color_map[stat_val],
                label=str(stat_val) if stat_val not in seen_labels else None,
                linewidth=3,
            )
            seen_labels.add(stat_val)
        # Shade from last group's max step to right edge
        if "prev_group" in locals():
            prev_max = prev_group["current_step"].max()
            shade_background(ax, background_colors[bg_color_idx % 2], prev_max, ax.get_xlim()[1])

        # Add a colorbar for the continuous group_stat
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        sm.set_clim(norm.vmin * 0.95, norm.vmax * 1.05)  # pyright: ignore[reportOptionalOperand]
        ax.set_xlim(-1000, plot_df.current_step.max() + 1000)
        ax2.set_ylim(norm.vmin * 0.95, norm.vmax * 1.05)  # pyright: ignore[reportOptionalOperand]
        # Make the colorbar 20% smaller by adjusting its fraction
        cbar = plt.colorbar(sm, ax=ax2, pad=0.01, fraction=0.075)  # default fraction is ~0.1
        # Remove all ticks from both sides of the colorbar
        cbar.ax.tick_params(axis="both", which="both", left=False, right=True, labelleft=False, labelright=False)
        cbar.set_ticks([])
        ax2.tick_params(axis="y", which="both", left=False, right=False, labelleft=False, labelright=True)

        # Right tick labels are bleeding into the colormap if we use fancytoolbox legend
        # Move right y-axis tick labels further right by 80%
        for label in ax2.get_yticklabels():
            label.set_x(label.get_position()[0] + 0.03)

        # Align the limits of ax2 and the colorbar to be identical

        # ax.set_title(f"Metric: {metric_key}")
        # ax.set_xlabel("Training Steps")
        ax.set_ylabel(metric_key)
        # Custom legend: one entry per unique pbt_group_key
        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicate labels, keep order
        seen = set()
        unique = []
        for h, l in zip(handles, labels, strict=True):
            if l not in seen:
                unique.append((h, l))
                seen.add(l)
        # Only keep entries where label is a pbt_group_key (not e.g. 'mean', etc.)
        group_keys = [lbl for _, lbl in unique if lbl is not None]
        # Remove duplicates while preserving order of appearance
        seen_keys = set()
        group_keys_ordered = []
        for k in group_keys:
            if k not in seen_keys:
                group_keys_ordered.append(k)
                seen_keys.add(k)

        # Sort group_keys_ordered by their group_stat value (convert to float if possible)
        def _try_float(x):
            try:
                return float(x)
            except Exception:
                return x

        group_keys_sorted = sorted(group_keys_ordered, key=_try_float, reverse=True)
        # Filter and order legend entries according to group_keys_sorted
        filtered_sorted = []
        for key in group_keys_sorted:
            for h, lbl in unique:
                if lbl == key:
                    filtered_sorted.append((h, lbl))
                    break
        if filtered_sorted:
            handles, labels = zip(*filtered_sorted, strict=True)
        # Place the legend below the plot in a fancybox
        legend = ax.legend(
            handles,
            labels,
            title=group_stat,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(len(labels), 5),
            fancybox=True,
            framealpha=0.8,
        )
        # Reduce legend font size by 2
        fontsize = max(legend.get_texts()[0].get_fontsize() - 2, 1) if legend.get_texts() else 10
        for text in legend.get_texts():
            text.set_fontsize(fontsize)
        if legend.get_title() is not None:  # pyright: ignore[reportUnnecessaryComparison]
            legend.get_title().set_fontsize(fontsize)
        if (legend2 := ax2.get_legend()) is not None:
            legend2.remove()
        ax.set_title(
            f"{df.iloc[0].config.cli_args.env_type.item()} "
            f"- {df.iloc[0].config.cli_args.agent_type.item()} "
            f"- {metrics[0]}"
        )
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_n_save(
    df: pd.DataFrame,
    metrics: Sequence[str | tuple[str, ...]],
    save_path: str | Path,
    experiment_keys: list[str] | None = None,
    group_stat: str | None = None,
    group_by=("pbt_epoch", "pbt_group_key"),
    *,
    figsize: tuple[int, int] = (12, 8),
    log: bool | None = None,
    format="pdf",
    **kwargs,
) -> None:
    # for metric in metrics: [metric]
    fig = plot_run_data(
        df,
        metrics,
        experiment_keys,
        figsize,
        group_stat,
        group_by,
        log=log,
        show=False,
        **kwargs,
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists() and not save_path.is_dir():
        # check extension
        assert save_path.suffix == f".{format}", (
            f"File {save_path} already exists with different format "
            f"({save_path.suffix} != .{format}). Change the name or format,"
        )
    fig.savefig(save_path, format=format, bbox_inches="tight")
    logger.info(f"Saved plot to {save_path}")  # noqa: G004


def _join_nested(m):
    if isinstance(m, tuple):
        return "_".join(m)
    return m


def export_run_data(
    experiment_path: str | Path | pd.DataFrame,
    experiment_keys: list[str] | None = None,
    metrics: Sequence[str | tuple[str, ...]] = ("episode_reward_mean",),
    group_stat: str | None = None,
    group_by: Sequence[str] = ("pbt_epoch", "pbt_group_key"),
    *,
    figsize: tuple[int, int] = (12, 8),
    log: bool | None = None,
    format="pdf",
    save_path: str | Path | None = None,
    **kwargs,
):
    """
    TODO:
        - Some older runs do not have a pbt_group_key, and likely also no pbt_epoch
    """
    metric_str = metrics if isinstance(metrics, str) else "-".join(map(_join_nested, metrics))
    if not isinstance(experiment_path, pd.DataFrame):
        experiment_path = Path(experiment_path)
        if save_path is None:
            save_path = experiment_path / "plots"
        else:
            save_path = Path(save_path)
        logger.info("Loading run data %s ...", experiment_path.name)
        data = load_run_data(experiment_path)
        logger.debug("Combining dataframes...")
        combined_df = combine_df(data)
        out_path = save_path / f"{experiment_path.name}_{metric_str}.{format}"
    elif save_path is None:
        raise ValueError("When providing a DataFrame directly, save_path must be specified.")
    else:
        out_path = Path(save_path)
        combined_df = experiment_path
    logger.info("Plotting and saving...")

    plot_n_save(
        combined_df,
        metrics=metrics,
        save_path=out_path,
        experiment_keys=experiment_keys,
        group_stat=group_stat,
        group_by=group_by,
        figsize=figsize,
        log=log,
        format=format,
        **kwargs,
    )
    return out_path


def _export_one(args, group_by, figsize, format, **kwargs):
    experiment_path, metric = args
    try:
        file_path = export_run_data(
            experiment_path,
            metrics=(metric,),
            group_by=group_by,
            figsize=figsize,
            format=format,
            **kwargs,
        )
    except Exception as e:  # noqa: PERF203
        tb = traceback.format_exc()
        logger.error(f"Failed to export run data for {experiment_path}: {e!r}", exc_info=True)  # noqa: G004
        return e, tb, experiment_path
    else:
        logger.info(f"Exported run data for {experiment_path} to {file_path}")  # noqa: G004
        return file_path


def _export_multiple(
    experiment_path: Path,
    metrics: Sequence[str | tuple[str, ...]],
    group_by: Sequence[str],
    figsize: tuple[int, int],
    format: str,
    **kwargs: OptionalRepeat[Any],
) -> tuple[list[Path], list[tuple[Exception, str, Path]]]:
    experiment_path = Path(experiment_path)
    saved_files: list[Path] = []
    try:
        data = load_run_data(experiment_path)
    except Exception as e:  # noqa: PERF203
        tb = traceback.format_exc()
        logger.error(f"Failed to load run data for {experiment_path}: {e!r}", exc_info=True)  # noqa: G004
        return saved_files, [(e, tb, experiment_path)]

    logger.debug("Combining dataframes...")
    try:
        combined_df = combine_df(data)
    except Exception as e:  # noqa: PERF203
        tb = traceback.format_exc()
        logger.error(f"Failed to combine dataframes for {experiment_path}: {e!r}", exc_info=True)  # noqa: G004
        return saved_files, [(e, tb, experiment_path)]
    # Expand kwargs if any value is an Options instance (cross product over all Options)

    # Identify keys in kwargs whose values are Options
    option_keys = [k for k, v in kwargs.items() if isinstance(v, Repeat)]
    errors: list[tuple[Exception, str, Path]] = []
    if option_keys:
        # Build a list of lists for each Options value, or [v] if not Options
        option_values: list[Repeat[object]] = [kwargs[k] for k in option_keys]  # pyright: ignore[reportAssignmentType]
        # For each combination in the cross product, build a dict of overrides
        for combo in product(*option_values):
            combo_kwargs = kwargs.copy()
            for k, v in zip(option_keys, combo, strict=True):
                combo_kwargs[k] = v
            combo_kwargs = cast("dict[str, Any]", combo_kwargs)
            assert not any(isinstance(v, Repeat) for v in combo_kwargs.values()), (
                "All Repeat values should have been expanded."
            )
            for metric in metrics:
                metric_str = metrics if isinstance(metrics, str) else "-".join(map(_join_nested, metrics))
                try:
                    file_path = export_run_data(
                        combined_df,
                        metrics=(metric,),
                        group_by=group_by,
                        figsize=figsize,
                        format=format,
                        save_path=experiment_path / "plots" / f"{experiment_path.name}_{metric_str}.{format}",
                        **combo_kwargs,
                    )
                except Exception as e:  # noqa: PERF203
                    tb = traceback.format_exc()
                    logger.error(f"Failed to export run data for {experiment_path}: {e!r}", exc_info=True)  # noqa: G004
                    errors.append((e, tb, experiment_path))
                else:
                    logger.info(f"Exported run data for {experiment_path} to {file_path}")  # noqa: G004
                    saved_files.append(file_path)
        return saved_files, errors
    kws = cast("dict[str, Any]", kwargs)
    for metric in metrics:
        metric_str = metrics if isinstance(metrics, str) else "-".join(map(_join_nested, metrics))
        try:
            file_path = export_run_data(
                combined_df,
                metrics=(metric,),
                group_by=group_by,
                figsize=figsize,
                format=format,
                save_path=experiment_path / "plots" / f"{experiment_path.name}_{metric_str}.{format}",
                **kws,
            )
        except Exception as e:  # noqa: PERF203
            tb = traceback.format_exc()
            logger.error(f"Failed to export run data for {experiment_path}: {e!r}", exc_info=True)  # noqa: G004
            errors.append((e, tb, experiment_path))
        else:
            logger.info(f"Exported run data for {experiment_path} to {file_path}")  # noqa: G004
            saved_files.append(file_path)
    return saved_files, errors


T = TypeVar("T")


class Repeat(list[T]):
    pass


OptionalRepeat: TypeAlias = Repeat[T] | T


def export_all_runs(
    output_dir: str | Path,
    *,
    single: bool = False,
    format="pdf",
    figsize=(14, 10),
    group_by=("pbt_epoch", "pbt_group_key"),
    metrics=("episode_reward_mean",),
    test=TEST,
    max_workers: int = 4,
    zip_plots: bool = False,
    excludes: Sequence[str] = (),
    redo: bool = False,
    main_only: OptionalRepeat[bool] = False,
    plot_reduced: OptionalRepeat[bool] = True,
    **kwargs,
):
    """
    Export all runs in the given output directory using multiple processes.

    Args:
        output_dir: Directory containing experiment runs.
        format: Output file format.
        figsize: Figure size.
        group_by: Grouping columns.
        metrics: Metrics to plot.
        test: If set, limits the number of runs processed.
        max_workers: Number of parallel processes to use.
        **kwargs: Additional keyword arguments for export_run_data.

    Returns:
        List of saved file paths.
    """
    output_dir = Path(output_dir)
    saved_files: list[Path] = []
    test = int(test) - 1 if test else float("inf")
    # Submit tasks as soon as we get experiment paths from glob
    file_paths: list[Path] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        i = 0
        skip_dirs = set()
        for experiment_path in output_dir.glob("*/*/*") if not single else [output_dir]:
            if any(excl in str(experiment_path) for excl in excludes) or "cometml" in str(experiment_path):
                logger.info(f"Excluding experiment path {experiment_path} due to exclude patterns.")  # noqa: G004
                continue
            if not experiment_path.is_dir():
                # possibly not yet sorted into groups
                experiment_path = experiment_path.parent  # noqa: PLW2901
                if not (experiment_path / ".validate_storage_marker").exists():
                    logger.error(f"Missing .validate_storage_marker in {experiment_path}, skipping.")
                    skip_dirs.add(experiment_path)
                    continue
                if experiment_path in skip_dirs:
                    logger.info(f"Skipping previously failed experiment path {experiment_path}.")  # noqa: G004
                    continue
                skip_dirs.add(experiment_path)
                # raise ValueError(f"Experiment path {experiment_path} is not a directory.")
            filtered_metrics = metrics.copy()
            # Prepare all combinations of main_only and plot_reduced if they are Repeat, else just use as is
            main_only_options = main_only if isinstance(main_only, Repeat) else [main_only]
            plot_reduced_options = plot_reduced if isinstance(plot_reduced, Repeat) else [plot_reduced]
            # If not redo, check for all combinations and remove metrics that already exist for all combinations
            if not redo:
                metrics_to_remove = set()
                for metric in metrics:
                    for mo in main_only_options:
                        for pr in plot_reduced_options:
                            metric_str = "-".join(metric) if isinstance(metric, tuple) else metric
                            suffixes = []
                            if mo:
                                suffixes.append("main_only")
                            if pr:
                                suffixes.append("reduced")
                            else:
                                suffixes.append("all")
                            suffix_str = ("_" + "_".join(suffixes)) if suffixes else ""
                            out_path = (
                                output_dir / "plots" / f"{experiment_path.name}_{metric_str}{suffix_str}.{format}"
                            )
                            if out_path.exists():
                                metrics_to_remove.add((metric, mo, pr))
                                # Include in zip file paths even if skipping
                                file_paths.append(out_path)
                                logger.info(
                                    f"Plot for metric {metric} (main_only={mo}, plot_reduced={pr}) already exists at {out_path}, skipping."  # noqa: G004
                                )
                # Remove metrics for which all combinations exist
                # If all combinations for a metric exist, remove it from filtered_metrics
                for metric in metrics:
                    all_exist = all(
                        (metric, mo, pr) in metrics_to_remove
                        for mo in main_only_options
                        for pr in plot_reduced_options
                    )  # fmt: skip
                    if all_exist and metric in filtered_metrics:
                        filtered_metrics.remove(metric)
            futures.append(
                executor.submit(
                    _export_multiple,
                    experiment_path=experiment_path,
                    metrics=filtered_metrics,
                    group_by=group_by,
                    figsize=figsize,
                    format=format,
                    **kwargs,
                )
            )
            if i >= test:
                break
            i += 1

        # Collect results as they complete
        tracebacks = []
        zipf = None
        try:
            if zip_plots:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_path = output_dir / f"exported_plots_{timestamp}.zip"
                zipf = ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=9)
                for file_path in file_paths:
                    arcname = output_dir.name / file_path.relative_to(output_dir)
                    zipf.write(file_path, arcname=str(arcname))
            try:
                # Collect results as they complete
                file_paths: list[Path]
                errors: list[tuple[Exception, str, Path]]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    file_paths, errors = future.result()
                    if isinstance(file_paths, (Path, str)):
                        file_paths = [Path(file_paths)]
                    for file_path in file_paths:
                        saved_files.extend(file_paths)
                        arcname = output_dir.name / file_path.relative_to(output_dir)
                        if zipf is not None:
                            zipf.write(file_path, arcname=str(arcname))
                    for error_return in errors:
                        error, tb, failed_path = error_return
                        logger.error(f"Error during export of {failed_path} : {error}\n{tb}")  # noqa: G004
                        tracebacks.append((failed_path, tb))
            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt received: cancelling all running futures.")
                for f in futures:
                    f.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
        finally:
            if zipf is not None:
                zipf.close()

    if tracebacks:
        logger.error("Some exports failed with errors:")
        for failed_path, tb in tracebacks:
            print("\n--------------------------------\n")
            logger.error(f"Failed export for {failed_path}:\n{tb}")  # noqa: G004
    logger.info(f"Exported {len(saved_files)} files in total:\n%s", "\n".join(str(f) for f in saved_files))  # noqa: G004
    if zip_plots:
        logger.info(f"Zipped plots saved to {zip_path}")  # pyright: ignore[reportPossiblyUnboundVariable] # noqa: G004
    return saved_files


if __name__ == "__main__":
    # cd /path/to/parent
    # find dir1 dir2 -type f -iname '*.pdf' -printf '%P\n' | sed 's|^[^/]*/||' | zip -@ all_pdfs.zip
    import argparse
    from ast import literal_eval
    from datetime import datetime

    from ray_utilities import nice_logger

    logger = nice_logger(logger, logging.INFO)
    # logging.info("Set handle")  # noqa: LOG015
    # logger.setLevel()
    logger.info("Running data export script...")

    parser = argparse.ArgumentParser(description="Export and plot RLlib experiment results.")
    parser.add_argument("path", type=str, help="Experiment output directory or run path.")
    parser.add_argument("--all", action="store_true", help="Export all runs in the directory.")
    parser.add_argument("--main_only", action="store_true", help="Plot only the main branch runs.")
    parser.add_argument("--format", type=str, default="pdf", help="Output file format (default: pdf).")
    parser.add_argument("--metrics", type=str, nargs="+", default=None, help="Metrics to plot.")
    parser.add_argument(
        "--figsize", type=int, nargs=2, default=[14, 10], help="Figure size as two integers (width height)."
    )
    parser.add_argument(
        "--test", nargs="?", const=True, default=False, type=int, help="Run in test mode with limited data."
    )
    parser.add_argument("--pbt_metric", type=str, default="episode_reward_mean", help="PBT selection metric.")
    parser.add_argument(
        "--workers", "-w", "-p", type=int, default=4, help="Number of parallel workers for exporting all runs."
    )
    parser.add_argument("--zip", "-z", action="store_true", default=False, help="Whether to zip the output plots.")
    parser.add_argument(
        "--excludes", nargs="*", type=str, default=["def-workspace", "TESTING"], help="Experiment keys to exclude."
    )
    parser.add_argument("--redo", action="store_true", help="Redo existing plots.")
    parser.add_argument("--single", "-s", action="store_true", help="Export a single run instead of all runs.")

    args = parser.parse_args()

    assert not args.all or not args.main_only, "Cannot use --all and --main_only together."
    if args.metrics is None:
        args.metrics = ["episode_reward_mean", ("training", "episode_return_mean")]
    else:
        args.metrics = list(map(literal_eval, args.metrics))
    export_all_runs(
        args.path,
        single=args.single,
        metrics=args.metrics,
        group_by=("pbt_epoch", "pbt_group_key"),
        figsize=tuple(args.figsize),
        format=args.format,
        main_only=Repeat([True, False]) if not args.main_only else True,
        plot_reduced=Repeat([True, False]),
        test=args.test,
        pbt_metric=args.pbt_metric,
        max_workers=args.workers,
        zip_plots=args.zip,
        excludes=args.excludes,
        redo=args.redo,
    )


# Example for grouping by MultiIndex columns:
# group_cols = [("config", "pbt_epoch"), ("config", "pbt_group_key")]
# group_values = [df[col] for col in group_cols]
# group_df = pd.concat(group_values, axis=1)
# group_df.columns = ["pbt_epoch", "pbt_group_key"]
# grouped = df.groupby([group_df["pbt_epoch"], group_df["pbt_group_key"]])
