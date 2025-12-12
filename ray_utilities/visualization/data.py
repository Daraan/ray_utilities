from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, cast, TYPE_CHECKING

import base62
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from tqdm import tqdm
from typing_extensions import Final


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
    # Maybe want to keep but not hashable
    ("config", "fork_from", "parent_time"),
]

LOG_SETTINGS = {"lr"}

nan: Final = float("nan")

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
    for result_file in tqdm(result_files):
        df = pd.read_json(result_file, lines=True)
        # Drop unwanted columns if they exist
        # Flatten nested dict columns and convert to MultiIndex
        # ptimes = df[("config", "fork_from", "parent_time")]
        # mask = ~ptimes.isna()

        # df.loc[mask.values, ("config", "fork_from", "parent_time")] = ptimes[mask.values].map(str).values
        df = pd.json_normalize(df.to_dict(orient="records"), sep="/")
        df.columns = pd.MultiIndex.from_tuples([tuple(col.split("/")) for col in df.columns])
        df = df.sort_index(axis=1).drop(columns=DROP_COLUMNS, errors="ignore")
        # df.columns = pd.MultiIndex.from_tuples(cast("list[tuple[str, ...]]", df.columns))
        experiment_key = df.config.experiment_key.iloc[-1].item()
        assert result_file.name == "result.json" or experiment_key in result_file.name, (
            f"Experiment key {experiment_key} does not match result file name {result_file.name}"
        )
        run_data[experiment_key] = df

    logger.info(f"Loaded data for run {offline_run} with {len(run_data.keys())} experiments")  # noqa: G004
    return run_data


def __base62_sort_key(s: str) -> int:
    if not isinstance(s, str):
        return s
    if s.endswith("Z"):
        return 0
    return base62.decode(s.split("S", 1)[-1])


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

    combined_df = pd.concat(dfs, keys=dataframes.keys(), names=["run_id", "training_iteration"]).sort_index(
        key=_base62_sort_key
    )
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
    continued_runs = which_continued(combined_df)
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

    combined_df.loc[:, ("config", "__pbt_main_branch__")] = main_branch_mask

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
            vmin=values.replace(0, float("nan")).min(),
            vmax=values.max(),
        )
    else:
        norm = mcolors.Normalize(
            vmin=values.min(),
            vmax=values.max(),
        )
    cmap = cm.get_cmap(name)

    # Map group_stat values to colors
    unique_stats = values.unique()
    color_map = {val: mcolors.to_hex(cmap(norm(val))) for val in unique_stats if val > 0}
    # For zero or negative values, fallback to a default color
    for val in unique_stats:
        if log and val <= 0:
            color_map[val] = "#cccccc"
    return color_map, cmap, norm


def which_continued(df: pd.DataFrame) -> pd.DataFrame:
    # there is no continued key we need to check which run_id (in the index) spans over multiple pbt_epochs
    pbt_epochs = df[("config", "pbt_epoch")]
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
    if experiment_keys is None:
        experiment_keys = df.index.get_level_values(0).unique().to_list()
        if TEST:
            experiment_keys = experiment_keys[:5]
    if group_stat is None:
        # assumes a name=... format
        group_stat = df.iloc[0].config[group_by[-1]].str.split("=").iloc[0][0]
        assert group_stat is not None
    log = log if log is not None else group_stat in LOG_SETTINGS
    depth = len(df.columns[0])

    def ifill(*cols):
        return (*cols, *(nan,) * (depth - len(cols)))

    # Secondary x-axis mappers
    first_change: tuple[str, int] = df[ifill("config", "pbt_epoch")].diff().iloc[1:].ne(0).idxmax()  # pyright: ignore[reportAssignmentType]
    perturbation_interval = df.current_step.loc[(first_change[0], first_change[1] - 1)].item()
    secondard_to_main = lambda xs: (xs + 0.5) * perturbation_interval  # noqa: E731
    main_to_secondary = lambda xs: (xs - perturbation_interval / 2) / perturbation_interval  # noqa: E731
    num_pbt_epochs = df[ifill("config", "pbt_epoch")].max().item() + 1

    # Select the group-by columns from MultiIndex columns
    group_cols = [("config", k) for k in group_by]
    # Extract the group-by columns as a DataFrame (each column is 1D)
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

        access_metric = metric if isinstance(metric, str) else ifill(*metric)
        metric_key = metric if isinstance(metric, str) else metric[-1]

        plot_df: pd.DataFrame = pd.concat(
            [
                df[["current_step", metric]]
                if isinstance(metric, str)
                else df[[ifill("current_step"), ifill(*metric)]],
                df[[ifill("config", group_by[0]), ifill("config", "__pbt_main_branch__")]],
                df[("config", group_stat)].round(10),
            ],
            axis=1,
        )
        plot_df.columns = ["current_step", metric_key, group_by[0], "__pbt_main_branch__", group_stat]
        plot_df["__pbt_main_branch__"] = plot_df["__pbt_main_branch__"].fillna(False)  # noqa: FBT003
        # plot_df.sort_values(["training_iteration", "__pbt_main_branch__", metric], inplace=True)
        assert group_stat in plot_df

        color_map, cmap, norm = make_cmap(plot_df[group_stat], log=log)
        # Sort each subgroup by their std from highest to lowest
        # Sort plot_df within each (group_by[0], group_stat) group by the std of the metric (descending)
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

        # Assign ranks for each epoch based on their last iteration metric value
        max_group_steps = plot_df.groupby([group_by[0], group_stat, "current_step"])[metric_key].mean()
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
        plot_df["__group_rank__"] = plot_df[[group_by[0], group_stat]].apply(
            lambda x: group_ranks.loc[(x[group_by[0]], x[group_stat])].item(), axis=1
        )

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
                main_group = group.index.get_level_values("run_id").isin(continued_runs.index).any().item()
                example = group.iloc[0]
                print(
                    "Filtering group: epoch",
                    example[group_by[0]],
                    example[group_stat],
                    "rank",
                    example["__group_rank__"],
                    "Main group:",
                    main_group,
                    "Keep:",
                    main_group or example["__group_rank__"] in ranks_to_keep or example[group_stat] in stats_to_keep,
                )

                def keep_row(row: pd.Series) -> bool:
                    r = row["__group_rank__"]
                    s = row[group_stat]

                    return main_group or r in ranks_to_keep or s in stats_to_keep

                return group[group.apply(keep_row, axis=1)]

            print("Applying group filtering to reduce clutter.... Shape before", plot_df.shape)
            plot_df = plot_df.groupby([group_by[0], group_stat], group_keys=False).apply(filter_groups)
            print("Shape after", plot_df.shape)
        plot_df = plot_df.sort_values(
            ["training_iteration", "__pbt_main_branch__", "__group_sort__", "__group_rank__", metric_key]
        )
        # Iterators:
        max_epoch = plot_df[group_by[0]].max()
        last_data = plot_df[plot_df[group_by[0]] == max_epoch]
        # TODO: What about those runs that were trained too long
        last_data = last_data.sort_values(["training_iteration", "__group_rank__", "__group_sort__", metric_key])
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
        for i, ((pbt_epoch, stat_val), group) in enumerate(grouper):
            # Add background shade when group changes
            if pbt_epoch == max_epoch:
                (pbt_epoch, stat_val), group = next(last_group_iter)  # noqa: PLW2901

            if pbt_epoch != last_epoch:
                last_epoch = pbt_epoch
            if TEST and pbt_epoch > 1:
                break
            is_main = False
            if group["__pbt_main_branch__"].any() or i == len(grouper) - 1:
                is_main = True
                if last_main_group is not None:
                    # If the last group correctly has 3 points then this is correct, if due to some crsh
                    # Add connection to last points
                    group = _connect_groups(last_main_group, group, group_stat)  # noqa: PLW2901
                print("Main group:", pbt_epoch, stat_val)
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
                print("Skipping non-main group:", pbt_epoch, stat_val)
                continue
            elif last_main_group is not None:
                print("Plotting non-main group:", pbt_epoch, stat_val)
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
                label=str(stat_val) if pbt_epoch == 0 else None,
                linewidth=3,
            )
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

        ax.set_title(f"Metric: {metric_key}")
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
    metrics: Sequence[str],
    save_path: str | Path,
    experiment_keys: list[str] | None = None,
    group_stat: str | None = None,
    group_by=("pbt_epoch", "pbt_group_key"),
    *,
    figsize: tuple[int, int] = (12, 8),
    log: bool | None = None,
    format="pdf",
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


def export_run_data(
    experiment_path: str | Path,
    experiment_keys: list[str] | None = None,
    metrics: Sequence[str] = ("episode_reward_mean",),
    group_stat: str | None = None,
    group_by=("pbt_epoch", "pbt_group_key"),
    *,
    figsize: tuple[int, int] = (12, 8),
    log: bool | None = None,
    format="pdf",
    save_path: str | Path | None = None,
):
    experiment_path = Path(experiment_path)
    if save_path is None:
        save_path = experiment_path / "plots"
    else:
        save_path = Path(save_path)
    print("Loading run data...", experiment_path.name)
    data = load_run_data(experiment_path)
    print("Combining dataframes...")
    combined_df = combine_df(data)
    print("Plotting and saving...")
    metric_str = "_".join(metrics)
    plot_n_save(
        combined_df,
        metrics=metrics,
        save_path=save_path / f"{experiment_path.name}_{metric_str}.{format}",
        experiment_keys=experiment_keys,
        group_stat=group_stat,
        group_by=group_by,
        figsize=figsize,
        log=log,
        format=format,
    )
    return save_path


# Example for grouping by MultiIndex columns:
# group_cols = [("config", "pbt_epoch"), ("config", "pbt_group_key")]
# group_values = [df[col] for col in group_cols]
# group_df = pd.concat(group_values, axis=1)
# group_df.columns = ["pbt_epoch", "pbt_group_key"]
# grouped = df.groupby([group_df["pbt_epoch"], group_df["pbt_group_key"]])
